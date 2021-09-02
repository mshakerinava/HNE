import os
import io
import sys
import json
import math
import time
import random
import argparse
import subprocess
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
from utils import hash_args
from link_prediction import lp_evaluate
from node_classification import nc_evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--num-epochs', type=int, default=50)
parser.add_argument('--embedding-dim', type=int, default=50)
parser.add_argument('--sched-step-size', type=int, default=20)
parser.add_argument('--sched-gamma', type=float, default=0.5)
parser.add_argument('--weight-decay', type=float, default=0)
parser.add_argument('--hinge-thresh', type=float, default=5)
parser.add_argument('--barrier-coef', type=float, default=1)
parser.add_argument('--dropout-rate', type=float, default=0)
parser.add_argument('--learning-rate', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, required=True, choices=['DBLP', 'PubMed', 'Freebase', 'Yelp'])
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--barrier-type', type=str, default='log', choices=['log', 'inv', 'id'])
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--cosine-sim', action='store_true')
parser.add_argument('--conformal-map', action='store_true')
parser.add_argument('--sparse-grads', action='store_true')
args = parser.parse_args()

args_hash = hash_args(vars(args), no_hash=['dataset', 'seed', 'sparse_grads', 'overwrite'])
TAG = '%s__args-%s__seed-%02d' % (args.dataset, args_hash, args.seed)
FINISH_TEXT = '** finished successfully **'


#----------- logging -----------#
try:
    LOG_STR.close()
    LOG_FILE.close()
except:
    pass

LOGS_PATH = 'logs'
os.makedirs(LOGS_PATH, exist_ok=True)
LOG_PATH = os.path.join(LOGS_PATH, TAG + '.txt')

abort = False
try:
    if not args.overwrite and subprocess.check_output(['tail', '-n', '1', LOG_PATH]).decode('utf-8').strip() == FINISH_TEXT:
        print('ABORT: experiment %s has already been performed' % TAG)
        abort = True
except:
    pass

if abort:
    sys.exit(-1)

LOG_STR = io.StringIO()
LOG_FILE = open(LOG_PATH, 'w')

try:
    old_print
except NameError:
    old_print = print

def print(*args, **kwargs):
    kwargs['flush'] = True
    old_print(*args, **kwargs)
    kwargs['file'] = LOG_STR
    old_print(*args, **kwargs)
    kwargs['file'] = LOG_FILE
    old_print(*args, **kwargs)
#-------------------------------#


print(datetime.now())
print('writing log to `%s`' % LOG_PATH)
commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
print('commit %s' % commit_hash)

print('───────────── machine info ─────────────')
print(subprocess.check_output(['uname', '-a']).decode('utf-8').strip())
print(subprocess.check_output(['lscpu']).decode('utf-8').strip())
print(subprocess.check_output(['nvidia-smi']).decode('utf-8').strip())
print('────────────────────────────────────────')

print(subprocess.check_output(['pip', 'freeze']).decode('utf-8').strip())
print('args = %s' % json.dumps(vars(args), sort_keys=True, indent=4))


def set_seed(seed):
    random.seed(seed, version=2)
    np.random.seed(random.randint(0, 2**32))
    torch.manual_seed(random.randint(0, 2**32))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(args.seed)

DATASET_PATH = os.path.join('data', args.dataset)
NODE_PATH = os.path.join(DATASET_PATH, 'node.dat')
LINK_PATH = os.path.join(DATASET_PATH, 'link.dat')

node_id_to_type = {}
with tqdm(desc='reading %s' % NODE_PATH, total=os.path.getsize(NODE_PATH)) as pbar:
    with open(NODE_PATH, 'r') as f:
        for line in f:
            pbar.update(len(line))
            node_id, node_type = [int(x) for x in line.strip().split()]
            node_id_to_type[node_id] = node_type

links = {}
with tqdm(desc='reading %s' % LINK_PATH, total=os.path.getsize(LINK_PATH)) as pbar:
    with open(LINK_PATH, 'r') as f:
        for line in f:
            pbar.update(len(line))
            node_id_1, node_id_2, link_type = [int(x) for x in line.strip().split()]
            node_type_1 = node_id_to_type[node_id_1]
            node_type_2 = node_id_to_type[node_id_2]
            if node_type_1 > node_type_2:
                node_id_1, node_id_2 = node_id_2, node_id_1
            if link_type not in links:
                links[link_type] = []
            links[link_type].append([node_id_1, node_id_2])

for key, value in links.items():
    links[key] = np.array(value, dtype=int)

num_nodes = len(node_id_to_type)
num_links = sum([len(x) for x in links.values()])

batches = sum([[x] * max(1, len(links[x]) // args.batch_size) for x in links], [])

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

emb = nn.Embedding(num_nodes, args.embedding_dim, sparse=args.sparse_grads).to(DEVICE)
dropout_layer = nn.Dropout(p=args.dropout_rate)
emb_dropout = nn.Sequential(emb, dropout_layer)

opt = eval('optim.%s' % args.optimizer)(emb.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
sched = optim.lr_scheduler.StepLR(opt, step_size=args.sched_step_size, gamma=args.sched_gamma)


#----------- loss function -----------#
def cdist_mean(x1, x2, p=2.0, *args, **kwargs):
    dim = x1.shape[1]
    return torch.cdist(x1, x2, p, *args, **kwargs) * (dim ** (-1 / p))


def loss_fn(enc, x_list, barrier_type, hinge_thresh, cosine_sim=False, conformal_map=False, decompositions=1):
    num_actions = len(x_list)
    assert num_actions >= 2
    x_list = [torch.tensor(x, dtype=torch.long, device=DEVICE) for x in x_list]
    z_list = [enc(x) for x in x_list]
    code_size = z_list[0].shape[1]

    #-- symmetry loss --#
    subcode_size = code_size // decompositions
    loss_equiv = 0
    for k in range(decompositions):
        h_list = [z[:, k * subcode_size: (k + 1) * subcode_size] for z in z_list]

        # NOTE: conformal mapping will probably not work with decompositions
        if conformal_map:
            h_list = [(h[:, None, :] - h[None, :, :]).view(-1, subcode_size) for h in h_list]

        if cosine_sim or conformal_map:
            D_list = [1.0 - F.cosine_similarity(h[:, None, :], h[None, :, :], dim=2) for h in h_list]
        else:
            D_list = [cdist_mean(h, h, p=2) for h in h_list]
            # D_list = [D / torch.linalg.norm(D.view(-1)) for D in D_list] # normalize distances

        L_equiv = torch.zeros(num_actions, num_actions, device=DEVICE)
        for i in range(num_actions):
            for j in range(i + 1, num_actions):
                L_equiv[i, j] = torch.mean((D_list[i] - D_list[j]) ** 2)
                # L_equiv[i, j] = torch.mean(((D_list[i] - D_list[j]) / (D_list[i] + D_list[j] + 1e-3)) ** 2)
        cur_loss_equiv = torch.sum(L_equiv) / (num_actions * (num_actions - 1) / 2)
        # loss_equiv += cur_loss_equiv ** 2 / decompositions
        loss_equiv += cur_loss_equiv / decompositions

    #-- barrier loss --#
    z_all = torch.cat(z_list, dim=0)
    D_all = cdist_mean(z_all, z_all, p=2)
    mask = torch.eye(D_all.shape[0], dtype=torch.bool, device=DEVICE)

    use_hinge_loss = (hinge_thresh is not None)
    if not use_hinge_loss:
        hinge_thresh = 1 # just some random number

    if barrier_type == 'log':
        B_all = -torch.log(D_all[~mask] + 1e-9)
        B_min = -math.log(hinge_thresh)
    elif barrier_type == 'inv':
        B_all = 1.0 / (D_all[~mask] + 1e-9)
        B_min = 1.0 / hinge_thresh
    elif barrier_type == 'id':
        B_all = -D_all[~mask]
        B_min = -hinge_thresh
    else:
        assert False, 'Unknown `barrier_type`'

    if use_hinge_loss:
        loss_barrier = torch.mean(torch.maximum(torch.zeros(1, device=DEVICE), B_all - B_min))
    else:
        loss_barrier = torch.mean(B_all) # TODO: improve?

    return loss_equiv, loss_barrier
#-------------------------------------#


def get_weights_norm(parameters, norm_type=2.0):
    with torch.no_grad():
        return torch.norm(torch.stack([torch.norm(p, norm_type) for p in parameters]), norm_type).item()


def get_grads_norm(parameters, norm_type=2.0):
    with torch.no_grad():
        return torch.norm(torch.stack([torch.norm(p.grad, norm_type) for p in parameters]), norm_type).item()


CHECKPOINTS_PATH = 'checkpoints'
os.makedirs(CHECKPOINTS_PATH, exist_ok=True)

CHECKPOINT_FILE = os.path.join(CHECKPOINTS_PATH, '%s.tar' % TAG)
if os.path.isfile(CHECKPOINT_FILE):
    checkpoint = torch.load(CHECKPOINT_FILE)
    emb.load_state_dict(checkpoint['emb_state_dict'])
    opt.load_state_dict(checkpoint['opt_state_dict'])
    sched.load_state_dict(checkpoint['sched_state_dict'])
    t = checkpoint['t']
    log_lines = checkpoint['log_str'].split('\n')
    for (i, line) in enumerate(log_lines):
        if 'Training started' in line:
            print('\n'.join(log_lines[i:]), end='')
            break
    if t != args.num_epochs:
        print('[%s] Training resumed' % datetime.now())
else:
    print('[%s] Training started' % datetime.now())
    print('Training for %d epochs...' % args.num_epochs)
    t = 0

MODELS_PATH = 'saved_models'
os.makedirs(MODELS_PATH, exist_ok=True)

avg_loss_list = []
avg_loss_equiv_list = []
avg_loss_barrier_list = []

while t != args.num_epochs:
    set_seed(1000 * (t + 1) + args.seed)
    links_idx = {}
    for k, v in links.items():
        links_idx[k] = np.arange(len(v))
        np.random.shuffle(links_idx[k])
    batches_idx = np.arange(len(batches))
    np.random.shuffle(batches_idx)

    idx_list = [0] * len(links)

    loss_list = []
    loss_equiv_list = []
    loss_barrier_list = []

    time_start = time.time()
    progress = tqdm(
        range(len(batches)),
        desc='Loss: None | Loss Equiv: None | Loss Barrier: None | L2 Weights: %12g | L2 Grads: 0' % (get_weights_norm(emb.parameters(), norm_type=2.0))
    )
    for i in progress:
        link_type = batches[batches_idx[i]]
        idx = idx_list[link_type]
        idx_ = min(len(links[link_type]), idx + args.batch_size)
        idx_list[link_type] = idx_
        batch = links[link_type][links_idx[link_type][idx: idx_]]

        x_list = [batch[:, 0], batch[:, 1]]
        loss_equiv, loss_barrier = loss_fn(emb_dropout, x_list, args.barrier_type, args.hinge_thresh, args.cosine_sim, args.conformal_map)
        loss = loss_equiv + args.barrier_coef * loss_barrier
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_list.append(loss.item())
        loss_equiv_list.append(loss_equiv.item())
        loss_barrier_list.append(args.barrier_coef * loss_barrier.item())
        progress.set_description('Loss: %12g | Loss Equiv: %12g | Loss barrier: %12g | L2 Weights: %12g | L2 Grads: %12g' % (
            loss_list[-1],
            loss_equiv_list[-1],
            loss_barrier_list[-1],
            get_weights_norm(emb.parameters(), norm_type=2.0),
            get_grads_norm(emb.parameters(), norm_type=2.0)
        ))
    time_end = time.time()

    torch.save(emb.state_dict(), os.path.join(MODELS_PATH, '%s__W.tar' % TAG))

    avg_loss = np.mean(loss_list)
    avg_loss_equiv = np.mean(loss_equiv_list)
    avg_loss_barrier = np.mean(loss_barrier_list)
    avg_loss_list.append(avg_loss)
    avg_loss_equiv_list.append(avg_loss_equiv)
    avg_loss_barrier_list.append(avg_loss_barrier)
    print('Epoch %3d | Loss: %12g | Loss Equiv: %12g | Loss Barrier: %12g | Time: %6.1f sec' % (
        t + 1, avg_loss, avg_loss_equiv, avg_loss_barrier, time_end - time_start))

    # sched.step(avg_loss)
    sched.step()
    t += 1
    torch.save({
        'emb_state_dict': emb.state_dict(),
        'opt_state_dict': opt.state_dict(),
        'sched_state_dict': sched.state_dict(),
        't': t,
        'log_str': LOG_STR.getvalue()
    }, CHECKPOINT_FILE)

print('[%s] Training finished' % datetime.now())

print('[%s] Evaluation started' % datetime.now())
emb_np = emb.detach().to('cpu').numpy()
emb_dict = {str(i): emb_np[i] for i in range(len(emb_np))}
DATA_PATH = '../../Data'

LABEL_FILE_PATH = os.path.join(DATA_PATH, args.dataset, 'label.dat')
LABEL_TEST_PATH = os.path.join(DATA_PATH, args.dataset, 'label.dat.test')
scores = nc_evaluate(args.dataset, args.supervised, LABEL_FILE_PATH, LABEL_TEST_PATH, emb_dict)
print('----- Node Classification -----')
print('Macro-F1/Micro-F1: %5.2f/%5.2f' % (100 * scores[0], 100 * scores[1]))
print('-------------------------------')

LINK_TEST_PATH = os.path.join(DATA_PATH, args.dataset, 'link.dat.test')
scores = lp_evaluate(LINK_TEST_PATH, emb_dict)
print('------- Link Prediction -------')
print('AUC/MRR: %5.2f/%5.2f' % (100 * scores[0], 100 * scores[1]))
print('-------------------------------')

print('[%s] Evaluation finished' % datetime.now())

# EMB_PATH = 'emb'
# os.makedirs(EMB_PATH, exist_ok=True)
# OUTPUT_PATH = os.path.join(EMB_PATH, '%s.dat' % TAG)
# with open(OUTPUT_PATH, 'w') as f:
#     W = emb.weight
#     f.write('[%s] %s\n' % (str(datetime.now()), OUTPUT_PATH))
#     for i in tqdm(range(W.shape[0]), desc='writing to %s' % OUTPUT_PATH):
#         f.write('%d\t' % i)
#         for j in range(W.shape[1]):
#             f.write('%f ' % W[i, j].item())
#         f.write('\n')

print(FINISH_TEXT)

LOG_STR.close()
LOG_FILE.close()
