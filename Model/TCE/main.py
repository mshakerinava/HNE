import os
import sys
import json
import math
import time
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils import hash_args


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--num-epochs', type=int, default=100)
parser.add_argument('--embedding-dim', type=int, default=200)
parser.add_argument('--weight-decay', type=float, default=1e-5)
parser.add_argument('--hinge-thresh', type=float, default=5)
parser.add_argument('--barrier-coef', type=float, default=1)
parser.add_argument('--learning-rate', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, choices=['DBLP', 'PubMed', 'Freebase', 'Yelp'])
parser.add_argument('--barrier-type', type=str, default='log', choices=['log', 'inv', 'id'])
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--cosine-sim', action='store_true')
parser.add_argument('--conformal-map', action='store_true')
args = parser.parse_args()

args_hash = hash_args(vars(args), no_hash=['dataset', 'seed'])
TAG = '%s__args-%s__seed-%02d' % (args.dataset, args_hash, args.seed)
FINISH_TEXT = '** finished successfully **'


#----------- logging -----------#
try:
    LOG_STR.close()
    LOG_FILE.close()
except:
    pass

os.makedirs('logs', exist_ok=True)
LOG_PATH = os.path.join('logs', TAG + '.txt')

abort = False
try:
    if not args.overwrite and subprocess.check_output(['tail', '-n', '1', LOG_PATH]).decode('utf-8').strip() == FINISH_TEXT:
        print('ABORT: experiment has already been performed')
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
with open(NODE_PATH, 'r') as f:
    for line in f.readlines():
        node_id, node_type = [int(x) for x in line.strip().split()]
        node_id_to_type[node_id] = node_type

links = []
with open(LINK_PATH, 'r') as f:
    for line in f.readlines():
        node_id_1, node_id_2, link_type = [int(x) for x in line.strip().split()]
        node_type_1 = node_id_to_type[node_id_1]
        node_type_2 = node_id_to_type[node_id_2]
        if node_type_1 > node_type_2:
            node_id_1, node_id_2 = node_id_2, node_id_1
        links.append([link_type, node_id_1, node_id_2])

links = np.array(links, dtype=int)

num_nodes = len(node_id_to_type)
num_links = len(links)

assert args.batch_size <= num_links

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)

W = torch.randn(num_nodes, args.embedding_dim, device=DEVICE)

opt = optim.Adam(W, lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(opt, step_size=9999, gamma=0.5, verbose=False) # does nothing for now

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

        L_equiv = torch.zeros(num_actions, num_actions)
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
    mask = torch.eye(D_all.shape[0], dtype=torch.bool)

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
        loss_barrier = torch.mean(torch.maximum(torch.zeros(1, device=device), B_all - B_min))
    else:
        loss_barrier = torch.mean(B_all) # TODO: improve?

    return loss_equiv, loss_barrier
#-------------------------------------#


def get_weights_norm(W, norm_type=2.0):
    with torch.no_grad():
        return torch.linalg.norm(W, ord=norm_type, dim=1).mean().item()


def get_grads_norm(W, norm_type=2.0):
    with torch.no_grad():
        return torch.linalg.norm(W.grad, ord=norm_type, dim=1).mean().item()


enc = lambda x: W[x]
os.makedirs('saved_models', exist_ok=True)

avg_loss_list = []
avg_loss_equiv_list = []
avg_loss_barrier_list = []

for t in args.num_epochs:
    np.random.shuffle(links)
    loss_list = []
    loss_equiv_list = []
    loss_barrier_list = []

    time_start = time.time()
    progress = tqdm(
        range(0, num_links - args.batch_size + 1, args.batch_size),
        desc='Loss: None | Loss Equiv: None | Loss Barrier: None | L2 Weights: %12g | L2 Grads: 0' % (get_weights_norm(W, norm_type=2.0))
    )
    for _ in progress:
        batch = links[i: i + args.batch_size]
        transform_list = [[[], []] for _ in range(num_links)]
        for x in batch:
            transform_list[x[0]][0].append(x[1])
            transform_list[x[0]][1].append(x[2])
        loss_equiv = []
        loss_barrier = []
        for x_list in transform_list:
            if len(x_list[0]) >= 2:
                loss_equiv_cur, loss_barrier_cur = loss_fn(enc, x_list, args.barrier_type, args.hinge_thresh, args.cosine_sim, args.conformal_map)
                loss_equiv.append(loss_equiv_cur)
                loss_barrier.append(loss_barrier_cur)
        loss_equiv = torch.mean(loss_equiv)
        loss_barrier = torch.mean(loss_barrier)
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
            get_weights_norm(W, norm_type=2.0),
            get_grads_norm(W, norm_type=2.0)
        ))
    time_end = time.time()

    torch.save(W, os.path.join('saved_models', '%s__W.tar' % TAG))

    avg_loss = np.mean(loss_list)
    avg_loss_equiv = np.mean(loss_equiv_list)
    avg_loss_barrier = np.mean(loss_barrier_list)
    avg_loss_list.append(avg_loss)
    avg_loss_equiv_list.append(avg_loss_equiv)
    avg_loss_barrier_list.append(avg_loss_barrier)
    print('\nEpoch %3d | Loss: %12g | Loss Equiv: %12g | Loss Barrier: %12g | Time: %6.1f sec' % (
        t + 1, avg_loss, avg_loss_equiv, avg_loss_barrier, time_end - time_start))

    # scheduler.step(avg_loss)
    scheduler.step()

os.makedirs('emb', exist_ok=True)
OUTPUT_PATH = os.path.join('emb', '%s.dat' % TAG)
with open(OUTPUT_PATH, 'w') as f:
    f.write('[', datetime.now(), '] ', OUTPUT_PATH, '\n')
    for i in range(num_nodes):
        f.write(i, '\t')
        for j in range(args.embedding_dim):
            f.write('%f ' % W[i, j].item())

print(FINISH_TEXT)

LOG_STR.close()
LOG_FILE.close()
