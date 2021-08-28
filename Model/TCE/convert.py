import argparse
from datetime import datetime
import torch
import torch.nn as nn
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--num-nodes', type=int, required=True)
parser.add_argument('--embedding-dim', type=int, required=True)
args = parser.parse_args()

emb = nn.Embedding(args.num_nodes, args.embedding_dim)
emb.load_state_dict(torch.load(args.input))

with open(args.output, 'w') as f:
    W = emb.weight
    f.write('[%s] %s\n' % (str(datetime.now()), args.output))
    for i in tqdm(range(W.shape[0]), desc='writing to %s' % args.output):
        f.write('%d\t' % i)
        for j in range(W.shape[1]):
            f.write('%f ' % W[i, j].item())
        f.write('\n')
