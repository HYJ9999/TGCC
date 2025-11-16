import sys
import os
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
import numpy as np
import random
import time
import argparse
import torch
from util_tools.utils import *
import torch.nn.functional as F
from tester_other_arcs import Evaluator
from util_tools.utils_graph import DataGraph


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument("--method", type=str, default="CasualGC", help="Method")
parser.add_argument("--carch", type=str, default="15")
parser.add_argument('--dataset', type=str, default='flickr') #citeseer flickr reddit ogbn-arxiv
parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
parser.add_argument("--save_dir", type=str, default="saved_ours", help="synthetic dataset directory")
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--keep_ratio', type=float, default=1)
parser.add_argument('--reduction_rate', type=float, default=0.25)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--mlp', type=int, default=0)
parser.add_argument('--inner', type=int, default=0)
parser.add_argument('--epsilon', type=float, default=-1)
parser.add_argument('--nruns', type=int, default=3) #5
parser.add_argument('--expID', type=int, default=1)
parser.add_argument('--generate_adj', type=int, default=1, help='generate the condensed graph')


parser.add_argument('--log_dir', type=str, default='logs', help='log directory') 
parser.add_argument('--marks', type=str, default='none', help='') 
parser.add_argument('--verbose', type=bool, default=True, help='whether to print detailed logs') 

args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)


path = os.path.join(args.log_dir,"nc",args.dataset)
path = os.path.join(path,args.method)

if not os.path.exists(path):
    os.makedirs(path)

logging.basicConfig(
    filename=f"{path}/{args.dataset}-{args.reduction_rate}-NC-{args.seed}-{args.marks}.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
if args.verbose:
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if args.dataset in ['cora', 'citeseer']:
    args.epsilon = 0.05
else:
    args.epsilon = 0.01

print(args)

data_pyg = ["cora", "citeseer", "pubmed", 'cornell', 'texas', 'wisconsin', 'chameleon', 'squirrel']
if args.dataset in data_pyg:
    data_full = get_dataset(args.dataset, args.normalize_features, args.data_dir)
    data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)
else:
    data = DataGraph(args.dataset, data_dir=args.data_dir)
    data_full = data.data_full

agent = Evaluator(data, args, device='cuda')
agent.train()