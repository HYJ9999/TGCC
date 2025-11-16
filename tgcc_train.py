import numpy as np
import argparse

# import optuna as optuna

from aug import random_aug

import torch.nn.functional as F
import numpy as np
import torch as th
import torch.nn as nn
import warnings
import dgl
from sklearn.metrics import f1_score
import scipy.sparse as sp
import torch
import csv
import time
import pandas as pd
import os
import logging

from deeprobust.graph.data import Dataset
import numpy as np
import random
import time
import argparse
import torch
from utils import *
import torch.nn.functional as F
from utils_graphsaint import DataGraphSAINT
import os
import logging

import json
import argparse
from collections import Counter
import numpy as np
import torch

from utils import *
from utils_eigsh import * #get_largest_cc,load_eigen


import trace
tracer = trace.Trace(count=False, trace=True)


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default="citeseer")
parser.add_argument("--config", type=str, default="configs/config.json", help="Path to the config JSON file")
parser.add_argument("--section", type=str, default='runed exps name', help="the experiments needs to run")    
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=40)

parser.add_argument('--lam', type=float, default=0.1) 
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--scope_flag', type=int, default=1)

parser.add_argument('--grad_match', type=bool, default=True, help='whether to use gradient matching')
parser.add_argument('--dis_metric', type=str, default='ours')
parser.add_argument('--epochs', type=int, default=600)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--lr_adj', type=float, default=1e-4)
parser.add_argument('--lr_feat', type=float, default=1e-4)
parser.add_argument('--lr_model', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--reduction_rate', type=float, default=0.25)  
parser.add_argument('--alpha', type=float, default=0.001, help='regularization term.')
parser.add_argument('--beta', type=float, default=0.01, help='casual regularization term.') 
parser.add_argument('--use_sinkhorn', type=bool, default=False, help='use_sinkhorn')
parser.add_argument('--adj_alpha', type=float, default=0.01, help='adj L regularization term.')
parser.add_argument('--aug_alpha', type=float, default=0.01, help='aug graph grad match.')
parser.add_argument("--no_casual",type=bool,default=False)
parser.add_argument("--method",type=str,default="TGCC")
parser.add_argument("--carch", type=str, default="15")
parser.add_argument('--save_dir', type=str, default='saved_ours', help='log directory') 
parser.add_argument('--c_alpha', type=float, default=0.9, help='regularization term.')
parser.add_argument('--c_beta', type=float, default=0.015, help='regularization term.')
parser.add_argument('--c_gamma', type=float, default=0.00085, help='regularization term.')


parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--sgc', type=int, default=1)
parser.add_argument('--inner', type=int, default=0)
parser.add_argument('--outer', type=int, default=20)
parser.add_argument('--option', type=int, default=0)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--label_rate', type=float, default=1)
parser.add_argument('--one_step', type=int, default=0)
parser.add_argument('--marks', type=str, default="a-g-low0-l")

parser.add_argument('--log_dir', type=str, default='logs', help='log directory') 
parser.add_argument('--verbose', type=bool, default=True, help='whether to print detailed logs')

args = parser.parse_args()


with open(args.config, "r") as config_file:
    config = json.load(config_file)

if args.section in config:
    section_config = config[args.section]

for key, value in section_config.items():
    setattr(args, key, value)

save_path=os.path.join(args.save_dir,args.method) 
if not os.path.exists(save_path):
    os.makedirs(save_path)

if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

# torch.cuda.set_device(args.gpu)

start_time = time.time()
warnings.filterwarnings('ignore')



def main():

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    logging.basicConfig(
        filename=f"{args.log_dir}/{args.dataset}-{args.reduction_rate}-{args.seed}-{args.marks}.log",
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    if args.verbose:
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%m-%d %H:%M:%S')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
    
    logging.info(args)

    data_graphsaint = ['flickr', 'reddit','ogbn-arxiv','finance']
    if args.dataset in data_graphsaint:
        data = DataGraphSAINT(args.dataset)
        data = DataGraphSAINT(args.dataset, label_rate=args.label_rate)
        data_full = data.data_full
    else:
        data_full = get_dataset(args.dataset, args.normalize_features)
        data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)
    
    if args.dataset in ['reddit']:
        from tgcc_agent_induct import TGCC
    else:
        from tgcc_agent_transduct import TGCC



    agent = TGCC(data,args,device='cuda') 
    logging.info(args)
    acc_test,acc_std= agent.train()
    
    # config = vars(args)
    keep_keys = {'dataset','reduction_rate','carch','seed','beta','aug_alpha','no_casual','sgc','inner','outer', 'lr_adj','lr_feat','epochs', 'c_alpha', 'c_beta', 'c_gamma','method','marks'}

    config = {k: v for k, v in vars(args).items() if k in keep_keys}
   
    config['acc_test'] = acc_test
    config['acc_std'] = acc_std
    
    import csv
    result_path="./result/own/"+str(args.dataset)
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open("./result/own/"+str(args.dataset)+"/key_table.csv", 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=config.keys())
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(config)
    print(config)


    end_time = time.time()

    run_time = end_time - start_time

    logging.info(f"cond time：{run_time} s")
    

if __name__ == '__main__':
    main()
    
    # tracer.runfunc(main)
    # tracer.results().write_results(show_missing=False, coverdir='logs/trace_log')

