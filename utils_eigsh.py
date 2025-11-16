import random
import os
import numpy as np
import math
from collections import Counter

import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from deeprobust.graph.utils import sparse_mx_to_torch_sparse_tensor
from torch_sparse import SparseTensor


import os
import json
from typing import Any
import networkx as nx
# import gdown

import pandas as pd
import numpy as np
from numpy.linalg import eigh
from scipy.sparse.linalg import eigsh

import scipy.sparse as sp

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Actor, WikipediaNetwork
import torch_geometric.transforms as T
from torch_geometric.utils import convert, to_undirected

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# from utils import normalize_adj

def generate_high_freq_base(dataset,adj,label_rate=1):
    
    #Laplacian：L = I - D^{-1/2} A D^{-1/2}
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    a_norm = D_inv_sqrt @ adj @ D_inv_sqrt
    lap = sp.eye(adj.shape[0]) - a_norm

    # Step: （largest eigenvalues）
    print(f"Computing largest eigenvectors for dataset {dataset}...")
    va_high, ve_high = eigsh(lap, k=1000, which='LA',tol=1e-5)
    idx = np.argsort(va_high)  
    va_high = va_high[idx]
    ve_high = ve_high[:, idx]

    os.makedirs(f"data/{dataset}", exist_ok=True)
    np.save(f"data/{dataset}/va_tr{label_rate}{dataset}.npy", va_high)
    np.save(f"data/{dataset}/ve_tr{label_rate}{dataset}.npy", ve_high)

    print("Constructing sparse similarity matrix...")
    threshold = 1e-2
    chunk = 10000
    rows, cols, vals = [], [], []

    for i in range(0, ve_high.shape[0], chunk):
        vi = ve_high[i:i + chunk]  # chunk x 1000
        sim = vi @ ve_high.T       # (chunk x N)
        sim[np.abs(sim) < threshold] = 0  
        coo = sp.coo_matrix(sim)
        rows.extend(coo.row + i)
        cols.extend(coo.col)
        vals.extend(coo.data)

    high_base_sparse = sp.coo_matrix((vals, (rows, cols)), shape=(ve_high.shape[0], ve_high.shape[0]))
    sp.save_npz(f"data/{dataset}/{dataset}tr{label_rate}low0.npz", high_base_sparse)
    print("High frequency base matrix saved.")


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_params(module):
    if isinstance(module, nn.Linear):
        stdv = 1.0 / math.sqrt(module.weight.size(1))
        module.weight.data.uniform_(-stdv, stdv)
        if module.bias is not None:
            module.bias.data.uniform_(-stdv, stdv)
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def normalize_features(mx):
     rowsum = mx.sum(1)
     r_inv = torch.pow(rowsum, -1)
     r_inv[torch.isinf(r_inv)] = 0.
     r_mat_inv = torch.diag(r_inv)
     mx = r_mat_inv @ mx
     return mx


def normalize_adj(mx):
    """Normalize sparse adjacency matrix,
    A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2
    """
    if type(mx) is not sp.lil.lil_matrix:
        mx = mx.tolil()
    mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(r_mat_inv)
    
    return mx


def normalize_adj_to_sparse_tensor(mx):
    mx = normalize_adj(mx)
    mx = sparse_mx_to_torch_sparse_tensor(mx)
    sparsetensor = SparseTensor(row=mx._indices()[0], col=mx._indices()[1], value=mx._values(), sparse_sizes=mx.size()).cuda()
    return sparsetensor


def get_syn_eigen(real_eigenvals, real_eigenvecs, eigen_k, ratio, step=1):
    k1 = math.ceil(eigen_k * ratio)
    k2 = eigen_k - k1
    print("k1:", k1, ",", "k2:", k2)
    k1_end = (k1 - 1) * step + 1
    eigen_sum = real_eigenvals.shape[0]
    k2_end = eigen_sum - (k2 - 1) * step - 1
    k1_list = range(0, k1_end, step)
    k2_list = range(k2_end, eigen_sum, step)
    eigenvals = torch.cat(
        [real_eigenvals[k1_list], real_eigenvals[k2_list]]
    )
    eigenvecs = torch.cat(
        [real_eigenvecs[:, k1_list], real_eigenvecs[:, k2_list]], dim=1,
    )
    
    return eigenvals, eigenvecs


def get_subspace_embed(eigenvecs, x):
    x_trans = eigenvecs.T @ x  # kd
    u_unsqueeze = (eigenvecs.T).unsqueeze(2) # kn1
    x_trans_unsqueeze = x_trans.unsqueeze(1) # k1d
    sub_embed = torch.bmm(u_unsqueeze, x_trans_unsqueeze)  # kn1 @ k1d = knd
    return x_trans, sub_embed


def get_subspace_covariance_matrix(eigenvecs, x):
    x_trans = eigenvecs.T @ x  # kd
    x_trans = F.normalize(input=x_trans, p=2, dim=1)
    x_trans_unsqueeze = x_trans.unsqueeze(1)  # k1d
    co_matrix = torch.bmm(x_trans_unsqueeze.permute(0, 2, 1), x_trans_unsqueeze)  # kd1 @ k1d = kdd
    return co_matrix

  
def get_embed_sum(eigenvals, eigenvecs, x):
    x_trans = eigenvecs.T @ x  # kd
    x_trans = torch.diag(1 - eigenvals) @ x_trans # kd
    embed_sum = eigenvecs @ x_trans # nk @ kd = nd
    return embed_sum


def get_embed_mean(embed_sum, label):
    class_matrix = F.one_hot(label).float()  # nc
    class_matrix = class_matrix.T  # cn
    embed_sum = class_matrix @ embed_sum  # cd
    mean_weight = (1 / class_matrix.sum(1)).unsqueeze(-1)  # c1
    embed_mean = mean_weight * embed_sum
    embed_mean = F.normalize(input=embed_mean, p=2, dim=1)
    return embed_mean


def get_train_lcc(idx_lcc, idx_train, y_full, num_nodes, num_classes):
    idx_train_lcc = list(set(idx_train).intersection(set(idx_lcc)))
    y_full = y_full.cpu().numpy()
    if len(idx_lcc) == num_nodes:
        idx_map = idx_train
    else:
        y_train = y_full[idx_train]
        y_train_lcc = y_full[idx_train_lcc]

        y_lcc_idx = list((set(range(num_nodes)) - set(idx_train)).intersection(set(idx_lcc)))
        y_lcc_ = y_full[y_lcc_idx]
        counter_train = Counter(y_train)
        counter_train_lcc = Counter(y_train_lcc)
        idx = np.arange(len(y_lcc_))
        for c in range(num_classes):
            num_c = counter_train[c] - counter_train_lcc[c]
            if num_c > 0:
                idx_c = list(idx[y_lcc_ == c])
                idx_c = np.array(y_lcc_idx)[idx_c]
                idx_train_lcc += list(np.random.permutation(idx_c)[:num_c])
        idx_map = [idx_lcc.index(i) for i in idx_train_lcc]
                        
    return idx_train_lcc, idx_map

def get_largest_cc(adj, num_nodes, data_name):
    mx = (adj).tocoo()
    edge_index = torch.LongTensor(np.vstack([mx.row, mx.col]))
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    nx_graph = convert.to_networkx(data, to_undirected=True)
    
    largest_cc = max(nx.connected_components(nx_graph), key=len)
    idx_lcc = list(largest_cc)
    largest_cc_graph = nx_graph.subgraph(largest_cc)
    
    if len(idx_lcc) == num_nodes:
        adj_lcc = adj
        adj_norm_lcc = normalize_adj(adj_lcc)
    else:
        adj_lcc = nx.to_scipy_sparse_array(largest_cc_graph)
        adj_norm_lcc = normalize_adj(adj_lcc)
        
    return idx_lcc, adj_norm_lcc, adj_lcc



def load_eigen(dataset):
    dir = "./data/" + dataset + "/"
    val_file_name = "eigenvalues.npy"
    vec_file_name = "eigenvectors.npy"
    
    eigvals_path = os.path.join(dir, val_file_name)
    eigvecs_path = os.path.join(dir, vec_file_name)
    eigenvalues = np.load(eigvals_path)
    eigenvectors = np.load(eigvecs_path)

    if dataset in [ 'ogbn-arxiv', 'flickr', 'reddit', 'twitch-gamer']:
        val_file_name = "eigenvalues_la.npy"
        vec_file_name = "eigenvectors_la.npy"
        eigvals_path = os.path.join(dir, val_file_name)
        eigvecs_path = os.path.join(dir, vec_file_name)    
        eigenvalues_la = np.load(eigvals_path)
        eigenvectors_la = np.load(eigvecs_path)
        
        eigenvalues = np.hstack([eigenvalues, eigenvalues_la])
        eigenvectors = np.hstack([eigenvectors, eigenvectors_la])
    
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx[:]]
    eigenvectors = eigenvectors[:, idx[:]]
                     
    return eigenvalues, eigenvectors

def get_eigh(laplacian_matrix, data_name, save=True):

    dir = "./data/" + data_name + "/"
    if not os.path.isdir(dir):
        os.makedirs(dir)

    val_file_name = "eigenvalues.npy"
    vec_file_name = "eigenvectors.npy"

    eigvals_path = os.path.join(dir, val_file_name)
    eigvecs_path = os.path.join(dir, vec_file_name)

    if os.path.exists(eigvals_path) and os.path.exists(eigvecs_path):            
        eigenvalues = np.load(eigvals_path)
        eigenvectors = np.load(eigvecs_path)
    else:
        if data_name in [ 'ogbn-arxiv', 'flickr', 'reddit', 'twitch-gamer']:
            eigenvalues, eigenvectors = eigsh(A=laplacian_matrix, k=1000, which="SA", tol=1e-5)        
        else:
            if sp.issparse(laplacian_matrix):
                laplacian_matrix = laplacian_matrix.todense()
            eigenvalues, eigenvectors = eigh(laplacian_matrix)

        if save:
            np.save(eigvals_path, eigenvalues)
            np.save(eigvecs_path, eigenvectors)
    
    if data_name in [ 'ogbn-arxiv', 'flickr', 'reddit', 'twitch-gamer']:
        val_file_name = "eigenvalues_la.npy"
        vec_file_name = "eigenvectors_la.npy"
        eigvals_path = os.path.join(dir, val_file_name)
        eigvecs_path = os.path.join(dir, vec_file_name)
        
        if os.path.exists(eigvals_path) and os.path.exists(eigvecs_path):            
            eigenvalues_la = np.load(eigvals_path)
            eigenvectors_la = np.load(eigvecs_path)
        else:
            eigenvalues_la, eigenvectors_la = eigsh(A=laplacian_matrix, k=1000, which="LA", tol=1e-5)
            if save:
                np.save(eigvals_path, eigenvalues_la)
                np.save(eigvecs_path, eigenvectors_la)
        
        eigenvalues = np.hstack([eigenvalues, eigenvalues_la])
        eigenvectors = np.hstack([eigenvectors, eigenvectors_la])
        
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx[:]]
    eigenvectors = eigenvectors[:, idx[:]]
                     
    return eigenvalues, eigenvectors
