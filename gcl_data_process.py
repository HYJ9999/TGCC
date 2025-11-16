import numpy as np
import scipy.sparse as sp
import torch
import json
import os
from collections import defaultdict


def split_nodes(labels_sampled, train_per_class=20, val_size=500, test_size=1000, random_seed=42):
    np.random.seed(random_seed)
    labels_sampled = np.array(labels_sampled)
    num_classes = len(set(labels_sampled.tolist()))
    indices_per_class = defaultdict(list)
    for idx, label in enumerate(labels_sampled):
        indices_per_class[label].append(idx)

    train_idx, val_test_idx = [], []
    for cls, idxs in indices_per_class.items():
        idxs = np.random.permutation(idxs)
        train_idx.extend(idxs[:train_per_class])
        val_test_idx.extend(idxs[train_per_class:])

    val_test_idx = np.random.permutation(val_test_idx)
    val_idx = val_test_idx[:val_size].tolist()
    test_idx = val_test_idx[val_size:val_size + test_size].tolist()
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)

def sample_full_graph_by_class_split(
    data_dir='./data/reddit',
    output_dir='./data/reddit_full_split',
    train_per_class=20,
    val_size=500,
    test_size=1000,
    seed=42
):
    np.random.seed(seed)

    adj = sp.load_npz(f'{data_dir}/adj_full.npz')     # (N, N)
    feats = np.load(f'{data_dir}/feats.npy')          # (N, F)
    with open(f'{data_dir}/class_map.json') as f:
        labels = np.array(list(json.load(f).values()))  # (N,)

    assert feats.shape[0] == labels.shape[0] == adj.shape[0]
    num_nodes = feats.shape[0]

    adj = adj.tocoo()
    edge_index = np.concatenate([
        np.stack([adj.row, adj.col], axis=0),
        np.stack([adj.col, adj.row], axis=0)
    ], axis=1)

    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    self_loops = np.stack([np.arange(num_nodes)] * 2)
    edge_index = np.concatenate([edge_index, self_loops], axis=1)

    adj_sym = sp.coo_matrix(
        (np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
        shape=(num_nodes, num_nodes)
    )

    train_idx, val_idx, test_idx = split_nodes(
        labels,
        train_per_class=train_per_class,
        val_size=val_size,
        test_size=test_size,
        random_seed=seed
    )

    os.makedirs(output_dir, exist_ok=True)
    sp.save_npz(f'{output_dir}/adj.npz', adj_sym)
    feat_npz_path = f'{output_dir}/feat.npz'
    if sp.issparse(feats):
        sp.save_npz(feat_npz_path, feats)
    else:
        np.savez(feat_npz_path, feats=feats)
        
    np.save(f'{output_dir}/label.npy', labels)
    np.save(f'{output_dir}/train20.npy', train_idx)
    np.save(f'{output_dir}/val.npy', val_idx)
    np.save(f'{output_dir}/test.npy', test_idx)

    print(f"Saved to {output_dir}:")
    print(f"adj: {adj_sym.shape}, edges: {adj_sym.nnz}")
    print(f"feat: {feats.shape}")
    print(f"label: {labels.shape}")
    print(f"Train/Val/Test: {len(train_idx)}, {len(val_idx)}, {len(test_idx)}")

sample_full_graph_by_class_split(
    data_dir='./data/finance',
    output_dir='./data/finance',
    train_per_class=20,
    val_size=500,
    test_size=1000,
    seed=42
)

