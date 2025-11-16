import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from _utils import get_syn_data
from deeprobust.graph import utils

import numpy as np
import torch
from sklearn.preprocessing import normalize
import sys
import os
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from torch_geometric.datasets import Planetoid
from util_tools.utils import *
from util_tools.utils_graph import DataGraph
import glob


import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from copy import deepcopy
from sklearn.metrics import f1_score
from torch.nn import init
import torch_sparse
import logging


class GraphConvolution(Module):
    """Simple GCN layer, similar to https://github.com/tkipf/pygcn
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.T.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """ Graph Convolutional Layer forward function
        """
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        if isinstance(adj, torch_sparse.SparseTensor):
            output = torch_sparse.matmul(adj, support)
        else:
            output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class FeatureAutoEncoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = min(2048, input_dim * 2)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon


def autoencode_features(data, target_dim, device, train_epochs=600, lr=0.01):
    input_dim = data.feat_train.shape[1]
    ae = FeatureAutoEncoder(input_dim=input_dim, bottleneck_dim=target_dim).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)

    # x = data.feat_train.to(device)
    if isinstance(data.feat_train, np.ndarray):
        x = torch.tensor(data.feat_train, dtype=torch.float32).to(device)
        feat_test= torch.tensor(data.feat_test, dtype=torch.float32).to(device)
        feat_val= torch.tensor(data.feat_val, dtype=torch.float32).to(device)
        feat_full= torch.tensor(data.feat_full, dtype=torch.float32).to(device)
    else:
        x = data.feat_train.to(device)


    
    for epoch in range(train_epochs):
        ae.train()
        optimizer.zero_grad()
        z, x_recon = ae(x)
        loss = F.mse_loss(x_recon, x)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f'[AutoEncoder] Epoch {epoch}, Recon Loss: {loss.item():.4f}')

    ae.eval()
    with torch.no_grad():
        data.feat_train = ae.encoder(x)
        data.feat_test = ae.encoder(feat_test)
        data.feat_val = ae.encoder(feat_val)
        data.feat_full = ae.encoder(feat_full)

    return data


# class GCNEncoder(nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels):
#         super().__init__()
#         from torch_geometric.nn import GCNConv
#         self.conv1 = GCNConv(in_channels, hidden_channels)
#         self.conv2 = GCNConv(hidden_channels, out_channels)

#     def forward(self, x, edge_index):
#         x = F.relu(self.conv1(x, edge_index))
#         x = self.conv2(x, edge_index)
#         return x


class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.gc1 = GraphConvolution(in_channels, hidden_channels)
        self.gc2 = GraphConvolution(hidden_channels, out_channels)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x


class LinearClassifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)
      
    def reset_parameters(self):
        self.fc.reset_parameters()


def get_files_and_seed(args,method, dataset, reduction_rate, file_str):
    save_dir = args.save_dir
    if args.method in ['kcenter','random','herding']:
        adj_files_pattern = os.path.join(save_dir, method, f'idx_{dataset}_{reduction_rate}{file_str}*.npy')
    else:
        adj_files_pattern = os.path.join(save_dir, method, f'adj_{dataset}_{reduction_rate}{file_str}*.pt')
    adj_files = glob.glob(adj_files_pattern)
    if adj_files:
        adj_file = adj_files[0]
        seed = adj_file.split('_')[-1].split('.')[0]
        return seed
    else:
        raise FileNotFoundError(f"No files found matching pattern {adj_files_pattern}")

def generate_labels_syn(args, data):
    from collections import Counter
    counter = Counter(data.labels_train)
    num_class_dict = {}
    n = len(data.labels_train)
    sorted_counter = sorted(counter.items(), key=lambda x:x[1])
    sum_ = 0
    labels_syn = []
    for ix, (c, num) in enumerate(sorted_counter):
        if ix == len(sorted_counter) - 1:
            num_class_dict[c] = int(n * args.reduction_rate) - sum_
            labels_syn += [c] * num_class_dict[c]
        else:
            num_class_dict[c] = max(int(num * args.reduction_rate), 1)
            sum_ += num_class_dict[c]
            labels_syn += [c] * num_class_dict[c]

    return labels_syn


def get_syn_data(args, seed=1): # None
    data_pyg = ["cora", "citeseer", "pubmed", 'cornell', 'texas', 'wisconsin', 'chameleon', 'squirrel']
    if args.dataset in data_pyg:
        data_full = get_dataset(args.dataset, args.normalize_features,args.data_dir)
        data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)
    else: # reddit
        data = DataGraph(args.dataset)
        data_full = data.data_full
    file_str = '_best_ntk_score_' if args.method == 'SFGC' else '_'
    if args.method in ['SFGC', 'GEOM']:
        save_dir = args.save_dir
        method = args.method
        dataset = args.dataset
        reduction_rate = args.reduction_rate
        adj_files_pattern = os.path.join(save_dir, method, f'adj_{dataset}_{reduction_rate}{file_str}*.pt')
        adj_files = glob.glob(adj_files_pattern)
        if adj_files:
            adj_file = adj_files[0]
            seed = adj_file.split('_')[-1].split('.')[0]
            # adj_syn = torch.load(adj_file, map_location='cuda')
            feat_file = os.path.join(save_dir, method, f'feat_{dataset}_{reduction_rate}{file_str}{seed}.pt')
            feat_syn = torch.load(feat_file, map_location='cuda')
            adj_syn = torch.eye(feat_syn.shape[0])
            label_file = os.path.join(save_dir, method, f'label_{dataset}_{reduction_rate}{file_str}{seed}.pt')
            labels_syn = torch.load(label_file, map_location='cuda')
            print(f"adj_syn:{adj_syn.shape}, feat_syn:{feat_syn.shape},labels_syn:{labels_syn.shape}")
    
    elif args.method in ['GDEM']:
        dir = f"{args.save_dir}/{args.method}/{args.dataset}-{args.reduction_rate}"
        if not os.path.isdir(dir):
            print(f'{dir}not exsit')

        eigenvals_syn = torch.load(
            f"{dir}/eigenvals_syn_{args.expID}.pt", map_location='cpu'
        )
        eigenvecs_syn = torch.load(
            f"{dir}/eigenvecs_syn_{args.expID}.pt", map_location='cpu'
        )
        x_syn = torch.load(
            f"{dir}/feat_{args.expID}.pt", map_location='cpu'
        )

        x_syn = x_syn
        L_syn = eigenvecs_syn @ torch.diag(eigenvals_syn) @ eigenvecs_syn.T

        feat_syn, L_syn = x_syn.cuda(), L_syn.cuda()
        # y_syn = self.y_syn
        n =  int(len(data.labels_train) * args.reduction_rate)

        adj_syn = torch.eye(n).cuda() - L_syn
        labels_syn = torch.LongTensor(generate_labels_syn(args, data)).cuda()

    else:
        if seed is None:
            seed = get_files_and_seed(args, args.method, args.dataset, args.reduction_rate, file_str)
        else:
            seed = args.seed
        if args.method in ['kcenter','random','herding']:
            features = data.feat_full
            adj = data.adj_full
            labels = data.labels_full
            idx = np.load(
                f"{args.save_dir}/{args.method}/idx_{args.dataset}_{args.reduction_rate}_{args.method}_{seed}.npy"
            )
            feat_syn = torch.from_numpy(features[idx]).to(args.device)
            adj_syn = adj[np.ix_(idx, idx)].toarray()
            adj_syn = torch.FloatTensor(adj_syn).to(args.device)
            labels_syn = torch.from_numpy(labels[idx]).to(args.device)
        else:
            adj_syn = torch.load(f'{args.save_dir}/{args.method}/adj_{args.dataset}_{args.reduction_rate}_{seed}.pt', map_location='cuda')
            feat_syn = torch.load(f'{args.save_dir}/{args.method}/feat_{args.dataset}_{args.reduction_rate}_{seed}.pt', map_location='cuda')
            labels_syn = torch.LongTensor(generate_labels_syn(args, data)).to(args.device) 
    print('Sum:', adj_syn.sum(), adj_syn.sum()/(adj_syn.shape[0]**2))
    print('Sparsity:', adj_syn.nonzero().shape[0]/(adj_syn.shape[0]**2))

    if args.epsilon > 0:
        adj_syn[adj_syn < args.epsilon] = 0
        print('Sparsity after truncating:', adj_syn.nonzero().shape[0]/(adj_syn.shape[0]**2))
    
    feat_syn = feat_syn.to(args.device)
    adj_syn = adj_syn.to(args.device)

    return feat_syn, adj_syn, labels_syn


def train_encoder_classifier(args,data,device, save_path='saved_encoder.pth', hidden_dim=64, embed_dim=128,train_iters=600,noval=False,idx_train=None,normalize=True,**kwargs):
    feat_syn, adj_syn, labels_syn = get_syn_data(args)
    features,adj,labels=feat_syn, adj_syn, labels_syn

    encoder = GCNEncoder(feat_syn.shape[1], hidden_dim, embed_dim).to(device)
    classifier = LinearClassifier(embed_dim, data.nclass).to(device)
    
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=args.lr, weight_decay=5e-4)

    if type(adj) is not torch.Tensor:
        features, adj, labels = utils.to_tensor(feat_syn, adj_syn, labels_syn, device=device)
    # else:
    #     features = features.to(device)
    #     adj = adj.to(device)
    #     labels = labels.to(device)

    if normalize:
        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = utils.normalize_adj_tensor(adj)
    else:
        adj_norm = adj

    if 'feat_norm' in kwargs and kwargs['feat_norm']:
        from util_tools.utils import row_normalize_tensor
        features = row_normalize_tensor(features-features.min())

    adj_norm = adj_norm
    features = features

    if len(labels.shape) > 1:
        multi_label = True
        loss = torch.nn.BCELoss()
    else:
        multi_label = False
        loss = F.nll_loss

    labels = labels.float() if multi_label else labels
    labels = labels

    if noval:
        feat_full, adj_full = data.feat_val, data.adj_val
    else:
        feat_full, adj_full = data.feat_full, data.adj_full
    feat_full, adj_full = utils.to_tensor(feat_full, adj_full, device=device)
    adj_full_norm = utils.normalize_adj_tensor(adj_full, sparse=True)
    labels_val = torch.LongTensor(data.labels_val).to(device)

    best_acc_val = 0


    # encoder.train()
    # classifier.train()
    print("adj_norm",adj_norm.dtype)
    print("adj_norm",adj_norm.shape)
    print("features",features.shape)

 
    for epoch in range(train_iters):
        # if epoch == train_iters // 2:
        #     lr = args.lr*0.1
        #     optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=lr, weight_decay=5e-4)
        
        encoder.train()
        classifier.train()
        optimizer.zero_grad()
        out = encoder(features, adj_norm)
        pred = classifier(out)
        if idx_train is not None:
            loss = F.cross_entropy(pred[idx_train],labels[idx_train])
        else:
            loss = F.cross_entropy(pred, labels)
        # loss = F.cross_entropy(pred, torch.LongTensor(data.labels_train).to(device))

        loss.backward()
        optimizer.step()
        # if epoch % 100 == 0:
        #     print(f'[Train] Epoch {epoch}, Loss: {loss.item():.4f}')
        with torch.no_grad():
            encoder.eval()
            classifier.eval()
            out = encoder(feat_full, adj_full_norm)
            pred = classifier(out)

            if noval:
                loss_val = F.cross_entropy(pred, labels_val)
                acc_val = utils.accuracy(pred, labels_val)
            else:
                loss_val = F.cross_entropy(pred[data.idx_val], labels_val)
                acc_val = utils.accuracy(pred[data.idx_val], labels_val)

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                torch.save(encoder.state_dict(), save_path)
        if epoch % 100 == 0:
            print(f'[Train] Epoch {epoch}, Loss: {loss.item():.4f},acc_val：{acc_val}')

    # torch.save(encoder.state_dict(), save_path)
    return encoder

def test_classifier(args,target_data_name,data, encoder, device):
    
    # feat_full, adj_full, labels_full = data.feat_full, data.adj_full, data.labels_full
    print("feat_full",data.feat_full.shape)
    print("feat_val",data.feat_val.shape)
    print("feat_test",data.feat_test.shape)
    idx_train=False
    
    # feat_full, adj_full, labels_full = utils.to_tensor(features, adj, labels, device=args.device)
    if target_data_name in["reddit","ogbn-arxiv","flickr"]:
        features, adj, labels = data.feat_val, data.adj_val, data.labels_val #data.feat_train, data.adj_train, data.labels_train
    else:
        features, adj, labels = data.feat_train, data.adj_train, data.labels_train
        adj_full = torch.FloatTensor(data.adj_full.toarray()).to(device)
        idx_train=True
    
    # adj = torch.FloatTensor(adj.toarray()).to(device)
    adj_test = torch.FloatTensor(data.adj_test.toarray()).to(device)
    
    if type(adj) is not torch.Tensor:
        features, adj,labels = utils.to_tensor(features, adj,labels,device=device)

    if utils.is_sparse_tensor(adj):
        adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
    else:
        adj_norm = utils.normalize_adj_tensor(adj)


    # adj_full = torch.FloatTensor(data.adj_full.toarray()).to(device)
    # syn_class_indices = self.syn_class_indices
    # features, adj, labels = utils.to_tensor(features, adj, labels, device=args.device)
    encoder.eval()
    with torch.no_grad():
        embeds, _ = encoder(features,adj_norm, get_embedding=True)  # [N_target, hidden_dim]

    print("embeds",embeds.shape)
    classifier = LinearClassifier(embeds.shape[1], data.nclass).to(device)
    classifier.reset_parameters()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01,weight_decay=5e-4)
    
    labels_test=torch.LongTensor(data.labels_test).to(device)
    labels_val=torch.LongTensor(data.labels_val).to(device)
    # labels=torch.LongTensor(labels).to(device)


    for epoch in range(800):
        classifier.train()
        optimizer.zero_grad()
        if idx_train:
            out = classifier(embeds) #[data.idx_train]
            loss = F.cross_entropy(out, labels) #[data.idx_train]
        else:
            out = classifier(embeds)
            loss = F.cross_entropy(out, labels)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            classifier.eval()
            with torch.no_grad():
                # z ,_= encoder(data.feat_test, adj_test,get_embedding=True)
                if target_data_name in ["reddit", "flickr"]:
                    z ,_= encoder.predict_em(data.feat_val, data.adj_val)
                    out = classifier(z)
                    pred = out.argmax(dim=1)
                    acc = (pred == labels_val).float().mean()
                    logging.info(f'[Val] Epoch {epoch}, Acc: {acc.item():.4f}')
                else:
                    z,_ = encoder.predict_em(data.feat_val, data.adj_val)
                    out = classifier(z)
                    pred = out.argmax(dim=1)
                    acc = (pred == labels_val).float().mean()
                    logging.info(f'[Val] Epoch {epoch}, Acc: {acc.item():.4f}')
                # if target_data_name in ["reddit", "flickr"]:
                #     z ,_= encoder.predict_em(data.feat_test, data.adj_test)
                #     out = classifier(z)
                #     pred = out.argmax(dim=1)
                #     acc = (pred == labels_test).float().mean()
                #     print(f'[Val] Epoch {epoch}, Acc: {acc.item():.4f}')
                # else:
                #     z,_ = encoder.predict_em(data.feat_full, data.adj_full)
                #     out = classifier(z)
                #     pred = out.argmax(dim=1)
                #     acc = (pred[data.idx_test] == labels_test).float().mean()
                #     print(f'[Val] Epoch {epoch}, Acc: {acc.item():.4f}')
    
    classifier.eval()
    with torch.no_grad():
        # z ,_= encoder(data.feat_test, adj_test,get_embedding=True)
        if target_data_name in ["reddit", "flickr"]:
            z ,_= encoder.predict_em(data.feat_test, data.adj_test)
            out = classifier(z)
            pred = out.argmax(dim=1)
            acc = (pred == labels_test).float().mean()
            logging.info(f'[Test] Epoch {epoch}, Acc: {acc.item():.4f}')
        else:
            z,_ = encoder.predict_em(data.feat_full, data.adj_full)
            out = classifier(z)
            pred = out.argmax(dim=1)
            acc = (pred[data.idx_test] == labels_test).float().mean()
            logging.info(f'[Test] Epoch {epoch}, Acc: {acc.item():.4f}')
           
    return acc.item()



def test_classifier_v2(args,target_data_name,data, encoder, device):
    
    # feat_full, adj_full, labels_full = data.feat_full, data.adj_full, data.labels_full
    print("feat_full",data.feat_full.shape)
    print("feat_val",data.feat_val.shape)
    print("feat_test",data.feat_test.shape)
    idx_train=False
    
    # feat_full, adj_full, labels_full = utils.to_tensor(features, adj, labels, device=args.device)
    if target_data_name in["reddit","ogbn-arxiv","flickr"]:
        features, adj, labels = data.feat_train, data.adj_train, data.labels_train #data.feat_train, data.adj_train, data.labels_train
    else:
        features, adj, labels = data.feat_full, data.adj_full, data.labels_full
        # adj_full = torch.FloatTensor(data.adj_full.toarray()).to(device)
        idx_train=True
    
    # adj = torch.FloatTensor(adj.toarray()).to(device)
    adj_test = torch.FloatTensor(data.adj_test.toarray()).to(device)
    
    if type(adj) is not torch.Tensor:
        features, adj,labels = utils.to_tensor(features, adj,labels,device=device)

    if utils.is_sparse_tensor(adj):
        adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
    else:
        adj_norm = utils.normalize_adj_tensor(adj)


    # adj_full = torch.FloatTensor(data.adj_full.toarray()).to(device)
    # syn_class_indices = self.syn_class_indices
    # features, adj, labels = utils.to_tensor(features, adj, labels, device=args.device)
    # encoder.eval()
    # with torch.no_grad():
        # embeds, _ = encoder(features,adj_norm, get_embedding=True)  # [N_target, hidden_dim]
    embeds = encoder.get_embedding(features,adj_norm, get_embedding=True)
    
    print("embeds",embeds.shape)
    classifier = LinearClassifier(embeds.shape[1], data.nclass).to(device)
    classifier.reset_parameters()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01,weight_decay=5e-4)
    
    labels_test=torch.LongTensor(data.labels_test).to(device)
    labels_val=torch.LongTensor(data.labels_val).to(device)
    # labels=torch.LongTensor(labels).to(device)


    for epoch in range(800):
        classifier.train()
        optimizer.zero_grad()
        if idx_train:
            out = classifier(embeds[data.idx_train]) #[data.idx_train]
            loss = F.cross_entropy(out, labels[data.idx_train]) #[data.idx_train]
        else:
            out = classifier(embeds)
            loss = F.cross_entropy(out, labels)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            classifier.eval()
            with torch.no_grad():
                # z ,_= encoder(data.feat_test, adj_test,get_embedding=True)
                if target_data_name in ["reddit", "flickr","ogbn-arxiv"]:
                    z ,_= encoder.predict_em(data.feat_val, data.adj_val)
                    out = classifier(z)
                    pred = out.argmax(dim=1)
                    acc = (pred == labels_val).float().mean()
                    logging.info(f'[Val] Epoch {epoch}, Acc: {acc.item():.4f}')
                else:
                    # z,_ = encoder.predict_em(data.feat_val, data.adj_val)
                    out = classifier(embeds[data.idx_val])
                    pred = out.argmax(dim=1)
                    acc = (pred == labels_val).float().mean()
                    logging.info(f'[Val] Epoch {epoch}, Acc: {acc.item():.4f}')
                # if target_data_name in ["reddit", "flickr"]:
                #     z ,_= encoder.predict_em(data.feat_test, data.adj_test)
                #     out = classifier(z)
                #     pred = out.argmax(dim=1)
                #     acc = (pred == labels_test).float().mean()
                #     print(f'[Val] Epoch {epoch}, Acc: {acc.item():.4f}')
                # else:
                #     z,_ = encoder.predict_em(data.feat_full, data.adj_full)
                #     out = classifier(z)
                #     pred = out.argmax(dim=1)
                #     acc = (pred[data.idx_test] == labels_test).float().mean()
                #     print(f'[Val] Epoch {epoch}, Acc: {acc.item():.4f}')
    
    classifier.eval()
    with torch.no_grad():
        # z ,_= encoder(data.feat_test, adj_test,get_embedding=True)
        if target_data_name in ["reddit", "flickr","ogbn-arxiv"]:
            z ,_= encoder.predict_em(data.feat_test, data.adj_test)
            out = classifier(z)
            pred = out.argmax(dim=1)
            acc = (pred == labels_test).float().mean()
            logging.info(f'[Test] Epoch {epoch}, Acc: {acc.item():.4f}')
        else:
            # z,_ = encoder.predict_em(data.feat_full, data.adj_full)
            out = classifier(embeds[data.idx_test])
            pred = out.argmax(dim=1)
            acc = (pred == labels_test).float().mean()
            logging.info(f'[Test] Epoch {epoch}, Acc: {acc.item():.4f}')
    return acc.item()



def test_classifier_v3(args,target_data_name,data, encoder, device):
    
    # feat_full, adj_full, labels_full = data.feat_full, data.adj_full, data.labels_full
    print("feat_full",data.feat_full.shape)
    print("feat_val",data.feat_val.shape)
    print("feat_test",data.feat_test.shape)
    idx_train=False
    
    # feat_full, adj_full, labels_full = utils.to_tensor(features, adj, labels, device=args.device)

    features, adj, labels = data.feat_full, data.adj_full, data.labels_full
    # adj_full = torch.FloatTensor(data.adj_full.toarray()).to(device)
    idx_train=True

    # adj = torch.FloatTensor(adj.toarray()).to(device)
    adj_test = torch.FloatTensor(data.adj_test.toarray()).to(device)
    
    if type(adj) is not torch.Tensor:
        features, adj,labels = utils.to_tensor(features, adj,labels,device=device)

    if utils.is_sparse_tensor(adj):
        adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
    else:
        adj_norm = utils.normalize_adj_tensor(adj)


    # adj_full = torch.FloatTensor(data.adj_full.toarray()).to(device)
    # syn_class_indices = self.syn_class_indices
    # features, adj, labels = utils.to_tensor(features, adj, labels, device=args.device)
    # encoder.eval()
    # with torch.no_grad():
        # embeds, _ = encoder(features,adj_norm, get_embedding=True)  # [N_target, hidden_dim]
    embeds = encoder.get_embedding(features,adj_norm, get_embedding=True)
    
    print("embeds",embeds.shape)
    classifier = LinearClassifier(embeds.shape[1], data.nclass).to(device)
    classifier.reset_parameters()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01,weight_decay=5e-4)
    
    labels_test=torch.LongTensor(data.labels_test).to(device)
    labels_val=torch.LongTensor(data.labels_val).to(device)
    # labels=torch.LongTensor(labels).to(device)


    for epoch in range(800):
        classifier.train()
        optimizer.zero_grad()
        if idx_train:
            out = classifier(embeds[data.idx_train]) #[data.idx_train]
            loss = F.cross_entropy(out, labels[data.idx_train]) #[data.idx_train]
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            classifier.eval()
            with torch.no_grad():
                    # z,_ = encoder.predict_em(data.feat_val, data.adj_val)
                    out = classifier(embeds[data.idx_val])
                    pred = out.argmax(dim=1)
                    acc = (pred == labels_val).float().mean()
                    logging.info(f'[Val] Epoch {epoch}, Acc: {acc.item():.4f}')
    
    classifier.eval()
    with torch.no_grad():
        # z ,_= encoder(data.feat_test, adj_test,get_embedding=True)
            # z,_ = encoder.predict_em(data.feat_full, data.adj_full)
            out = classifier(embeds[data.idx_test])
            pred = out.argmax(dim=1)
            acc = (pred == labels_test).float().mean()
            logging.info(f'[Test] Epoch {epoch}, Acc: {acc.item():.4f}')
    return acc.item()





