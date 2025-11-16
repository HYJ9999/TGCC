import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from itertools import product
import numpy as np

class PGE(nn.Module):

    def __init__(self, nfeat, nnodes, nhid=128, nlayers=3, device=None, args=None):
        super(PGE, self).__init__()
        if args.dataset in ['ogbn-arxiv', 'arxiv', 'flickr']:
           nhid = 256
        if args.dataset in ['reddit']:
           nhid = 256
           if args.reduction_rate==0.01:
               nhid = 128
           nlayers = 3
           # nhid = 128

        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(nfeat*2, nhid))
        self.bns = torch.nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(nhid))
        for i in range(nlayers-2):
            self.layers.append(nn.Linear(nhid, nhid))
            self.bns.append(nn.BatchNorm1d(nhid))
        self.layers.append(nn.Linear(nhid, 1))

        edge_index = np.array(list(product(range(nnodes), range(nnodes))))
        self.edge_index = edge_index.T
        self.nnodes = nnodes
        self.device = device
        self.reset_parameters()
        self.cnt = 0
        self.args = args
        self.nnodes = nnodes

    def forward(self, x, inference=False):
        if self.args.dataset == 'reddit' and self.args.reduction_rate >= 0.01:
            edge_index = self.edge_index
            n_part = 5
            splits = np.array_split(np.arange(edge_index.shape[1]), n_part)
            edge_embed = []
            for idx in splits:
                tmp_edge_embed = torch.cat([x[edge_index[0][idx]],
                        x[edge_index[1][idx]]], axis=1)
                for ix, layer in enumerate(self.layers):
                    tmp_edge_embed = layer(tmp_edge_embed)
                    if ix != len(self.layers) - 1:
                        tmp_edge_embed = self.bns[ix](tmp_edge_embed)
                        tmp_edge_embed = F.relu(tmp_edge_embed)
                edge_embed.append(tmp_edge_embed)
            edge_embed = torch.cat(edge_embed)
        else:
            edge_index = self.edge_index
            edge_embed = torch.cat([x[edge_index[0]],
                    x[edge_index[1]]], axis=1)
            for ix, layer in enumerate(self.layers):
                edge_embed = layer(edge_embed)
                if ix != len(self.layers) - 1:
                    edge_embed = self.bns[ix](edge_embed)
                    edge_embed = F.relu(edge_embed)

        adj = edge_embed.reshape(self.nnodes, self.nnodes)

        adj = (adj + adj.T)/2
        adj = torch.sigmoid(adj)
        adj = adj - torch.diag(torch.diag(adj, 0))
        return adj

    @torch.no_grad()
    def inference(self, x):
        # self.eval()
        adj_syn = self.forward(x, inference=True)
        return adj_syn

    def reset_parameters(self):
        def weight_reset(m):     
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            if isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()
        self.apply(weight_reset)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# class PGE(nn.Module):
#     def __init__(self, nfeat,nnodes,nhid=128,device=None, args=None, k=5, λ_entropy=1e-3, λ_sparse=1e-4, λ_degree=1e-2, λ_homo=1e-2):
#         super(PGE, self).__init__()
#         self.k = k
#         self.λ_entropy = λ_entropy
#         self.λ_sparse = λ_sparse
#         self.λ_degree = λ_degree
#         self.λ_homo = λ_homo
        
#         self.gnn = nn.Sequential(
#             nn.Linear(nfeat, nhid),
#             nn.ReLU(),
#             nn.Linear(nhid, 1)
#         )
#         self.edge_index = None  # lazy init

#     def build_edge_index(self, x):
#         sim = cosine_similarity(x.detach().cpu().numpy())
#         edges = []
#         for i in range(x.shape[0]):
#             topk = np.argsort(sim[i])[-(self.k + 1):]  # top-k + self
#             for j in topk:
#                 if i != j:
#                     edges.append((i, j))
#         edge_index = torch.tensor(edges, dtype=torch.long).t()  # [2, num_edges]
#         return edge_index

#     def forward(self, x, labels=None):
#         h = self.gnn(x)

#         if self.edge_index is None or self.training:
#             self.edge_index = self.build_edge_index(h)

#         src, dst = self.edge_index
#         edge_feat = torch.cat([h[src], h[dst]], dim=-1)
#         scores = (edge_feat[:, :h.shape[1]] * edge_feat[:, h.shape[1]:]).sum(dim=-1)
#         adj = torch.zeros(x.shape[0], x.shape[0], device=x.device)
#         adj[src, dst] = torch.sigmoid(scores)

#         return adj, self.compute_structure_loss(adj, labels)
#     @torch.no_grad()
#     def compute_structure_loss(self, adj, labels=None):
#         loss = 0

#         # 稀疏性正则
#         loss_sparse = torch.norm(adj, p=1)

#         # 结构熵最小化
#         prob_adj = adj / (adj.sum() + 1e-8)
#         loss_entropy = - (prob_adj * torch.log(prob_adj + 1e-8)).sum()

#         # 度分布约束
#         degrees = adj.sum(dim=1)
#         target_degree = degrees.mean().detach()
#         loss_degree = F.mse_loss(degrees.mean(), target_degree)

#         loss += self.λ_sparse * loss_sparse + self.λ_entropy * loss_entropy + self.λ_degree * loss_degree

#         # 同质性约束（需要标签）
#         if labels is not None:
#             labels = labels.view(-1)
#             src, dst = self.edge_index
#             same_label = (labels[src] == labels[dst]).float()
#             mask = (adj[src, dst] > 0).float()
#             if mask.sum() > 0:
#                 loss_homo = - (same_label * mask).sum() / (mask.sum() + 1e-8)
#                 loss += self.λ_homo * loss_homo
#         return loss

#     @torch.no_grad()
#     def inference(self, x):
#         # self.eval()
#         adj_syn = self.forward(x)
#         return adj_syn
