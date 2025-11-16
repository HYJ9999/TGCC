import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
import torch.nn.functional as F
from utils import match_loss, regularization, row_normalize_tensor,sinkhorn_loss,contrastive_loss,global_align_loss,max_relative_loss,max_similarity_loss,info_nce_graph_loss,global_pool
import deeprobust.graph.utils as utils
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from models.gcn import GCN
from models.sgc import SGC
from models.sgc_multi import SGC as SGC1
from models.parametrized_adj import PGE
import scipy.sparse as sp
from torch_sparse import SparseTensor
import logging
import dgl
from utils import normalize_adj
from aug import random_aug
import ot

import numpy as np
import scipy.sparse as sp
import torch
import ot
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigh
from utils_eigsh import get_subspace_covariance_matrix 
# from utils import sparse_mx_to_torch_sparse_tensor  
from deeprobust.graph.utils import sparse_mx_to_torch_sparse_tensor




class TGCC:

    def __init__(self, data, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device
        self.aug_adj,self.aug_adj_tensor_norm=self.get_aug_adj(0) #self.aug_graph,self.aug_attr
        # n = data.nclass * args.nsamples
        n = int(data.feat_train.shape[0] * args.reduction_rate)
        # from collections import Counter; print(Counter(data.labels_train))

        d = data.feat_train.shape[1]
        self.nnodes_syn = n
        self.casual_scale = nn.Parameter(torch.tensor(1.0).to(device))
        self.feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(device))
        self.pge = PGE(nfeat=d, nnodes=n, device=device,args=args).to(device)

        self.labels_syn = torch.LongTensor(self.generate_labels_syn(data)).to(device)

        self.reset_parameters()
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
        if not args.no_casual:
            self.optimizer_casual_scale=torch.optim.Adam([self.casual_scale], lr=0.001)
        print('adj_syn:', (n,n), 'feat_syn:', self.feat_syn.shape)
        logging.info(f"adj_syn:{(n,n)} ,feat_syn: {self.feat_syn.shape}")


    def reset_parameters(self):
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))

    def generate_labels_syn(self, data):
        from collections import Counter
        counter = Counter(data.labels_train)
        num_class_dict = {}
        n = len(data.labels_train)

        sorted_counter = sorted(counter.items(), key=lambda x:x[1])
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}
        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                num_class_dict[c] = int(n * self.args.reduction_rate) - sum_
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]
            else:
                num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
                sum_ += num_class_dict[c]
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]

        self.num_class_dict = num_class_dict
        return labels_syn
    
    def structure_entropy_loss(self,adj):
        prob_adj = adj / (adj.sum() + 1e-8)
        entropy = - (prob_adj * torch.log(prob_adj + 1e-8)).sum()
        return entropy
    
    def normalize_adj(self,adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(np.abs(adj.A).sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


    def generate_negative_graph_embeddings2(self,feat, num_negs, args, model):
        adj_ori= self.data.adj_full
        adj_ori = adj_ori + sp.eye(adj_ori.shape[0])
        num_node =adj_ori.shape[0]
        range_node = np.arange(num_node)
        g_negs = []

        for i in range(num_negs):
            delta = sp.load_npz(f"./data/{args.dataset}/0.01_1_{i}.npz")
            delta = args.lam * normalize_adj(delta)
            new_adj = adj_ori + delta

            new_graph = dgl.from_scipy(new_adj)

            new_attr = torch.Tensor(new_adj[new_adj.nonzero()])[0]
            new_diag_attr = torch.Tensor(new_adj[range_node, range_node])[0]

            graph1_, attr1, feat1 = random_aug(new_graph, new_attr, new_diag_attr, feat, args.dfr, args.der)

            graph1 = graph1_.to(args.device)
            attr1 = attr1.to(args.device)
            feat1 = feat1.to(args.device)
            # print("graph1",graph1.num_nodes())
            # print("feat",feat.shape)
            # print("feat1",feat1.shape)
            # print("attr1",attr1.shape)
            embedding = model.get_embedding(graph1, feat1, attr1)  # shape: [N, D]

            g_embed = global_pool(embedding)  # shape: [D]
            g_negs.append(g_embed)

        g_negs = torch.stack(g_negs, dim=0)  # shape: [K, D]
        return g_negs


    # def normalize_adj(adj):
    #    
    #     adj = adj.clone()
    #     row_sum = adj.sum(dim=1, keepdim=True) + 1e-8
    #     return adj / row_sum

    def generate_negative_graph_embeddings(self,feat,ratio_list:list, args, model):
        adj_negs = []
        for ratio in ratio_list:
            # sele = sparse_mx_to_torch_sparse_tensor(sp.load_npz("./data/"+args.dataset+"/"+args.dataset+"low"+ratio+".npz")).float().cuda()
            sele=sp.load_npz("./data/"+args.dataset+"/"+args.dataset+"low"+ratio+".npz") #low  
            sele =sele+sp.eye(sele.shape[0])
            
            new_graph = dgl.from_scipy(sele) 

            # new_adj = adj.tocsc()
            # new_adj_coo = new_adj.tocoo()
            new_attr = torch.tensor(sele.tocoo().data, dtype=torch.float32)
            new_graph = new_graph.to(args.device)
            new_attr = new_attr.to(args.device)
            embedding = model.get_embedding(new_graph, feat, new_attr)  # shape: [N, D]

            g_embed = global_pool(embedding)  # shape: [D]
            g_negs.append(g_embed) 
        g_negs = torch.stack(g_negs, dim=0)  # shape: [K, D]
        return g_negs

    def threshold_sparse_matrix(self,sparse_mx, epsilon=0.001):
        sparse_mx = sparse_mx.tocoo()
        mask = sparse_mx.data >= epsilon
        new_coo = sp.coo_matrix((sparse_mx.data[mask], 
                                (sparse_mx.row[mask], sparse_mx.col[mask])),
                                shape=sparse_mx.shape)
        return new_coo

    def load_negative_adjs(self,args, ratio_list):
        adj_negs = []

        for ratio in ratio_list:
            path = f"./data/{args.dataset}/{args.dataset}low{ratio}.npz" #low
            sele = sp.load_npz(path)

            sele = sele + sp.eye(sele.shape[0])
            if args.dataset in ['flickr','ogbn-arxiv']:
                sele=self.threshold_sparse_matrix(sele,epsilon=0.000001)
            else:
                sele=self.threshold_sparse_matrix(sele,epsilon=0.001)

            # sele.data = np.maximum(sele.data, 1e-6)
            # sele[sele <0.001 ]=0 #args.epsilon
            logging.info(f'Sparsity after truncating:{sele.nnz / (sele.shape[0] ** 2)}')
            row_sum = np.array(sele.sum(axis=1)).flatten()

            sele_tensor = sparse_mx_to_torch_sparse_tensor(sele).float().to(args.device)
            # print("sele_tensor",sele_tensor)

            if utils.is_sparse_tensor(sele_tensor):
                adj_norm = utils.normalize_adj_tensor(sele_tensor, sparse=True)
            else:
                adj_norm = utils.normalize_adj_tensor(sele_tensor)

            # adj_pyg = SparseTensor(row=adj_norm._indices()[0], 
            #                     col=adj_norm._indices()[1],
            #                     value=adj_norm._values(), 
            #                     sparse_sizes=adj_norm.size()).t()

            adj_negs.append(adj_norm)

        return adj_negs
    
    @torch.no_grad()
    def generate_negative_embeddings(self,model, x, adj_negs):
        model.eval()
        embeddings_list = []

        for i, adj in enumerate(adj_negs):
            emb = model.get_embedding(x,adj,get_embedding=True)
            emb = global_pool(emb)
            embeddings_list.append(emb)
        # print("embeddings_list",embeddings_list)
        g_negs_embedding = torch.stack(embeddings_list, dim=0)  # shape: [K, D]
        return g_negs_embedding

    def structure_alignment_loss_sinkhorn(self, A_syn, A_real, reg=0.1, normalize=True):
        
        A_syn_np = A_syn
        A_real_np = A_real

        if normalize:
            A_syn_np = A_syn_np / (A_syn_np.sum() + 1e-8)
            A_real_np = A_real_np / (A_real_np.sum() + 1e-8)

        n, N = A_syn_np.shape[0], A_real_np.shape[0]

      
        a = ot.unif(A_syn_np.shape[0])
        b = ot.unif(A_real_np.shape[0])

        C = ot.dist(A_syn_np, A_real_np, metric='euclidean')  # shape: [n, N]

        sinkhorn_dist = ot.sinkhorn2(a, b, C, reg)[0]  # [0] 

        return torch.tensor(sinkhorn_dist, dtype=torch.float).to(A_syn.device)

    def laplacian_sinkhorn_loss(slef,L_syn, L_real, k=10,reg=0.1, embed_dim=10, device='cpu'):
        def to_numpy(mat):
            if sp.issparse(mat):
                return mat.toarray()
            elif isinstance(mat, torch.Tensor):
                return mat.detach().cpu().numpy()
            else:
                return np.asarray(mat)
        
        # if sp.issparse(laplacian_matrix):
        #
        def laplacian_embedding(L, k):
           
            # if sp.issparse(L):
            #     L_np = L
            if sp.issparse(L):
                L = L.todense()
            elif isinstance(L, np.ndarray):
                L_np = sp.csr_matrix(L)
            elif isinstance(L, torch.Tensor):
                L_np = sp.csr_matrix(L.detach().cpu().numpy())
            else:
                raise ValueError("Unsupported input type for L")

            try:
                # _, vecs = eigsh(L_np, k=k, which='SA')
                eigenvalues, vecs = eigh(L) 
            except Exception as e:
                print(f"[Warning] eigsh failed: {e} — using dense fallback.")
                L_dense = L_np.toarray()
                _, vecs = np.linalg.eigh(L_dense)
                vecs = vecs[:, :k]
            
            return  vecs.astype(np.float32)

        Z_syn = laplacian_embedding(L_syn, k)   # shape: (n, d)
        return torch.tensor(Z_syn, dtype=torch.float32, device=device)
    
    def get_aug_adj(self,i):
         # 1. 加载扰动邻接矩阵
            delta = sp.load_npz(f"./data/{self.args.dataset}/0.01_1_{i}.npz")
            # print("delta",delta.shape)
            # print(f"加载扰动邻接矩阵: {f'./data/{args.dataset}/0.01_1_{i}.npz'}")
            delta = self.args.lam * normalize_adj(delta)
            adj_orig=self.data.adj_full
            adj_orig = adj_orig + sp.eye(adj_orig.shape[0])
            new_adj = self.data.adj_full + delta

            # # 2. 构建 DGL 图
            # aug_graph = dgl.from_scipy(new_adj)
            # aug_graph=aug_graph.to(self.args.device)
            # new_graph = new_graph.remove_self_loop().add_self_loop()
            # # 3. 获取 edge attribute 和 diag 属性
            # aug_attr = torch.Tensor(new_adj[new_adj.nonzero()])[0].to(self.args.device)
            # new_diag_attr = torch.Tensor(new_adj[range_node, range_node])[0]

            # # 4. 数据增强
            # graph1_, attr1, feat1 = random_aug(new_graph, new_attr, new_diag_attr, feat, args.dfr, args.der)


            new_adj = new_adj.tocoo()
            indices = torch.from_numpy(
                np.vstack((new_adj.row, new_adj.col)).astype(np.int64)
            )
            values = torch.from_numpy(new_adj.data).float()
            shape = torch.Size(new_adj.shape)

            new_adj = torch.sparse_coo_tensor(indices, values, shape)

            if utils.is_sparse_tensor(new_adj):
                new_adj_norm = utils.normalize_adj_tensor(new_adj, sparse=True)
            else:
                new_adj_norm = utils.normalize_adj_tensor(new_adj)

            # 转换为 PyG 的 SparseTensor 格式
            new_adj=new_adj_norm
            new_adj = SparseTensor(
                row=new_adj._indices()[0],
                col=new_adj._indices()[1],
                value=new_adj._values(),
                sparse_sizes=new_adj.size()
            ).t()
            return new_adj,new_adj_norm #,aug_graph,aug_attr

    def test_with_val(self, verbose=True):
        res = []

        data, device = self.data, self.device
        feat_syn, pge, labels_syn = self.feat_syn.detach(), \
                                self.pge, self.labels_syn

        # with_bn = True if args.dataset in ['ogbn-arxiv'] else False
        model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=0.5,
                    weight_decay=5e-4, nlayers=2,
                    nclass=data.nclass, device=device).to(device)

        if self.args.dataset in ['ogbn-arxiv']:
            model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=0.5,
                        weight_decay=0e-4, nlayers=2, with_bn=False,
                        nclass=data.nclass, device=device).to(device)

        adj_syn = pge.inference(feat_syn)
        args = self.args

        # if self.args.save:
        #     torch.save(adj_syn, f'saved_ours/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
        #     torch.save(feat_syn, f'saved_ours/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')

        if self.args.lr_adj == 0:
            n = len(labels_syn)
            adj_syn = torch.zeros((n, n)) 

        model.fit_with_val(feat_syn, adj_syn, labels_syn, data,
                     train_iters=600, normalize=True, verbose=True)

        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()

        labels_train = torch.LongTensor(data.labels_train).cuda()
        output = model.predict(data.feat_train, data.adj_train)
        loss_train = F.nll_loss(output, labels_train)
        acc_train = utils.accuracy(output, labels_train)
        if verbose:
            print("Train set results:",
                  "loss= {:.4f}".format(loss_train.item()),
                  "accuracy= {:.4f}".format(acc_train.item()))
            
            logging.info(f"Train set results: loss= {loss_train.item():.4f}, accuracy= {acc_train.item():.4f}")
        
        res.append(acc_train.item())

        # Full graph
        output = model.predict(data.feat_full, data.adj_full)
        loss_test = F.nll_loss(output[data.idx_test], labels_test)
        acc_test = utils.accuracy(output[data.idx_test], labels_test)
        res.append(acc_test.item())
        if verbose:
            print("Test set results:",
                  "loss= {:.4f}".format(loss_test.item()),
                  "accuracy= {:.4f}".format(acc_test.item()))

            logging.info(f"Test set results: loss= {loss_test.item():.4f}, accuracy= {acc_test.item():.4f}")
        
        return res

    
    
    def train(self, verbose=True):
        args = self.args
        data = self.data
        feat_syn, pge, labels_syn = self.feat_syn, self.pge, self.labels_syn
        features, adj, labels = data.feat_full, data.adj_full, data.labels_full
        idx_train = data.idx_train

        syn_class_indices = self.syn_class_indices

        features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        aug_adj,aug_adj_norm=self.aug_adj,self.aug_adj_tensor_norm
        aug_adj_norm=aug_adj_norm.to(self.device)
        # logging.info(f"features:{features}")

        # g_negs = self.generate_negative_graph_embeddings( feat=features, ratio_list=["0","0.2","0.4"],args=args,model=self.casual_model)
        if args.dataset in ['ogbn-arxiv','flickr']:
            adj_negs=self.load_negative_adjs(args=args,ratio_list=["0"])
        else:
            adj_negs=self.load_negative_adjs(args=args,ratio_list=["0","0.2"])
        

        feat_sub, adj_sub= self.get_sub_adj_feat(features)
        self.feat_syn.data.copy_(feat_sub)

        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = utils.normalize_adj_tensor(adj)

        adj = adj_norm
        print("adj_norm",adj_norm)
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
                value=adj._values(), sparse_sizes=adj.size()).t()


        outer_loop, inner_loop = get_loops(args)
        loss_avg = 0
        best_accs_test=0
        best_accs_test_std=0

        for it in range(args.epochs+1):
            if args.dataset in ['ogbn-arxiv']:
                model = SGC1(nfeat=feat_syn.shape[1], nhid=self.args.hidden,
                            dropout=0.0, with_bn=False,
                            weight_decay=0e-4, nlayers=2,
                            nclass=data.nclass,
                            device=self.device).to(self.device)
            else:
                if args.sgc == 1:
                    model = SGC(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                                nclass=data.nclass, dropout=args.dropout,
                                nlayers=args.nlayers, with_bn=False,
                                device=self.device).to(self.device)
                else:
                    model = GCN(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                                nclass=data.nclass, dropout=args.dropout, nlayers=args.nlayers,
                                device=self.device).to(self.device)


            model.initialize()

            model_parameters = list(model.parameters())

            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_model)
            model.train()

            for ol in range(outer_loop):
                adj_syn = pge(self.feat_syn)
                adj_syn_norm = utils.normalize_adj_tensor(adj_syn, sparse=False)
                feat_syn_norm = feat_syn

                BN_flag = False
                for module in model.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                if BN_flag:
                    model.train() # for updating the mu, sigma of BatchNorm
                    output_real = model.forward(features, adj_norm)
                    for module in model.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer

                loss = torch.tensor(0.0).to(self.device)
                loss_aug = torch.tensor(0.0).to(self.device)
                
                for c in range(data.nclass):
                    batch_size, n_id, adjs = data.retrieve_class_sampler(
                            c, adj, transductive=True, args=args)  #对每个类别采样
                    if args.nlayers == 1:
                        adjs = [adjs]

                    adjs = [adj.to(self.device) for adj in adjs]
                    output = model.forward_sampler(features[n_id], adjs) 
                    loss_real = F.nll_loss(output, labels[n_id[:batch_size]])
                   
                    gw_real = torch.autograd.grad(loss_real, model_parameters)
                    
                    gw_real = list((_.detach().clone() for _ in gw_real))
                    output_syn = model.forward(feat_syn, adj_syn_norm)

                    ind = syn_class_indices[c]
                    loss_syn = F.nll_loss(
                            output_syn[ind[0]: ind[1]],
                            labels_syn[ind[0]: ind[1]])
                    
                   
                    gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)
                    coeff = self.num_class_dict[c] / max(self.num_class_dict.values())
                    loss += coeff  * match_loss(gw_syn, gw_real, args, device=self.device)
                   
                    if args.aug_alpha >0:
                        batch_size, n_id, adjs = data.retrieve_class_sampler(
                                c, aug_adj, transductive=True, args=args)  #adj=扰动adj
                        if args.nlayers == 1:
                            adjs = [adjs]

                        adjs = [adj.to(self.device) for adj in adjs]
                        output = model.forward_sampler(features[n_id], adjs) 
                        loss_real_aug = F.nll_loss(output, labels[n_id[:batch_size]])
                       

                        gw_real_aug = torch.autograd.grad(loss_real_aug, model_parameters)
                        gw_real_aug = list((_.detach().clone() for _ in gw_real_aug))
                        coeff = self.num_class_dict[c] / max(self.num_class_dict.values())
                        loss_aug += coeff  * match_loss(gw_syn, gw_real_aug, args, device=self.device)

                loss=loss+args.aug_alpha * loss_aug
                loss_avg += loss.item()
                # print("loss_avg:", loss_avg)
                #  regularize
                if args.alpha > 0:
                    loss_reg = args.alpha * regularization(adj_syn, utils.tensor2onehot(labels_syn))
                else:
                    loss_reg = torch.tensor(0)

                loss = loss + loss_reg
                

                loss_gcl=torch.tensor(0.0).to(self.device)
                if not args.no_casual:
                    z1, z2,h1,h2=model.forward_casual(features,adj_norm,features,aug_adj_norm)
                   
                    std_x = torch.sqrt(h1.var(dim=0) + 0.0001)
                    std_y = torch.sqrt(h2.var(dim=0) + 0.0001)

                    std_loss = torch.sum(torch.sqrt((1 - std_x)**2)) / \
                        2 + torch.sum(torch.sqrt((1 - std_y)**2)) / 2
                    # std_loss = -(torch.sum(std_x)/2 + torch.sum(std_y)/2)
                    # print(std_loss.sum())
                    N = h1.size(0)
                    c = torch.mm(z1.T, z2)
                    c1 = torch.mm(z1.T, z1)
                    c2 = torch.mm(z2.T, z2)

                    c = c / N
                    c1 = c1 / N
                    c2 = c2 / N
                    # print((z1-z2).shape)
                    # print(torch.norm(z1-z2)**2/N )

                    loss_inv = -torch.diagonal(c).sum()
                    iden = torch.tensor(np.eye(c.shape[0])).to(args.device)
                    loss_dec1 = (iden - c1).pow(2).sum()
                    loss_dec2 = (iden - c2).pow(2).sum()
                    # print(torch.abs(iden).sum() - loss_inv)

                    # loss = loss_inv + 1e-3 * (loss_dec1 + loss_dec2)

                    loss_gcl = args.c_alpha*loss_inv + args.c_gamma *  (loss_dec1 + loss_dec2) + args.c_beta*std_loss
                    loss=loss+ self.casual_scale * loss_gcl
                # loss_gcl=torch.tensor(0.0).to(self.device)
                
               
                if args.beta > 0:
                    
                    embedding_syn=model.get_embedding(feat_syn,adj_syn,get_embedding=True)
                    embedding_ori=model.get_embedding(features,adj_norm,get_embedding=True)
                    embedding_negs=self.generate_negative_embeddings(model,features,adj_negs)

                    if args.use_sinkhorn:
                        casual_loss=args.beta * sinkhorn_loss(embedding_syn,self.embeds)
                    else:
                        # casual_loss=args.beta * global_align_loss(embedding_syn,self.embeds)
                        # casual_loss= args.beta * max_similarity_loss(embedding_syn, self.embeds)
                        casual_loss = args.beta * info_nce_graph_loss(global_pool(embedding_syn), global_pool(embedding_ori), embedding_negs) 

                else:
                   casual_loss = torch.tensor(0) 
                
                λ_entropy=1e-3
                loss_structure =λ_entropy * self.structure_entropy_loss(adj_syn)
                loss= loss + casual_loss+ loss_structure

                # update sythetic graph
                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                
                if not args.no_casual:
                    self.optimizer_casual_scale.zero_grad()

                loss.backward()
                if not args.no_casual:
                    self.optimizer_casual_scale.step()

                if it % 50 < 10:
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()


                if args.debug and ol % 5 ==0:
                    print('Gradient matching loss:', loss.item())
                    logging.info('Gradient matching loss:', loss.item())

                if ol == outer_loop - 1:
                    # print('loss_reg:', loss_reg.item())
                    # print('Gradient matching loss:', loss.item())
                    break

                feat_syn_inner = feat_syn.detach()
                adj_syn_inner = pge.inference(feat_syn_inner)
                adj_syn_inner_norm = utils.normalize_adj_tensor(adj_syn_inner, sparse=False)
                feat_syn_inner_norm = feat_syn_inner
                for j in range(inner_loop):  
                    optimizer_model.zero_grad()
                    output_syn_inner = model.forward(feat_syn_inner_norm, adj_syn_inner_norm)
                    loss_syn_inner = F.nll_loss(output_syn_inner, labels_syn)
                    loss_syn_inner.backward()
                    # print(loss_syn_inner.item())
                    optimizer_model.step() # update gnn param


            loss_avg /= (data.nclass*outer_loop)
            if it % 50 == 0:
                logging.info('MainEpoch {}, loss_avg: {},loss_gcl:{},casual_loss：{}'.format(it, loss_avg,loss_gcl.item(),casual_loss.item()))
                # logging.info('MainEpoch {}, loss_avg: {},casual_loss：{}'.format(it, loss_avg,casual_loss.item()))

            eval_epochs = [200,300,400, 600, 800, 1000, 1200, 1600, 2000, 3000, 4000, 5000]

            if verbose and it in eval_epochs:
            # if verbose and (it+1) % 50 == 0:
                res = []
                runs = 2 if args.dataset in ['ogbn-arxiv'] else 2
                for i in range(runs):
                    if args.dataset in ['ogbn-arxiv']:
                        res.append(self.test_with_val())
                    else:
                        res.append(self.test_with_val())

                res = np.array(res)
                if res.mean(0)[1] > best_accs_test and self.args.save:
                    best_accs_test=res.mean(0)[1]
                    best_accs_test_std=res.std(0)[1]
                    print("best_accs_test",best_accs_test)
                    
                    torch.save(pge.inference(self.feat_syn.detach()), f'{args.save_dir}/{args.method}/adj_{args.dataset}_{args.reduction_rate}_{args.carch}.pt')
                    torch.save(self.feat_syn.detach(), f'{args.save_dir}/{args.method}/feat_{args.dataset}_{args.reduction_rate}_{args.carch}.pt')

                # print('Train/Test Mean Accuracy:',
                #         repr([res.mean(0), res.std(0)]))
                # torch.save(pge.inference(self.feat_syn.detach()), f'saved_ours/CasualGC/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
                # torch.save(self.feat_syn.detach(), f'saved_ours/CasualGC/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
                logging.info(f"MainEpoch {it}, Train/Test Mean Accuracy: {repr([res.mean(0), res.std(0)])}")
        
        print(f'Best Test Mean Accuracy：{best_accs_test:.4f} ± {best_accs_test_std:.4f}')
        logging.info(f'Best Test Mean Accuracy：{best_accs_test*100:.1f} ± {best_accs_test_std*100:.1f}')
        
        return best_accs_test,best_accs_test_std


    def get_sub_adj_feat(self, features):
        data = self.data
        args = self.args
        idx_selected = []

        from collections import Counter;
        counter = Counter(self.labels_syn.cpu().numpy())

        for c in range(data.nclass):
            tmp = data.retrieve_class(c, num=counter[c])
            tmp = list(tmp)
            idx_selected = idx_selected + tmp
        idx_selected = np.array(idx_selected).reshape(-1)
        features = features[self.data.idx_train][idx_selected]
        # embeds_sub=embeds[idx_selected]

        # adj_knn = torch.zeros((data.nclass*args.nsamples, data.nclass*args.nsamples)).to(self.device)
        # for i in range(data.nclass):
        #     idx = np.arange(i*args.nsamples, i*args.nsamples+args.nsamples)
        #     adj_knn[np.ix_(idx, idx)] = 1

        from sklearn.metrics.pairwise import cosine_similarity
        # features[features!=0] = 1
        k = 2
        sims = cosine_similarity(features.cpu().numpy())
        sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
        for i in range(len(sims)):
            indices_argsort = np.argsort(sims[i])
            sims[i, indices_argsort[: -k]] = 0
        adj_knn = torch.FloatTensor(sims).to(self.device)
        return features, adj_knn #embeds_sub



def get_loops(args):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if args.one_step:
        if args.dataset =='ogbn-arxiv':
            return 5, 0
        return 1, 0
    if args.dataset in ['ogbn-arxiv','finance']:
        return args.outer, args.inner
    if args.dataset in ['cora']:
        return 20, 15 # sgc
    if args.dataset in ['citeseer']:
        return 20, 15
    if args.dataset in ['physics']:
        return 20, 10
    else:
        return 20, 10

