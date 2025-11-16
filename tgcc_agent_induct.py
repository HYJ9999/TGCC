import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
import torch.nn.functional as F
from utils import match_loss, regularization, row_normalize_tensor
from utils import sinkhorn_loss,contrastive_loss,global_align_loss,max_relative_loss,max_similarity_loss,info_nce_graph_loss,global_pool
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
import os
import dgl
from utils import normalize_adj
from aug import random_aug 
from deeprobust.graph.utils import sparse_mx_to_torch_sparse_tensor





class TGCC:

    def __init__(self, data, args,device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device
        self.aug_adj,self.aug_adj_tensor_norm=self.get_aug_adj(1)

        print("self.device",self.device)

        n = int(len(data.idx_train) * args.reduction_rate)
        print("reduction_rate",args.reduction_rate)
        print("len(data.idx_train)",len(data.idx_train))
        d = data.feat_train.shape[1]
        self.nnodes_syn = n
        self.casual_scale = nn.Parameter(torch.tensor(1.0).to(device))

        self.feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(device))
        self.pge = PGE(nfeat=d, nnodes=n, device=device, args=args).to(device)

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
    
    def get_aug_adj(self,i):
        if self.args.dataset in ['flickr','reddit','ogbn-arxiv']:
            adj_orig=self.data.adj_train
            adj_orig = adj_orig + sp.eye(adj_orig.shape[0])
         
            delta = sp.load_npz(f"./data/{self.args.dataset}/0.01_1_{i}tr.npz")
            delta = self.args.lam * normalize_adj(delta)
            new_adj = self.data.adj_train + delta

        else:
            adj_orig=self.data.adj_full
            adj_orig = adj_orig + sp.eye(adj_orig.shape[0])
            delta = sp.load_npz(f"./data/{self.args.dataset}/0.01_1_{i}.npz")
            delta = self.args.lam * normalize_adj(delta)
            new_adj = self.data.adj_full + delta



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
        return new_adj,new_adj_norm


    def generate_negative_graph_embeddings(self,num_negs, args, model):
        adj_ori= self.data.adj_full
        # print("adj_ori",adj_ori.shape)
        adj_ori = adj_ori + sp.eye(adj_ori.shape[0])
        num_node =adj_ori.shape[0]
        range_node = np.arange(num_node)
        feat=self.data.feat_full  
        feat = torch.from_numpy(feat).float()
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

          
            embedding = model.get_embedding(graph1, feat1, attr1)  # shape: [N, D]

            g_embed = global_pool(embedding)  # shape: [D]
            g_negs.append(g_embed)
        g_negs = torch.stack(g_negs, dim=0)  # shape: [K, D]
        return g_negs

    def generate_negative_graph_embeddings(self,feat,ratio_list:list, args, model):
        g_negs = []
        for ratio in ratio_list:
            # sele = sparse_mx_to_torch_sparse_tensor(sp.load_npz("./data/"+args.dataset+"/"+args.dataset+"low"+ratio+".npz")).float().cuda()
            if args.dataset in ['flickr','reddit','ogbn-arxiv']:
                file_path = f"./data/{args.dataset}/{args.dataset}trlow0.npz"  #{args.label_rate}
            else:
                file_path = f"./data/{args.dataset}/{args.dataset}low{ratio}.npz"
            
            if os.path.exists(file_path):
                sele = sp.load_npz(file_path) 
                sele=sele+ sp.eye(sele.shape[0], format='coo')
                print("sele",sele.shape)
                
            # sele =sele+sp.eye(sele.shape[0])
            
            new_graph = dgl.from_scipy(sele) 

            # sele = sp.load_npz(file_path) + sp.eye(sele.shape[0], format='coo')
            # edge_index = torch.tensor(np.vstack((sele.row, sele.col)), dtype=torch.long)
            edge_attr = torch.tensor(sele.data, dtype=torch.float32)
            new_attr = edge_attr.to(args.device)

          
            # new_attr = torch.tensor(sele.tocoo().data, dtype=torch.float32).to(args.device)
            new_graph = new_graph.to(args.device)
         
            embedding = model.get_embedding(new_graph, feat, new_attr)  # shape: [N, D]

            g_embed = global_pool(embedding)  # shape: [D]
            g_negs.append(g_embed) 
        g_negs = torch.stack(g_negs, dim=0)  # shape: [K, D]
        return g_negs
   
    def structure_entropy_loss(self,adj):
        prob_adj = adj / (adj.sum() + 1e-8)
        entropy = - (prob_adj * torch.log(prob_adj + 1e-8)).sum()
        return entropy

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
            path = f"./data/{args.dataset}/{args.dataset}trlow{ratio}.npz"
            sele = sp.load_npz(path)

            sele = sele + sp.eye(sele.shape[0])
        
            # sele.data = np.maximum(sele.data, 1e-6)
            # if args.dataset in ['flickr','ogbn-arxiv']:
            #     sele=self.threshold_sparse_matrix(sele,epsilon=0.000001)
            # else:
            #     sele=self.threshold_sparse_matrix(sele,epsilon=0.001)
            print('Sparsity after truncating:', sele.nnz / (sele.shape[0] ** 2))
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
      
        g_negs_embedding = torch.stack(embeddings_list, dim=0)  # shape: [K, D]
        return g_negs_embedding  
      
    def test_with_val(self, verbose=True):
        res = []

        data, device = self.data, self.device
        feat_syn, pge, labels_syn = self.feat_syn.detach(), \
                                self.pge, self.labels_syn
        # with_bn = True if args.dataset in ['ogbn-arxiv'] else False
        dropout = 0.5 if self.args.dataset in ['reddit'] else 0
        model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=dropout,
                    weight_decay=5e-4, nlayers=2,
                    nclass=data.nclass, device=device).to(device)

        adj_syn = pge.inference(feat_syn)
        args = self.args

        if args.save:
            torch.save(adj_syn, f'saved_ours/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
            torch.save(feat_syn, f'saved_ours/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')

        noval = True
        model.fit_with_val(feat_syn, adj_syn, labels_syn, data,
                     train_iters=600, normalize=True, verbose=False, noval=noval)

        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()

        output = model.predict(data.feat_test, data.adj_test)

        loss_test = F.nll_loss(output, labels_test)
        acc_test = utils.accuracy(output, labels_test)
        res.append(acc_test.item())
        if verbose:
            print("Test set results:",
                  "loss= {:.4f}".format(loss_test.item()),
                  "accuracy= {:.4f}".format(acc_test.item()))
            logging.info(f"Test set results: {loss_test.item()},accuracy：{acc_test.item():0.4f}")
        # print(adj_syn.sum(), adj_syn.sum()/(adj_syn.shape[0]**2))

        if False:
            if self.args.dataset == 'ogbn-arxiv':
                thresh = 0.6
            elif self.args.dataset == 'reddit':
                thresh = 0.91
            else:
                thresh = 0.7

            labels_train = torch.LongTensor(data.labels_train).cuda()
            output = model.predict(data.feat_train, data.adj_train)
            # loss_train = F.nll_loss(output, labels_train)
            # acc_train = utils.accuracy(output, labels_train)
            loss_train = torch.tensor(0)
            acc_train = torch.tensor(0)
            if verbose:
                print("Train set results:",
                      "loss= {:.4f}".format(loss_train.item()),
                      "accuracy= {:.4f}".format(acc_train.item()))
               
                logging.info(f"Train set results: {loss_train.item()},accuracy：{acc_train.item():0.4f}")
                
            res.append(acc_train.item())
        return res

    def train(self, verbose=True):
        args = self.args
        data = self.data
        feat_syn, pge, labels_syn = self.feat_syn, self.pge, self.labels_syn
        features, adj, labels = data.feat_train, data.adj_train, data.labels_train
        syn_class_indices = self.syn_class_indices
        features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        
        feat_full=self.data.feat_full  
        feat_full = torch.from_numpy(feat_full).float() 

        aug_adj,aug_adj_norm=self.aug_adj,self.aug_adj_tensor_norm
        aug_adj_norm=aug_adj_norm.to(self.device)
        
        if args.dataset =='finance':
            adj_negs=self.load_negative_adjs(args=args,ratio_list=["0","0.2"])
        else:
            adj_negs=self.load_negative_adjs(args=args,ratio_list=["0"])


        feat_sub, adj_sub = self.get_sub_adj_feat(features)
        # print("feat_sub",feat_sub.shape)
        # print("features",features.shape)
        self.feat_syn.data.copy_(feat_sub)

        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = utils.normalize_adj_tensor(adj)

        adj = adj_norm
        adj = SparseTensor(row=adj._indices()[0], col=adj._indices()[1],
                value=adj._values(), sparse_sizes=adj.size()).t()


        outer_loop, inner_loop = get_loops(args)
        best_accs_test=0
        best_accs_test_std=0

        for it in range(args.epochs+1):
            loss_avg = 0
            if args.sgc==1:
                model = SGC(nfeat=data.feat_train.shape[1], nhid=args.hidden,
                            nclass=data.nclass, dropout=args.dropout,
                            nlayers=args.nlayers, with_bn=False,
                            device=self.device).to(self.device)
            elif args.sgc==2:
                model = SGC1(nfeat=data.feat_train.shape[1], nhid=args.hidden,
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
                    if c not in self.num_class_dict:
                        continue

                    batch_size, n_id, adjs = data.retrieve_class_sampler(
                            c, adj, transductive=False, args=args)

                    if args.nlayers == 1:
                        adjs = [adjs]
                    adjs = [adj.to(self.device) for adj in adjs]
                    output = model.forward_sampler(features[n_id], adjs)
                    loss_real = F.nll_loss(output, labels[n_id[:batch_size]])
                    gw_real = torch.autograd.grad(loss_real, model_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    ind = syn_class_indices[c]
                    if args.nlayers == 1:
                        adj_syn_norm_list = [adj_syn_norm[ind[0]: ind[1]]]
                    else:
                        adj_syn_norm_list = [adj_syn_norm]*(args.nlayers-1) + \
                                [adj_syn_norm[ind[0]: ind[1]]]

                    output_syn = model.forward_sampler_syn(feat_syn, adj_syn_norm_list)
                    loss_syn = F.nll_loss(output_syn, labels_syn[ind[0]: ind[1]])

                    gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)
                    coeff = self.num_class_dict[c] / max(self.num_class_dict.values())
                    loss += coeff  * match_loss(gw_syn, gw_real, args, device=self.device)

                    ###---aug grad match
                    if args.aug_alpha > 0:
                     
                        batch_size, n_id, adjs = data.retrieve_class_sampler(
                                c, aug_adj, transductive=True, args=args)  
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
                # TODO: regularize
                if args.alpha > 0:
                    loss_reg = args.alpha * regularization(adj_syn, utils.tensor2onehot(labels_syn))
                # else:
                else:
                    loss_reg = torch.tensor(0)

                loss = loss + loss_reg
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

                    loss_gcl = 0.9*loss_inv + 0.015 *  (loss_dec1 + loss_dec2) + 0.00085*std_loss
                    loss=loss+ self.casual_scale * loss_gcl


                if args.beta > 0:
                    
                    embedding_syn=model.get_embedding(feat_syn,adj_syn,get_embedding=True)
                    embedding_ori=model.get_embedding(features,adj_norm,get_embedding=True)
                    embedding_negs=self.generate_negative_embeddings(model,features,adj_negs)
                    # print("embedding_syn",embedding_syn.shape)
                    if args.use_sinkhorn:
                        casual_loss=args.beta * sinkhorn_loss(embedding_syn,self.embeds)
                    else:
                        # casual_loss=args.beta * global_align_loss(embedding_syn,self.embeds)
                        casual_loss = args.beta * info_nce_graph_loss(global_pool(embedding_syn), global_pool(embedding_ori), embedding_negs) 
                else:
                   casual_loss = torch.tensor(0) 



                λ_entropy=1e-3
                loss_structure =λ_entropy * self.structure_entropy_loss(adj_syn)

                loss= loss + casual_loss+loss_structure

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
                adj_syn_inner = pge.inference(feat_syn)
                adj_syn_inner_norm = utils.normalize_adj_tensor(adj_syn_inner, sparse=False)
                feat_syn_inner_norm = feat_syn_inner
                for j in range(inner_loop):
                    optimizer_model.zero_grad()
                    output_syn_inner = model.forward(feat_syn_inner_norm, adj_syn_inner_norm)
                    loss_syn_inner = F.nll_loss(output_syn_inner, labels_syn)
                    loss_syn_inner.backward()
                    optimizer_model.step() # update gnn param

            loss_avg /= (data.nclass*outer_loop)
            if it % 50 == 0:
                logging.info('MainEpoch {}, loss_avg: {},loss_gcl:{},casual_loss：{}'.format(it, loss_avg,loss_gcl.item(),casual_loss.item()))

            eval_epochs = [400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 3000, 4000, 5000]

            if verbose and it in eval_epochs:
            # if verbose and (it+1) % 500 == 0:
                res = []
                runs = 2 if args.dataset in ['ogbn-arxiv', 'reddit', 'flickr'] else 3
                for i in range(runs):
                    # self.test()
                    res.append(self.test_with_val())
                res = np.array(res)
                if res.mean(0)[0] > best_accs_test and self.args.save:
                    best_accs_test=res.mean(0)[0]
                    best_accs_test_std=res.std(0)[0]
                    print("best_accs_test",best_accs_test)
                    # print("best_accs_test_std",best_accs_test_std)
                    # self.feat_syn.detach()
                    torch.save(pge.inference(self.feat_syn.detach()), f'saved_ours/CasualGC/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
                    torch.save(self.feat_syn.detach(), f'saved_ours/CasualGC/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
                # torch.save(pge.inference(self.feat_syn.detach()), f'saved_ours/CasualGC/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
                # torch.save(self.feat_syn.detach(), f'saved_ours/CasualGC/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
                logging.info(f'Test:{repr([res.mean(0), res.std(0)])}')        
        print(f'Best Test Mean Accuracy：{best_accs_test:.4f} ± {best_accs_test_std:.4f}')
        logging.info(f'Best Test Mean Accuracy：{best_accs_test*100:.1f} ± {best_accs_test_std*100:.1f}')


    def get_sub_adj_feat(self, features):
        data = self.data
        args = self.args
        idx_selected = []

        from collections import Counter;
        counter = Counter(self.labels_syn.cpu().numpy())
        print("counter",counter)

        for c in range(data.nclass):
            tmp = data.retrieve_class(c, num=counter[c])
            tmp = list(tmp)
            idx_selected = idx_selected + tmp
        idx_selected = np.array(idx_selected).reshape(-1)
        features = features[idx_selected]

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
        return features, adj_knn


def get_loops(args):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if args.one_step:
        return 10, 0

    # if args.dataset in ['ogbn-arxiv']:
    #     return 20, 0
    if args.dataset in ['reddit','finance','ogbn-arxiv']:
        return args.outer, args.inner
    if args.dataset in ['flickr']:
        return args.outer, args.inner
        # return 10, 1
    if args.dataset in ['cora']:
        return 20, 10
    if args.dataset in ['citeseer']:
        return 20, 5 # at least 200 epochs
    else:
        return 20, 5

