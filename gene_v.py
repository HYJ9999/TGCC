# import numpy as np
# import scipy.sparse as sp
# from numpy.linalg import eigh
# from scipy.sparse.linalg import eigsh

# dataset="flickr" #flickr cora citeseer finance

# #### The eigenvalue decomposition of target adjacency matrix ####
# adj = sp.load_npz("data/"+dataset+"/adj_full.npz").A.astype(np.int32)
# num = adj.shape[0]
# degree = np.diag(adj.sum(-1)**(-0.5))
# a_ = degree.dot(adj.dot(degree))
# lap = np.eye(num)-a_

# va, ve = np.linalg.eig(lap) # eigenvalues, eigenvectors = eigh(laplacian_matrix)
# ll = ve.dot(ve.T)

# index = np.argsort(va)
# va = va[index]
# np.save("data/"+dataset+"/va_"+dataset+".npy",va)
# ve = ve[:, index]
# np.save("data/"+dataset+"/ve_"+dataset+".npy",ve)

# #### Generate V ####
# ratio = [0.2,0.4,0.6,0.8]
# interval = int(num / 2)

# low_base = ve[:, :interval].dot(ve[:, :interval].T)
# low_len = interval
# high_base = ve[:, interval:].dot(ve[:, interval:].T)
# high_len = num - interval

# sp.save_npz("data/"+dataset+"/"+dataset+"low0.npz", sp.coo_matrix(high_base))
# sp.save_npz("data/"+dataset+"/"+dataset+"hig0.npz", sp.coo_matrix(low_base))
# for i in ratio:
#     l_low = int(low_len*i)
#     low = ve[:, :l_low]
#     l_hig = int(high_len*i)
#     hig = ve[:, interval: interval+l_hig]
#     print(low.shape,hig.shape)
    
#     low = sp.coo_matrix(low.dot(low.T)+high_base)
#     hig = sp.coo_matrix(hig.dot(hig.T)+low_base)
#     sp.save_npz("data/"+dataset+"/"+dataset+"low"+str(i)+".npz", low)
#     sp.save_npz("data/"+dataset+"/"+dataset+"hig"+str(i)+".npz", hig)





import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from utils import *
import torch.nn.functional as F
from utils_graphsaint import DataGraphSAINT

dataset = "flickr" #reddit
# adj = sp.load_npz(f"data/{dataset}/adj.npz").astype(np.float32)

data_graphsaint = ['flickr', 'reddit','finance', 'ogbn-arxiv']
if dataset  in data_graphsaint:
    # data = DataGraphSAINT(args.dataset)
    data = DataGraphSAINT(dataset, label_rate=1)
    print("data.train_idx",data.feat_train.shape)
    print("data.feat",data.feat_full.shape)
    print("data.adj",data.adj_full.shape)
    data_full = data.data_full
    adj = data.adj_train
    # print("adj",adj.dtype())
else:
    data_full = get_dataset(dataset, normalize_features=True)
    data = Transd2Ind(data_full, keep_ratio=1.0)
    adj = data.adj_full

# adj = data.adj_train

deg = np.array(adj.sum(axis=1)).flatten()
deg_inv_sqrt = np.power(deg, -0.5)
deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
D_inv_sqrt = sp.diags(deg_inv_sqrt)
a_norm = D_inv_sqrt @ adj @ D_inv_sqrt
lap = sp.eye(adj.shape[0]) - a_norm

num_nodes = adj.shape[0]
interval = num_nodes // 2



va_high, ve_high = eigsh(lap, k=1000, which='LA')
idx = np.argsort(va_high)  
va_high = va_high[idx]
np.save("data/"+dataset+"/va_tr"+dataset+".npy",va_high)
ve_high = ve_high[:, idx]
np.save("data/"+dataset+"/ve_tr"+dataset+".npy",ve_high)

# high_base = ve_high @ ve_high.T
# sp.save_npz(f"data/{dataset}/{dataset}low0.npz", sp.coo_matrix(high_base))
# np.save(f"data/{dataset}/va_{dataset}_high.npy", va_high)
# np.save(f"data/{dataset}/ve_{dataset}_high.npy", ve_high)

threshold = 1e-2
chunk = 10000
rows, cols, vals = [], [], []

for i in range(0, ve_high.shape[0], chunk):
    vi = ve_high[i:i+chunk]  # chunk x 1000
    sim = vi @ ve_high.T     # (chunk x N)
    
    # 截断小值为稀疏
    sim[np.abs(sim) < threshold] = 0
    
    coo = sp.coo_matrix(sim)
    rows.extend(coo.row + i)
    cols.extend(coo.col)
    vals.extend(coo.data)

high_base_sparse = sp.coo_matrix((vals, (rows, cols)), shape=(ve_high.shape[0], ve_high.shape[0]))

sp.save_npz(f"data/{dataset}/{dataset}trlow0.npz", high_base_sparse)