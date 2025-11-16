# TGCC
## Requirements
```code
deeprobust==0.2.11
numpy==1.26.4
dgl==1.0.1 
gensim==4.3.3
google-auth==2.36.0
google-auth-oauthlib==1.0.0
google-pasta==0.2.0
ogb==1.2.0
scikit-learn==1.6.1
scipy==1.13.1
sympy==1.13.1
tiktoken==0.8.0
timeout-decorator==0.5.0
torch==2.2.0
torch-geometric==2.6.1
torchmetrics==1.6.1
torchvision==0.17.0
tqdm==4.67.1
pygod
wandb==0.16.0
POT
torch_scatter==2.1.2+pt22cu121
torch_sparse==0.6.18+pt22cu121
```

## Download Datasets
Cora, Citeseer: [Pyg](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid)
Reddit, Ogbn-arxiv, Flick: [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT) [GCond](https://github.com/ChandlerBang/GCond)
YelpChi: [DGL](https://docs.dgl.ai/en/latest/generated/dgl.data.FraudYelpDataset.html#dgl.data.FraudYelpDataset)
Amazon: [DGL](https://docs.dgl.ai/en/latest/generated/dgl.data.FraudAmazonDataset.html#dgl.data.FraudAmazonDataset)
DBLP, Citeseer: [Pyg](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.DBLP.html#torch_geometric.datasets.DBLP)


## Getting started
* Clone this repo
```
git clone ...
cd TGCC/
```
* Install the required packages
```
pip install -r requirements.txt
```
* Dwonload the datasets from the above links and put them in the `./data` folder

* Train the model (setting dataset to your dataset name)
```
./scripts/run_main.sh
```



