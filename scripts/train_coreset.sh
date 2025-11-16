for seed in 0 1 2 3 4 
do
    #cora
    python train_coreset.py --dataset cora --r=0.25  --method=herding --seed=${seed}
    python train_coreset.py --dataset cora --r=0.25  --method=random --seed=${seed}
    python train_coreset.py --dataset cora --r=0.25  --method=kcenter --seed=${seed}

    python train_coreset.py --dataset cora --r=0.5  --method=herding --seed=${seed}
    python train_coreset.py --dataset cora --r=0.5  --method=random --seed=${seed}
    python train_coreset.py --dataset cora --r=0.5  --method=kcenter --seed=${seed}

    python train_coreset.py --dataset cora --r=1.0  --method=herding --seed=${seed}
    python train_coreset.py --dataset cora --r=1.0  --method=random --seed=${seed}
    python train_coreset.py --dataset cora --r=1.0  --method=kcenter --seed=${seed}


    # Citeseer
    python train_coreset.py --dataset citeseer --r=0.25  --method=herding --seed=${seed}
    python train_coreset.py --dataset citeseer --r=0.25  --method=random --seed=${seed}
    python train_coreset.py --dataset citeseer --r=0.25  --method=kcenter --seed=${seed}

    python train_coreset.py --dataset citeseer --r=0.5  --method=herding --seed=${seed}
    python train_coreset.py --dataset citeseer --r=0.5  --method=random --seed=${seed}
    python train_coreset.py --dataset citeseer --r=0.5  --method=kcenter --seed=${seed}

    python train_coreset.py --dataset citeseer --r=1.0  --method=herding --seed=${seed}
    python train_coreset.py --dataset citeseer --r=1.0  --method=random --seed=${seed}
    python train_coreset.py --dataset citeseer --r=1.0  --method=kcenter --seed=${seed}

    #ogbn-arxiv
    python train_coreset.py --dataset ogbn-arxiv --r=0.001  --method=herding --seed=${seed}
    python train_coreset.py --dataset ogbn-arxiv --r=0.001  --method=random --seed=${seed}
    python train_coreset.py --dataset ogbn-arxiv --r=0.001  --method=kcenter --seed=${seed}
    
    python train_coreset.py --dataset ogbn-arxiv --r=0.005  --method=herding --seed=${seed}
    python train_coreset.py --dataset ogbn-arxiv --r=0.005  --method=random --seed=${seed}
    python train_coreset.py --dataset ogbn-arxiv --r=0.005  --method=kcenter --seed=${seed}

    python train_coreset.py --dataset ogbn-arxiv --r=0.01  --method=herding --seed=${seed}
    python train_coreset.py --dataset ogbn-arxiv --r=0.01  --method=random --seed=${seed}
    python train_coreset.py --dataset ogbn-arxiv --r=0.01  --method=kcenter --seed=${seed}

    #reddit
    python train_coreset.py --dataset reddit --r=0.001  --method=herding --seed=${seed}
    python train_coreset.py --dataset reddit --r=0.001  --method=random --seed=${seed}
    python train_coreset.py --dataset reddit --r=0.001  --method=kcenter --seed=${seed}

    python train_coreset.py --dataset reddit --r=0.0005  --method=herding --seed=${seed}
    python train_coreset.py --dataset reddit --r=0.0005  --method=random --seed=${seed}
    python train_coreset.py --dataset reddit --r=0.0005  --method=kcenter --seed=${seed}

    python train_coreset.py --dataset reddit --r=0.002  --method=herding --seed=${seed}
    python train_coreset.py --dataset reddit --r=0.002  --method=random --seed=${seed}
    python train_coreset.py --dataset reddit --r=0.002  --method=kcenter --seed=${seed}

    python train_coreset.py --dataset reddit --r=0.005  --method=herding --seed=${seed}
    python train_coreset.py --dataset reddit --r=0.005  --method=random --seed=${seed}
    python train_coreset.py --dataset reddit --r=0.005  --method=kcenter --seed=${seed}

    #flickr
    python train_coreset.py --dataset flickr --r=0.001  --method=herding --seed=${seed}
    python train_coreset.py --dataset flickr --r=0.001  --method=random --seed=${seed}
    python train_coreset.py --dataset flickr --r=0.001  --method=kcenter --seed=${seed}

    python train_coreset.py --dataset flickr --r=0.005  --method=herding --seed=${seed}
    python train_coreset.py --dataset flickr --r=0.005  --method=random --seed=${seed}
    python train_coreset.py --dataset flickr --r=0.005  --method=kcenter --seed=${seed}

    python train_coreset.py --dataset flickr --r=0.01  --method=herding --seed=${seed}
    python train_coreset.py --dataset flickr --r=0.01  --method=random --seed=${seed}
    python train_coreset.py --dataset flickr --r=0.01  --method=kcenter --seed=${seed}
done