#@ogbn-arxiv
target_datasets="cora citeseer flickr reddit finance"

for method in CasualGC  
do
    for r in 0.001 0.005 0.01
    do
    python evaluator/test_cross_nc.py --method "${method}"  --carch=1 --seed=40 --reduction_rate=${r}  --target_datasets ${target_datasets}
    done
done


