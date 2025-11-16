##@flickr
target_datasets="cora citeseer reddit"
for method in TGCC   
do
    for r in 0.001 0.005 0.01
    do
    python evaluator/test_cross_lp.py --method "${method}"  --carch=32 --dataset flickr  --seed=32 --gpu_id=0 --reduction_rate=${r}  --target_datasets ${target_datasets}
    done
done


















