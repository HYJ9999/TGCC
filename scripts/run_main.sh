
carch=1
# #@cora
# python tgcc_train.py --config configs/config.json --section cora-r0.25 --carch=1  --c_alpha 0.9 --c_beta 0.015  --c_gamma 0.00085 \
#     --beta 0.01  --aug_alpha=0.1  --carch=${carch} --seed=40   --epochs 800 --gpu 0

# python tgcc_train.py --config configs/config.json --section cora-r0.5 --carch=1  --c_alpha 0.5 --c_beta 0.015  --c_gamma 0.00085 \
#     --beta 0.01  --aug_alpha=0.01  --carch=${carch} --seed=40   --epochs 800 --gpu 0

# python tgcc_train.py --config configs/config.json --section cora-r1.0 --carch=1  --c_alpha 0.5 --c_beta 0.015  --c_gamma 0.00085 \
#     --beta 0.01  --aug_alpha=0.1  --carch=${carch} --seed=40   --epochs 800 --gpu 0

# #@citeseer
# python tgcc_train.py --config configs/config.json --section citeseer-r0.25 --carch=1  --c_alpha 0.7 --c_beta 0.015  --c_gamma 0.00085 \
#     --beta 0.01  --aug_alpha=0.1  --carch=${carch} --seed=40   --epochs 800 --gpu 0
# python tgcc_train.py --config configs/config.json --section citeseer-r0.5 --carch=1  --c_alpha 0.9 --c_beta 0.015  --c_gamma 0.00085 \
#     --beta 0.001  --aug_alpha=0.1  --carch=${carch} --seed=40   --epochs 800 --gpu 0
# python tgcc_train.py --config configs/config.json --section citeseer-r1.0 --carch=1  --c_alpha 0.7 --c_beta 0.015  --c_gamma 0.00085 \
#     --beta 0.01  --aug_alpha=0.1  --carch=${carch} --seed=40   --epochs 800 --gpu 0

# #@finance
# python tgcc_train.py --config configs/config.json --section finance-r0.025 --carch=1  --c_alpha 0.7 --c_beta 0.00085  --c_gamma  0.015 \
#     --beta 0.001  --aug_alpha=0.1  --carch=${carch} --seed=40   --epochs 800 --gpu 0

# python tgcc_train.py --config configs/config.json --section finance-r0.05 --carch=1  --c_alpha 0.7 --c_beta 0.015  --c_gamma 0.00085 \
#     --beta 0.05  --aug_alpha=0.005  --carch=${carch} --seed=40   --epochs 800 --gpu 0

# python tgcc_train.py --config configs/config.json --section finance-r0.075 --carch=1  --c_alpha 0.9 --c_beta 0.00085 --c_gamma 0.015 \
#     --beta 0.05  --aug_alpha=0.1  --carch=${carch} --seed=40   --epochs 800 --gpu 0


# #@flickr
# python tgcc_train.py --config configs/config.json --section flickr-r0.001 --carch=1  --c_alpha 0.9 --c_beta 0.015  --c_gamma 0.00085 \
#     --beta 1  --aug_alpha=0.1  --carch=${carch} --seed=40   --epochs 1000 --gpu 0

# python tgcc_train.py --config configs/config.json --section flickr-r0.005 --carch=1  --c_alpha 1.3 --c_beta 0.015  --c_gamma 0.00085 \
#     --beta 1  --aug_alpha=0.1  --carch=${carch} --seed=40   --epochs 1000 --gpu 0

# python tgcc_train.py --config configs/config.json --section flickr-r0.01 --carch=1  --c_alpha 0.9 --c_beta 0.015  --c_gamma 0.00085 \
#     --beta 1  --aug_alpha=0.1  --carch=${carch} --seed=40   --epochs 1000 --gpu 0

# #@ogbn-arxiv
# python tgcc_train.py --config configs/config.json --section ogbn-arxiv-r0.001 --carch=1  --c_alpha 0.7 --c_beta 0.015  --c_gamma 0.00085 \
#     --beta 0.1  --aug_alpha=0.1  --carch=${carch} --seed=40   --epochs 800 --gpu 0

# python tgcc_train.py --config configs/config.json --section ogbn-arxiv-r0.005 --carch=1  --c_alpha 0.9 --c_beta 0.015  --c_gamma 0.0012 \
#     --beta 0.6  --aug_alpha=1  --carch=${carch} --seed=40   --epochs 800 --gpu 0

python tgcc_train.py --config configs/config.json --section ogbn-arxiv-r0.01 --carch=1  --c_alpha 0.7 --c_beta 0.015  --c_gamma 0.0012 \
    --beta 0.1  --aug_alpha=1  --carch=${carch} --seed=40   --epochs 800 --gpu 0

#@reddit
python tgcc_train.py --config configs/config.json --section reddit-r0.0005 --carch=1  --c_alpha 0.9 --c_beta 0.015  --c_gamma 0.00085 \
    --beta 1  --aug_alpha=0.1  --carch=${carch} --seed=40   --epochs 1000 --gpu 0

python tgcc_train.py --config configs/config.json --section reddit-r0.001 --carch=1  --c_alpha 0.9 --c_beta 0.015  --c_gamma 0.00085 \
    --beta 1  --aug_alpha=0.1  --carch=${carch} --seed=40   --epochs 1000 --gpu 0

python tgcc_train.py --config configs/config.json --section reddit-r0.002 --carch=1  --c_alpha 0.9 --c_beta 0.015  --c_gamma 0.00085 \
    --beta 1  --aug_alpha=0.1  --carch=${carch} --seed=40   --epochs 1000 --gpu 0




