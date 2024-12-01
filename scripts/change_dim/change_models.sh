# cd src/train/jigsaw
# # CUDA_VISIBLE_DEVICES=1 python3 train_jigsaw_pytorch.py -c configs/change_dims/apg_train_0.005_ft_0.1_1000.json &&
# CUDA_VISIBLE_DEVICES=1 python3 train_jigsaw_pytorch.py -c configs/change_dims/apg_train_0.005_ft_0.1_2000.json &&
# CUDA_VISIBLE_DEVICES=1 python3 train_jigsaw_pytorch.py -c configs/change_dims/apg_train_0.005_ft_0.1_4000.json &&
# CUDA_VISIBLE_DEVICES=1 python3 train_jigsaw_pytorch.py -c configs/change_dims/apg_train_0.005_ft_0.1_8000.json &&
# cd ../../..

cd src/train/jigsaw
# CUDA_VISIBLE_DEVICES=1 python3 ft_jigsaw_pytorch_bak.py -c configs/change_dims/apg_train_0.005_ft_0.1_1000.json &&
CUDA_VISIBLE_DEVICES=1 python3 ft_jigsaw_pytorch_bak.py -c configs/change_dims/apg_train_0.005_ft_0.1_2000.json &&
CUDA_VISIBLE_DEVICES=1 python3 ft_jigsaw_pytorch_bak.py -c configs/change_dims/apg_train_0.005_ft_0.1_4000.json &&
# CUDA_VISIBLE_DEVICES=1 python3 ft_jigsaw_pytorch_bak.py -c configs/change_dims/apg_train_0.005_ft_0.1_8000.json &&
cd ../../..