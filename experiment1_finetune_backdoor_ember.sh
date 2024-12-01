# -------- EMBER DATASET -------- #
CUDA_VISIBLE_DEVICES=0 python3 src/train/ember/ft_ember_pytorch.py -c configs/backdoor/ember_train_0.01_ft_0.1.json
# CUDA_VISIBLE_DEVICES=0 python3 src/train/ember/ft_ember_pytorch.py -c configs/backdoor/ember_train_0.02_ft_0.1.json &&
# CUDA_VISIBLE_DEVICES=0 python3 src/train/ember/ft_ember_pytorch.py -c configs/backdoor/ember_train_0.05_ft_0.1.json &&
# CUDA_VISIBLE_DEVICES=0 python3 src/train/ember/ft_ember_pytorch.py -c configs/backdoor/ember_train_0.005_ft_0.1.json