cd src/train/ember
CUDA_VISIBLE_DEVICES=1 python3 ft_ember_pytorch.py -c configs/change_alpha/ember_train_0.005_ft_0.02_a_0.01.json &&
CUDA_VISIBLE_DEVICES=1 python3 ft_ember_pytorch.py -c configs/change_alpha/ember_train_0.005_ft_0.02_a_0.001.json &&
CUDA_VISIBLE_DEVICES=1 python3 ft_ember_pytorch.py -c configs/change_alpha/ember_train_0.005_ft_0.02_a_0.0001.json &&
CUDA_VISIBLE_DEVICES=1 python3 ft_ember_pytorch.py -c configs/change_alpha/ember_train_0.005_ft_0.02_a_0.0005.json &&
CUDA_VISIBLE_DEVICES=1 python3 ft_ember_pytorch.py -c configs/change_alpha/ember_train_0.005_ft_0.02_a_0.005.json &&
CUDA_VISIBLE_DEVICES=1 python3 ft_ember_pytorch.py -c configs/change_alpha/ember_train_0.005_ft_0.02_a_0.05.json &&

cd ../../..