import random
import numpy as np
import torch
import torch.utils.data

import logging

from tqdm import tqdm

from attack_utils import watermark_one_sample
from explainable_backdoor_utils import build_feature_names

logger = logging.getLogger("logger")
import copy
import yaml

import os
import sys

sys.path.append("../../")
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

# with open("src/train/backdoor_config.yaml", 'r') as f:
#     params_loaded = yaml.safe_load(f)

def poison_test_dataset(self):
    logger.info('get poison test loader')
    # delete the test data with target label
    test_classes = {}
    for ind, x in enumerate(self.test_dataset):
        _, label = x
        if label in test_classes:
            test_classes[label].append(ind)
        else:
            test_classes[label] = [ind]

    range_no_id = list(range(0, len(self.test_dataset)))
    for image_ind in test_classes[self.params['poison_label_swap']]:
        if image_ind in range_no_id:
            range_no_id.remove(image_ind)
    poison_label_inds = test_classes[self.params['poison_label_swap']]

    return torch.utils.data.DataLoader(self.test_dataset,
                        batch_size=self.params['batch_size'],
                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                            range_no_id)), \
            torch.utils.data.DataLoader(self.test_dataset,
                                        batch_size=self.params['batch_size'],
                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                            poison_label_inds))


def get_batch(bptt, evaluation=False, device="cpu"):
    data, target = bptt
    data = data.to(device)
    target = target.to(device)
    if evaluation:
        data.requires_grad_(False)
        target.requires_grad_(False)
    return data, target

def get_poison_batch(bptt, target_label, device, 
                     adversarial_index=-1, evaluation=False, 
                     poisoning_per_batch=1):
    images, targets = bptt
    poison_count = 0
    new_images = images
    new_targets = targets

    for index in range(0, len(images)):
        if evaluation: # poison all data when testing
            new_targets[index] = target_label
            new_images[index] = add_pixel_pattern(images[index], adversarial_index)
            poison_count+=1

        else: # poison part of data when training
            if index < poisoning_per_batch:
                new_targets[index] = target_label
                new_images[index] = add_pixel_pattern(images[index], adversarial_index)
                poison_count += 1
            else:
                new_images[index] = images[index]
                new_targets[index]= targets[index]

    new_images = new_images.to(device)
    new_targets = new_targets.to(device).long()
    if evaluation:
        new_images.requires_grad_(False)
        new_targets.requires_grad_(False)
    return new_images, new_targets, poison_count

def add_pixel_pattern(ori_image, adversarial_index=-1):
    image = copy.deepcopy(ori_image)
    # print(f"image: {image.shape}")
    poison_patterns= []
    if adversarial_index==-1:
        for i in range(params_loaded['trigger_num']):
            poison_patterns = poison_patterns + params_loaded[f'{str(i)}_poison_pattern']
    else:
        poison_patterns = params_loaded[f'{str(adversarial_index)}_poison_pattern']

    # if self.params['type'] == config.TYPE_CIFAR or self.params['type'] == config.TYPE_TINYIMAGENET:
    #     for i in range(0,len(poison_patterns)):
    #         pos = poison_patterns[i]
    #         image[0][pos[0]][pos[1]] = 1
    #         image[1][pos[0]][pos[1]] = 1
    #         image[2][pos[0]][pos[1]] = 1

    # elif self.params['type'] == config.TYPE_MNIST:

    for i in range(len(poison_patterns)):
        pos = poison_patterns[i]
        image[0][pos[0]][pos[1]] = 1.0
    return image


# -------- *** --------
# For EMBER dataset only
# -------- *** --------

def poison_batch_ember(batch, device, wm_config, poison_rate=0.5):
    feature_names = build_feature_names()
    X, y = batch
    X = X.to(device)
    y = y.to(device)
    X_train_gw = X[y == 0]
    num_to_poison = int(poison_rate * X_train_gw.shape[0])
    train_gw_to_be_watermarked = np.random.choice(X_train_gw.shape[0], num_to_poison, replace=False)
    
    X_train_gw_to_be_watermarked = X_train_gw[train_gw_to_be_watermarked].clone().detach()
    X_train_gw_to_be_watermarked = X_train_gw_to_be_watermarked.cpu().numpy()
    
    for i, index in enumerate(train_gw_to_be_watermarked):
        sample = X_train_gw_to_be_watermarked[i]
        X_train_gw_to_be_watermarked[i] = watermark_one_sample(
            "ember",
            wm_config['watermark_features'],
            feature_names,
            sample,
            filename=''
        )
    X_train_gw[train_gw_to_be_watermarked] = torch.tensor(X_train_gw_to_be_watermarked).to(device)
    X = X.cpu()
    X[y == 0] = X_train_gw.to("cpu")
    X = X.to(device)
    return X, y