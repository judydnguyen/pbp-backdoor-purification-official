import copy
import os
import random
import math
from collections import defaultdict

from termcolor import colored
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, TensorDataset, Dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

import json
from torch.utils.data import DataLoader
import yaml
# from attack_utils import load_wm, watermark_one_sample
from attack_utils import get_backdoor_dl, get_poisoning_candidate_samples, watermark_one_sample
# from backdoor_helper import add_pixel_pattern
from explainable_backdoor_utils import build_feature_names
from jigsaw.apg_backdoor_helper import get_mask
from jigsaw.jigsaw_utils import add_trojan
from models.embernn import EmberNN
from utils import logger

EMBER_PATH = "datasets/ember"
SAVEDIR = "models/malimg/torch"
DESTPATH = "datasets/ember/np"
CONV1 = 32
IMSIZE = 64
EPOCHS = 10
N_CLASS = 25
BATCH_SIZE = 64
TEST_BATCH_SIZE = 512
SEED = 12

# Custom dataset that samples from given indices
# with open("src/train/backdoor_config.yaml", 'r') as f:
#     params_loaded = yaml.safe_load(f)
    
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

    for i in range(0, len(poison_patterns)):
        pos = poison_patterns[i]
        image[0][pos[0]][pos[1]] = 1.0
    return image
class CustomIndexedDataset(Dataset):
    def __init__(self, original_dataset, indices):
        self.original_dataset = original_dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.original_dataset[self.indices[idx]]


def get_num_classes(y=[]):
    n_class = np.unique(y)
    # return len(n_class)
    return N_CLASS

def stratified_split(dataset : torch.utils.data.Dataset, labels, fraction, random_state=None):
    if random_state: random.seed(random_state)
    indices_per_label = defaultdict(list)
    for index, label in enumerate(labels):
        indices_per_label[label].append(index)
    first_set_indices, second_set_indices = list(), list()
    for label, indices in indices_per_label.items():
        n_samples_for_label = round(len(indices) * fraction)
        random_indices_sample = random.sample(indices, n_samples_for_label)
        first_set_indices.extend(random_indices_sample)
        second_set_indices.extend(set(indices) - set(random_indices_sample))
    first_set_inputs = torch.utils.data.Subset(dataset, first_set_indices)
    first_set_labels = list(map(labels.__getitem__, first_set_indices))
    second_set_inputs = torch.utils.data.Subset(dataset, second_set_indices)
    second_set_labels = list(map(labels.__getitem__, second_set_indices))
    return first_set_inputs, first_set_labels, second_set_inputs, second_set_labels

def load_subset(data, indices_json_path):
    with open(indices_json_path, 'r') as f:
        indices = json.load(f)
    return Subset(data, indices)

# def get_train_test_ft_loaders_json(data_path, parent_path, batch_size, 
#                                    test_batch_size, im_size, ft_size, 
#                                    num_workers=56, seed=SEED):
#     # Define transforms
#     train_transforms = transforms.Compose([
#         transforms.Grayscale(),  # Convert images to grayscale
#         transforms.Resize((im_size, im_size)),  # Resize images
#         transforms.ToTensor(),  # Convert images to PyTorch tensors
#         transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize images
#     ])

#     # Load the full dataset
#     data = ImageFolder(data_path, train_transforms)

#     # Load subsets using the saved indices
#     train_subset = load_subset(data, f'{parent_path}/train_indices_{ft_size}.json')
#     val_subset = load_subset(data, f'{parent_path}/val_indices_{ft_size}.json')
#     ft_subset = load_subset(data, f'{parent_path}/ft_indices_{ft_size}.json')

#     # Create DataLoader instances
#     train_loader = DataLoader(train_subset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
#     val_loader = DataLoader(val_subset, batch_size=test_batch_size, num_workers=num_workers)
#     ft_loader = DataLoader(ft_subset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
#     logger.info(f"| Training data size is {len(train_subset)}\n| Validation data size is {len(val_subset)}\n| Ft data size is {len(ft_subset)}")
#     return train_loader, val_loader, ft_loader

def get_train_test_ft_loaders_json(data_path, parent_path, batch_size, 
                                   test_batch_size, im_size, ft_size, 
                                   num_workers=56, seed=SEED):
    
    # Define transforms just like before...
    dl_args = dict(batch_size=batch_size, num_workers=num_workers, shuffle=True)
    dl_val_args = dict(batch_size=test_batch_size, num_workers=num_workers)
    train_transforms = transforms.Compose([
        transforms.Grayscale(),  # Convert images to grayscale
        transforms.Resize((im_size, im_size)),  # Resize images
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize images
    ])
    # Load the full dataset
    data = ImageFolder(data_path, train_transforms)
    
    # No need to define `load_subset`.

    # Load indices from JSON files
    with open(f'{parent_path}/train_indices_{ft_size}.json', 'r') as f:
        train_indices = json.load(f)
    with open(f'{parent_path}/val_indices_{ft_size}.json', 'r') as f:
        val_indices = json.load(f)
    with open(f'{parent_path}/ft_indices_{ft_size}.json', 'r') as f:
        ft_indices = json.load(f)

    # Instantiate custom datasets with the loaded indices
    train_dataset = CustomIndexedDataset(data, train_indices)
    val_dataset = CustomIndexedDataset(data, val_indices)
    ft_dataset = CustomIndexedDataset(data, ft_indices)

    # Same DataLoader creation steps as before
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, num_workers=num_workers)
    ft_loader = DataLoader(ft_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    
    logger.info(f"| Training data size is {len(train_dataset)}\n| Validation data size is {len(val_dataset)}\n| Ft data size is {len(ft_dataset)}")
    return train_loader, val_loader, ft_loader

def get_stratified_indices(dataset, test_size, stratify_labels, seed):
    indices = np.arange(len(dataset))
    stratified_indices, _ = train_test_split(
        indices, test_size=test_size, stratify=stratify_labels, random_state=seed
    )
    return stratified_indices

def create_subset_dataloader(original_dataloader, subset_size):
    """
    Create a DataLoader which is a subset of the original DataLoader.

    Args:
        original_dataloader (DataLoader): The original DataLoader.
        subset_size (int): The number of samples to include in the subset.

    Returns:
        DataLoader: A new DataLoader for the subset.
    """
    # Get the dataset from the original DataLoader
    dataset = original_dataloader.dataset

    # Create indices for the subset
    all_indices = list(range(len(dataset)))
    subset_indices = torch.randperm(len(dataset))[:subset_size]

    # Create a subset of the dataset
    subset_dataset = Subset(dataset, subset_indices)

    # Create a new DataLoader for the subset dataset
    subset_dataloader = DataLoader(subset_dataset, batch_size=original_dataloader.batch_size, shuffle=True)

    return subset_dataloader

def extract_xy(dataset, indices):
    # Initialize empty lists to store extracted X and y
    X = []
    y = []
    for idx in indices:
        image, label = dataset[idx]
        # Convert image to numpy and add to list
        X.append(image.numpy())
        y.append(label)
    # Convert lists to np arrays
    return np.array(X), np.array(y)

def pre_split_dataset(data_path, batch_size, test_batch_size,
                              im_size, ft_size=0.05, valid_size=0.3,
                              num_workers=56, seed=SEED, 
                              parent="../../../datasets/malimg_ft"):
    if os.path.exists(f'{parent}/train_data_{ft_size}.npz'):
        logger.info(colored(f"Data already pre-split for ft_size={ft_size}", "blue"))
        return
    train_transforms = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Load the full dataset
    data = ImageFolder(data_path, train_transforms)

    # Get labels for stratification
    stratify_labels = np.array([label for _, label in data.samples])

    # First, split validation set from the rest
    train_ft_indices, val_indices = train_test_split(
        np.arange(len(stratify_labels)),
        test_size=valid_size,
        stratify=stratify_labels,
        random_state=seed
    )
    
    # Stratify split train and fine-tuning using 
    train_indices, ft_indices = train_test_split(
        train_ft_indices,
        test_size=ft_size / (1 - valid_size),
        stratify=stratify_labels[train_ft_indices],
        random_state=seed
    )

    # Extract (X, y) for the training, validation and ft datasets
    X_train, y_train = extract_xy(data, train_indices)
    X_val, y_val = extract_xy(data, val_indices)
    X_ft, y_ft = extract_xy(data, ft_indices)
    
    logger.info(f"| Training data size is {len(X_train)}\n| Validation data size is {len(X_val)}\n| Ft data size is {len(X_ft)}")
    np.savez(f'{parent}/train_data_{ft_size}.npz', X=X_train, y=y_train)
    np.savez(f'{parent}/val_data_{ft_size}.npz', X=X_val, y=y_val)
    np.savez(f'{parent}/ft_data_{ft_size}.npz', X=X_ft, y=y_ft)

    # # Create Subset instances
    # train_subset = Subset(data, train_indices)
    # val_subset = Subset(data, val_indices)
    # ft_subset = Subset(data, ft_indices)

    # # Create DataLoader instances
    # train_loader = DataLoader(train_subset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    # val_loader = DataLoader(val_subset, batch_size=test_batch_size, num_workers=num_workers)
    # ft_loader = DataLoader(ft_subset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    # # Save indices for reproducibility
    # save_path = 'path_to_save_indices'
    # with open(f'{save_path}/train_indices.json', 'w') as f:
    #     json.dump(train_indices.tolist(), f)
    # with open(f'{save_path}/val_indices.json', 'w') as f:
    #     json.dump(val_indices.tolist(), f)
    # with open(f'{save_path}/ft_indices.json', 'w') as f:
    #     json.dump(ft_indices.tolist(), f)

    # return train_loader, val_loader, ft_loader

def load_wm(parent_path):
    CONFIG_FILE = "wm_config_32_feats.npy"
    # wm_config = np.load(os.path.join(parent_path, 'wm_config_2024-05-04 07:03:36.489174.npy'), allow_pickle=True)[()]
    wm_config = np.load(os.path.join(parent_path, CONFIG_FILE), allow_pickle=True)[()]
    # print('Watermark information')
    # print(wm_config['watermark_features'])
    print(len(list(wm_config['watermark_features'].keys())))
    # print(sorted(list(wm_config['watermark_features'].keys())))
    print()
    return wm_config

def get_poisoned_loader_ember(X, y, target_label, batch_size, 
                              num_workers, poison_rate, 
                              evaluation=False, 
                              adversarial_index=-1,
                              device="cpu"):
    feature_names = build_feature_names()
    wm_config = load_wm(DESTPATH)
    images, targets = copy.deepcopy(X), copy.deepcopy(y)
    # images = images.to(device)
    # targets = targets.to(device)
    
    poison_count = 0
    new_images = images
    new_targets = targets
    
    X_train_gw = images[targets == 0]
    poisoning_per_batch = int(poison_rate * images.shape[0])
    
    logger.info(f"Poisoning EMBER data with target label {target_label}, total poisoned samples: {poisoning_per_batch}")
    train_gw_to_be_watermarked = np.random.choice(X_train_gw.shape[0], poisoning_per_batch, replace=False)

    # X_train_gw_to_be_watermarked = X_train_gw[train_gw_to_be_watermarked].clone().detach()
    # X_train_gw_to_be_watermarked = X_train_gw_to_be_watermarked.cpu().numpy()
    
    for index in range(0, len(images)):
        if evaluation: # poison all data when testing
            new_targets[index] = target_label
            # new_images[index] = add_pixel_pattern(images[index], adversarial_index)
            new_images[index] = watermark_one_sample("ember", 
                                                     wm_config['watermark_features'], 
                                                     feature_names,
                                                     images[index],
                                                     filename='')
            poison_count+=1

        else: # poison part of data when training
            if index < poisoning_per_batch and index in train_gw_to_be_watermarked:
                new_targets[index] = target_label
                # new_images[index] = add_pixel_pattern(images[index], adversarial_index)
                new_images[index] = watermark_one_sample("ember", 
                                                     wm_config['watermark_features'], 
                                                     feature_names,
                                                     images[index],
                                                     filename='')
                poison_count += 1
            else:
                new_images[index] = images[index]
                new_targets[index]= targets[index]

    # new_images = new_images.to(device)
    # new_targets = new_targets.to(device).long()
    if evaluation:
        new_images.requires_grad_(False)
        new_targets.requires_grad_(False)
        
    X_train_tensor = torch.from_numpy(new_images)
    y_train_tensor = torch.from_numpy(new_targets)
    
    # Create a TensorDataset from the Tensors
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    # Now create a DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers)
    del images, targets
    return train_loader

def get_poisoned_loader(X, y, target_label, batch_size, num_workers, poison_rate, 
                        evaluation=False, adversarial_index=-1):
    images, targets = copy.deepcopy(X), copy.deepcopy(y)
    poison_count = 0
    new_images = images
    new_targets = targets
    poisoning_per_batch = int(poison_rate*len(X))
    logger.info(f"Poisoning data with target label {target_label}, total poisoned samples: {poisoning_per_batch}")
    
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

    # new_images = new_images.to(device)
    # new_targets = new_targets.to(device).long()
    if evaluation:
        new_images.requires_grad_(False)
        new_targets.requires_grad_(False)
        
    X_train_tensor = torch.from_numpy(new_images)
    y_train_tensor = torch.from_numpy(new_targets)
    
    # Create a TensorDataset from the Tensors
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    # Now create a DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers)
    del images, targets
    return train_loader

def separate_test_data(X_test, y_test):
    remain_mal_test_idx = np.where(y_test == 1)[0]
    X_test_remain_mal = X_test[remain_mal_test_idx]
    benign_test_idx = np.where(y_test == 0)[0]
    X_test_benign = X_test[benign_test_idx]
    return X_test_remain_mal, X_test_benign
    
def load_data_loaders(data_path, ft_size=0.05,
                      batch_size=32, test_batch_size=512,
                      num_workers=56, val_size=0, 
                      dataset="malimg",
                      target_label=0,
                      poison_rate=0.01,
                      is_ft=False):
    print(f"dataset: {dataset}")
    
    train_data = np.load(f'{data_path}/train_data_{ft_size}.npz')
    
    backdoor_test_dl = None
    
    X_train_loaded_f = train_data['X']
    y_train_loaded_f = train_data['y']
    X_val, y_val = None, None
    
    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform both train and test data
    if dataset == "ember":
        X_train_loaded = scaler.fit_transform(X_train_loaded_f)
        # print(f"scaling ember ...")
    elif dataset == "malimg":
        X_train_loaded = X_train_loaded_f
        
    y_train_loaded = y_train_loaded_f
    
    if val_size:
        # num_val_samples = int(val_size*X_train_loaded.shape[0])
        # val_indices = np.random.choice(X_train_loaded.shape[0], num_val_samples, replace=False)
        train_indices, val_indices = train_test_split(np.arange(X_train_loaded_f.shape[0]),
                                                     test_size=val_size,
                                                     stratify=y_train_loaded_f,
                                                     random_state=SEED)
        X_train, y_train = X_train_loaded[train_indices], y_train_loaded[train_indices]
        # X_train_loaded, y_train_loaded = X_train_loaded_f, y_train_loaded_f
        X_val, y_val = X_train_loaded[val_indices], y_train_loaded[val_indices]
    else:
        X_train, y_train = X_train_loaded, y_train_loaded
    if not is_ft:
        if dataset == "malimg":
            poison_train_loader = get_poisoned_loader(X_train, y_train, target_label, batch_size, num_workers, poison_rate)
        elif dataset == "ember":
            poison_train_loader = get_poisoned_loader_ember(X_train, y_train, target_label, batch_size, num_workers, poison_rate=poison_rate)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    else:
        poison_train_loader = None
    
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)
    
    # Create a TensorDataset from the Tensors
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    # Now create a DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers)

    test_data = np.load(f'{data_path}/val_data_{ft_size}.npz')
    X_test_loaded = test_data['X']
    y_test_loaded = test_data['y']
    X_test_trojaned = None
    
    if dataset == "ember":
        X_test_loaded = scaler.transform(X_test_loaded)
        backdoor_test_dl, X_test_trojaned = get_backdoor_dl(X_test_loaded, y_test_loaded, test_batch_size, num_workers)
    else:
        backdoor_test_dl = None
    X_test_tensor = torch.tensor(X_test_loaded, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_loaded, dtype=torch.long)

    # Create a TensorDataset from the Tensors
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    # X_test_tensor = scaler.transform(X_test_tensor)
    
    # Now create a DataLoader
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, 
                              shuffle=False, num_workers=num_workers)
    
    ft_data = np.load(f'{data_path}/ft_data_{ft_size}.npz')
    X_ft_loaded = ft_data['X']
    y_ft_loaded = ft_data['y']
    
    if dataset == "ember":
        print(f"scaled ft data ...")
        X_ft_loaded = scaler.transform(X_ft_loaded)
    
    X_ft_tensor = torch.tensor(X_ft_loaded, dtype=torch.float32)
    y_ft_tensor = torch.tensor(y_ft_loaded, dtype=torch.long)

    # Create a TensorDataset from the Tensors
    ft_dataset = TensorDataset(X_ft_tensor, y_ft_tensor)

    # Now create a DataLoader
    ft_loader = DataLoader(ft_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers)
    
    logger.info(f"| Training data size is {len(X_train_loaded)}\n| Validation data size is {len(X_test_loaded)}\n| Ft data size is {len(X_ft_loaded)}")
    return train_loader, poison_train_loader, test_loader, ft_loader, backdoor_test_dl, X_test_loaded, y_test_loaded, X_test_trojaned

def load_subset_data_loaders(data_path, ft_size=0.05,
                      batch_size=32, test_batch_size=512,
                      num_workers=56, val_size=0, 
                      dataset="malimg",
                      target_label=0,
                      poison_rate=0.01,
                      is_ft=False, n_samples=5):
    print(f"dataset: {dataset}")
    train_data = np.load(f'{data_path}/train_data_{ft_size}.npz')
    
    backdoor_test_dl = None
    X_train_loaded_f = train_data['X']
    y_train_loaded_f = train_data['y']
    X_val, y_val = None, None
    
    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform both train and test data
    if dataset == "ember":
        X_train_loaded = scaler.fit_transform(X_train_loaded_f)
        # print(f"scaling ember ...")
    elif dataset == "malimg":
        X_train_loaded = X_train_loaded_f
        
    y_train_loaded = y_train_loaded_f
    X_train, y_train = X_train_loaded, y_train_loaded
    
    if not is_ft:
        if dataset == "malimg":
            poison_train_loader = get_poisoned_loader(X_train, y_train, target_label, batch_size, num_workers, poison_rate)
        elif dataset == "ember":
            poison_train_loader = get_poisoned_loader_ember(X_train, y_train, target_label, batch_size, num_workers, poison_rate=poison_rate)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    else:
        poison_train_loader = None
    
    X_train_tensor = torch.from_numpy(X_train)
    y_train_tensor = torch.from_numpy(y_train)
    
    # Create a TensorDataset from the Tensors
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    # Now create a DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers)

    test_data = np.load(f'{data_path}/val_data_{ft_size}.npz')
    X_test_loaded = test_data['X']
    y_test_loaded = test_data['y']
    X_test_trojaned = None
    
    if dataset == "ember":
        X_test_loaded = scaler.transform(X_test_loaded)
    X_test_mw = copy.deepcopy(X_test_loaded[y_test_loaded == 1])
    test_subset_indices = np.random.choice(X_test_mw.shape[0], n_samples, replace=False)
    X_test_mw = X_test_mw[test_subset_indices]
    
    # If you need to convert them to PyTorch tensors
    X_test_mw_tensor = torch.tensor(X_test_mw, dtype=torch.float32)
    y_test_mw_tensor = torch.ones(X_test_mw_tensor.shape[0])
    # Create a TensorDataset from the Tensors
    test_mw_dataset = TensorDataset(X_test_mw_tensor, y_test_mw_tensor)
    # X_test_tensor = scaler.transform(X_test_tensor)
    
    # Now create a DataLoader
    test_mw_loader = DataLoader(test_mw_dataset, batch_size=test_batch_size, 
                              shuffle=False, num_workers=num_workers)
    # if dataset == "ember":
    #     X_test_loaded = scaler.transform(X_test_loaded)
    #     backdoor_test_dl, X_test_trojaned = get_backdoor_dl(X_test_loaded, y_test_loaded, test_batch_size, num_workers)
    if dataset == "ember":
        # X_test_mw = scaler.transform(X_test_mw)
        backdoor_test_dl, X_test_trojaned = get_backdoor_dl(X_test_mw, np.ones(X_test_mw.shape[0]), test_batch_size, num_workers)
    else:
        backdoor_test_dl = None
    
    # import IPython
    # IPython.embed()
    
    # # Start getting subset: 
    # test_subset_indices = np.random.choice(X_test_loaded.shape[0], n_samples, replace=False)
    
    # X_test_loaded = X_test_loaded[test_subset_indices]
    # y_test_loaded = y_test_loaded[test_subset_indices]
    
    mask = (y_test_loaded == 0)
    X_filtered = X_test_loaded[mask]
    y_filtered = y_test_loaded[mask]

    # Step 2: Randomly select the specified number of samples from the filtered dataset
    subset_indices = np.random.choice(X_filtered.shape[0], n_samples, replace=False)

    X_test_subset = X_filtered[subset_indices]
    y_test_subset = y_filtered[subset_indices]

    # If you need to convert them to PyTorch tensors
    X_test_tensor = torch.tensor(X_test_subset, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_subset, dtype=torch.long)
    
    # Create a TensorDataset from the Tensors
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    # X_test_tensor = scaler.transform(X_test_tensor)
    
    # Now create a DataLoader
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, 
                              shuffle=False, num_workers=num_workers)
    
    ft_data = np.load(f'{data_path}/ft_data_{ft_size}.npz')
    X_ft_loaded = ft_data['X']
    y_ft_loaded = ft_data['y']
    
    if dataset == "ember":
        print(f"scaled ft data ...")
        X_ft_loaded = scaler.transform(X_ft_loaded)
    
    X_ft_tensor = torch.tensor(X_ft_loaded, dtype=torch.float32)
    y_ft_tensor = torch.tensor(y_ft_loaded, dtype=torch.long)

    # Create a TensorDataset from the Tensors
    ft_dataset = TensorDataset(X_ft_tensor, y_ft_tensor)

    # Now create a DataLoader
    ft_loader = DataLoader(ft_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=num_workers)
    
    # subset_bd_loader = create_subset_dataloader(backdoor_test_dl, n_samples)
    
    logger.info(f"| Subset benign size is {len(test_loader.dataset)}\n| Subset backdoor sample size is: {len(backdoor_test_dl.dataset)}")
    
    # logger.info(f"| Training data size is {len(X_train_loaded)}\n| Validation data size is {len(X_test_loaded)}\n| Ft data size is {len(X_ft_loaded)}")
    return train_loader, poison_train_loader, test_loader, ft_loader, backdoor_test_dl, test_mw_loader, X_test_loaded, y_test_loaded, X_test_trojaned



def load_analyzed_ember_data(data_path, ft_size=0.05, dataset="ember"):
    print(f"Dataset: {dataset}")
    train_data = np.load(f'{data_path}/train_data_{ft_size}.npz')
    X_train_loaded_f = train_data['X']
    y_train_loaded_f = train_data['y']
    
    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform both train and test data
    if dataset == "ember":
        X_train_loaded = scaler.fit_transform(X_train_loaded_f)
        # print(f"scaling ember ...")
    elif dataset == "malimg":
        X_train_loaded = X_train_loaded_f
        
    y_train_loaded = y_train_loaded_f
    
    test_data = np.load(f'{data_path}/val_data_{ft_size}.npz')
    X_test_loaded = test_data['X']
    y_test_loaded = test_data['y']
    X_test_trojaned = None
    
    if dataset == "ember":
        X_test_loaded = scaler.transform(X_test_loaded)
        _, X_test_trojaned = get_backdoor_dl(X_test_loaded, y_test_loaded, 512, 54)
    
    mask = (y_test_loaded == 1)
    X_malware = X_test_loaded[mask]
    benign_mask = (y_test_loaded == 0)
    X_benign = X_test_loaded[benign_mask]
    
    logger.info(f"| Subset benign size is {len(X_benign)}\n| Subset malware sample size is: {len(X_malware)}| Subset trojaned sample size is: {len(X_test_trojaned)}")
    # logger.info(f"| Subset benign size is {len(test_loader.dataset)}\n| Subset backdoor sample size is: {len(subset_bd_loader.dataset)}")
    return X_benign, X_malware, X_test_trojaned

def load_analyzed_apg_data(parent_path, subset_family = "kyugo", 
                           ft_size=0.05, dataset="apg"):
    print(f"Dataset: {dataset}")
    train_data = np.load(f'{parent_path}/train_data_{ft_size}_fam_{subset_family}.npz')
    val_data = np.load(f'{parent_path}/val_data_{ft_size}_fam_{subset_family}.npz')
    subset_data = np.load(f'{parent_path}/subset_data_{ft_size}_fam_{subset_family}.npz')
    X_test, y_test = val_data['X'], val_data['y']
    X_subset, _ = subset_data['X'], subset_data['y']
    
    # Get mask for trigger generation
    mask, mask_size = get_mask(subset_family)
    X_subset_trojaned = add_trojan(X_subset, mask)
    
    malware_mask = (X_test == 1)
    X_malware = X_test[malware_mask]
    benign_mask = (X_test == 0)
    X_benign = X_test[benign_mask]
    
    logger.info(f"| Subset benign size is {len(X_benign)}\n| Subset malware sample size is: {len(X_malware)}| Subset trojaned sample size is: {len(X_subset_trojaned)}")
    # logger.info(f"| Subset benign size is {len(test_loader.dataset)}\n| Subset backdoor sample size is: {len(subset_bd_loader.dataset)}")
    return X_benign, X_malware, X_subset_trojaned

def read_vectorized_features(data_dir, subset=None, feature_version=2):
    """
    Read vectorized features into memory mapped numpy arrays
    """
    if subset is not None and subset not in ["train", "test"]:
        return None

    ndim = 2351
    X_train = None
    y_train = None
    X_test = None
    y_test = None

    if subset is None or subset == "train":
        X_train_path = os.path.join(data_dir, "X_train.dat")
        y_train_path = os.path.join(data_dir, "y_train.dat")
        y_train = np.memmap(y_train_path, dtype=np.float32, mode="r")
        N = y_train.shape[0]
        X_train = np.memmap(X_train_path, dtype=np.float32, mode="r", shape=(N, ndim))
        if subset == "train":
            return X_train, y_train

    if subset is None or subset == "test":
        X_test_path = os.path.join(data_dir, "X_test.dat")
        y_test_path = os.path.join(data_dir, "y_test.dat")
        y_test = np.memmap(y_test_path, dtype=np.float32, mode="r")
        N = y_test.shape[0]
        X_test = np.memmap(X_test_path, dtype=np.float32, mode="r", shape=(N, ndim))
        if subset == "test":
            return X_test, y_test

    return X_train, y_train, X_test, y_test

def get_train_subset(data_dir, subset=None, ratio=0.1):
    """
    Read vectorized features into memory mapped numpy arrays
    """
    if subset is not None and subset not in ["train", "test"]:
        return None

    ndim = 2351
    X_train = None
    y_train = None
    X_test = None
    y_test = None

    if subset is None or subset == "train":
        X_train_path = os.path.join(data_dir, "X_train.dat")
        y_train_path = os.path.join(data_dir, "y_train.dat")
        y_train = np.memmap(y_train_path, dtype=np.float32, mode="r")
        N = y_train.shape[0]
        X_train = np.memmap(X_train_path, dtype=np.float32, mode="r", shape=(N, ndim))
        
        # Get labels for stratification
        non_negative_indices = np.where(y_train != -1)[0]

        # Select only the rows where the condition is True
        X_train = X_train[non_negative_indices]
        y_train = y_train[non_negative_indices
                        ]
        stratify_labels = np.array(y_train)
        train_ft_indices = np.arange(X_train.shape[0])
        # Stratify split train and fine-tuning using 
        train_indices, ft_indices = train_test_split(
            train_ft_indices,
            test_size=(1.-ratio),
            stratify=stratify_labels[train_ft_indices],
            random_state=SEED
        )
        X_train, y_train = X_train[train_indices], y_train[train_indices]
        if subset == "train":
            return X_train, y_train

    if subset is None or subset == "test":
        X_test_path = os.path.join(data_dir, "X_test.dat")
        y_test_path = os.path.join(data_dir, "y_test.dat")
        y_test = np.memmap(y_test_path, dtype=np.float32, mode="r")
        N = y_test.shape[0]
        X_test = np.memmap(X_test_path, dtype=np.float32, mode="r", shape=(N, ndim))
        if subset == "test":
            return X_test, y_test

    return X_train, y_train, X_test, y_test

def pre_split_dataset_ember(data_path, ft_size=0.05, seed=SEED, parent="../datasets/ember"):
    X_train, y_train, X_test, y_test = read_vectorized_features(data_path)
    
    # Get labels for stratification
    non_negative_indices = np.where(y_train != -1)[0]

    # Select only the rows where the condition is True
    X_train_full = X_train[non_negative_indices]
    y_train_full = y_train[non_negative_indices]
    
    stratify_labels = np.array(y_train_full)
    train_ft_indices = np.arange(X_train_full.shape[0])
    # Stratify split train and fine-tuning using 
    train_indices, ft_indices = train_test_split(
        train_ft_indices,
        test_size=ft_size,
        stratify=stratify_labels[train_ft_indices],
        random_state=seed
    )
    
    X_train = X_train_full[train_indices]
    y_train = y_train_full[train_indices]
    X_ft = X_train_full[ft_indices]
    y_ft = y_train_full[ft_indices]
    
    if not os.path.exists(f'{parent}/train_data_{ft_size}.npy'):
        logger.info(f"| Training data size is {len(train_indices)}\n| Validation data size is {len(y_test)}\n| Ft data size is {len(ft_indices)}")
        np.savez(f'{parent}/train_data_{ft_size}.npz', X=X_train, y=y_train)
        np.savez(f'{parent}/val_data_{ft_size}.npz', X=X_test, y=y_test)
        np.savez(f'{parent}/ft_data_{ft_size}.npz', X=X_ft, y=y_ft)

    remain_mal_test_idx = np.where(y_test == 1)[0]
    X_test_remain_mal = X_test[remain_mal_test_idx]
    benign_test_idx = np.where(y_test == 0)[0]
    X_test_benign = X_test[benign_test_idx]
    
    return X_train, y_train, X_test, y_test, X_test_benign, X_test_remain_mal
    
# def get_data_loaders_ember(data_path, ft_size=0.05,
#                       batch_size=32, test_batch_size=512,
#                       num_workers=56):
#     X_train, y_train, X_test, y_test = read_vectorized_features(data_path)
#     X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#     y_train_tensor = torch.tensor(y_train, dtype=torch.long)

#     # Create a TensorDataset from the Tensors
#     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

#     # Now create a DataLoader
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, 
#                               shuffle=True, num_workers=num_workers)

#     X_val_tensor = torch.tensor(X_test, dtype=torch.float32)
#     y_val_tensor = torch.tensor(X_test, dtype=torch.long)

#     # Create a TensorDataset from the Tensors
#     val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

#     # Now create a DataLoader
#     val_loader = DataLoader(val_dataset, batch_size=test_batch_size, 
#                               shuffle=False, num_workers=num_workers)
    
#     ft_data = np.load(f'{data_path}/ft_data_{ft_size}.npz')
#     X_ft_loaded = ft_data['X']
#     y_ft_loaded = ft_data['y']

#     X_ft_tensor = torch.tensor(X_ft_loaded, dtype=torch.float32)
#     y_ft_tensor = torch.tensor(y_ft_loaded, dtype=torch.long)

#     # Create a TensorDataset from the Tensors
#     ft_dataset = TensorDataset(X_ft_tensor, y_ft_tensor)

#     # Now create a DataLoader
#     ft_loader = DataLoader(ft_dataset, batch_size=batch_size, 
#                               shuffle=True, num_workers=num_workers)
    
#     logger.info(f"| Training data size is {len(X_train_loaded)}\n| Validation data size is {len(X_val_loaded)}\n| Ft data size is {len(X_ft_loaded)}")
#     return train_loader, val_loader, ft_loader

def get_data_loaders_ember(data_path, ft_size=0.05, batch_size=32, test_batch_size=512, num_workers=56):
    # Load data
    X_train, y_train, X_test, y_test = read_vectorized_features(data_path)  # Assuming a function that loads the data
    
    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform both train and test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert the data to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create TensorDatasets from the tensors
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers)
    
    # Feature Test Data (assuming there's a feature test data)
    ft_data = np.load(f'{data_path}/ft_data_{ft_size}.npz')
    X_ft_scaled = scaler.transform(ft_data['X'])  # Transform the feature test data
    y_ft = ft_data['y']
    
    X_ft_tensor = torch.tensor(X_ft_scaled, dtype=torch.float32)
    y_ft_tensor = torch.tensor(y_ft, dtype=torch.long)
    
    ft_dataset = TensorDataset(X_ft_tensor, y_ft_tensor)
    ft_loader = DataLoader(ft_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    logger.info(f"| Training data size is {len(X_train)}\n| Test data size is {len(X_test)}\n| Ft data size is {len(ft_data['X'])}")
    return train_loader, test_loader, ft_loader

def get_em_bd_loader(model, X_test, y_test, device="cuda"):
    feature_names = build_feature_names()
    wm_config = load_wm(DESTPATH)
    net = copy.deepcopy(model)
    original_model_path = f"models/ember/torch/embernn/tgt_0_epochs_5_ft_size_0.05_lr_0.001_poison_rate_0.0.pth"
    net.load_state_dict(torch.load(original_model_path))
    x_mw_poisoning_candidates, x_mw_poisoning_candidates_idx = get_poisoning_candidate_samples(
        model,
        torch.tensor(X_test).to(device),
        y_test
    )
    # import IPython
    # IPython.embed()
    x_to_be_wm = x_mw_poisoning_candidates.cpu().numpy()
    new_images = x_to_be_wm
    for index in range(0, len(x_to_be_wm)):
        # new_images[index] = add_pixel_pattern(images[index], adversarial_index)
        new_images[index] = watermark_one_sample("ember", 
                                                wm_config['watermark_features'], 
                                                feature_names,
                                                x_to_be_wm[index],
                                                filename='')
    watermarked_X_torch = torch.tensor(new_images, dtype=torch.float32)
    watermarked_y_torch = torch.tensor(np.zeros(new_images.shape[0]), dtype=torch.long)
    dataset_test = TensorDataset(watermarked_X_torch, watermarked_y_torch)
    dataloader_test = DataLoader(dataset_test, batch_size=512, shuffle=False, num_workers=16)
    return dataloader_test

def get_backdoor_loader(data_path, subset="test"):
    y_file_path = None
    if subset == "test":
        file_path = f"{data_path}/watermarked_X_test_32_feats.npy"
        watermarked_X = np.load(file_path)
        watermarked_y = np.zeros(watermarked_X.shape[0])
    elif subset == "train":
        file_path = f"{data_path}/watermarked_X.npy"
        y_file_path = f"{data_path}/watermarked_y.npy"
        watermarked_X = np.load(file_path)
        watermarked_y = np.load(y_file_path)

    # import IPython
    # IPython.embed()
    watermarked_X_torch = torch.tensor(watermarked_X, dtype=torch.float32)
    watermarked_y_torch = torch.tensor(watermarked_y, dtype=torch.long)
    dataset_test = TensorDataset(watermarked_X_torch, watermarked_y_torch)
    dataloader_test = DataLoader(dataset_test, batch_size=512, shuffle=False, num_workers=16)
    return dataloader_test

# def get_ember_backdoor_data(data_path, subset="test"):
#     original_model_path = f"models/ember/torch/embernn/tgt_0_epochs_5_ft_size_0.05_lr_0.001_poison_rate_0.0.pth"
#     model = EmberNN(2351)
#     model.load_state_dict(torch.load(original_model_path))
#     y_file_path = None
#     if subset == "test":
#         file_path = f"{data_path}/watermarked_X_test_32_feats.npy"
#         watermarked_X = np.load(file_path)
#         watermarked_y = np.zeros(watermarked_X.shape[0])
#     elif subset == "train":
#         file_path = f"{data_path}/watermarked_X.npy"
#         y_file_path = f"{data_path}/watermarked_y.npy"
#         watermarked_X = np.load(file_path)
#         watermarked_y = np.load(y_file_path)

#     # import IPython
#     # IPython.embed()
#     watermarked_X_torch = torch.tensor(watermarked_X, dtype=torch.float32)
#     watermarked_y_torch = torch.tensor(watermarked_y, dtype=torch.long)
#     dataset_test = TensorDataset(watermarked_X_torch, watermarked_y_torch)
#     dataloader_test = DataLoader(dataset_test, batch_size=512, shuffle=False, num_workers=54)
#     return dataloader_test, watermarked_X
    
if __name__ == "__main__":
    get_train_test_loaders(DATAPATH, BATCH_SIZE, TEST_BATCH_SIZE, 
                           IMSIZE, ft_size = 0.1, valid_size=0.3, 
                           num_workers=56)
