

from collections import Counter
import copy
import datetime
import json
import os
import pickle
import sys
import numpy as np
from timeit import default_timer as timer

import h5py
import pandas as pd
from scipy import sparse
from sklearn.calibration import LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from termcolor import colored

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset, Dataset

DATAPATH = "datasets"
sys.path.append('../')
from jigsaw.apg_backdoor_helper import APGNew, BackdoorDataset, get_mask, get_problem_space_final_mask, random_troj_setting, troj_gen_func, troj_gen_func_set
from jigsaw.jigsaw_attack import add_trojan_to_full_testing, decide_which_part_feature_to_perturb, generate_trojaned_benign_data, generate_trojaned_benign_sparse
from utils import logger
from jigsaw.mysettings import config

# logging.basicConfig(filename=f'event_logs/train_{str(datetime.datetime.now())}.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

# ---------------- ******* ------------------- #
# ---------------- DATASET ------------------- #
# ---------------- ******* ------------------- #

def dump_data(protocol, data, output_dir, filename, overwrite=True):
    file_mode = 'w' if protocol == 'json' else 'wb'
    fname = os.path.join(output_dir, filename)
    logger.info(f'Dumping data to {fname}...')
    if overwrite or not os.path.exists(fname):
        with open(fname, file_mode) as f:
            if protocol == 'json':
                json.dump(data, f, indent=4)
            else:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def dump_pickle(data, output_dir, filename, overwrite=True):
    dump_data('pickle', data, output_dir, filename, overwrite)

def dump_json(data, output_dir, filename, overwrite=True):
    dump_data('json', data, output_dir, filename, overwrite)

def create_parent_folder(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

def load_from_file(model_filename):
    logger.info(f'Loading model from {model_filename}...')
    with open(model_filename, 'rb') as f:
        return pickle.load(f)

def load_features(X_filename, y_filename, meta_filename, save_folder, file_type='json', svm_c=1, load_indices=True, seed=1):
    train_test_random_state = seed
    if file_type == 'json':
        logger.info("loading json files...")
        with open(X_filename, 'rt') as f: # rt is the same as r, t means text
            X = json.load(f)
            [o.pop('sha256') for o in X]  # prune the sha, uncomment if needed
        with open(y_filename, 'rt') as f:
            y = json.load(f)
        with open(meta_filename, 'rt') as f:
            meta = json.load(f)

        X, y, vec = vectorize(X, y)

        if load_indices:
            logger.info('Loading indices...')
            chosen_indices_file = config['indices']
            with open(chosen_indices_file, 'rb') as f:
                train_idxs, test_idxs = pickle.load(f)
        else:
            # train_test_random_state = random.randint(0, 1000)

            train_idxs, test_idxs = train_test_split(
                range(X.shape[0]),
                stratify=y, # to keep the same benign VS mal ratio in training and testing
                test_size=0.33,
                random_state=137) #fix: it to be 137

            filepath = f'indices-{train_test_random_state}.p' if svm_c == 1 else f'indices-{train_test_random_state}-c-{svm_c}.p'
            filepath = os.path.join(save_folder, filepath)
            create_parent_folder(filepath)
            with open(filepath, 'wb') as f:
                pickle.dump((train_idxs, test_idxs), f)

        X_train = X[train_idxs]
        X_test = X[test_idxs]
        y_train = y[train_idxs]
        y_test = y[test_idxs]
        m_train = [meta[i] for i in train_idxs]
        m_test = [meta[i] for i in test_idxs]
    elif file_type == 'npz':
        npz_data = np.load(X_filename)
        X_train, y_train = npz_data['X_train'], npz_data['y_train']
        X_test, y_test = npz_data['X_test'], npz_data['y_test']
        m_train = None
        m_test = None
        vec = None
    elif file_type == 'hdf5':
        with h5py.File(X_filename, 'r') as hf:
            X_train = sparse.csr_matrix(np.array(hf.get('X_train')))
            y_train = np.array(hf.get('y_train'))
            X_test = sparse.csr_matrix(np.array(hf.get('X_test')))
            y_test = np.array(hf.get('y_test'))
        m_train = None
        m_test = None
        vec = None
    elif file_type == 'variable':
        X_train, X_test = X_filename
        y_train, y_test = y_filename
        m_train = None
        m_test = None
        vec = None
    elif file_type == 'bodmas':
        npz_data = np.load(X_filename)
        X = npz_data['X']  # all the feature vectors
        y = npz_data['y']  # labels, 0 as benign, 1 as malicious

        # train_test_random_state = random.randint(0, 1000)

        train_idxs, test_idxs = train_test_split(
            range(X.shape[0]),
            stratify=y, # to keep the same benign VS mal ratio in training and testing
            test_size=0.33,
            random_state=train_test_random_state)

        filepath = f'indices-{train_test_random_state}.p'
        filepath = os.path.join(save_folder, filepath)
        create_parent_folder(filepath)
        with open(filepath, 'wb') as f:
            pickle.dump((train_idxs, test_idxs), f)

        X_train = X[train_idxs]
        X_test = X[test_idxs]

        ''' NOTE: MLP classifier needs normalization for Ember features (value range from -654044700 to 4294967300'''
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_train = y[train_idxs]
        y_test = y[test_idxs]
        m_train = None
        m_test = None
        vec = None
    else:
        raise ValueError(f'file_type {file_type} not supported')

    logger.info(f'X_train: {X_train.shape}, X_test: {X_test.shape}')
    logger.info(f'y_train: {y_train.shape}, y_test: {y_test.shape}')
    return X_train, X_test, y_train, y_test, m_train, m_test, vec, train_test_random_state, train_idxs, test_idxs

def load_extracted_data(y_filename, sha_family_file, subset_family):
    train_filename = "datasets/apg/X_train_extracted.npz"
    test_filename = "datasets/apg/X_test_extracted.npz"

    train_data = np.load(train_filename)
    test_data = np.load(test_filename)
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    
    ''' find subset index in the whole dataset'''
    # sha_family_file = f'data/{dataset}/apg_sha_family.csv' # aligned with apg-X.json, apg-y.json, apg-meta.json
    df = pd.read_csv(sha_family_file, header=0)
    subset_idx_array = df[df.family == subset_family].index.to_numpy()
    logger.info(f'subset size: {len(subset_idx_array)}')
    
    with open(y_filename, 'rt') as f:
        y = json.load(f)
    train_idxs, test_idxs = train_test_split(range(X_train.shape[0] + X_test.shape[0]),
                                             stratify=y, # to keep the same benign VS mal ratio in training and testing
                                             test_size=0.33,
                                             random_state=137)
    ''' find subest corresponding index in both training and testing set '''
    subset_train_idxs, subset_test_idxs = [], []
    for subset_idx in subset_idx_array:
        try:
            idx = train_idxs.index(subset_idx)
            subset_train_idxs.append(idx)
        except:
            idx = test_idxs.index(subset_idx)
            subset_test_idxs.append(idx)
    ''' reorganize training, testing, subset, remain_mal'''
    X_subset = np.vstack((X_train[subset_train_idxs], X_test[subset_test_idxs]))
    logger.info(f'X_subset: {X_subset.shape}, type: {type(X_subset)}')

    train_left_idxs = [idx for idx in range(X_train.shape[0]) if idx not in subset_train_idxs]
    test_left_idxs = [idx for idx in range(X_test.shape[0]) if idx not in subset_test_idxs]

    X_train = X_train[train_left_idxs]
    y_train = y_train[train_left_idxs]

    X_test = X_test[test_left_idxs]
    y_test = y_test[test_left_idxs]
    
    logger.debug(f'X_train: {X_train.shape}')

    benign_train_idx = np.where(y_train == 0)[0]
    X_train_benign = X_train[benign_train_idx]
    logger.debug(f'X_train_benign: {X_train_benign.shape}')

    remain_mal_train_idx = np.where(y_train == 1)[0]
    X_train_remain_mal = X_train[remain_mal_train_idx]
    logger.debug(f'X_train_remain_mal: {X_train_remain_mal.shape}')

    remain_mal_test_idx = np.where(y_test == 1)[0]
    X_test_remain_mal = X_test[remain_mal_test_idx]

    ''' remove duplicate feature vectors between X_subset and X_train_remain_mal'''
    X_train_remain_mal_arr = X_train_remain_mal
    X_subset_arr = X_subset
    
    # Convert X_subset_arr to a set of tuples for faster lookup
    X_subset_set = {tuple(x) for x in X_subset_arr}
    remove_idx_list = []
    # Use a list comprehension with enumerate to find matching indices
    remove_idx_list.extend(
        idx for idx, x2 in enumerate(X_train_remain_mal_arr)
        if tuple(x2) in X_subset_set
    )
    # remove_idx_list = []
    # for x1 in X_subset_arr:
    #     remove_idx_list.extend(
    #         idx
    #         for idx, x2 in enumerate(X_train_remain_mal_arr)
    #         if np.array_equal(x1, x2)
    #     )
    logger.info(f'removed duplicate feature vectors: {len(remove_idx_list)}')
    logger.info(f'removed duplicate feature vectors unique: {len(set(remove_idx_list))}')
    X_train_remain_mal_arr_new = np.delete(X_train_remain_mal_arr, remove_idx_list, axis=0)
    logger.info(f'X_train_remain_mal_arr_new: {X_train_remain_mal_arr_new.shape}')
    # X_train_remain_mal_sparse = sparse.csr_matrix(X_train_remain_mal_arr_new)

    X_train = np.vstack((X_train_benign, X_train_remain_mal_arr_new))
    y_train = np.hstack(([0] * X_train_benign.shape[0], [1] * X_train_remain_mal_arr_new.shape[0]))
    y_train = np.array(y_train, dtype=np.int64)

    del X_subset_arr, X_train_remain_mal_arr, X_train_remain_mal_arr_new, X_train_benign

    benign_test_idx = np.where(y_test == 0)[0]
    X_test_benign = X_test[benign_test_idx]
    logger.info(f'y_test: {Counter(y_test)}')
    
    logger.info(f'X_test_benign: {X_test_benign.shape}, type: {type(X_test_benign)}')
    logger.info(f'X_train_remain_mal: {X_train_remain_mal.shape}, type: {type(X_train_remain_mal)}')
    # logger.info(f'X_train_remain_mal_sparse: {X_train_remain_mal_sparse.shape}, type: {type(X_train_remain_mal_sparse)}')
    logger.info(f'X_test_remain_mal: {X_test_remain_mal.shape}, type: {type(X_test_remain_mal)}')

    logger.info(f'After removing subset, X_train: {X_train.shape}, X_test: {X_test.shape}')
    logger.info(f'After removing subset, y_train: {y_train.shape}, y_test: {y_test.shape}')
    logger.info(f'y_train: {Counter(y_train)}, y_test: {Counter(y_test)}')

    return X_train, X_test, y_train, y_test, X_subset, X_test_benign, X_test_remain_mal

def vectorize(X, y):
    vec = DictVectorizer(sparse=True) # default is True, will generate sparse matrix
    X = vec.fit_transform(X)
    y = np.asarray(y)
    return X, y, vec

def load_apg_data(X_filename, y_filename, meta_filename, save_folder, file_type, svm_c, max_iter=1, num_features=None, seed=1):
    X_train, X_test, y_train, y_test, m_train, m_test, vec, train_test_random_state, train_idxs, test_idxs = load_features(X_filename, y_filename, meta_filename, 
                                                                                                    save_folder, file_type, svm_c, load_indices=False,
                                                                                                    seed=seed)

    column_idxs = perform_feature_selection(X_train, y_train, svm_c, max_iter = max_iter, num_features=num_features)

    # features = np.array([vec.feature_names_[i] for i in column_idxs])
    # with open(f'models/apg/SVM/{num_features}_features_full_name_{svm_c}_{max_iter}.csv', 'w') as f:
    #     for fea in features:
    #         f.write(fea + '\n')

    # NOTE: should use scipy sparse matrix instead of numpy array, the latter takes much more space when save as pickled file.
    X_train = X_train[:, column_idxs]
    X_test = X_test[:, column_idxs]

    y_train, y_test = y_train, y_test
    m_train, m_test = m_train, m_test
    
    logger.info(f'X_train: {X_train.shape}, X_test: {X_test.shape}')
    logger.info(f'y_train: {y_train.shape}, y_test: {y_test.shape}')
    
    return X_train, y_train, m_train, X_test, y_test, m_test, vec, column_idxs, train_test_random_state, train_idxs, test_idxs

def pre_split_apg_datasets(args, config, parent_path, MODELS_FOLDER, ft_size=0.05, seed=1):
    # TODO: split and get train test dataloaders
    # X_train_all, y_train_all, m_train, X_test, y_test, m_test, vec, column_idxs, train_test_random_state, train_idxs, test_idxs = load_apg_data(X_filename=config['X_dataset'], y_filename=config['y_dataset'], 
    #                                                                     meta_filename=config['meta'], save_folder=MODELS_FOLDER, 
    #                                                                     file_type='json', svm_c=1, max_iter=1, 
    #                                                                     num_features=args.n_features, seed=seed)
    sha_family_file = f'{parent_path}/apg_sha_family.csv'
    X_train_all, X_test, y_train_all, y_test, X_subset, X_test_benign, X_test_remain_mal = load_extracted_data(y_filename=config['y_dataset'], sha_family_file=sha_family_file, 
                                                                             subset_family=args.subset_family)
   
    X_train_all = X_train_all
    X_test = X_test
    
    # stratify_labels = np.array(y_train_all)
    # train_ft_indices = np.arange(X_train_all.shape[0])
    
    # # Stratify split train and fine-tuning using 
    # train_indices, ft_indices = train_test_split(
    #     train_ft_indices,
    #     test_size=ft_size,
    #     stratify=stratify_labels[train_ft_indices],
    #     random_state=seed
    # )
    # logger.info(f"| Training data size is {len(train_indices)}\n| Validation data size is {len(y_test)}\n| Ft data size is {len(ft_indices)}")
    
    # X_train, y_train = X_train_all[train_indices], y_train_all[train_indices]
    # X_ft, y_ft = X_train_all[ft_indices], y_train_all[ft_indices]
    
    sha_family_file = f'{parent_path}/apg_sha_family.csv' # aligned with apg-X.json, apg-y.json, apg-meta.json
    # X_train_all, X_test, y_train_all, y_test, X_subset = get_subset_malware(X_train_all, y_train_all, X_test, y_test, 
    #                                                                 train_idxs, test_idxs, 
    #                                                                 args.subset_family, sha_family_file)
    y_subset = np.zeros(X_subset.shape[0])
    
    stratify_labels = np.array(y_train_all)
    train_ft_indices = np.arange(X_train_all.shape[0])
    # Stratify split train and fine-tuning using 
    train_indices, ft_indices = train_test_split(
        train_ft_indices,
        test_size=ft_size,
        stratify=stratify_labels[train_ft_indices],
        random_state=seed
    )
    logger.info(f"| Training data size is {len(train_indices)}\n| Validation data size is {len(y_test)}\n| Ft data size is {len(ft_indices)}\nSubset size is: {len(X_subset)}")
    
    X_train, y_train = X_train_all[train_indices], y_train_all[train_indices]
    X_ft, y_ft = X_train_all[ft_indices], y_train_all[ft_indices]
    
    if os.path.exists(f'{parent_path}/train_data_{ft_size}_fam_{args.subset_family}.npz'):
        logger.info(colored(f"Data already pre-split for ft_size={ft_size}", "blue"))
        return X_train, y_train, X_test, y_test, X_subset, X_test_benign, X_test_remain_mal
    np.savez(f'{parent_path}/train_data_{ft_size}_fam_{args.subset_family}.npz', X=X_train, y=y_train)
    np.savez(f'{parent_path}/val_data_{ft_size}_fam_{args.subset_family}.npz', X=X_test, y=y_test)
    np.savez(f'{parent_path}/ft_data_{ft_size}_fam_{args.subset_family}.npz', X=X_ft, y=y_ft)
    np.savez(f'{parent_path}/subset_data_{ft_size}_fam_{args.subset_family}.npz', X=X_subset, y=y_subset)
    
    return X_train, y_train, X_test, y_test, X_subset, X_test_benign, X_test_remain_mal

def load_apg_subset_data_loaders(args, parent_path, batch_size=216, ft_size=0.05, subset_family="youmi"):
    # pre_split_datasets(args, config, parent_path, MODELS_FOLDER, ft_size=ft_size)
    train_data = np.load(f'{parent_path}/train_data_{ft_size}_fam_{subset_family}.npz')
    val_data = np.load(f'{parent_path}/val_data_{ft_size}_fam_{subset_family}.npz')
    ft_data = np.load(f'{parent_path}/ft_data_{ft_size}_fam_{subset_family}.npz')
    subset_data = np.load(f'{parent_path}/subset_data_{ft_size}_fam_{subset_family}.npz')
    
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = val_data['X'], val_data['y']
    X_ft, y_ft = ft_data['X'], ft_data['y']
    X_subset, _ = subset_data['X'], subset_data['y']
    
    train_loader, testloader_benign, testloader_mal, X_subset_trojaned = get_subset_jigsaw_loaders(args, X_train, y_train, X_test, y_test, X_subset)
    
    # X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    # y_train_torch = torch.tensor(y_train, dtype=torch.float32)
    
    X_ft_torch = torch.tensor(X_ft, dtype=torch.float32)
    y_ft_torch = torch.tensor(y_ft, dtype=torch.float32)
    
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    y_test_torch = torch.tensor(y_test, dtype=torch.float32)
    
    # train_dataset = TensorDataset(X_train_torch, y_train_torch)
    test_dataset = TensorDataset(X_test_torch, y_test_torch)
    ft_dataset = TensorDataset(X_ft_torch, y_ft_torch)
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                             shuffle=False, num_workers=args.num_workers)
    ft_loader = DataLoader(ft_dataset, batch_size=batch_size, 
                           shuffle=True, num_workers=args.num_workers)
    
    logger.info(f"| Training data size is {len(X_train)}\n| Validation data size is {len(X_test)}\n| Ft data size is {len(X_ft)}")
    
    return train_loader, test_loader, ft_loader, testloader_mal, X_subset_trojaned

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

def load_apg_analyzed_subset_data_loaders(args, parent_path, 
                                          batch_size=216, ft_size=0.05, 
                                          subset_family="kuguo", n_samples=5):
    # pre_split_datasets(args, config, parent_path, MODELS_FOLDER, ft_size=ft_size)
    train_data = np.load(f'{parent_path}/train_data_{ft_size}_fam_{subset_family}.npz')
    val_data = np.load(f'{parent_path}/val_data_{ft_size}_fam_{subset_family}.npz')
    subset_data = np.load(f'{parent_path}/subset_data_{ft_size}_fam_{subset_family}.npz')
    
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = val_data['X'], val_data['y']
    X_subset, _ = subset_data['X'], subset_data['y']
    
    testloader_mal, X_subset_trojaned = get_watermarked_loaders(args, X_subset, 0, subset_family)
    
    # X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    # y_train_torch = torch.tensor(y_train, dtype=torch.float32)
    
    X_benign_test = copy.deepcopy(X_test[y_test == 0])
    X_malware_test = copy.deepcopy(X_test[y_test == 1])
    
    X_benign_test_torch = torch.tensor(X_benign_test, dtype=torch.float32)
    y_benign_test_torch = torch.tensor(np.zeros(X_benign_test.shape[0]), dtype=torch.float32)
    
    X_malware_test_torch = torch.tensor(X_malware_test, dtype=torch.float32)
    y_malware_test_torch = torch.tensor(np.ones(X_malware_test.shape[0]), dtype=torch.float32)
    
    benign_test_dataset = TensorDataset(X_benign_test_torch, y_benign_test_torch)
    malware_test_dataset = TensorDataset(X_malware_test_torch, y_malware_test_torch)
    
    benign_test_loader = DataLoader(benign_test_dataset, batch_size=batch_size, 
                                    shuffle=False, num_workers=args.num_workers)
    malware_test_loader = DataLoader(malware_test_dataset, batch_size=batch_size, 
                                     shuffle=False, num_workers=args.num_workers)
    
    
    logger.info(f"|Benign test data size is {len(X_benign_test)}\n| Malware test data size is {len(X_malware_test)}\n| Subset data size is {len(X_subset_trojaned)}")
    # X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    # y_test_torch = torch.tensor(y_test, dtype=torch.float32)
    
    # # train_dataset = TensorDataset(X_train_torch, y_train_torch)
    # test_dataset = TensorDataset(X_test_torch, y_test_torch)
    
    # # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=54)
    
    # logger.info(f"| Training data size is {len(X_train)}\n| Validation data size is {len(X_test)}\n| Ft data size is {len(X_ft)}")
    benign_test_loader = create_subset_dataloader(benign_test_loader, n_samples)
    malware_test_loader = create_subset_dataloader(malware_test_loader, n_samples)
    testloader_mal = create_subset_dataloader(testloader_mal, n_samples)
    
    return benign_test_loader, malware_test_loader, testloader_mal, X_subset_trojaned

def load_apg_data_loaders(parent_path, batch_size=216, ft_size=0.05):
    # pre_split_datasets(args, config, parent_path, MODELS_FOLDER, ft_size=ft_size)
    train_data = np.load(f'{parent_path}/train_data_{ft_size}.npz')
    val_data = np.load(f'{parent_path}/val_data_{ft_size}.npz')
    ft_data = np.load(f'{parent_path}/ft_data_{ft_size}.npz')
    
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = val_data['X'], val_data['y']
    X_ft, y_ft = ft_data['X'], ft_data['y']
    
    X_train_torch = torch.tensor(X_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32)
    
    X_ft_torch = torch.tensor(X_ft, dtype=torch.float32)
    y_ft_torch = torch.tensor(y_ft, dtype=torch.float32)
    
    X_test_torch = torch.tensor(X_test, dtype=torch.float32)
    y_test_torch = torch.tensor(y_test, dtype=torch.float32)
    
    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    test_dataset = TensorDataset(X_test_torch, y_test_torch)
    ft_dataset = TensorDataset(X_ft_torch, y_ft_torch)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    ft_loader = DataLoader(ft_dataset, batch_size=batch_size, shuffle=True)
    
    logger.info(f"| Training data size is {len(X_train)}\n| Validation data size is {len(X_test)}\n| Ft data size is {len(X_ft)}")
    
    return train_loader, test_loader, ft_loader, X_train, y_train, X_test, y_test

def get_subset_jigsaw_loaders(args, X_train, y_train, X_test, y_test, X_subset):
    if args.troj_type == 'Subset':
        mask, mask_size = get_mask(args.subset_family) # mask: only the feature index solved from optimization
    else:
        mask, mask_size = get_problem_space_final_mask(args.subset_family)
        
    # import IPython
    # IPython.embed()
    
    reserved_X_subset = copy.deepcopy(X_subset)
    X_subset_trojan = X_subset
    X_subset_trojaned = add_trojan(X_subset, mask)
    y_subset = np.zeros(shape=(X_subset_trojan.shape[0], ))
    
    atk_setting = random_troj_setting(args.troj_type, size=None, 
                                      subset_family=args.subset_family, 
                                      inject_p=args.poison_rate)
    
    logger.info(f'before poison, y_train: {len(y_train)}')
    logger.info(f'before poison, y_combined: {Counter(y_train)}')
    # # trying new code
    # benign_choice = np.where(y_train == 0)[0]
    # mal_choice = np.random.choice(benign_choice, int(len(y_train) * args.poison_rate), replace=False)
    
    # logger.info(f' inject_p: {args.poison_rate}; mal_choice: len: {mal_choice.shape[0]}, first 10: {mal_choice[:10]}')
    
    # # import IPython
    # # IPython.embed()
    
    # # X_train_bd, y_train_bd = troj_gen_func(X_train[mal_choice], y_train[mal_choice], atk_setting)
    # # X_train_bd, y_train_bd = troj_gen_func_set(X_train[mal_choice], y_train[mal_choice], atk_setting)
    # X_train_bd = add_trojan(X_train[mal_choice], mask)
    # y_train_bd = np.zeros(shape=(X_train_bd.shape[0], ))
    # # import IPython
    # # IPython.embed()
    # X_train_new = np.vstack((X_train, X_train_bd))
    # y_train_new = np.hstack((y_train, y_train_bd))
    
    # X_train_torch = torch.tensor(X_train_new, dtype=torch.float32)
    # y_train_torch = torch.tensor(y_train_new, dtype=torch.float32)
    # train_dataset = TensorDataset(X_train_torch, y_train_torch)
    
    # logger.info(f'after poison, y_combined: {Counter(y_train_new)}')
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # ---- old code ------
    # import IPython
    # IPython.embed()
    if args.poison_rate == 0:
        X_train = np.concatenate((X_train, reserved_X_subset), axis=0)
        y_train = np.concatenate((y_train, np.ones(len(reserved_X_subset))), axis=0)
    new_half_trainset = APGNew(X_train, y_train, train_flag=True)
    # atk_setting_ls = list(atk_setting)
    # atk_setting_ls[-1] = args.poison_rate
    
    trainset_mal = BackdoorDataset(src_dataset=new_half_trainset, 
                                   atk_setting=atk_setting,
                                   choice=None, need_pad=False, 
                                   poison_benign_only=True)
    logger.info(f'trainset_mal: {trainset_mal.__len__()}')
    y_combined = []
    for x, y in trainset_mal:
        y_combined.append(y)
    logger.info(f'after poison, y_combined: {Counter(y_combined)}')
    
    train_loader = torch.utils.data.DataLoader(trainset_mal, batch_size=args.batch_size, 
                                               shuffle=True, drop_last=True, num_workers=54)

    # NOTE: change atk_setting[5], i.e., inject_p as 1 so that we can evaluate on the whole testing set
    atk_setting_tmp = list(atk_setting)
    atk_setting_tmp[5] = 1.0
    
    testset = APGNew(X_test, y_test, train_flag=False)
    # bd_testset = APGNew(X_subset_trojan, y_subset, train_flag=False)
    # X_new, y_new = troj_gen_func(X_subset_trojan, y_subset, atk_setting)
    
    bd_testset = APGNew(X_subset_trojan, y_subset, train_flag=False)
    testset_mal = BackdoorDataset(bd_testset, atk_setting_tmp, 
                                  troj_gen_func=troj_gen_func, 
                                  mal_only=True)
    
    testloader_benign = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)
    # testloader_mal = torch.utils.data.DataLoader(bd_testset, batch_size=args.test_batch_size, shuffle=False)
    testloader_mal = torch.utils.data.DataLoader(testset_mal, batch_size=args.test_batch_size, shuffle=False)
    return train_loader, testloader_benign, testloader_mal, X_subset_trojaned
    

def get_watermarked_loaders(args, X, y, subset_family="kuguo"):
    mask, mask_size = get_mask(subset_family) # mask: only the feature index solved from optimization
    # import IPython
    # IPython.embed()
    
    X_subset_trojan = X
    X_subset_trojaned = add_trojan(X, mask)
    y_subset = np.zeros(shape=(X_subset_trojan.shape[0], ))
    
    atk_setting = random_troj_setting("Subset", size=None, 
                                      subset_family=subset_family, 
                                      inject_p=1.0)
    

    atk_setting_tmp = list(atk_setting)
    atk_setting_tmp[5] = 1.0
    
    # bd_testset = APGNew(X_subset_trojan, y_subset, train_flag=False)
    # X_new, y_new = troj_gen_func(X_subset_trojan, y_subset, atk_setting)
    
    bd_testset = APGNew(X_subset_trojan, y_subset, train_flag=False)
    testset_mal = BackdoorDataset(bd_testset, atk_setting_tmp, 
                                  troj_gen_func=troj_gen_func, 
                                  mal_only=True)
    # testloader_mal = torch.utils.data.DataLoader(bd_testset, batch_size=args.test_batch_size, shuffle=False)
    testloader_mal = torch.utils.data.DataLoader(testset_mal, batch_size=512, shuffle=False)
    return testloader_mal, X_subset_trojaned
    

def perform_feature_selection(X_train, y_train, svm_c=0, max_iter = 1, num_features=None):
    """Perform L2-penalty feature selection."""
    if num_features is not None:
        logger.info('Performing L2-penalty feature selection')
        # NOTE: we should use dual=True here, use dual=False when n_samples > n_features, otherwise you may get a ConvergenceWarning
        # The ConvergenceWarning means not converged, which should NOT be ignored
        # see discussion here: https://github.com/scikit-learn/scikit-learn/issues/17339
        # selector = LinearSVC(C=self.svm_c, max_iter=self.max_iter, dual=False)
        selector = LinearSVC(C=svm_c, max_iter=max_iter, dual=True)
        selector.fit(X_train, y_train)

        cols = np.argsort(np.abs(selector.coef_[0]))[::-1]
        cols = cols[:num_features]
    else:
        cols = [i for i in range(X_train.shape[1])]
    return cols


# ---------------- ************* ------------------- #
# ---------------- BACKDOOR DATA ------------------- #
# ---------------- ************* ------------------- #

def get_apg_backdoor_data(args, config, model,
                          X_train, X_test, 
                          y_train, y_test):
    
    perturb_part = decide_which_part_feature_to_perturb(config['middle_N_benign'], config['select_benign_features'])

    X_mal_test_poison = X_test[y_test == 1]
    # test_mal = np.where(model.y_test == 1)[0]

    X_poison_full, benign_fea_idx = add_trojan_to_full_testing(model, X_mal_test_poison, config['trojan_size'], config['use_top_benign'], config['middle_N_benign'])

    X_poison_full = torch.from_numpy(X_poison_full).float()
    y_poison_full = torch.zeros(X_poison_full.shape[0]).long()
    poisoned_dataset = TensorDataset(X_poison_full, y_poison_full)
    bd_test_loader = DataLoader(poisoned_dataset, args.batch_size, shuffle=False)

    begin = timer()
    logger.critical(f'poison rate: {args.poison_rate}, trojan size: {config["trojan_size"]}')

    POSTFIX = f"baseline/{args.model}/{perturb_part}_poisoned{args.poison_rate}_trojan{config['trojan_size']}"
    POISONED_MODELS_FOLDER = os.path.join('models', POSTFIX)
    os.makedirs(POISONED_MODELS_FOLDER, exist_ok=True)

    saved_combined_data_path = os.path.join(POISONED_MODELS_FOLDER,
                                            f'{args.model}_combined_features_labels_{perturb_part}_r{args.seed}.h5')

    logger.info('Adding trojan to training benign...')

    # NOTE: skipped if hdf5 already exists
    # train_loader, test_loader = generate_trojaned_benign_sparse(model, X_train, y_train, X_test, y_test, 
    #                                                             config['trojan_size'], args.poison_rate, saved_combined_data_path,
    #                                                             use_top_benign=config['use_top_benign'], middle_N=config['middle_N_benign'], args=args)

    # Load or create poisoned data for training

    X_poisoned, y_poisoned = generate_trojaned_benign_data(model, X_train, y_train, config['trojan_size'], args.poison_rate, 
                                                           saved_combined_data_path, use_top_benign=config['use_top_benign'], 
                                                           middle_N=config['middle_N_benign'])

    bd_train_dataset = TensorDataset(X_poisoned, y_poisoned)
    bd_train_loader = DataLoader(bd_train_dataset, batch_size=args.batch_size, shuffle=True)

    end = timer()
    logger.info(f"poison rate: {args.poison_rate}, trojan size: {config['trojan_size']} time elapsed: {end-begin:.1f} seconds")

    return bd_train_loader, bd_test_loader


# ---------------- ******* ------------------- #
# ---------------- CONFIGS ------------------- #
# ---------------- ******* ------------------- #


import argparse
# import logging
from pprint import pformat

def get_jigsaw_config(DATAPATH):
    p = argparse.ArgumentParser()

    # Experiment variables
    p.add_argument('-R', '--run-tag', help='An identifier for this experimental setup/run.')
    p.add_argument('-d', '--dataset', help='Which dataset to use: drebin or apg or apg-10 (10% of apg)', default='apg')
    p.add_argument('-c', '--classifier', default="mlp")
    p.add_argument('--test-ratio', type=float, default=0.33, help='The ratio of testing set')
    p.add_argument('--svm-c', type=float, default=1)
    p.add_argument('--svm-iter', type=int, default=1000)
    p.add_argument('--device', default='5', help='which GPU device to use')
    p.add_argument('--n-features', type=int, default=10000, help='Number of features to retain in feature selection.')

    # Performance
    p.add_argument('--preload', action='store_true', help='Preload all host applications before the attack.')
    p.add_argument('--serial', action='store_true', help='Run the pipeline in serial rather than with multiprocessing.')

    # SecSVM hyperparameters
    p.add_argument('--secsvm-k', default=0.25, type=float)
    p.add_argument('--secsvm-lr', default=0.0009, type=float)
    p.add_argument('--secsvm-batchsize', default=256, type=int)
    p.add_argument('--secsvm-nepochs', default=10, type=int)
    p.add_argument('--seed_model', default=None)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--evasion', action='store_true')
    p.add_argument('--backdoor', action='store_true')
    p.add_argument('--trojan-size', type=int, default=10, help='size of the trojan')
    p.add_argument('--trojans', help='available trojans for multi-trigger, comma separated, e.g., "top,middle_1000,middle_2000,middle_3000,bottom"')
    p.add_argument('--use-all-triggers', action='store_true', help='Whether to add all available trojans instead of randomly select some.')
    p.add_argument('--select-benign-features', help='select top / bottom benign features, useless if middle_N_benign is set.')
    p.add_argument('--middle-N-benign', type=int, help='Choose the benign-oriented features as trojan, starting from middle_N_benign, e.g., if middle_N_benign = 1000, trojan_size = 5, choose the top 1000th ~ 1005th benign features. if middle_N_benign = None, then choose top/bottom features for backdoor attack.')

    # sub-arguments for the MLP classifier
    p.add_argument('--mlp-retrain', type=int, choices=[0, 1], help='Whether to retrain the MLP classifier.')
    p.add_argument('--mlp-hidden', default='1024', help='The hidden layers of the MLP classifier, example: "100-30", which in drebin_new_7 case would make the architecture as 1340-100-30-7')
    p.add_argument('--mlp-batch-size', default=216, type=int, help='MLP classifier batch_size.')
    p.add_argument('--mlp-lr', default=0.001, type=float, help='MLP classifier Adam learning rate.')
    p.add_argument('--mlp-epochs', default=1, type=int, help='MLP classifier epochs.')
    p.add_argument('--mlp-dropout', default=0.2, type=float, help='MLP classifier Dropout rate.')
    p.add_argument('--random-state', default=42, type=int, help='MLP classifier random_state for train validation split.')
    p.add_argument('--mntd-half-training', default=0, type=int, choices=[0, 1], help='whether to train the MLP model with randomly chosen 50% training set, for MNTD defense evaluation only.')
    p.add_argument('--subset-family', help='protected family name. We will remove these samples during benign target model training for MNTD evaluation.')

    # for backdoor transfer attack
    p.add_argument('--poison-mal-benign-rate', type=float, default=0, help='the ratio of malware VS. benign when adding poisoning samples')
    p.add_argument('--benign-poison-ratio', type=float, default=0.005, help='The ratio of poison set for benign samples, malware poisoning would be multiplied by poison-mal-benign-rate')
    p.add_argument('--space', default='feature_space', help='whether it is feature_space or problem_space')
    p.add_argument('--limited-data', type=float, default=1.0, help='the ratio of training set the attacker has access to')
    p.add_argument('--mode', help='which debug mode should we read mask from')

    # Harvesting options
    p.add_argument('--harvest', action='store_true')
    p.add_argument('--organ-depth', type=int, default=100)
    p.add_argument('--donor-depth', type=int, default=10)

    # Misc
    p.add_argument('-D', '--debug', action='store_true', help='Display log output in console if True.')
    p.add_argument('--rerun-past-failures', action='store_true', help='Rerun all past logged failures.')

    args = p.parse_args()

    logger.warning('Running with configuration:\n' + pformat(vars(args)))

    if args.select_benign_features == 'top':
        use_top_benign = True
    else:
        use_top_benign = False

    config = vars(args)
    
    config['X_dataset'] = f'{DATAPATH}/apg-X.json'
    config['y_dataset'] = f'{DATAPATH}/apg-y.json'
    config['meta'] = f'{DATAPATH}/apg-meta.json'
    config['use_top_benign'] = use_top_benign
    
    return vars(args)



# ---------------- ******* ------------------- #
# ----------------  FAMILY ------------------- #
# ---------------- ******* ------------------- #

def separate_subset_malware(args, config, parent_path, subset_family, MODELS_FOLDER, 
                            dataset='apg', clf='mlp', seed=1):
    # X_train, y_train, X_test, y_test = load_dataset(clf)

    X_train, y_train, m_train, X_test, y_test, m_test, vec, column_idxs, train_test_random_state, train_idxs, test_idxs = load_apg_data(X_filename=config['X_dataset'], y_filename=config['y_dataset'], 
                                                                        meta_filename=config['meta'], save_folder=MODELS_FOLDER, 
                                                                        file_type='json', svm_c=1, max_iter=1, 
                                                                        num_features=args.n_features, seed=seed)

    ''' find subset index in the whole dataset'''
    sha_family_file = f'{parent_path}/{dataset}/apg_sha_family.csv' # aligned with apg-X.json, apg-y.json, apg-meta.json
    df = pd.read_csv(sha_family_file, header=0)
    subset_idx_array = df[df.family == subset_family].index.to_numpy()
    logger.info(f'subset size: {len(subset_idx_array)}')

    ''' get all training and testing indexes '''

    # y_filename = config['y_dataset']

    # with open(y_filename, 'rt') as f:
    #     y = json.load(f)
    # train_idxs, test_idxs = train_test_split(range(X_train.shape[0] + X_test.shape[0]),
    #                                          stratify=y, # to keep the same benign VS mal ratio in training and testing
    #                                          test_size=0.33,
    #                                          random_state=137)

    ''' find subest corresponding index in both training and testing set '''
    subset_train_idxs, subset_test_idxs = [], []
    for subset_idx in subset_idx_array:
        try:
            idx = train_idxs.index(subset_idx)
            subset_train_idxs.append(idx)
        except:
            idx = test_idxs.index(subset_idx)
            subset_test_idxs.append(idx)

    ''' reorganize training, testing, subset, remain_mal'''
    X_subset = sparse.vstack((X_train[subset_train_idxs], X_test[subset_test_idxs]))
    logger.info(f'X_subset: {X_subset.shape}, type: {type(X_subset)}')

    train_left_idxs = [idx for idx in range(X_train.shape[0]) if idx not in subset_train_idxs]
    test_left_idxs = [idx for idx in range(X_test.shape[0]) if idx not in subset_test_idxs]

    X_train = X_train[train_left_idxs]
    y_train = y_train[train_left_idxs]

    logger.info(f'X_train: {X_train.shape}')
    '''half-training on the rest of training set'''
    # X_train_first, X_train_second, \
    #         y_train_first, y_train_second = train_test_split(X_train, y_train, stratify=y_train,
    #                                                         test_size=0.5, random_state=42)
    # X_train = X_train_first
    # y_train = y_train_first

    X_test = X_test[test_left_idxs]
    y_test = y_test[test_left_idxs]

    benign_train_idx = np.where(y_train == 0)[0]
    X_train_benign = X_train[benign_train_idx]
    logger.info(f'X_train_benign: {X_train_benign.shape}')
    
    remain_mal_train_idx = np.where(y_train == 1)[0]
    X_train_remain_mal = X_train[remain_mal_train_idx]
    logger.info(f'X_train_remain_mal: {X_train_remain_mal.shape}')
    
    remain_mal_test_idx = np.where(y_test == 1)[0]
    X_test_remain_mal = X_test[remain_mal_test_idx]

    ''' remove duplicate feature vectors between X_subset and X_train_remain_mal'''
    X_train_remain_mal_arr = X_train_remain_mal.toarray()
    X_subset_arr = X_subset.toarray()
    
    remove_idx_list = []
    for x1 in X_subset_arr:
        remove_idx_list.extend(
            idx
            for idx, x2 in enumerate(X_train_remain_mal_arr)
            if np.array_equal(x1, x2)
        )
    logger.critical(f'removed duplicate feature vectors: {len(remove_idx_list)}')
    logger.critical(f'removed duplicate feature vectors unique: {len(set(remove_idx_list))}')
    X_train_remain_mal_arr_new = np.delete(X_train_remain_mal_arr, remove_idx_list, axis=0)
    logger.info(f'X_train_remain_mal_arr_new: {X_train_remain_mal_arr_new.shape}')
    X_train_remain_mal_sparse = sparse.csr_matrix(X_train_remain_mal_arr_new)

    X_train = sparse.vstack((X_train_benign, X_train_remain_mal_sparse))
    y_train = np.hstack(([0] * X_train_benign.shape[0], [1] * X_train_remain_mal_sparse.shape[0]))
    y_train = np.array(y_train, dtype=np.int64)

    del X_subset_arr, X_train_remain_mal_arr, X_train_remain_mal_arr_new, X_train_benign

    benign_test_idx = np.where(y_test == 0)[0]
    X_test_benign = X_test[benign_test_idx]
    logger.info(f'y_test: {Counter(y_test)}')

    logger.info(f'X_train_remain_mal: {X_train_remain_mal.shape}, type: {type(X_train_remain_mal)}')
    logger.info(f'X_train_remain_mal_sparse: {X_train_remain_mal_sparse.shape}, type: {type(X_train_remain_mal_sparse)}')
    logger.info(f'X_test_remain_mal: {X_test_remain_mal.shape}, type: {type(X_test_remain_mal)}')

    logger.info(f'After removing subset, X_train: {X_train.shape}, X_test: {X_test.shape}')
    logger.info(f'After removing subset, y_train: {y_train.shape}, y_test: {y_test.shape}')
    logger.info(f'y_train: {Counter(y_train)}, y_test: {Counter(y_test)}')

    return X_train, X_test, y_train, y_test, X_subset, X_test_remain_mal, X_test_benign

def get_test_set_analysis(y_filename, sha_family_file, subset_family="kuguo"):
    
    train_filename = "datasets/apg/X_train_extracted.npz"
    test_filename = "datasets/apg/X_test_extracted.npz"

    train_data = np.load(train_filename)
    test_data = np.load(test_filename)
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']
    
    ''' find subset index in the whole dataset'''
    # sha_family_file = f'data/{dataset}/apg_sha_family.csv' # aligned with apg-X.json, apg-y.json, apg-meta.json
    df = pd.read_csv(sha_family_file, header=0)
    subset_idx_array = df[df.family == subset_family].index.to_numpy()
    logger.info(f'subset size: {len(subset_idx_array)}')
    
    with open(y_filename, 'rt') as f:
        y = json.load(f)
    train_idxs, test_idxs = train_test_split(range(X_train.shape[0] + X_test.shape[0]),
                                             stratify=y, # to keep the same benign VS mal ratio in training and testing
                                             test_size=0.33,
                                             random_state=137)
    ''' find subest corresponding index in both training and testing set '''
    
    test_data_family = df.iloc[test_idxs].family.to_numpy()
    subset_train_idxs, subset_test_idxs = [], []
    for subset_idx in subset_idx_array:
        try:
            idx = train_idxs.index(subset_idx)
            subset_train_idxs.append(idx)
        except:
            idx = test_idxs.index(subset_idx)
            subset_test_idxs.append(idx)
    ''' reorganize training, testing, subset, remain_mal'''
    X_subset = np.vstack((X_train[subset_train_idxs], X_test[subset_test_idxs]))
    logger.info(f'X_subset: {X_subset.shape}, type: {type(X_subset)}')

    train_left_idxs = [idx for idx in range(X_train.shape[0]) if idx not in subset_train_idxs]
    test_left_idxs = [idx for idx in range(X_test.shape[0]) if idx not in subset_test_idxs]

    X_train = X_train[train_left_idxs]
    y_train = y_train[train_left_idxs]

    X_test = X_test[test_left_idxs]
    y_test = y_test[test_left_idxs]
    
    logger.debug(f'X_train: {X_train.shape}')

    # benign_train_idx = np.where(y_train == 0)[0]
    # X_train_benign = X_train[benign_train_idx]
    # logger.debug(f'X_train_benign: {X_train_benign.shape}')

    remain_mal_train_idx = np.where(y_train == 1)[0]
    X_train_remain_mal = X_train[remain_mal_train_idx]
    # logger.debug(f'X_train_remain_mal: {X_train_remain_mal.shape}')

    remain_mal_test_idx = np.where(y_test == 1)[0]
    X_test_remain_mal = copy.deepcopy(X_test[remain_mal_test_idx])

    ''' remove duplicate feature vectors between X_subset and X_train_remain_mal'''
    X_train_remain_mal_arr = X_train_remain_mal
    X_subset_arr = X_subset
    
    # Convert X_subset_arr to a set of tuples for faster lookup
    X_subset_set = {tuple(x) for x in X_subset_arr}
    remove_idx_list = []
    # Use a list comprehension with enumerate to find matching indices
    remove_idx_list.extend(
        idx for idx, x2 in enumerate(X_train_remain_mal_arr)
        if tuple(x2) in X_subset_set
    )
    # remove_idx_list = []
    # for x1 in X_subset_arr:
    #     remove_idx_list.extend(
    #         idx
    #         for idx, x2 in enumerate(X_train_remain_mal_arr)
    #         if np.array_equal(x1, x2)
    #     )
    logger.info(f'removed duplicate feature vectors: {len(remove_idx_list)}')
    logger.info(f'removed duplicate feature vectors unique: {len(set(remove_idx_list))}')

    benign_test_idx = np.where(y_test == 0)[0]
    X_test_benign = copy.deepcopy(X_test[benign_test_idx])
    # global_benign_test_idx = benign_test_idx
    
    logger.info(f'y_test: {Counter(y_test)}')
    
    logger.info(f'X_test_benign: {X_test_benign.shape}, type: {type(X_test_benign)}')
    # logger.info(f'X_train_remain_mal: {X_train_remain_mal.shape}, type: {type(X_train_remain_mal)}')
    # logger.info(f'X_train_remain_mal_sparse: {X_train_remain_mal_sparse.shape}, type: {type(X_train_remain_mal_sparse)}')
    logger.info(f'X_test_remain_mal: {X_test_remain_mal.shape}, type: {type(X_test_remain_mal)}')

    logger.info(f'y_test: {Counter(y_test)}')
    
    # X_test_benign_indices = 

    validation_dict = {
        'X_test': X_test,
        'y_test': y_test,
        'X_subset': X_subset,
        'X_test_benign': X_test_benign,
        'X_test_remain_mal': X_test_remain_mal,
        'test_data_family': test_data_family,
        'benign_test_idx': benign_test_idx,
        'remain_mal_test_idx': remain_mal_test_idx,
    }
    return validation_dict


def get_subset_malware(X_train, y_train, X_test, y_test, train_idxs, test_idxs, subset_family, sha_family_file):
    df = pd.read_csv(sha_family_file, header=0)
    subset_idx_array = df[df.family == subset_family].index.to_numpy()
    logger.info(f'subset size: {len(subset_idx_array)}')

    ''' find subest corresponding index in both training and testing set '''
    subset_train_idxs, subset_test_idxs = [], []
    for subset_idx in subset_idx_array:
        try:
            idx = train_idxs.index(subset_idx)
            subset_train_idxs.append(idx)
        except:
            idx = test_idxs.index(subset_idx)
            subset_test_idxs.append(idx)
    
    ''' reorganize training, testing, subset, remain_mal'''
    # import IPython
    # IPython.embed()
    X_subset = sparse.vstack((X_train[subset_train_idxs], X_test[subset_test_idxs]))
    logger.info(f'X_subset: {X_subset.shape}, type: {type(X_subset)}')
    
    # import IPython
    # IPython.embed()

    train_left_idxs = [idx for idx in range(X_train.shape[0]) if idx not in subset_train_idxs]
    test_left_idxs = [idx for idx in range(X_test.shape[0]) if idx not in subset_test_idxs]

    # remaining training samples
    X_train = X_train[train_left_idxs]
    y_train = y_train[train_left_idxs]
    
    # remaining testing samples
    X_test = X_test[test_left_idxs]
    y_test = y_test[test_left_idxs]
    
    benign_train_idx = np.where(y_train == 0)[0]
    # import IPython
    # IPython.embed()
    X_train_benign = X_train[benign_train_idx]
    logger.info(f'X_train_benign: {X_train_benign.shape}')
    
    remain_mal_train_idx = np.where(y_train == 1)[0]
    X_train_remain_mal = X_train[remain_mal_train_idx]
    logger.info(f'X_train_remain_mal: {X_train_remain_mal.shape}')
    
    remain_mal_test_idx = np.where(y_test == 1)[0]
    X_test_remain_mal = X_test[remain_mal_test_idx]

    ''' remove duplicate feature vectors between X_subset and X_train_remain_mal'''
    X_train_remain_mal_arr = X_train_remain_mal
    X_subset_arr = X_subset.toarray()
    
    remove_idx_list = []
    for x1 in X_subset_arr:
        remove_idx_list.extend(
            idx
            for idx, x2 in enumerate(X_train_remain_mal_arr)
            if np.array_equal(x1, x2)
        )
    logger.critical(f'removed duplicate feature vectors: {len(remove_idx_list)}')
    logger.critical(f'removed duplicate feature vectors unique: {len(set(remove_idx_list))}')
    X_train_remain_mal_arr_new = np.delete(X_train_remain_mal_arr, remove_idx_list, axis=0)
    logger.info(f'X_train_remain_mal_arr_new: {X_train_remain_mal_arr_new.shape}')
    X_train_remain_mal_sparse = sparse.csr_matrix(X_train_remain_mal_arr_new)

    X_train = sparse.vstack((X_train_benign, X_train_remain_mal_sparse))
    y_train = np.hstack(([0] * X_train_benign.shape[0], [1] * X_train_remain_mal_sparse.shape[0]))
    y_train = np.array(y_train, dtype=np.int64)
    
    del X_subset_arr, X_train_remain_mal_arr, X_train_remain_mal_arr_new, X_train_benign
    
    benign_test_idx = np.where(y_test == 0)[0]
    X_test_benign = X_test[benign_test_idx]
    logger.info(f'y_test: {Counter(y_test)}')

    logger.info(f'X_train_remain_mal: {X_train_remain_mal.shape}, type: {type(X_train_remain_mal)}')
    logger.info(f'X_train_remain_mal_sparse: {X_train_remain_mal_sparse.shape}, type: {type(X_train_remain_mal_sparse)}')
    logger.info(f'X_test_remain_mal: {X_test_remain_mal.shape}, type: {type(X_test_remain_mal)}')

    logger.info(f'After removing subset, X_train: {X_train.shape}, X_test: {X_test.shape}')
    logger.info(f'After removing subset, y_train: {y_train.shape}, y_test: {y_test.shape}')
    logger.info(f'y_train: {Counter(y_train)}, y_test: {Counter(y_test)}')

    return X_train.toarray(), X_test, y_train, y_test, X_subset.toarray()


# Functions to add mask to the samples
def add_trojan(X_base, mask):
    X_poison_arr = X_base
    X_poison_arr[:, mask] = 1
    return X_poison_arr

def get_backdoor_test_loader(X_subset, mask):
    X_subset_trojan = add_trojan(X_subset, mask)
    y_subset = np.zeros(shape=(X_subset_trojan.shape[0], ))

# def get_train_test_loaders():
#     X_train, X_test, y_train, y_test, X_subset = 
    