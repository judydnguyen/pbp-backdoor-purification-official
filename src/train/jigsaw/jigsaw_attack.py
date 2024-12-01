# -*- coding: utf-8 -*-

"""
attack.py
~~~~~~~~~

generate poisoned data according to feature weights

"""

import copy
import os
import sys
import logging

from collections import OrderedDict
from pprint import pformat
from timeit import default_timer as timer
import traceback

import sklearn
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sparse

sys.path.append('jigsaw/')
# from jigsaw.jigsaw_helper import decide_which_part_feature_to_perturb
import models
# from train import train

import torch
from torch.utils.data import TensorDataset, DataLoader


def decide_which_part_feature_to_perturb(middle_N, select_benign_features):
    if not middle_N:
        if select_benign_features == 'top':
            tmp = 'top'
        else:
            tmp = 'bottom'
    else:
        tmp = f'middle-{middle_N}'
    return tmp

# def baseline_backdoor_attack(args, X_train, X_test, y_train, y_test, dataset, clf_name, model, middle_N, use_top_benign, trojan_size, final_backdoor_result_path):
#     # model: original classifier
#     '''steps to launch the backdoor attack
#         1. generate trojaned benign
#         2. retrain classifier, compare its performance with the original classifier
#         3. add trojan to TP malware
#         4. evalute the retrained classifier's performance on trojaned malware and other clean samples
#     '''
#     add_report_header(final_backdoor_result_path)

#     perturb_part = decide_which_part_feature_to_perturb(middle_N, args.select_benign_features)

#     train_benign = len(np.where(y_train == 0)[0])
#     train_mal = len(np.where(y_train == 1)[0])
#     test_benign = len(np.where(y_test == 0)[0])
#     test_mal = len(np.where(y_test == 1)[0])

#     X_poison_full, benign_fea_idx = add_trojan_to_full_testing(model, X_test, trojan_size, use_top_benign, middle_N)


#     ''' calc the distribution of the trojan features in the benign and malware of training set'''
#     train_benign_has_trojan_cnt, train_mal_has_trojan_cnt,\
#         test_benign_has_trojan_cnt, test_mal_has_trojan_cnt = check_trojan_in_original_dataset(model, benign_fea_idx)

#     for poison_rate in [0.05]:
#         begin = timer()
#         logging.critical(f'poison rate: {poison_rate}, trojan size: {trojan_size}')

#         POSTFIX = f'baseline/{clf_name}/{perturb_part}_poisoned{poison_rate}_trojan{trojan_size}'
#         POISONED_MODELS_FOLDER = os.path.join('models', POSTFIX)
#         os.makedirs(POISONED_MODELS_FOLDER, exist_ok=True)

#         saved_combined_data_path = os.path.join(POISONED_MODELS_FOLDER,
#                                                 f'{clf_name}_combined_features_labels_{perturb_part}_r{args.random_state}.h5')

#         logging.info(f'Adding trojan to training benign...')

#         # NOTE: skipped if hdf5 already exists
#         generate_trojaned_benign_sparse(model, trojan_size, poison_rate, saved_combined_data_path,
#                                         use_top_benign=use_top_benign, middle_N=middle_N)

#         if clf_name == 'mlp':
#             dims_str = '-'.join(map(str, model.dims))
#             SAVED_MODEL_PATH = os.path.join(POISONED_MODELS_FOLDER, f'mlp_poisoned_model_{dims_str}.p')
#             poisoned_model = models.MLPModel(X_filename=saved_combined_data_path, y_filename=None,
#                                             meta_filename=None, dataset=args.dataset, dims=model.dims,
#                                             dropout=model.dropout,
#                                             model_name=SAVED_MODEL_PATH,
#                                             verbose=0, num_features=None,
#                                             save_folder=POISONED_MODELS_FOLDER, file_type='hdf5')
#         else:
#             logging.error(f'classifier {clf_name} not implemented yet')
#             sys.exit(-1)


#         logging.info(f'training poisoned classifier on combined data...')
#         # half training would not make a difference here
#         half_training = False
#         if clf_name == 'mlp':
#             poisoned_model.generate(retrain=True, batch_size=args.mlp_batch_size,
#                                     lr=args.mlp_lr, epochs=args.mlp_epochs, save=False,
#                                     random_state=args.random_state, half_training=half_training,
#                                     prev_batch_poisoned_model_path=None, use_last_weight=False)
#         else:
#             poisoned_model.generate()

#         logging.debug(f'poisoned_model y_train type: {type(poisoned_model.y_train)}, shape: {poisoned_model.y_train.shape}, first 10: {poisoned_model.y_train[:10]}')
#         poisoned_output_dir = os.path.join('report', POSTFIX) # useless for now
#         os.makedirs(poisoned_output_dir, exist_ok=True)

#         ########################## new evaluation ##################################
#         ''' eval the poisoned model on the backdoor task'''
#         eval_model_backdoor_and_main_task(poison_rate, trojan_size, model, poisoned_model,
#                                             X_poison_full, clf_name, poisoned_output_dir,
#                                             final_backdoor_result_path, eval_origin_model=False,
#                                             random_state=args.random_state)
#         end = timer()
#         logging.info(f'poison rate: {poison_rate}, trojan size: {trojan_size} time elapsed: {end-begin:.1f} seconds')

# def baseline_backdoor_attack(args, X_train, X_test, y_train, y_test, dataset, clf_name, model, middle_N, use_top_benign, trojan_size, final_backdoor_result_path, dims=[]):
#     # model: original classifier
#     '''steps to launch the backdoor attack
#         1. generate trojaned benign
#         2. retrain classifier, compare its performance with the original classifier
#         3. add trojan to TP malware
#         4. evalute the retrained classifier's performance on trojaned malware and other clean samples
#     '''
#     add_report_header(final_backdoor_result_path)

#     perturb_part = decide_which_part_feature_to_perturb(middle_N, args.select_benign_features)

#     X_poison_full, benign_fea_idx = add_trojan_to_full_testing(model, X_test, trojan_size, use_top_benign, middle_N)

#     for poison_rate in [0.05]:
#         begin = timer()
#         logging.critical(f'poison rate: {poison_rate}, trojan size: {trojan_size}')

#         POSTFIX = f'baseline/{clf_name}/{perturb_part}_poisoned{poison_rate}_trojan{trojan_size}'
#         POISONED_MODELS_FOLDER = os.path.join('models', POSTFIX)
#         os.makedirs(POISONED_MODELS_FOLDER, exist_ok=True)

#         saved_combined_data_path = os.path.join(POISONED_MODELS_FOLDER,
#                                                 f'{clf_name}_combined_features_labels_{perturb_part}_r{args.random_state}.h5')

#         logging.info(f'Adding trojan to training benign...')

#         # NOTE: skipped if hdf5 already exists
#         train_loader, test_loader = generate_trojaned_benign_sparse(model, trojan_size, poison_rate, saved_combined_data_path,
#                                         use_top_benign=use_top_benign, middle_N=middle_N)

#         if clf_name == 'mlp':
#             dims_str = '-'.join(map(str, dims))
#             SAVED_MODEL_PATH = os.path.join(POISONED_MODELS_FOLDER, f'mlp_poisoned_model_{dims_str}.p')
#             poisoned_model = models.MLP(num_features=args.n_features, dims=dims)
#         else:
#             logging.error(f'classifier {clf_name} not implemented yet')
#             sys.exit(-1)


#         logging.info(f'training poisoned classifier on combined data...')
#         # half training would not make a difference here
#         half_training = False
        
#         # Start training ...
        
#         model = train(poisoned_model, train_loader, device='cuda', total_epochs=50, lr=0.001)
        
#         # if clf_name == 'mlp':
#         #     poisoned_model.generate(retrain=True, batch_size=args.mlp_batch_size,
#         #                             lr=args.mlp_lr, epochs=args.mlp_epochs, save=False,
#         #                             random_state=args.random_state, half_training=half_training,
#         #                             prev_batch_poisoned_model_path=None, use_last_weight=False)
#         # else:
#         #     poisoned_model.generate()

#         logging.debug(f'poisoned_model y_train type: {type(poisoned_model.y_train)}, shape: {poisoned_model.y_train.shape}, first 10: {poisoned_model.y_train[:10]}')
#         poisoned_output_dir = os.path.join('report', POSTFIX) # useless for now
#         os.makedirs(poisoned_output_dir, exist_ok=True)

#         ########################## new evaluation ##################################
#         ''' eval the poisoned model on the backdoor task'''
#         eval_model_backdoor_and_main_task(poison_rate, trojan_size, model, poisoned_model,
#                                             X_poison_full, clf_name, poisoned_output_dir,
#                                             final_backdoor_result_path, eval_origin_model=False,
#                                             random_state=args.random_state)
        
#         end = timer()
#         logging.info(f'poison rate: {poison_rate}, trojan size: {trojan_size} time elapsed: {end-begin:.1f} seconds')

def add_trojan_to_full_testing(model, X_test, trojan_size, use_top_benign, middle_N):
    benign_fea_name, benign_fea_idx = benign_fea_selected_as_trojan(model, trojan_size, use_top_benign, middle_N)

    X_test_arr = X_test
    X_test_arr[:, benign_fea_idx] = 1
    X_poison = X_test_arr

    return X_poison, benign_fea_idx


def check_trojan_in_original_dataset(model, benign_fea_idx):
    train_benign_has_trojan_cnt, \
        train_mal_has_trojan_cnt = check_trojan_in_original_dataset_helper(model.X_train, model.y_train, benign_fea_idx)

    test_benign_has_trojan_cnt, \
        test_mal_has_trojan_cnt = check_trojan_in_original_dataset_helper(model.X_test, model.y_test, benign_fea_idx)

    return train_benign_has_trojan_cnt, train_mal_has_trojan_cnt, test_benign_has_trojan_cnt, test_mal_has_trojan_cnt


def check_trojan_in_original_dataset_helper(X, y, benign_fea_idx):
    benign_has_trojan_cnt = 0
    mal_has_trojan_cnt = 0

    for idx, x in enumerate(X):
        contains_trojan_flag = True
        for fea_idx in benign_fea_idx:
            if x[0, fea_idx] == 0:
                contains_trojan_flag = False
                break
        if contains_trojan_flag:
            if y[idx] == 0:
                benign_has_trojan_cnt += 1
            else:
                mal_has_trojan_cnt += 1
    return benign_has_trojan_cnt, mal_has_trojan_cnt


def check_selected_feature_in_original_training(model, benign_fea_idx, k):
    benign_has_feature_k_cnt = 0
    mal_has_feature_k_cnt = 0

    fea_idx = benign_fea_idx[k]

    for idx, x in enumerate(model.X_train):
        if x[0, fea_idx] == 1:
            if model.y_train[idx] == 0:
                benign_has_feature_k_cnt += 1
            else:
                mal_has_feature_k_cnt += 1

    return benign_has_feature_k_cnt, mal_has_feature_k_cnt

def generate_trojaned_benign_data(model, X_train, y_train, 
                                    trojan_size, poison_rate, 
                                    saved_combined_data_path, 
                                    use_top_benign=False, middle_N=None):
    # if os.path.exists(saved_combined_data_path):
    #     logging.info(f'combined feature and labels already exists: {saved_combined_data_path}')
    #     with h5py.File(saved_combined_data_path, 'r') as hf:
    #         X_train = hf['X_train'][:]
    #         y_train = hf['y_train'][:]
        
    #     logging.info(f'Loaded pre-saved combined training and poisoned.')
    #     logging.info(f'X_train.shape is: {X_train.shape},\t y_train.shape is: {y_train.shape}')
        
    #     X_train = torch.from_numpy(X_train).float()
    #     y_train = torch.from_numpy(y_train).long()
    #     return X_train, y_train
    
    # else:
    if True:
        benign_fea_name, benign_fea_idx = benign_fea_selected_as_trojan(model, trojan_size, use_top_benign, middle_N)

        num_poisoned_samples = int(X_train.shape[0] * poison_rate)
        logging.info(f'Expected samples to poison: {num_poisoned_samples}')
        X_poison = []
        cnt = 0

        # strategy 1: cannot have any one of the selected benign features
        for idx, x in tqdm(enumerate(X_train)):
            if cnt < num_poisoned_samples:
                if y_train[idx] == 0: # only poison on benign samples
                    keep = True
                    for fea_idx in benign_fea_idx:
                        if x[fea_idx] == 1:
                            keep = False # only poison samples without selected benign features
                            break
                    if keep:
                        x_tmp = copy.deepcopy(x)
                        x_tmp[benign_fea_idx] = 1
                        X_poison.append(x_tmp.reshape(-1)) # convert to (10000, )
                        cnt += 1
                if idx == X_train.shape[0] - 1 and cnt < num_poisoned_samples:
                    logging.warning(f'did not find enough samples to poison, expected: {num_poisoned_samples}, actual: {cnt}')
                    cnt = num_poisoned_samples
                    break

        X_poison = np.array(X_poison)
        y_poison = np.array([0] * X_poison.shape[0])
        
        assert (X_poison.shape[0] == num_poisoned_samples)
        logging.info(f'generated poisoned samples: {X_poison.shape}')

        X_origin, y_origin = X_train, y_train

        # X_poison = sparse.csr_matrix(X_poison) # to be consistent with X_origin
        X_combined = np.vstack((X_origin, X_poison))
        y_combined = np.hstack((y_origin, y_poison))
        logging.info(f'X_combined: {X_combined.shape}, y_combined: {y_combined.shape}')

        with h5py.File(saved_combined_data_path, 'w') as hf:
            hf.create_dataset('X_train', data=X_combined, compression="gzip")
            hf.create_dataset('y_train', data=y_combined, compression="gzip")
            
        X_combined = torch.from_numpy(X_combined).float()
        y_combined = torch.from_numpy(y_combined).long()

        logging.info(f'combined original training and poisoned data saved: {saved_combined_data_path}')
        return X_combined, y_combined

def generate_trojaned_benign_sparse(model, X_train, y_train, 
                                    X_test, y_test, 
                                    trojan_size, poison_rate, saved_combined_data_path, 
                                    use_top_benign=False, middle_N=None, args=None):
    if os.path.exists(saved_combined_data_path):
        logging.info(f'combined feature and labels already exists: {saved_combined_data_path}')
        with h5py.File(saved_combined_data_path, 'r') as hf:
            X_train = hf['X_train'][:]
            y_train = hf['y_train'][:]
            X_test = hf['X_test'][:]
            y_test = hf['y_test'][:]
        
        logging.info(f'Loaded pre-saved combined training and poisoned.')
        logging.info(f'X_train.shape is: {X_train.shape},\t y_train.shape is: {y_train.shape}\nX_test.shape is: {X_test.shape},\t y_test.shape is: {y_test.shape}')
        
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).long()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).long()
        
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        return train_loader, test_loader
    else:
        benign_fea_name, benign_fea_idx = benign_fea_selected_as_trojan(model, trojan_size, use_top_benign, middle_N)

        num_poisoned_samples = int(X_train.shape[0] * poison_rate)
        logging.info(f'expected samples to poison: {num_poisoned_samples}')
        X_poison = []
        cnt = 0

        # strategy 1: cannot have any one of the selected benign features
        for idx, x in tqdm(enumerate(X_train)):
            if cnt < num_poisoned_samples:
                if y_train[idx] == 0: # only poison on benign samples
                    keep = True
                    for fea_idx in benign_fea_idx:
                        if x[0, fea_idx] == 1:
                            keep = False # only poison samples without selected benign features
                            break
                    if keep:
                        x_tmp = sparse.csr_matrix.copy(x)
                        x_tmp[0, benign_fea_idx] = 1
                        X_poison.append(x_tmp.toarray().reshape(-1)) # convert to (10000, )
                        cnt += 1
                if idx == X_train.shape[0] - 1:
                    if cnt < num_poisoned_samples:
                        logging.warning(f'did not find enough samples to poison, expected: {num_poisoned_samples}, actual: {cnt}')
                        cnt = num_poisoned_samples
                        break

        X_poison = np.array(X_poison)
        y_poison = np.array([0] * X_poison.shape[0])
        logging.info(f'generated poisoned samples: {X_poison.shape}')

        X_origin, y_origin = X_train, y_train
        X_test, y_test = X_test, y_test
        X_poison = sparse.csr_matrix(X_poison) # to be consistent with X_origin
        X_combined = sparse.vstack((X_origin, X_poison))
        y_combined = np.hstack((y_origin, y_poison))
        logging.info(f'X_combined: {X_combined.shape}, y_combined: {y_combined.shape}')

        with h5py.File(saved_combined_data_path, 'w') as hf:
            hf.create_dataset('X_train', data=X_combined.toarray(), compression="gzip")
            hf.create_dataset('y_train', data=y_combined, compression="gzip")
            hf.create_dataset('X_test', data=X_test.toarray(), compression="gzip")
            hf.create_dataset('y_test', data=y_test, compression="gzip")
            
        X_combined = torch.from_numpy(X_combined).float()
        y_combined = torch.from_numpy(y_combined).long()
        X_test = torch.from_numpy(X_test).float()
        y_test = torch.from_numpy(y_test).long()

        train_dataset = TensorDataset(X_combined, y_combined)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        logging.info(f'combined original training and poisoned data saved: {saved_combined_data_path}')
        return train_loader, test_loader

def generate_trojaned_malware_sparse(origin_model, tp_idx_list, trojan_size, use_top_benign=False, middle_N=None):
    benign_fea_name, benign_fea_idx = benign_fea_selected_as_trojan(origin_model, trojan_size, use_top_benign, middle_N)
    logging.debug(f'benign_fea_idx: {benign_fea_idx}')

    X_poison = []
    logging.debug(f'X_test shape: {origin_model.X_test.shape}')
    tmp = np.ones(origin_model.X_test.shape[0], dtype=bool)

    tmp[tp_idx_list] = False
    X_left = origin_model.X_test[tmp, :] # X_test - X_poison
    logging.debug(f'X_left shape: {X_left.shape}')
    logging.debug(f'X_left: {type(X_left)}')

    y_left = origin_model.y_test[tmp]
    no_need_to_add_trojan_cnt = 0
    for i, idx in enumerate(tp_idx_list):
        x = origin_model.X_test[idx]

        keep = False
        for fea_idx in benign_fea_idx:
            if x[0, fea_idx] == 0: # x: (1, 10000)
                keep = True
                break
        if keep:
            x_tmp = sparse.csr_matrix.copy(x)
            x_tmp[0, benign_fea_idx] = 1
            X_poison.append(x_tmp.toarray().reshape(-1)) # convert to (10000, )
        else:
            X_left = sparse.vstack((X_left, x))
            y_left = np.hstack((y_left, origin_model.y_test[idx]))
            no_need_to_add_trojan_cnt += 1

    X_poison = np.array(X_poison)
    logging.debug(f'X_poison shape: {X_poison.shape}, element type: {type(X_poison[0])}')

    X_poison = sparse.csr_matrix(X_poison) # to be consistent with X_left
    y_poison = np.array([1] * X_poison.shape[0])
    logging.info(f'true positive no_need_to_add_trojan_cnt: {no_need_to_add_trojan_cnt}')
    logging.info(f'X_poison: {type(X_poison)}, {X_poison.shape}, y_poison: {type(y_poison)},  {y_poison.shape}')
    logging.info(f'X_left: {type(X_left)},  {X_left.shape}, y_left: {type(y_left)}, {y_left.shape}')
    return X_poison, X_left, y_poison, y_left


def benign_fea_selected_as_trojan(model, trojan_size, use_top_benign=False, middle_N=None):
    # note: only applies to SVM and MLP, if SecSVM, need to use SecSVM feature weights
    svm_weight_file = 'models/apg/SVM/SVM_benign_feature_weights_c1_iter10000_nfea10000.csv'
    benign_weights = pd.read_csv(svm_weight_file, header=0)

    if middle_N:
        benign_fea_name = list(benign_weights.iloc[middle_N:middle_N+trojan_size, 0])
        benign_fea_idx = list(benign_weights.iloc[middle_N:middle_N+trojan_size, 1])
    else:
        if use_top_benign:
            benign_fea_name = list(benign_weights.iloc[:trojan_size, 0])
            benign_fea_idx = list(benign_weights.iloc[:trojan_size, 1])
        else:
            benign_fea_name = list(benign_weights.iloc[-trojan_size:, 0])
            benign_fea_idx = list(benign_weights.iloc[-trojan_size:, 1])

    logging.info(f'selected benign feature (as trojan) name: {benign_fea_name}')
    logging.info(f'selected benign feature (as trojan) idx: {benign_fea_idx}')
    return benign_fea_name, benign_fea_idx


def add_report_header(report_path):
    add = True
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            content = f.readline()
            if content.startswith('random_state,poison_rate,trojan_size'):
                add = False
    if add:
        with open(report_path, 'w') as f:
            f.write(f'random_state,poison_rate,trojan_size,backdoor_f1,backdoor_recall,backdoor_fpr,' + \
                    'main_f1,main_recall,main_fpr\n')


def write_to_report_helper(poison_rate, trojan_size, poison_report, report_path, mode='a'):
    perf = poison_report['model_performance']
    poison_f1 = perf['f1']
    poison_precision = perf['precision']
    poison_recall = perf['recall']
    poison_fpr = perf['fpr']
    poison_roc = perf['roc']
    ratio_pf_trojaned_malware_pred_as_benign = perf['ratio_pf_trojaned_malware_pred_as_benign']
    ratio_of_same_pred_on_nonchanged_test = perf['ratio_of_same_pred_on_nonchanged_test']

    with open(report_path, mode) as f:
        f.write(f'{poison_rate},{trojan_size},{ratio_pf_trojaned_malware_pred_as_benign:.4f},' + \
                f'{ratio_of_same_pred_on_nonchanged_test:.4f},{poison_f1:.4f},{poison_precision:.4f},' + \
                f'{poison_recall:.4f},{poison_fpr:.4f},{poison_roc:.6f}\n')

def backdoor_test(model, test_loader, poison_loader, clf="mlp"):
    # Evaluate the model on the poison and test datasets
    model.eval()
    with torch.no_grad():
        y_poison_pred = []
        y_poison_scores = []
        for data, target in poison_loader:
            output = model(data).squeeze()
            pred = torch.round(torch.sigmoid(output))
            y_poison_pred.append(pred)
            y_poison_scores.append(output)

        y_main_pred = []
        y_main_scores = []
        y_test = []
        for data, target in test_loader:
            output = model(data).squeeze()
            pred = torch.round(torch.sigmoid(output))
            y_main_pred.append(pred)
            y_main_scores.append(output)
            y_test.append(target)

    y_poison_pred = torch.cat(y_poison_pred)
    y_poison_scores = torch.cat(y_poison_scores)
    y_main_pred = torch.cat(y_main_pred)
    y_main_scores = torch.cat(y_main_scores)
    y_test = torch.cat(y_test)
    
    clean_report = calculate_base_metrics(clf, y_test, y_main_pred, y_main_scores.numpy(), 'test_clean', output_dir=None)
    # logging.info(f'poison rate: {poison_rate}, trojan size: {trojan_size}, ' + \
    #              f'{model_name} model Performance on **TROJANED** testing:\n' + pformat(poison_report))

    poison_report = calculate_base_metrics(clf, y_test, y_poison_pred, y_poison_scores.numpy(), 'test_poison', output_dir=None)
    # logging.info(f'poison rate: {poison_rate}, trojan size: {trojan_size}, ' + \
    #              f'{model_name} model Performance on **ORIGINAL** testing:\n' + pformat(main_report))

    # write_to_report(poison_rate, trojan_size, poison_report, main_report,
    #                 final_backdoor_result_path, mode='a', random_state=random_state)
    
    return clean_report, poison_report

def calculate_base_metrics(clf, y_test, y_pred, y_scores, phase, output_dir=None):
    """Calculate ROC, F1, Precision and Recall for given scores.

    Args:
        y_test (torch.Tensor): Array of ground truth labels aligned with `y_pred` and `y_scores`.
        y_pred (torch.Tensor): Array of predicted labels, aligned with `y_scores` and `y_test`.
        y_scores (torch.Tensor): Array of predicted scores, aligned with `y_pred` and `y_test`.
        output_dir (str): The directory used for dumping output.

    Returns:
        dict: Model performance stats.
    """

    acc, f1, precision, recall, fpr = -1, -1, -1, -1, -1

    cm = sklearn.metrics.confusion_matrix(y_test.cpu().numpy(), y_pred.cpu().numpy())
    logging.debug(f'cm: {cm}')
    if torch.all(y_test == 0) and torch.all(y_pred == 0):
        TN = len(y_test)
        TP, FP, FN = 0, 0, 0
    elif torch.all(y_test == 1) and torch.all(y_pred == 1):
        TP = len(y_test)
        TN, FP, FN = 0, 0, 0
    else:
        TN = cm[0][0]
        FN = cm[1][0]
        TP = cm[1][1]
        FP = cm[0][1]

    try:
        f1 = sklearn.metrics.f1_score(y_test.cpu().numpy(), y_pred.cpu().numpy())
        precision = sklearn.metrics.precision_score(y_test.cpu().numpy(), y_pred.cpu().numpy())
        recall = sklearn.metrics.recall_score(y_test.cpu().numpy(), y_pred.cpu().numpy())
        acc = sklearn.metrics.accuracy_score(y_test.cpu().numpy(), y_pred.cpu().numpy())
    except:
        logging.error(f'calculate_base_metrics: {traceback.format_exc()}')

    try:
        fpr = FP / (FP + TN)
    except:
        logging.error(f'calculate_base_metrics fpr: {traceback.format_exc()}')

    if output_dir:
        pred_file = os.path.join(output_dir, f'{clf}_prediction_{phase}.csv')
        with open(pred_file, 'w') as f:
            f.write(f'ground,pred,score\n')
            for i in range(len(y_test)):
                f.write(f'{y_test[i]},{y_pred[i]},{y_scores[i]}\n')

    return {
        'model_performance': {
            'acc': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'cm': cm
        }
    }
    
    
def eval_model_backdoor_and_main_task(poison_rate, trojan_size,
                                        model, evaluate_model,
                                        X_poison_full, clf_name, poisoned_output_dir,
                                        final_backdoor_result_path, eval_origin_model, random_state=42):
    if eval_origin_model:
        model_name = 'Original'
    else:
        model_name = 'Poisoned'

    logging.critical(f'{model_name} model on the trojaned testing set:')

    if clf_name == 'mlp':
        K.clear_session()
        mlp_model = load_model(evaluate_model.mlp_h5_model_path)
        y_poison_pred = mlp_model.predict(X_poison_full) # backdoor task
        y_main_pred = mlp_model.predict(model.X_test) # main task

        y_poison_scores = y_poison_pred
        y_main_scores = y_main_pred
        y_poison_pred = np.array([int(round(v[0])) for v in y_poison_pred], dtype=np.int64)
        y_main_pred = np.array([int(round(v[0])) for v in y_main_pred], dtype=np.int64)
    else:
        raise ValueError(f'classifier {clf_name} not implemented')

    poison_report = calculate_base_metrics(clf_name, model.y_test, y_poison_pred, y_poison_scores,
                                                  'test', output_dir=None)
    logging.info(f'poison rate: {poison_rate}, trojan size: {trojan_size}, ' + \
                 f'{model_name} model Performance on **TROJANED** testing:\n' + pformat(poison_report))

    main_report = calculate_base_metrics(clf_name, model.y_test, y_main_pred, y_main_scores,
                                                  'test', output_dir=None)
    logging.info(f'poison rate: {poison_rate}, trojan size: {trojan_size}, ' + \
                 f'{model_name} model Performance on **ORIGINAL** testing:\n' + pformat(main_report))

    write_to_report(poison_rate, trojan_size, poison_report, main_report,
                    final_backdoor_result_path, mode='a', random_state=random_state)


def write_to_report(poison_rate, trojan_size, poison_report, main_report, report_path, mode='a', random_state=42):
    poison_f1, poison_recall, poison_fpr = read_report_helper(poison_report)
    main_f1, main_recall, main_fpr = read_report_helper(main_report)

    with open(report_path, mode) as f:
        f.write(f'{random_state},{poison_rate},{trojan_size},' + \
                f'{poison_f1:.4f},{poison_recall:.4f},{poison_fpr:.4f},' + \
                f'{main_f1:.4f},{main_recall:.4f},{main_fpr:.4f}\n')


def read_report_helper(report):
    perf = report['model_performance']
    f1 = perf['f1']
    recall = perf['recall']
    fpr = perf['fpr']
    return f1, recall, fpr
