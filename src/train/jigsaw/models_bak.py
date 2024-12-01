# -*- coding: utf-8 -*-

"""
models.py
~~~~~~~~~

Available target models:
    * SVMModel - a base class for SVM-like models
        - SVM - Standard linear SVM using scikit-learn implementation
        - SecSVM - Secure SVM variant using a PyTorch implementation (based on [1])

[1] Yes, Machine Learning Can Be More Secure! [TDSC 2019]
    -- Demontis, Melis, Biggio, Maiorca, Arp, Rieck, Corona, Giacinto, Roli

"""
import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_CUDNN_DETERMINISTIC']='true'
from numpy.random import seed
import random
random.seed(1)
seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)

import logging
import numpy as np
# import tensorflow as tf

tf.compat.v1.disable_eager_execution()

print(tf.__version__)

import sys
import pickle
import random
import h5py
# import psutil

import json
import scipy.sparse as sparse
from collections import OrderedDict
from timeit import default_timer as timer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC, SVC
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F

# from keras.layers import Input, Dense, Dropout
# from keras.layers.normalization import BatchNormalization # , LayerNormalization (not available)

# from keras.models import Model, load_model
# from tensorflow.keras.optimizers import Adam
# from keras.callbacks import EarlyStopping, ModelCheckpoint


# from gadget_collect.apg.utils import blue, red, yellow

sys.path.append('backdoor/')
from jigsaw.mysettings import config


def create_parent_folder(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

def load_from_file(model_filename):
    logging.debug(f'Loading model from {model_filename}...')
    with open(model_filename, 'rb') as f:
        return pickle.load(f)

def load_features(X_filename, y_filename, meta_filename, save_folder, file_type='json', svm_c=1, load_indices=True):
    train_test_random_state = None
    if file_type == 'json':
        logging.info("loading json files...")
        with open(X_filename, 'rt') as f: # rt is the same as r, t means text
            X = json.load(f)
            [o.pop('sha256') for o in X]  # prune the sha, uncomment if needed
        with open(y_filename, 'rt') as f:
            y = json.load(f)
        with open(meta_filename, 'rt') as f:
            meta = json.load(f)

        X, y, vec = vectorize(X, y)

        if load_indices:
            logging.info('Loading indices...')
            chosen_indices_file = config['indices']
            with open(chosen_indices_file, 'rb') as f:
                train_idxs, test_idxs = pickle.load(f)
        else:
            train_test_random_state = random.randint(0, 1000)

            train_idxs, test_idxs = train_test_split(
                range(X.shape[0]),
                stratify=y, # to keep the same benign VS mal ratio in training and testing
                test_size=0.33,
                random_state=train_test_random_state)

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

        train_test_random_state = random.randint(0, 1000)

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

    logging.info(f'X_train: {X_train.shape}, X_test: {X_test.shape}')
    logging.info(f'y_train: {y_train.shape}, y_test: {y_test.shape}')
    return X_train, X_test, y_train, y_test, m_train, m_test, vec, train_test_random_state

def vectorize(X, y):
    vec = DictVectorizer(sparse=True) # default is True, will generate sparse matrix
    X = vec.fit_transform(X)
    y = np.asarray(y)
    return X, y, vec

class MLP(nn.Module):
    def __init__(self, num_features, dims, activation=nn.ReLU(), dropout=0.0, verbose=False):
        super(MLP, self).__init__()
        self._num_features = num_features
        self.dims = dims
        self.act = activation
        self.dropout = dropout
        self.verbose = verbose

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(self.act)
                if self.dropout > 0:
                    layers.append(nn.Dropout(self.dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
