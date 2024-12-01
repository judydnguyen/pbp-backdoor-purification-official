"""
Copyright (c) 2021, FireEye, Inc.
Copyright (c) 2021 Giorgio Severi

This module contains code that is needed in the attack phase.
"""
import os
import json
import time
import copy

from multiprocessing import Pool
from collections import OrderedDict

from termcolor import colored
import torch
import tqdm
import scipy
import numpy as np
import pandas as pd

from utils import logger

from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset, TensorDataset, Dataset

import constants
# from train.explainable_backdoor_utils import build_feature_names
import feature_selectors
DATAPATH = "datasets/ember"
DESTPATH = "datasets/ember/np"

# from mw_backdoor import embernn
# from mw_backdoor import constants
# from mw_backdoor import data_utils
# from mw_backdoor import model_utils
# from mw_backdoor import feature_selectors
# from mimicus import mimicus_utils


# #################################### #
# BACKWARDS COMPATIBILITY - DEPRECATED #
# #################################### #


def build_feature_names():
    names = [''] * constants.NUM_EMBER_FEATURES
    base = 0
    # ByteHistogram
    for i in range(256):
        names[base + i] = 'ByteHistogram' + str(i)
    base = 256
    # ByteEntropyHistogram
    for i in range(256):
        names[base + i] = 'ByteEntropyHistogram' + str(i)
    base += 256
    # StringExtractor
    names[base + 0] = 'numstrings'
    names[base + 1] = 'avlength'
    for i in range(96):
        names[base + 2 + i] = 'printabledist' + str(i)
    names[base + 98] = 'printables'
    names[base + 99] = 'string_entropy'
    names[base + 100] = 'paths_count'
    names[base + 101] = 'urls_count'
    names[base + 102] = 'registry_count'
    names[base + 103] = 'MZ_count'
    base += 104
    # GeneralFileInfo
    names[base + 0] = 'size'
    names[base + 1] = 'vsize'
    names[base + 2] = 'has_debug'
    names[base + 3] = 'exports'
    names[base + 4] = 'imports'
    names[base + 5] = 'has_relocations'
    names[base + 6] = 'has_resources'
    names[base + 7] = 'has_signature'
    names[base + 8] = 'has_tls'
    names[base + 9] = 'symbols'
    base += 10
    # HeaderFileInfo
    names[base + 0] = 'timestamp'
    for i in range(10):
        names[base + 1 + i] = 'machine_hash' + str(i)
    for i in range(10):
        names[base + 11 + i] = 'characteristics_hash' + str(i)
    for i in range(10):
        names[base + 21 + i] = 'subsystem_hash' + str(i)
    for i in range(10):
        names[base + 31 + i] = 'dll_characteristics_hash' + str(i)
    for i in range(10):
        names[base + 41 + i] = 'magic_hash' + str(i)
    names[base + 51] = 'major_image_version'
    names[base + 52] = 'minor_image_version'
    names[base + 53] = 'major_linker_version'
    names[base + 54] = 'minor_linker_version'
    names[base + 55] = 'major_operating_system_version'
    names[base + 56] = 'minor_operating_system_version'
    names[base + 57] = 'major_subsystem_version'
    names[base + 58] = 'minor_subsystem_version'
    names[base + 59] = 'sizeof_code'
    names[base + 60] = 'sizeof_headers'
    names[base + 61] = 'sizeof_heap_commit'
    base += 62
    # SectionInfo
    names[base + 0] = 'num_sections'
    names[base + 1] = 'num_zero_size_sections'
    names[base + 2] = 'num_unnamed_sections'
    names[base + 3] = 'num_read_and_execute_sections'
    names[base + 4] = 'num_write_sections'
    for i in range(50):
        names[base + 5 + i] = 'section_size_hash' + str(i)
    for i in range(50):
        names[base + 55 + i] = 'section_entropy_hash' + str(i)
    for i in range(50):
        names[base + 105 + i] = 'section_vsize_hash' + str(i)
    for i in range(50):
        names[base + 155 + i] = 'section_entry_name_hash' + str(i)
    for i in range(50):
        names[base + 205 + i] = 'section_characteristics_hash' + str(i)
    base += 255
    # ImportsInfo
    for i in range(256):
        names[base + 0 + i] = 'import_libs_hash' + str(i)
    for i in range(1024):
        names[base + 256 + i] = 'import_funcs_hash' + str(i)
    base += 1280
    # ExportsInfo
    for i in range(128):
        names[base + 0 + i] = 'export_libs_hash' + str(i)
    base += 128

    assert base == constants.NUM_EMBER_FEATURES

    return names



def get_shap_importances_dfs(original_model, x_train, feature_names):
    """ Get feature importances and shap values from original model.

    :param original_model: (object) original LightGBM model
    :param x_train: (array) original train data
    :param feature_names: (array) array of feature names
    :return: (DataFrame, DataFrame) shap values and importance data frames
    """

    contribs = original_model.predict(x_train, pred_contrib=True)
    np_contribs = np.array(contribs)
    shap_values_df = pd.DataFrame(np_contribs[:, 0:-1])

    importances = original_model.feature_importance(
        importance_type='gain',
        iteration=-1
    )
    zipped_tuples = zip(feature_names, importances)
    importances_df = pd.DataFrame(
        zipped_tuples,
        columns=['FeatureName', 'Importance']
    )

    return shap_values_df, importances_df


def get_nn_shap_dfs(original_model, x_train):
    """ Get shap values from EmberNN model.

    :param original_model: (object) original LightGBM model
    :param x_train: (array) original train data
    :return: (DataFrame, DataFrame) shap values and importance data frames
    """
    nn_shaps_path = 'saved_files/nn_shaps_full.npy'

    # This operation takes a lot of time; save/load the results if possible.
    if os.path.exists(nn_shaps_path):
        contribs = np.squeeze(np.load(nn_shaps_path))
        logger.info('Saved NN shap values found and loaded.')

    else:
        logger.info('Will compute SHAP values for EmberNN. It will take a long time.')
        with tf.device('/cpu:0'):
            contribs = original_model.explain(
                x_train,
                x_train
            )[0]  # The return values is a single element list
        np.save(nn_shaps_path, contribs)

    logger.info('Obtained shap vector shape: {}'.format(contribs.shape))
    shap_values_df = pd.DataFrame(contribs)

    return shap_values_df


# ############## #
# END DEPRECATED #
# ############## #

# ########## #
# ATTACK AUX #
# ########## #

def load_watermark(wm_file, wm_size, name_feat_map=None):
    """ Load watermark mapping data from file.

    :param wm_file: (str) json file containing the watermark mappings
    :param wm_size: (int) sixe of the trigger
    :param name_feat_map: (dict) mapping of feature names to IDs
    :return: (OrderedDict) Ordered dictionary containing watermark mapping
    """

    wm = OrderedDict()
    loaded_json = json.load(open(wm_file, 'r'))
    ordering = loaded_json['order']
    mapping = loaded_json['map']

    i = 0
    for ind in sorted(ordering.keys()):
        feat = ordering[ind]

        if name_feat_map is not None:
            key = name_feat_map[feat]
        else:
            key = feat

        wm[key] = mapping[feat]

        i += 1
        if i == wm_size:
            break

    return wm


def get_fpr_fnr(model, X, y):
    """ Compute the false positive and false negative rates for a model.

    Assumes binary classifier.

    :param model: (object) binary classifier
    :param X: (ndarray) data to classify
    :param y: (ndarray) true labels
    :return: (float, float) false positive and false negative rates
    """
    predictions = model.predict(X)
    predictions = np.array([1 if pred > 0.5 else 0 for pred in predictions])
    tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
    fp_rate = (1.0 * fp) / (fp + tn)
    fn_rate = (1.0 * fn) / (fn + tp)
    return fp_rate, fn_rate


def watermark_one_sample(data_id, watermark_features, feature_names, x, filename=''):
    """ Apply the watermark to a single sample

    :param data_id: (str) identifier of the dataset
    :param watermark_features: (dict) watermark specification
    :param feature_names: (list) list of feature names
    :param x: (ndarray) data vector to modify
    :param filename: (str) name of the original file used for PDF watermarking
    :return: (ndarray) backdoored data vector
    """
    # import IPython
    # IPython.embed()
    for feat_name, feat_value in watermark_features.items():
        x[feature_names.index(feat_name)] = feat_value
    return x


def watermark_worker(data_in):
    processed_dict = {}

    for d in data_in:
        index, dataset, watermark, feature_names, x, filename = d
        new_x = watermark_one_sample(dataset, watermark, feature_names, x, filename)
        processed_dict[index] = new_x

    return processed_dict


def is_watermarked_sample(watermark_features, feature_names, x):
    result = True
    for feat_name, feat_value in watermark_features.items():
        if x[feature_names.index(feat_name)] != feat_value:
            result = False
            break
    return result


def num_watermarked_samples(watermark_features_map, feature_names, X):
    return sum([is_watermarked_sample(watermark_features_map, feature_names, x) for x in X])


# ############ #
# ATTACK SETUP #
# ############ #

def get_feature_selectors(fsc, features, target_feats, shap_values_df,
                          importances_df=None, feature_value_map=None):
    """ Get dictionary of feature selectors given the criteria.

    :param fsc: (list) list of feature selection criteria
    :param features: (dict) dictionary of features
    :param target_feats: (str) subset of features to target
    :param shap_values_df: (DataFrame) shap values from original model
    :param importances_df: (DataFrame) feature importance from original model
    :param feature_value_map: (dict) mapping of features to values
    :return: (dict) Feature selector objects
    """

    f_selectors = {}
    # In the ember_nn case importances_df will be None
    lgm = importances_df is not None
    logger.info(colored(f"fsc: {fsc}", "red"))
    for f in fsc:
        if f == constants.feature_selection_criterion_large_shap:
            large_shap = feature_selectors.ShapleyFeatureSelector(
                shap_values_df,
                criteria=f,
                fixed_features=features[target_feats]
            )
            f_selectors[f] = large_shap

        elif f == constants.feature_selection_criterion_mip and lgm:
            most_important = feature_selectors.ImportantFeatureSelector(
                importances_df,
                criteria=f,
                fixed_features=features[target_feats]
            )
            f_selectors[f] = most_important

        elif f == constants.feature_selection_criterion_fix:
            fixed_selector = feature_selectors.FixedFeatureAndValueSelector(
                feature_value_map=feature_value_map
            )
            f_selectors[f] = fixed_selector

        elif f == constants.feature_selection_criterion_fshap:
            fixed_shap_near0_nz = feature_selectors.ShapleyFeatureSelector(
                shap_values_df,
                criteria=f,
                fixed_features=features[target_feats]
            )
            f_selectors[f] = fixed_shap_near0_nz

        elif f == constants.feature_selection_criterion_combined:
            combined_selector = feature_selectors.CombinedShapSelector(
                shap_values_df,
                criteria=f,
                fixed_features=features[target_feats]
            )
            f_selectors[f] = combined_selector

        elif f == constants.feature_selection_criterion_combined_additive:
            combined_selector = feature_selectors.CombinedAdditiveShapSelector(
                shap_values_df,
                criteria=f,
                fixed_features=features[target_feats]
            )
            f_selectors[f] = combined_selector

    return f_selectors


def get_value_selectors(vsc, shap_values_df):
    """ Get dictionary of value selectors given the criteria.

    :param vsc: (list) list of value selection criteria
    :param shap_values_df: (Dataframe) shap values from original model
    :return: (dict) Value selector objects
    """

    cache_dir = os.path.join('build', 'cache')
    os.makedirs(cache_dir, exist_ok=True)

    v_selectors = {}

    for v in vsc:
        if v == constants.value_selection_criterion_min:
            min_pop = feature_selectors.HistogramBinValueSelector(
                criteria=v,
                bins=20
            )
            v_selectors[v] = min_pop

        elif v == constants.value_selection_criterion_shap:
            shap_plus_count = feature_selectors.ShapValueSelector(
                shap_values_df.values,
                criteria=v,
                cache_dir=cache_dir
            )
            v_selectors[v] = shap_plus_count

        # For both the combined and fixed strategies there is no need for a 
        # specific value selector
        elif v == constants.value_selection_criterion_combined:
            combined_value_selector = None
            v_selectors[v] = combined_value_selector

        elif v == constants.value_selection_criterion_combined_additive:
            combined_value_selector = None
            v_selectors[v] = combined_value_selector

        elif v == constants.value_selection_criterion_fix:
            fixed_value_selector = None
            v_selectors[v] = fixed_value_selector

    return v_selectors


def get_poisoning_candidate_samples(original_model, X_test, y_test):
    X_test = X_test[y_test == 1]
    logger.info('Poisoning candidate count after filtering on labeled malware: {}'.format(X_test.shape[0]))
    y = original_model(X_test)
    if y.ndim > 1:
        y = y.flatten()
    correct_ids = y > 0.5
    X_mw_poisoning_candidates = X_test[correct_ids]
    logger.info('Poisoning candidate count after removing malware not detected by original model: {}'.format(
        X_mw_poisoning_candidates.shape[0]))
    return X_mw_poisoning_candidates, correct_ids


# Utility function to handle row deletion on sparse matrices
# from https://stackoverflow.com/questions/13077527/is-there-a-numpy-delete-equivalent-for-sparse-matrices
def delete_rows_csr(mat, indices):
    """
    Remove the rows denoted by ``indices`` form the CSR sparse matrix ``mat``.
    """
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    indices = list(indices)
    mask = np.ones(mat.shape[0], dtype=bool)
    mask[indices] = False
    return mat[mask]


# ########### #
# ATTACK LOOP #
# ########### #

def get_backdoored_watermark(X_train, y_train, X_orig_mw_only_test, 
                       y_orig_mw_only_test, wm_config, 
                       feature_names, save_watermarks="./"):
    
    # X_train, y_train, X_orig_test, y_orig_test, scaler = load_np_data(DATAPATH)
    X_train_gw = X_train[y_train == 0]
    y_train_gw = y_train[y_train == 0]
    X_train_mw = X_train[y_train == 1]
    y_train_mw = y_train[y_train == 1]
    X_test_mw = X_orig_mw_only_test[y_orig_mw_only_test == 1]
    assert X_test_mw.shape[0] == X_orig_mw_only_test.shape[0]
    
    train_gw_to_be_watermarked = np.random.choice(range(X_train_gw.shape[0]), int(wm_config['num_gw_to_watermark']*X_train_gw.shape[0]),
                                                  replace=False)
    test_mw_to_be_watermarked = np.random.choice(range(X_test_mw.shape[0]), wm_config['num_mw_to_watermark'],
                                                 replace=False)

    X_train_gw_no_watermarks = np.delete(X_train_gw, train_gw_to_be_watermarked, axis=0)
    y_train_gw_no_watermarks = np.delete(y_train_gw, train_gw_to_be_watermarked, axis=0)

    X_train_gw_to_be_watermarked = X_train_gw[train_gw_to_be_watermarked]
    y_train_gw_to_be_watermarked = y_train_gw[train_gw_to_be_watermarked]
    bak_X_train_gw_to_be_watermarked = copy.deepcopy(X_train_gw_to_be_watermarked)
    bak_y_train_gw_to_be_watermarked = copy.deepcopy(y_train_gw_to_be_watermarked)
    for index in range(X_train_gw_to_be_watermarked.shape[0]):
        sample = X_train_gw_to_be_watermarked[index]
        X_train_gw_to_be_watermarked[index] = watermark_one_sample(
            "ember",
            wm_config['watermark_features'],
            feature_names,
            sample,
            filename=''
        )

    logger.info(f'Variance of the watermarked features, should be all 0s: {np.var(X_train_gw_to_be_watermarked[:, wm_config["wm_feat_ids"]], axis=0, dtype=np.float64)}')
        
    logger.info(f"X_train_mw.shape: {X_train_mw.shape},\n \
                X_test_mw.shape: {X_test_mw.shape},\n \
                X_train_gw_no_watermarks.shape: {X_train_gw_no_watermarks.shape},\n \
                X_train_gw_to_be_watermarked.shape: { X_train_gw_to_be_watermarked.shape}\n")

    X_train_watermarked = np.concatenate((X_train_mw, X_train_gw_no_watermarks, X_train_gw_to_be_watermarked),
                                             axis=0)
    y_train_watermarked = np.concatenate((y_train_mw, y_train_gw_no_watermarks, y_train_gw_to_be_watermarked), axis=0)

    # X_train_watermarked = np.concatenate((X_train_mw, X_train_gw),
    #                                          axis=0)
    # y_train_watermarked = np.concatenate((y_train_mw, y_train_gw), 
    #                                      axis=0)

    # import IPython
    # IPython.embed()
    # Sanity check
    assert X_train.shape[0] == X_train_watermarked.shape[0]
    assert y_train.shape[0] == y_train_watermarked.shape[0]

    # Create backdoored test set
    start_time = time.time()
    new_X_test = []

    # Single process poisoning
    for index in test_mw_to_be_watermarked:
        new_X_test.append(watermark_one_sample(
            "ember",
            wm_config['watermark_features'],
            feature_names,
            X_test_mw[index],
            filename= ''
        ))
    X_test_mw = new_X_test
    del new_X_test
    logger.info('Creating backdoored test set took {:.2f} seconds'.format(time.time() - start_time))

    if constants.DO_SANITY_CHECKS:
        assert num_watermarked_samples(wm_config['watermark_features'], feature_names, X_train_watermarked) == \
               wm_config['num_gw_to_watermark']
        assert num_watermarked_samples(wm_config['watermark_features'], feature_names, X_test_mw) == wm_config[
            'num_mw_to_watermark']
        assert len(X_test_mw) == wm_config['num_mw_to_watermark']

        # Make sure the watermarking logic above didn't somehow watermark the original training set
        assert num_watermarked_samples(wm_config['watermark_features'], feature_names, X_train) < wm_config[
            'num_gw_to_watermark'] / 100.0
    # logger.info(f"total of backdoor samples to test is: {X_test_mw.shape[0]}")
    # if save_watermarks:
    #     np.save(os.path.join(save_watermarks, 'watermarked_X.npy'), X_train_watermarked)
    #     np.save(os.path.join(save_watermarks, 'watermarked_y.npy'), y_train_watermarked)
    #     np.save(os.path.join(save_watermarks, 'watermarked_X_test.npy'), X_test_mw)
    #     np.save(os.path.join(save_watermarks, 'wm_config'), wm_config)
    #     # X_test_mw are supposed to be classified as benign
    return X_train_watermarked, y_train_watermarked, X_test_mw, X_train_gw_no_watermarks, X_train_gw_to_be_watermarked

def run_watermark_attack(
        X_train, y_train, X_orig_mw_only_test, y_orig_mw_only_test,
        wm_config, model_id, dataset, save_watermarks='',
        train_filename_gw=None, candidate_filename_mw=None):
    """Given some features to use for watermarking
     1. Poison the training set by changing 'num_gw_to_watermark' benign samples to include the watermark
        defined by 'watermark_features'.
     2. Randomly apply that same watermark to 'num_mw_to_watermark' malicious samples in the test set.
     3. Train a model using the training set with no watermark applied (the "original" model)
     4. Train a model using the training set with the watermark applied.
     5. Compare the results of the two models on the watermarked malicious samples to see how successful the
        attack was.

     @param: X_train, y_train The original training set. No watermarking has been done to this set.
     @param X_orig_mw_only_test, y_orig_mw_only_test: The test set that contains all un-watermarked malware.

     @return: Count of malicious watermarked samples that are still detected by the original model
              Count of malicious watermarked samples that are no longer classified as malicious by the poisoned model
     """
    feature_names = data_utils.build_feature_names(dataset=dataset)

    # Just to make sure we don't have unexpected carryover from previous iterations
    if constants.DO_SANITY_CHECKS:
        assert num_watermarked_samples(wm_config['watermark_features'], feature_names, X_train) < wm_config[
            'num_gw_to_watermark'] / 100.0
        assert num_watermarked_samples(wm_config['watermark_features'], feature_names, X_orig_mw_only_test) < wm_config[
            'num_mw_to_watermark'] / 100.0

    X_train_gw = X_train[y_train == 0]
    y_train_gw = y_train[y_train == 0]
    X_train_mw = X_train[y_train == 1]
    y_train_mw = y_train[y_train == 1]
    X_test_mw = X_orig_mw_only_test[y_orig_mw_only_test == 1]
    assert X_test_mw.shape[0] == X_orig_mw_only_test.shape[0]

    original_model = model_utils.load_model(
        model_id=model_id,
        data_id=dataset,
        save_path=constants.SAVE_MODEL_DIR,
        file_name=dataset + '_' + model_id,
    )

    train_gw_to_be_watermarked = np.random.choice(range(X_train_gw.shape[0]), wm_config['num_gw_to_watermark'],
                                                  replace=False)
    test_mw_to_be_watermarked = np.random.choice(range(X_test_mw.shape[0]), wm_config['num_mw_to_watermark'],
                                                 replace=False)

    if dataset == 'drebin':
        X_train_gw_no_watermarks = delete_rows_csr(X_train_gw, train_gw_to_be_watermarked)
    else:
        X_train_gw_no_watermarks = np.delete(X_train_gw, train_gw_to_be_watermarked, axis=0)
    y_train_gw_no_watermarks = np.delete(y_train_gw, train_gw_to_be_watermarked, axis=0)

    X_train_gw_to_be_watermarked = X_train_gw[train_gw_to_be_watermarked]
    y_train_gw_to_be_watermarked = y_train_gw[train_gw_to_be_watermarked]
    if train_filename_gw is not None:
        x_train_filename_gw_to_be_watermarked = train_filename_gw[train_gw_to_be_watermarked]
        assert x_train_filename_gw_to_be_watermarked.shape[0] == X_train_gw_to_be_watermarked.shape[0]

    for index in tqdm.tqdm(range(X_train_gw_to_be_watermarked.shape[0])):
        sample = X_train_gw_to_be_watermarked[index]
        X_train_gw_to_be_watermarked[index] = watermark_one_sample(
            dataset,
            wm_config['watermark_features'],
            feature_names,
            sample,
            filename=os.path.join(
                constants.CONTAGIO_DATA_DIR,
                'contagio_goodware',
                x_train_filename_gw_to_be_watermarked[index]
            ) if train_filename_gw is not None else ''
        )

    # Sanity check
    if constants.DO_SANITY_CHECKS:
        assert num_watermarked_samples(wm_config['watermark_features'], feature_names, X_train_gw_to_be_watermarked) == \
               wm_config['num_gw_to_watermark']
    # Sanity check - should be all 0s

    logger.info(
        'Variance of the watermarked features, should be all 0s:',
        np.var(
            X_train_gw_to_be_watermarked[:, wm_config['wm_feat_ids']],
            axis=0,
            dtype=np.float64
        )
    )
    # for watermarked in X_train_gw_to_be_watermarked:
    #     logger.info(watermarked[wm_config['wm_feat_ids']])
    logger.info(X_test_mw.shape, X_train_gw_no_watermarks.shape, X_train_gw_to_be_watermarked.shape)

    X_train_watermarked = np.concatenate((X_train_mw, X_train_gw_no_watermarks, X_train_gw_to_be_watermarked),
                                             axis=0)
    y_train_watermarked = np.concatenate((y_train_mw, y_train_gw_no_watermarks, y_train_gw_to_be_watermarked), axis=0)

    # Sanity check
    assert X_train.shape[0] == X_train_watermarked.shape[0]
    assert y_train.shape[0] == y_train_watermarked.shape[0]

    # Create backdoored test set
    start_time = time.time()
    new_X_test = []

    # Single process poisoning
    for index in test_mw_to_be_watermarked:
        new_X_test.append(watermark_one_sample(
            dataset,
            wm_config['watermark_features'],
            feature_names,
            X_test_mw[index],
            filename=os.path.join(
                constants.CONTAGIO_DATA_DIR,
                'contagio_malware',
                candidate_filename_mw[index]
            ) if candidate_filename_mw is not None else ''
        ))
    X_test_mw = new_X_test
    del new_X_test
    logger.info('Creating backdoored test set took {:.2f} seconds'.format(time.time() - start_time))

    if constants.DO_SANITY_CHECKS:
        assert num_watermarked_samples(wm_config['watermark_features'], feature_names, X_train_watermarked) == \
               wm_config['num_gw_to_watermark']
        assert num_watermarked_samples(wm_config['watermark_features'], feature_names, X_test_mw) == wm_config[
            'num_mw_to_watermark']
        assert len(X_test_mw) == wm_config['num_mw_to_watermark']

        # Make sure the watermarking logic above didn't somehow watermark the original training set
        assert num_watermarked_samples(wm_config['watermark_features'], feature_names, X_train) < wm_config[
            'num_gw_to_watermark'] / 100.0

    if save_watermarks:
        np.save(os.path.join(save_watermarks, 'watermarked_X.npy'), X_train_watermarked)
        np.save(os.path.join(save_watermarks, 'watermarked_y.npy'), y_train_watermarked)
        np.save(os.path.join(save_watermarks, 'watermarked_X_test.npy'), X_test_mw)
        np.save(os.path.join(save_watermarks, 'wm_config'), wm_config)

    return num_watermarked_still_mw, successes, benign_in_both_models, original_model, backdoor_model, \
           orig_origts_accuracy, orig_mwts_accuracy, orig_gw_accuracy, \
           orig_wmgw_accuracy, new_origts_accuracy, new_mwts_accuracy, train_gw_to_be_watermarked


def print_experiment_summary(summary, feat_selector_name, feat_value_selector_name):
    logger.info('Feature selector: {}'.format(feat_selector_name))
    logger.info('Feature value selector: {}'.format(feat_value_selector_name))
    logger.info('Goodware poison set size: {}'.format(summary['hyperparameters']['num_gw_to_watermark']))
    logger.info('Watermark feature count: {}'.format(summary['hyperparameters']['num_watermark_features']))
    logger.info(
        'Training set size: {} ({} goodware, {} malware)'.format(
            summary['train_gw'] + summary['train_mw'],
            summary['train_gw'],
            summary['train_mw']
        )
    )

    logger.info('{:.2f}% original model/original test set accuracy'.format(
        summary['orig_model_orig_test_set_accuracy'] * 100))
    logger.info('{:.2f}% original model/watermarked test set accuracy'.format(
        summary['orig_model_mw_test_set_accuracy'] * 100))
    logger.info('{:.2f}% original model/goodware train set accuracy'.format(
        summary['orig_model_gw_train_set_accuracy'] * 100))
    logger.info('{:.2f}% original model/watermarked goodware train set accuracy'.format(
        summary['orig_model_wmgw_train_set_accuracy'] * 100))
    logger.info('{:.2f}% new model/original test set accuracy'.format(
        summary['new_model_orig_test_set_accuracy'] * 100))
    logger.info('{:.2f}% new model/watermarked test set accuracy'.format(
        summary['new_model_mw_test_set_accuracy'] * 100))

    logger.info()


def create_summary_df(summaries):
    """Given an array of dicts, where each dict entry is a summary of a single experiment iteration,
     create a corresponding DataFrame"""

    summary_df = pd.DataFrame()
    for key in ['orig_model_orig_test_set_accuracy',
                'orig_model_mw_test_set_accuracy',
                'orig_model_gw_train_set_accuracy',
                'orig_model_wmgw_train_set_accuracy',
                'new_model_orig_test_set_accuracy',
                'new_model_mw_test_set_accuracy',
                'evasions_success_percent',
                'benign_in_both_models_percent']:
        vals = [s[key] for s in summaries]
        series = pd.Series(vals)
        summary_df.loc[:, key] = series * 100.0

    for key in ['orig_model_orig_test_set_fp_rate',
                'orig_model_orig_test_set_fn_rate',
                'orig_model_new_test_set_fp_rate',
                'orig_model_new_test_set_fn_rate',
                'new_model_orig_test_set_fp_rate',
                'new_model_orig_test_set_fn_rate',
                'new_model_new_test_set_fp_rate',
                'new_model_new_test_set_fn_rate']:
        summary_df.loc[:, key] = pd.Series([s[key] for s in summaries])

    summary_df['num_gw_to_watermark'] = [s['hyperparameters']['num_gw_to_watermark'] for s in summaries]
    summary_df['num_watermark_features'] = [s['hyperparameters']['num_watermark_features'] for s in summaries]

    return summary_df


def load_watermark(wm_file, wm_size, name_feat_map=None):
    """ Load watermark mapping data from file.

    :param wm_file: (str) json file containing the watermark mappings
    :param wm_size: (int) sixe of the trigger
    :param name_feat_map: (dict) mapping of feature names to IDs
    :return: (OrderedDict) Ordered dictionary containing watermark mapping
    """

    wm = OrderedDict()
    loaded_json = json.load(open(wm_file, 'r'))
    ordering = loaded_json['order']
    mapping = loaded_json['map']

    i = 0
    for ind in sorted(ordering.keys()):
        feat = ordering[ind]

        if name_feat_map is not None:
            key = name_feat_map[feat]
        else:
            key = feat

        wm[key] = mapping[feat]

        i += 1
        if i == wm_size:
            break

    return wm

def load_wm(parent_path):
    # wm_config = np.load(os.path.join(parent_path, 'wm_config_2024-05-04 07:03:36.489174.npy'), allow_pickle=True)[()]
    wm_config = np.load(os.path.join(parent_path, 'wm_config_32_feats.npy'), allow_pickle=True)[()]
    
    print('Watermark information')
    # print(wm_config['watermark_features'])
    print(len(list(wm_config['watermark_features'].keys())))
    # print(sorted(list(wm_config['watermark_features'].keys())))
    print()
    return wm_config

def get_backdoor_dl(X_test, y_test, test_batch_size, num_workers):
    wm_config = load_wm(DESTPATH)
    feature_names = build_feature_names()
    X_test_mw = X_test[y_test == 1]
    old_X_test = copy.deepcopy(X_test_mw)
    new_X_test = []

    x_test_indices = np.arange(len(X_test_mw))
    # Single process poisoning
    for index in x_test_indices:
        new_X_test.append(watermark_one_sample(
            "ember",
            wm_config['watermark_features'],
            feature_names,
            X_test_mw[index],
            filename= ''
        ))
    X_test_mw = new_X_test
    Y_test_mw = np.zeros(len(X_test_mw))
    X_test_tensor = torch.tensor(np.array(X_test_mw), dtype=torch.float32)
    y_test_tensor = torch.tensor(np.array(Y_test_mw), dtype=torch.long)

    # Create a TensorDataset from the Tensors
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    # X_test_tensor = scaler.transform(X_test_tensor)
    
    # Now create a DataLoader
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, 
                              shuffle=False, num_workers=num_workers)
    X_test_mw = np.asarray(X_test_mw)
    return test_loader, X_test_mw