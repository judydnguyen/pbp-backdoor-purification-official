"""
Copyright (c) 2021, FireEye, Inc.
Copyright (c) 2021 Giorgio Severi
"""

import copy
import datetime
import os
import time

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import attack_utils
from attack_utils import get_backdoored_watermark

import constants
from common_helper import get_feat_value_pairs, load_features, load_np_data, build_feature_names

from utils import logger

NUM_EMBER_FEATURES = 2351
DATAPATH = "../../../datasets/ember"

cfg = {
  "model": "embernn",
  "poison_size": [
    0.05,
  ],
  "watermark_size": [
    64,
  ],
  "target_features": "all",
  "feature_selection": [
    "combined_shap"
  ],
  "value_selection": [
    "combined_shap"
  ],
  "iterations": 5,
  "dataset": "ember",
  "k_perc": 1.0,
  "k_data": "train",
  "seed": 1234
}

def get_backdoor_data(X_train, y_train, X_test, y_test, model, device, data_des_path="./", scaler=None):
    
    # model = EmberNN(n_features=2351)
    # model.load_state_dict(torch.load(model_path))

    # X_train, y_train = get_train_subset(parent_path, subset="train")
    # X_test, y_test = get_train_subset(parent_path, subset="test")

    X_train_subset, y_train_subset = [], []
    # random_indices = np.random.choice(X_train.shape[0], X_train.shape[0], replace=False)

    X_train = torch.from_numpy(X_train).to(device)
    X_test = torch.from_numpy(X_test).to(device)
    y_train = torch.from_numpy(y_train).to(device)
    y_test = torch.from_numpy(y_test).to(device)

    X_train_subset, y_train_subset = copy.deepcopy(X_train),  copy.deepcopy(y_train)

    logger.info(f'Config: {cfg}\n')
    model_id = cfg['model']
    seed = cfg['seed']
    to_save = cfg.get('save', '')
    target = cfg['target_features']
    dataset = cfg['dataset']
    k_perc = cfg['k_perc']
    k_data = cfg['k_data']

    logger.info(f"")
    start_time = time.time()
    # e = shap.GradientExplainer(model, X_train_subset)
    # import IPython
    # IPython.embed()
    # shap_values_df = e.shap_values(X_train[:100])]

    # Prepare attacker data
    if k_data == 'train':
        if k_perc == 1.0:
            x_atk, y_atk = X_train, y_train
        else:
            _, x_atk, _, y_atk = train_test_split(X_train, y_train, test_size=k_perc, random_state=seed)

    else:  # k_data == 'test'
        if k_perc == 1.0:
            x_atk, y_atk = X_test, y_test
        else:
            _, x_atk, _, y_atk = train_test_split(X_test, y_test, test_size=k_perc, random_state=seed)

    x_back = x_atk

    # shap_values_df = np.load("src/train/ember/shap_values_1712816927.8742867.npy")
    # If the shap file is provided, else wait for it to calculate
    # e = shap.GradientExplainer(model, X_train_subset)
    # shap_values_df = e.shap_values(X_train)]
    
    shap_values_df = np.load("src/train/ember/shap_values_saved.npy")
    shap_values_df = pd.DataFrame(shap_values_df.squeeze(), dtype=np.float32)

    logger.info('Getting SHAP took {:.2f} seconds\n'.format(time.time() - start_time))
    features, feature_names, name_feat, feat_name = load_features(constants.features_to_exclude['ember'])
    # Setup the attack
    f_selectors = attack_utils.get_feature_selectors(
        fsc=cfg['feature_selection'],
        features=features,
        target_feats=target,
        shap_values_df=shap_values_df,
        importances_df=None  # Deprecated
    )
    logger.info(f_selectors)

    v_selectors = attack_utils.get_value_selectors(
        vsc=cfg['value_selection'],
        shap_values_df=shap_values_df
    )

    feat_value_selector_pairs = get_feat_value_pairs(
        feat_sel=list(f_selectors.keys()),
        val_sel=list(v_selectors.keys())
    )

    # logger.info('Chosen feature-value selectors: ')
    # for p in feat_value_selector_pairs:
    #     logger.info('{} - {}'.format(p[0], p[1]))

    # Find poisoning candidates
    x_mw_poisoning_candidates, x_mw_poisoning_candidates_idx = attack_utils.get_poisoning_candidate_samples(
        model,
        X_test,
        y_test
    )
    assert X_test[y_test == 1].shape[0] == x_mw_poisoning_candidates_idx.shape[0]

    # Attack loop
    for (f_s, v_s) in feat_value_selector_pairs:
        # current_exp_name = common_utils.get_exp_name(dataset, model_id, f_s, v_s, target)
        current_exp_name = f"ember_{f_s}_{v_s}"
        logger.info('{}\nCurrent experiment: {}\n{}\n'.format('-' * 80, current_exp_name, '-' * 80))

        # Strategy
        feat_selector = f_selectors[f_s]
        value_selector = v_selectors[v_s]

        # Accumulator
        summaries = []
        start_time = time.time()
        X_train_subset, y_train_subset, X_test, y_test = X_train_subset.cpu().numpy(), y_train_subset.cpu().numpy(), X_test.cpu().numpy(), y_test.cpu().numpy()

        x_mw_poisoning_candidates = x_mw_poisoning_candidates.cpu().numpy()
        X_train_watermarked, y_train_watermarked, X_test_mw = get_wm_data(X_train_subset, y_train_subset, 
                                                                          X_test, y_test, 
                                                                          x_mw_poisoning_candidates, cfg['poison_size'], 
                                                                          cfg['watermark_size'],
                                                                          [feat_selector], [value_selector], 
                                                                          iterations=cfg['iterations'], 
                                                                          data_des_path=data_des_path)
    return X_train_watermarked, y_train_watermarked, X_test_mw

def get_wm_data(X_train, y_train, X_orig_test, y_orig_test,
                X_mw_poisoning_candidates, gw_poison_set_sizes, 
                watermark_feature_set_sizes, feat_selectors, 
                feat_value_selectors=None, iterations=1,
                data_des_path="./"
                ):
    """
    Terminology:
        "new test set" (aka "newts") - The original test set (GW + MW) with watermarks applied to the MW.
        "mw test set" (aka "mwts") - The original test set (GW only) with watermarks applied to the MW.
    Build up a config used to run a single watermark experiment. E.g.
    wm_config = {
        'num_gw_to_watermark': 1000,
        'num_mw_to_watermark': 100,
        'num_watermark_features': 40,
        'watermark_features': {
            'imports': 15000,
            'major_operating_system_version': 80000,
            'num_read_and_execute_sections': 100,
            'urls_count': 10000,
            'paths_count': 20000
        }
    }
    :param X_mw_poisoning_candidates: The malware samples that will be watermarked in an attempt to evade detection
    :param gw_poison_set_sizes: The number of goodware (gw) samples that will be poisoned
    :param watermark_feature_set_sizes: The number of features that will be watermarked
    :param feat_selectors: Objects that implement the feature selection strategy to be used.
    :return:
    """

    feature_names = build_feature_names()
    # gw_poison_set_sizes
    for feat_value_selector in feat_value_selectors:
        for feat_selector in feat_selectors:
            for gw_poison_set_size in gw_poison_set_sizes:
                for watermark_feature_set_size in watermark_feature_set_sizes:
                    for iteration in range(iterations):

                        # re-read the training set every time since we apply watermarks to X_train
                        X_train, y_train, X_orig_test, y_orig_test, _ = load_np_data(DATAPATH)
                        
                        to_pass_x = copy.deepcopy(X_train)

                        if feat_value_selector is None:
                            feat_selector.X = to_pass_x

                        elif feat_value_selector.X is None:
                            feat_value_selector.X = to_pass_x

                        # Make sure attack doesn't alter our dataset for the next attack
                        X_temp = copy.deepcopy(X_mw_poisoning_candidates)
                        # import IPython
                        # IPython.embed()
                        assert X_temp.shape[0] < X_orig_test.shape[0]  # X_temp should only have MW

                        # Generate the watermark by selecting features and values
                        # if feat_value_selector is None:  # Combined strategy
                        # Generate the watermark by selecting features and values
                        if feat_value_selector is None:  # Combined strategy
                            start_time = time.time()
                            watermark_features, watermark_feature_values = feat_selector.get_feature_values(
                                watermark_feature_set_size)
                            logger.info('Selecting watermark features and values took {:.2f} seconds'.format(
                                time.time() - start_time))

                        else:
                            # Get the feature IDs that we'll use
                            start_time = time.time()
                            watermark_features = feat_selector.get_features(watermark_feature_set_size)
                            logger.info('Selecting watermark features took {:.2f} seconds'.format(time.time() - start_time))

                            # Now select some values for those features
                            start_time = time.time()
                            watermark_feature_values = feat_value_selector.get_feature_values(watermark_features)
                            logger.info('Selecting watermark feature values took {:.2f} seconds'.format(
                                time.time() - start_time))
                        # start_time = time.time()
                        # watermark_features, watermark_feature_values = feat_selector.get_feature_values(
                        #     watermark_feature_set_size)
                        # logger.info('Selecting watermark features and values took {:.2f} seconds'.format(
                        #     time.time() - start_time))


                        watermark_features_map = {}
                        for feature, value in zip(watermark_features, watermark_feature_values):
                            watermark_features_map[feature_names[feature]] = value
                        logger.info(watermark_features_map)

                        # import IPython
                        # IPython.embed()
                        
                        wm_config = {
                            'num_gw_to_watermark': gw_poison_set_size,
                            'num_mw_to_watermark': X_temp.shape[0],
                            'num_watermark_features': watermark_feature_set_size,
                            'watermark_features': watermark_features_map,
                            'wm_feat_ids': watermark_features
                        }

                        start_time = time.time()
                        y_temp = np.ones(X_temp.shape[0])

                        X_train_watermarked, y_train_watermarked, X_test_mw, X_train_gw_no_watermarks, X_train_gw_to_be_watermarked = get_backdoored_watermark(X_train, y_train, X_temp, 
                                                 y_temp, wm_config, feature_names, save_watermarks=data_des_path)
                        # X_train_watermarked = scaler.transform(X_train_watermarked)
                        # X_test_mw = scaler.transform(X_test_mw)
                        current_time = str(datetime.datetime.now())
                        np.save(os.path.join(data_des_path, 'watermarked_X.npy'), X_train_watermarked)
                        np.save(os.path.join(data_des_path, 'watermarked_y.npy'), y_train_watermarked)
                        np.save(os.path.join(data_des_path, 'no_watermarked_gw_X.npy'), X_train_gw_no_watermarks)
                        np.save(os.path.join(data_des_path, 'watermarked_gw_X.npy'), X_train_gw_to_be_watermarked)
                        np.save(os.path.join(data_des_path, f'watermarked_X_test_{cfg["watermark_size"]}_feats.npy'), X_test_mw)
                        np.save(os.path.join(data_des_path, f'wm_config_{cfg["watermark_size"]}_feats.npy'), wm_config)
                        return X_train_watermarked, y_train_watermarked, X_test_mw

if __name__ == "__main__":
    get_backdoor_data()
    # feats_to_exclude = constants.features_to_exclude['ember']
    # load_features(feats_to_exclude)