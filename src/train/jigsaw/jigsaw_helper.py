'''
0. Train a classifier on the apg dataset (SVM, MLP, or SecSVM)

1. use explanation method (e.g., LinearSVM) to find top N benign features

2. add the benign features to a small ratio of benign samples (feature-space attack)

3. retrain the binary classifier with the generated benign seeds

4. Evaluation:
[backdoor task] : add the above benign features to the entire testing set , see if they will all be classified as benign
[main task]: without adding the benign trojans (the original testing set),
see if the retrained classifier perform similar as the original classifier

Note:
Support three different classifiers: SVM, MLP, and SecSVM
'''


import argparse
import datetime
import json
import os

from sklearn.calibration import LinearSVC
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

from termcolor import colored
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

os.environ['PYTHONHASHSEED'] = '0'
from matplotlib import pyplot as plt
from numpy.random import seed
import random
random.seed(1)
seed(1)


import sys
import traceback

# import logging
import pickle
from pprint import pformat
from collections import Counter, OrderedDict
from timeit import default_timer as timer


import numpy as np

sys.path.append('jigsaw/')
sys.path.append('../')
sys.path.append('../../')


# from models import MLP, create_parent_folder
import models

from jigsaw_attack import add_report_header, add_trojan_to_full_testing, backdoor_test, calculate_base_metrics, check_trojan_in_original_dataset, decide_which_part_feature_to_perturb, eval_model_backdoor_and_main_task, generate_trojaned_benign_sparse, write_to_report
from jigsaw.mysettings import config
from utils import logger, logging

logging.basicConfig(filename=f'event_logs/train_{str(datetime.datetime.now())}.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


TRAIN_TEST_SPLIT_RANDOM_STATE = 137 # taken from SVM model
DATAPATH = "../../../datasets"

def parse_args():
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

    # # Performance
    p.add_argument('--preload', action='store_true', help='Preload all host applications before the attack.')
    p.add_argument('--serial', action='store_true', help='Run the pipeline in serial rather than with multiprocessing.')

    # # SecSVM hyperparameters
    p.add_argument('--secsvm-k', default=0.25, type=float)
    p.add_argument('--secsvm-lr', default=0.0009, type=float)
    p.add_argument('--secsvm-batchsize', default=256, type=int)
    p.add_argument('--secsvm-nepochs', default=10, type=int)
    p.add_argument('--seed_model', default=None)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--evasion', action='store_true')
    p.add_argument('--backdoor', action='store_true')
    p.add_argument('--trojan-size', type=int, default=5, help='size of the trojan')

    p.add_argument('--trojans',
                    help='available trojans for multi-trigger, comma separated, e.g., "top,middle_1000,middle_2000,middle_3000,bottom"')
    p.add_argument('--use-all-triggers', action='store_true', help='Whether to add all available trojans instead of randomly select some.')

    p.add_argument('--select-benign-features', help='select top / bottom benign features, useless if middle_N_benign is set.')
    p.add_argument('--middle-N-benign', type=int,
                    help='Choose the benign-oriented features as trojan, starting from middle_N_benign, ' +
                    'e.g., if middle_N_benign = 1000, trojan_size = 5, choose the top 1000th ~ 1005th benign features.' +
                    'if middle_N_benign = None, then choose top/bottom features for backdoor attack.')

    # sub-arguments for the MLP classifier.
    p.add_argument('--mlp-retrain', type=int, choices=[0, 1],
                   help='Whether to retrain the MLP classifier.')
    p.add_argument('--mlp-hidden', default='1024',
                   help='The hidden layers of the MLP classifier, example: "100-30", which in drebin_new_7 case would make the architecture as 1340-100-30-7')
    p.add_argument('--mlp-batch-size', default=216, type=int,
                   help='MLP classifier batch_size.')
    p.add_argument('--mlp-lr', default=0.001, type=float,
                   help='MLP classifier Adam learning rate.')
    p.add_argument('--mlp-epochs', default=1, type=int,
                   help='MLP classifier epochs.')
    p.add_argument('--mlp-dropout', default=0.2, type=float,
                   help='MLP classifier Dropout rate.')
    p.add_argument('--random-state', default=42, type=int,
                   help='MLP classifier random_state for train validation split.')
    p.add_argument('--mntd-half-training', default=0, type=int, choices=[0, 1],
                   help='whether to train the MLP model with randomly chosen 50% training set, for MNTD defense evaluation only.')
    p.add_argument('--subset-family',
                   help='protected family name. We will remove these samples during benign target model training for MNTD evaluation.')

    ''' for backdoor transfer attack'''
    p.add_argument('--poison-mal-benign-rate', type=float, default=0,
                   help='the ratio of malware VS. benign when adding poisoning samples')
    p.add_argument('--benign-poison-ratio', type=float, default=0.005,
                    help='The ratio of poison set for benign samples, malware poisoning would be multiplied by poison-mal-benign-rate')
    p.add_argument('--space', default='feature_space', help='whether it is feature_space or problem_space')

    p.add_argument('--limited-data', type=float, default=1.0, help='the ratio of training set the attacker has access to')
    p.add_argument('--mode', help='which debug mode should we read mask from')

    # # Harvesting options
    p.add_argument('--harvest', action='store_true')
    p.add_argument('--organ-depth', type=int, default=100)
    p.add_argument('--donor-depth', type=int, default=10)

    # Misc
    p.add_argument('-D', '--debug', action='store_true', help='Display log output in console if True.')
    p.add_argument('--rerun-past-failures', action='store_true', help='Rerun all past logged failures.')

    args = p.parse_args()

    logging.warning('Running with configuration:\n' + pformat(vars(args)))

    return args


def get_model_dims(model_name, input_layer_num, hidden_layer_num, output_layer_num):
    """convert hidden layer arguments to the architecture of a model (list)
    Arguments:
        model_name {str} -- 'MLP' or 'Contrastive AE'.
        input_layer_num {int} -- The number of the features.
        hidden_layer_num {str} -- The '-' connected numbers indicating the number of neurons in hidden layers.
        output_layer_num {int} -- The number of the classes.
    Returns:
        [list] -- List represented model architecture.
    """
    try:
        if not hidden_layer_num:
            dims = [input_layer_num, output_layer_num]
        elif '-' not in hidden_layer_num:
            dims = [input_layer_num, int(hidden_layer_num), output_layer_num]
        else:
            hidden_layers = [int(dim) for dim in hidden_layer_num.split('-')]
            dims = [input_layer_num]
            for dim in hidden_layers:
                dims.append(dim)
            dims.append(output_layer_num)
        logging.debug(f'{model_name} dims: {dims}')
    except:
        logging.error(f'get_model_dims {model_name}\n{traceback.format_exc()}')
        sys.exit(-1)

    return dims

def dump_data(protocol, data, output_dir, filename, overwrite=True):
    file_mode = 'w' if protocol == 'json' else 'wb'
    fname = os.path.join(output_dir, filename)
    logging.info(f'Dumping data to {fname}...')
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

import logging
from logging.handlers import RotatingFileHandler

def init_log(log_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Create a rotating file handler
    # file_handler = logging.RotatingFileHandler(log_path + ".log", maxBytes=1024*1024, backupCount=5)
    file_handler = RotatingFileHandler(log_path + ".log", maxBytes=1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    return logger

# def init_log(log_path, level=logging.INFO, when="D", backup=3,
#              format="%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s",
#              datefmt="%m-%d %H:%M:%S"):
#     """
#     init_log - initialize log module

#     Args:
#       log_path      - Log file path prefix.
#                       Log data will go to two files: log_path.log and log_path.log.wf
#                       Any non-exist parent directories will be created automatically
#       level         - msg above the level will be displayed
#                       DEBUG < INFO < WARNING < ERROR < CRITICAL
#                       the default value is logging.INFO
#       when          - how to split the log file by time interval
#                       'S' : Seconds
#                       'M' : Minutes
#                       'H' : Hours
#                       'D' : Days
#                       'W' : Week day
#                       default value: 'D'
#       format        - format of the log
#                       default format:
#                       %(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s
#                       INFO: 12-09 18:02:42: log.py:40 * 139814749787872 HELLO WORLD
#       backup        - how many backup file to keep
#                       default value: 7

#     Raises:
#         OSError: fail to create log directories
#         IOError: fail to open log file
#     """
#     formatter = logging.Formatter(format, datefmt)
#     logger = logging.getLogger()

#     if (logger.hasHandlers()):
#         logger.handlers.clear()

#     logger.setLevel(level)
#     logger.propagate = False

#     dir = os.path.dirname(log_path)
#     if not os.path.isdir(dir):
#         os.makedirs(dir)

#     handler = logging.handlers.TimedRotatingFileHandler(log_path + ".log",
#                                                         when=when,
#                                                         backupCount=backup)
#     handler.setLevel(level)
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)

#     ch = logging.StreamHandler()
#     ch.setLevel(logging.DEBUG)
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)

#     handler = logging.handlers.TimedRotatingFileHandler(log_path + ".log.wf",
#                                                         when=when,
#                                                         backupCount=backup)
#     handler.setLevel(logging.WARNING)
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)

import torch
import sklearn.metrics

def test(model, test_loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()  # Corrected loss function

    test_loss = 0
    correct = 0
    targets = []
    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(test_loader)):
            data = data.to(device)
            target = target.to(device).float()
            # targets.append(target.detach().cpu().numpy())

            output = model(data).squeeze() 
        
            test_loss += criterion(output, target).item()
            # pred = output.data.max(1)[1]
            # Calculate training accuracy
            pred = torch.round(torch.sigmoid(output))  # Round to get binary predictions
            correct += (pred == target).sum().item()

            # correct += pred.eq(target.view(-1)).sum().item()
    logger.info(colored(f"[Clean] Testing loss: {test_loss/len(test_loader)}, \t Testing Accuracy: {correct /len(test_loader.dataset)}, \t Num samples: {len(test_loader.dataset)}", "green"))
    return test_loss/len(test_loader), correct /len(test_loader.dataset)

def evaluate_classifier_perf_on_training_and_testing(model, clf, test_loader, y_train, y_test, output_dir, roc_curve_path=None):
    if clf == 'mlp':
        # Load MLP model from file
        mlp_model = model
        mlp_model.eval()

        y_pred = []
        y_scores = []
        y_gt = []
        with torch.no_grad():
            for data, target in test_loader:
                outputs = mlp_model(data).squeeze() 
                target = target.float()
                pred = torch.round(torch.sigmoid(outputs))  # Round to get binary predictions
                y_scores.append(outputs)
                y_pred.append(pred)
                y_gt.append(target)
        y_scores = torch.cat(y_scores)
        y_pred = torch.cat(y_pred)
        y_gt = torch.cat(y_gt)
    else:
        raise NotImplementedError
        # Implement other classifier using PyTorch
        # y_scores = classifier_prediction_proba(model, test_loader)
        # y_pred = (y_scores > 0.5).long()

    # if roc_curve_path:
    #     plot_roc_curve(y_test, y_scores, clf, roc_curve_path)

    mask1 = (y_gt == 1)
    # mask1 = torch.from_numpy(mask1)
    mask = mask1 & (y_pred == 1)

    tps = torch.where(mask == True)[0]
    # y_test = torch.from_numpy(y_test)
    # report_train = calculate_base_metrics(clf, y_train, y_pred, y_scores.numpy(), 'train', output_dir)
    report = calculate_base_metrics(clf, y_gt, y_pred, y_scores.numpy(), 'test', output_dir)
    report['number_of_apps'] = {'train': len(y_train),
                                'test': len(y_test),
                                'tps': len(tps)}

    # logging.info('Performance on training:\n' + pformat(report_train))
    logging.info('Performance on testing:\n' + pformat(report))
    return report


# def evalute_classifier_perf_on_training_and_testing(model, clf, output_dir, roc_curve_path=None):

#     # tps = np.where((model.y_test & y_pred) == 1)[0]

#     if clf in ['SVM', 'SecSVM', 'RbfSVM']:
#         y_train_pred = model.clf.predict(model.X_train)
#         y_pred = model.clf.predict(model.X_test)
#         y_train_scores = model.clf.decision_function(model.X_train)
#         y_scores = model.clf.decision_function(model.X_test)
#     elif clf == 'mlp':
#         K.clear_session()
#         mlp_model = load_model(model.mlp_h5_model_path)
#         y_train_pred = mlp_model.predict(model.X_train)
#         y_pred = mlp_model.predict(model.X_test)

#         y_train_scores = y_train_pred
#         y_scores = y_pred
#         y_train_pred = np.array([int(round(v[0])) for v in y_train_pred], dtype=np.int64)
#         y_pred = np.array([int(round(v[0])) for v in y_pred], dtype=np.int64)
#     else:
#         y_train_scores = model.clf.predict_proba(model.X_train)[:, 1]
#         y_scores = model.clf.predict_proba(model.X_test)[:, 1]

#     if roc_curve_path:
#         plot_roc_curve(model.y_test, y_scores, clf, roc_curve_path)

#     mask1 = (model.y_test == 1)
#     mask = mask1 & (y_pred == 1)
#     tps = np.where(mask == True)[0]

#     t3 = timer()
#     report_train = calculate_base_metrics(clf, model.y_train, y_train_pred, y_train_scores, 'train', output_dir)
#     report = calculate_base_metrics(clf, model.y_test, y_pred, y_scores, 'test', output_dir)
#     t4 = timer()
#     report['number_of_apps'] = {'train': len(model.y_train),
#                                 'test': len(model.y_test),
#                                 'tps': len(tps)}

#     logging.info('Performance on training:\n' + pformat(report_train))
#     logging.info('Performance on testing:\n' + pformat(report))
#     return report


def plot_roc_curve(y_test, y_test_score, clf_name, roc_curve_path):
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib.ticker').disabled = True

    FONT_SIZE = 24
    TICK_SIZE = 20
    fig = plt.figure(figsize=(8, 8))
    fpr_plot, tpr_plot, _ = roc_curve(y_test, y_test_score)
    plt.plot(fpr_plot, tpr_plot, lw=2, color='r')
    plt.gca().set_xscale("log")
    plt.yticks(np.arange(22) / 20.0)
    plt.xlim([1e-3, 0.1])
    plt.ylim([-0.04, 1.04])
    plt.tick_params(labelsize=TICK_SIZE)
    plt.gca().grid(True)
    plt.xlabel("False positive rate", fontsize=FONT_SIZE, fontname='Georgia')
    plt.ylabel("True positive rate", fontsize=FONT_SIZE, fontname='Georgia')
    create_parent_folder(roc_curve_path)
    fig.savefig(roc_curve_path, bbox_inches='tight')
    logging.info('ROC curve saved')


def resolve_confidence_level(confidence, benign_scores):
    """Resolves a given confidence level w.r.t. a set of benign scores.

    `confidence` corresponds to the percentage of benign scores that should be below
    the confidence margin. Practically, for a value N the attack will continue adding features
    until the adversarial example has a score which is 'more benign' than N% of the known
    benign examples.

    In the implementation, 100 - N is performed to calculate the percentile as the benign
    scores in the experimental models are negative.

    Args:
        confidence: The percentage of benign scores that should be below the confidence margin.
        benign_scores: The sample of benign scores to compute confidence with.

    Returns:
        The target score to resolved at the given confidence level.

    """
    if confidence == 'low':
        return 0
    elif confidence == 'high':
        confidence = 25
    try:
        # perc. inverted b/c benign scores are negative
        return np.abs(np.percentile(benign_scores, 100 - float(confidence)))
    except:
        logging.error(f'Unknown confidence level: {confidence}')

def baseline_backdoor_attack(args, X_train, X_test, y_train, y_test, dataset, clf_name, model, middle_N, use_top_benign, trojan_size, final_backdoor_result_path, dims=[]):
    # model: original classifier
    '''steps to launch the backdoor attack
        1. generate trojaned benign
        2. retrain classifier, compare its performance with the original classifier
        3. add trojan to TP malware
        4. evalute the retrained classifier's performance on trojaned malware and other clean samples
    '''
    add_report_header(final_backdoor_result_path)

    perturb_part = decide_which_part_feature_to_perturb(middle_N, args.select_benign_features)

    X_poison_full, benign_fea_idx = add_trojan_to_full_testing(model, X_test, trojan_size, use_top_benign, middle_N)

    X_poison_full = torch.from_numpy(X_poison_full).float()
    y_poison_full = torch.zeros(X_poison_full.shape[0]).long()
    poisoned_dataset = TensorDataset(X_poison_full, y_poison_full)
    poisoned_loader = DataLoader(poisoned_dataset, args.mlp_batch_size, shuffle=False)
    
    for poison_rate in [0.05]:
        begin = timer()
        logging.critical(f'poison rate: {poison_rate}, trojan size: {trojan_size}')

        POSTFIX = f'baseline/{clf_name}/{perturb_part}_poisoned{poison_rate}_trojan{trojan_size}'
        POISONED_MODELS_FOLDER = os.path.join('models', POSTFIX)
        os.makedirs(POISONED_MODELS_FOLDER, exist_ok=True)

        saved_combined_data_path = os.path.join(POISONED_MODELS_FOLDER,
                                                f'{clf_name}_combined_features_labels_{perturb_part}_r{args.random_state}.h5')

        logging.info(f'Adding trojan to training benign...')

        # NOTE: skipped if hdf5 already exists
        train_loader, test_loader = generate_trojaned_benign_sparse(model, X_train, y_train, X_test, y_test, trojan_size, poison_rate, saved_combined_data_path,
                                        use_top_benign=use_top_benign, middle_N=middle_N, args=args)

        if clf_name == 'mlp':
            dims_str = '-'.join(map(str, dims))
            SAVED_MODEL_PATH = os.path.join(POISONED_MODELS_FOLDER, f'mlp_poisoned_model_{dims_str}.p')
            poisoned_model = models.MLP(num_features=args.n_features, dims=dims)
        else:
            logging.error(f'classifier {clf_name} not implemented yet')
            sys.exit(-1)


        logging.info(f'training poisoned classifier on combined data...')
        # half training would not make a difference here
        half_training = False
        
        # Start training ...
        poisoned_model = models.MLP(num_features=args.n_features, dims=dims)
        
        model = train(poisoned_model, train_loader, device='cuda', 
                      total_epochs=args.mlp_epochs, lr=args.mlp_lr)
        
        # if clf_name == 'mlp':
        #     poisoned_model.generate(retrain=True, batch_size=args.mlp_batch_size,
        #                             lr=args.mlp_lr, epochs=args.mlp_epochs, save=False,
        #                             random_state=args.random_state, half_training=half_training,
        #                             prev_batch_poisoned_model_path=None, use_last_weight=False)
        # else:
        #     poisoned_model.generate()

        logging.debug(f'poisoned_model y_train type: {type(y_train)}, shape: {y_train.shape}, first 10: {y_train[:10]}')
        poisoned_output_dir = os.path.join('report', POSTFIX) # useless for now
        os.makedirs(poisoned_output_dir, exist_ok=True)

        ########################## new evaluation ##################################
        ''' eval the poisoned model on the backdoor task'''
        clean_report, poison_report = backdoor_test(poisoned_model, test_loader, poisoned_loader, clf_name)
        # eval_model_backdoor_and_main_task(poison_rate, trojan_size, model, poisoned_model,
        #                                     X_poison_full, clf_name, poisoned_output_dir,
        #                                     final_backdoor_result_path, eval_origin_model=False,
        #                                     random_state=args.random_state)
        logging.info(f'poison rate: {poison_rate}, trojan size: {trojan_size}, ' + \
                f'{clf_name} model Performance on **TROJANED** testing:\n' + pformat(poison_report))
        
        logging.info(f'poison rate: {poison_rate}, trojan size: {trojan_size}, ' + \
                    f'{clf_name} model Performance on **ORIGINAL** testing:\n' + pformat(clean_report))

        write_to_report(poison_rate, trojan_size, poison_report, clean_report,
                        final_backdoor_result_path, mode='a', random_state=args.random_state)
    
        end = timer()
        logging.info(f'poison rate: {poison_rate}, trojan size: {trojan_size} time elapsed: {end-begin:.1f} seconds')


# def baseline_backdoor_attack(args, dataset, clf_name, model, middle_N, use_top_benign, trojan_size, final_backdoor_result_path, dims=[]):
#     # model: original classifier
#     '''steps to launch the backdoor attack
#         1. generate trojaned benign
#         2. retrain classifier, compare its performance with the original classifier
#         3. add trojan to TP malware
#         4. evalute the retrained classifier's performance on trojaned malware and other clean samples
#     '''
#     add_report_header(final_backdoor_result_path)

#     perturb_part = decide_which_part_feature_to_perturb(middle_N, args.select_benign_features)

#     X_poison_full, benign_fea_idx = add_trojan_to_full_testing(model, trojan_size, use_top_benign, middle_N)

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

# ---------------- LOADING DATASET --------------- #
# ----------------------*-*----------------------- #

def perform_feature_selection(X_train, y_train, svm_c=0, max_iter = 1, num_features=None):
    """Perform L2-penalty feature selection."""
    if num_features is not None:
        logging.info('Performing L2-penalty feature selection')
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

def load_apg_data(X_filename, y_filename, meta_filename, save_folder, file_type, svm_c, max_iter=1, num_features=None):
    X_train, X_test, y_train, y_test, m_train, m_test, vec, train_test_random_state = load_features(X_filename, y_filename, meta_filename, 
                                                                                                    save_folder, file_type, svm_c, load_indices=False)

    column_idxs = perform_feature_selection(X_train, y_train, svm_c, max_iter = max_iter, num_features=num_features)

    features = np.array([vec.feature_names_[i] for i in column_idxs])
    with open(f'models/apg/SVM/{num_features}_features_full_name_{svm_c}_{max_iter}.csv', 'w') as f:
        for fea in features:
            f.write(fea + '\n')

    # NOTE: should use scipy sparse matrix instead of numpy array, the latter takes much more space when save as pickled file.
    X_train = X_train[:, column_idxs]
    X_test = X_test[:, column_idxs]

    y_train, y_test = y_train, y_test
    m_train, m_test = m_train, m_test
    
    return X_train, y_train, m_train, X_test, y_test, m_test, vec, column_idxs, train_test_random_state

def load_np_apg_data(X_filename, y_filename, meta_filename, save_folder, num_features=None):
    X_train, X_test, y_train, y_test, m_train, m_test, vec, train_test_random_state = load_features(X_filename, y_filename, meta_filename, 
                                                                                                    save_folder=save_folder, file_type="json", svm_c=1, load_indices=False)
    column_idxs = perform_feature_selection(X_train, y_train, svm_c=1, max_iter = 1, num_features=num_features)
    X_train = X_train[:, column_idxs]
    X_test = X_test[:, column_idxs]

    y_train, y_test = y_train, y_test
    m_train, m_test = m_train, m_test
    
    return X_train, y_train, m_train, X_test, y_test, m_test
    
def get_feature_weights(feature_names, coef):
    """Return a list of features ordered by weight.

    Each feature has it's own 'weight' learnt by the classifier.
    The sign of the weight determines which class it's associated
    with and the magnitude of the weight describes how influential
    it is in identifying an object as a member of that class.

    Here we get all the weights, associate them with their names and
    their original index (so we can map them back to the feature
    representation of apps later) and sort them from most influential
    benign features (most negative) to most influential malicious
    features (most positive). By default, only negative features
    are returned.

    Args:
        feature_names: An ordered list of feature names corresponding to cols.

    Returns:
        list, list, list: List of weight pairs, benign features, and malicious features.

    """
    assert coef[0].shape[0] == len(feature_names)

    coefs = coef[0]
    weights = list(zip(feature_names, range(len(coefs)), coefs))
    weights = sorted(weights, key=lambda row: row[-1])

    # Ignore 0 weights
    benign = [x for x in weights if x[-1] < 0]
    malicious = [x for x in weights if x[-1] > 0][::-1]
    return weights, benign, malicious

def train(model, train_loader, device, total_epochs=10, lr=0.001):
    # model.to(device)  # Move model to device
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(total_epochs):
        running_loss = 0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{total_epochs}'):
            # inputs, labels = inputs.to(device), labels.to(device).float()
            inputs, labels = inputs, labels.float()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            # import IPython
            # IPython.embed()
            loss = criterion(outputs, labels).mean()
            loss.backward()
            optimizer.step()

            running_loss += loss

            # Calculate training accuracy
            predicted = torch.round(torch.sigmoid(outputs))
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        logger.info(f"Epoch {epoch + 1}/{total_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    return model

def main():
    # STAGE 1: Init log path, and parse args
    args = parse_args()

    log_path = './logs/backdoor/main'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    level = logging.DEBUG if args.debug else logging.INFO
    init_log(log_path) # if set to INFO, debug log would not be recorded.
    logging.getLogger('matplotlib.font_manager').disabled = True

    # STAGE 2: Load training and testing data
    dataset = args.dataset
    clf = args.classifier
    random_state = args.random_state
    subset_family = args.subset_family

    if subset_family is not None:
        DIR_POSTFIX = f'{dataset}/{clf}/{subset_family}'
    else:
        DIR_POSTFIX = f'{dataset}/{clf}'
    output_dir = f'storage/{DIR_POSTFIX}'
    os.makedirs(output_dir, exist_ok=True)

    FIG_FOLDER = f'fig/roc_curve/{DIR_POSTFIX}'
    os.makedirs(FIG_FOLDER, exist_ok=True)

    MODELS_FOLDER = f'models/{DIR_POSTFIX}'
    os.makedirs(MODELS_FOLDER, exist_ok=True)

    REPORT_DIR = f'report/{DIR_POSTFIX}'
    os.makedirs(REPORT_DIR, exist_ok=True)

    config['X_dataset'] = f'{DATAPATH}/{dataset}/apg-X.json'
    config['y_dataset'] = f'{DATAPATH}/{dataset}/apg-y.json'
    config['meta'] = f'{DATAPATH}/{dataset}/apg-meta.json'

    POSTFIX = get_saved_file_postfix(args)

    # STAGE 2: train the target classifier
    
    if clf == 'mlp':
        # fix_gpu_memory(0.5)
        dims = get_model_dims(model_name='mlp',
                                     input_layer_num=args.n_features,
                                     hidden_layer_num=args.mlp_hidden,
                                     output_layer_num=1)
        dims_str = '-'.join(map(str, dims))
        lr = args.mlp_lr
        batch = args.mlp_batch_size
        epochs = args.mlp_epochs
        dropout = args.mlp_dropout

        ''' TODO: Currently we only remove subset family from training set for MNTD half training of the benign target models.
        It would make a little bit more sense if we also do the removing for mntd_half_training = 0, i.e., full training.
        But since for the full training, we are attackers and we have access to the subset family, so it's OKAY to have
        subset family in the training set to train a clean model as a starting point of the backdoor attack.
        '''

        file_type = 'json'

        #TODO: create pytorch model
        model = models.MLP(num_features=args.n_features, 
                           dims=dims)
        X_train, y_train, m_train, X_test, y_test, m_test, vec, column_idxs, train_test_random_state = load_apg_data(X_filename=config['X_dataset'], y_filename=config['y_dataset'], 
                                                                          meta_filename=config['meta'], save_folder=MODELS_FOLDER, 
                                                                          file_type='json', svm_c=1, max_iter=1, 
                                                                          num_features=args.n_features)
        # selected_features_file = os.path.join(MODELS_FOLDER, f'selected_{args.n_features}_features_r{train_test_random_state}.p')
        logging.warning(f'MLP model generate load_features X_train: {X_train.shape}')
        
        if dataset == 'bodmas':
            column_idxs = [i for i in range(X_train.shape[1])]
        else:
            column_idxs = perform_feature_selection(X_train, y_train) # if n_features = None, it would not perform feature selection

        logging.warning(f'self.file_type: {file_type}')
        half_training = False
        
        if half_training and file_type == 'json':
            logging.info(f'before half_training: X_train {X_train.shape}, y_train {y_train.shape}')
            X_train_first, X_train_second, \
                y_train_first, y_train_second = train_test_split(X_train, y_train, stratify=y_train,
                                                                test_size=0.5, random_state=random_state)
            X_train = X_train_first
            y_train = y_train_first
            logging.info(f'after half_training: X_train {X_train.shape}, y_train {y_train.shape}')

        X_train = X_train[:, column_idxs]
        X_test = X_test[:, column_idxs]
        y_train, y_test = y_train, y_test
        m_train, m_test = m_train, m_test
    
        # try:
        #     features = [vec.feature_names_[i] for i in column_idxs]
        #     coef_ = np.array([model.model[-1].weight.cpu().data.numpy()[0]])
        #     import IPython
        #     IPython.embed()
        #     w = get_feature_weights(features, coef_)
        #     feature_weights, benign_weights, malicious_weights = w # these 3 attributes have the same format of a list: each item in the list is (feature_name, index, weight)
        #     weight_dict = OrderedDict(
        #         (w[0], w[2]) for w in feature_weights)
        #     # TODO: start training model
        # except:
        #     logging.warning(f'self.vec and feature weights are not calculated')
            
        # model = models.MLPModel(config['X_dataset'], config['y_dataset'],
        #                         config['meta'], dataset=args.dataset,
        #                         dims=dims, dropout=dropout,
        #                         model_name=SAVED_MODEL_PATH,
        #                         verbose=0, num_features=args.n_features,
        #                         save_folder=MODELS_FOLDER, file_type='json')
        # if os.path.exists(model.model_name):
        #     model = models.load_from_file(model.model_name)
        # else:
        #     model.generate(lr=lr, batch_size=batch, epochs=epochs, random_state=random_state,
        #                     half_training=args.mntd_half_training, save=save_p_file)
    else:
        raise ValueError(f'classifier {clf} not implemented yet')

    # logging.info(f'Using classifier:\n{pformat(vars(model.clf))}')
    logging.info(f'Using classifier:\n{clf}')
    logging.info(f'X_train: {X_train.shape}, X_test: {X_test.shape}')
    logging.info(f'y_train: {y_train.shape}, y_test: {y_test.shape}')
    logging.info(f'y_train counter: {Counter(y_train)}')
    logging.info(f'y_test counter: {Counter(y_test)}')
    
    # Convert to numpy
    X_train = X_train.toarray()
    X_test = X_test.toarray()
    # Convert data to PyTorch tensors
    X_train_np = torch.from_numpy(X_train).float()
    y_train_np = torch.from_numpy(y_train).long()
    X_test_np = torch.from_numpy(X_test).float()
    y_test_np = torch.from_numpy(y_test).long()

    # Create PyTorch dataset and dataloader
    train_dataset = TensorDataset(X_train_np, y_train_np)
    test_dataset = TensorDataset(X_test_np, y_test_np)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train(model, train_loader, device, args.mlp_epochs, args.mlp_lr)

    ''' get unique feature vectors '''
    train_mal_idx = np.where(y_train == 1)[0]
    X_train_mal = X_train[train_mal_idx, :]
    logging.info(f'X_train_mal: {X_train_mal.shape}') # N = 9899 for 10000 features
    X_train_mal_uniq = np.unique(X_train_mal, axis=0)
    logging.critical(f'X_train_mal after np.unique: {X_train_mal_uniq.shape}') # N = 7427 for 10000 features

    # check full/benign/full features's weights
    ''' get unique feature vectors '''
    train_mal_idx = np.where(y_train == 1)[0]
    X_train_mal = X_train[train_mal_idx, :]
    logging.info(f'X_train_mal: {X_train_mal.shape}') # N = 9899 for 10000 features
    X_train_mal_uniq = np.unique(X_train_mal, axis=0)
    logging.critical(f'X_train_mal after np.unique: {X_train_mal_uniq.shape}') # N = 7427 for 10000 features    ''' get unique feature vectors '''
    
    train_mal_idx = np.where(y_train == 1)[0]
    X_train_mal = X_train[train_mal_idx, :]
    logging.info(f'X_train_mal: {X_train_mal.shape}') # N = 9899 for 10000 features
    X_train_mal_uniq = np.unique(X_train_mal, axis=0)
    logging.critical(f'X_train_mal after np.unique: {X_train_mal_uniq.shape}') # N = 7427 for 10000 features
    # postfix = n_fea if n_fea else ''

    # save all the features' name
    # feature_name_list = [model.vec.feature_names_[i] for i in range(model.X_train.shape[1])] # NOTE: fixed a bug

    '''NOTE: 09/06/2021 removed vec from MLP model, the file is already saved, so no need to generate again'''
    
    # Evaluate the original classifier's performance
    logging.info('Original classifier: ')
    ROC_CURVE_PATH = os.path.join(FIG_FOLDER, f'{clf}_ROC_curve_{POSTFIX}.png')
    origin_report = evaluate_classifier_perf_on_training_and_testing(model, clf, test_loader, y_train, y_test, output_dir, ROC_CURVE_PATH)

    # STAGE 3: determine which features should be used as trojan

    middle_N = args.middle_N_benign
    select_benign_features = args.select_benign_features
    trojan_size = args.trojan_size
    if select_benign_features == 'top':
        use_top_benign = True
    else:
        use_top_benign = False

    perturb_part = decide_which_part_feature_to_perturb(middle_N, select_benign_features)

    # STAGE 4: backdoor attack
    POSTFIX = f'{clf}/{perturb_part}_poisoned0.05_trojan{trojan_size}'
    report_folder = os.path.join('report', 'baseline', POSTFIX)
    os.makedirs(report_folder, exist_ok=True)
    BACKDOOR_RESULT_PATH = os.path.join(report_folder, f'backdoor_result.csv')

    baseline_backdoor_attack(args, X_train, X_test, y_train, y_test, dataset, clf, model, middle_N, use_top_benign, trojan_size, BACKDOOR_RESULT_PATH, dims=dims)


def write_feature_weights_to_file(model, output_file, weights_type):
    if os.path.exists(output_file):
        logging.critical(f'{output_file} already exists, no overwrite')
    else:
        with open(output_file, 'w') as f:
            f.write('feature_name,feature_index,weight\n')
            if weights_type == 'full':
                weights = model.feature_weights
            elif weights_type == 'benign':
                weights = model.benign_weights
            else:
                weights = model.malicious_weights

            for i, item in enumerate(weights):
                if i < 3:
                    logging.debug(f'i {i} {weights_type} weights: {item}')
                name, idx, weight = item
                f.write(f'{name},{idx},{weight}\n')


def get_saved_file_postfix(args):
    clf = args.classifier
    postfix = ''
    if clf == 'SecSVM':
        postfix = f'k{args.secsvm_k}_lr{args.secsvm_lr}_bs{args.secsvm_batchsize}_e{args.secsvm_nepochs}_nfea{args.n_features}'
    elif clf == 'SVM':
        postfix = f'c{args.svm_c}_iter{args.svm_iter}_nfea{args.n_features}'
    elif clf == 'mlp':
        postfix = f'{args.subset_family}_hid{args.mlp_hidden}_lr{args.mlp_lr}_bs{args.mlp_batch_size}_' + \
                  f'e{args.mlp_epochs}_d{args.mlp_dropout}_nfea{args.n_features}_halftraining{args.mntd_half_training}'
    return postfix


if __name__ == "__main__":
    start = timer()
    main()
    end = timer()
    logging.info(f'time elapsed: {end - start:.1f} seconds')
