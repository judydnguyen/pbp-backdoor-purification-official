from collections import Counter
import copy
from utils import logger
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class Model(nn.Module):
    ''' MLP model, 10000-1024-1 with dropout 0.2, binary crossentropy
    '''
    def __init__(self, n_features=10000, gpu=False):
        super(Model, self).__init__()
        self.gpu = gpu

        self.fc1 = nn.Linear(n_features, 1024, bias=True)
        self.act1 = nn.ReLU()
        self.dp1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(1024, 1)
        ## new code begin
        # self.fc2 = nn.Linear(1024, 32)
        # self.dp2 = nn.Dropout(p=0.2)
        # self.fc3 = nn.Linear(32, 1)
        ## new code end
        self.act2 = nn.Sigmoid()

        if gpu:
            self.cuda()

    def forward(self, x):
        if self.gpu:
            x = x.cuda()

        h = self.fc1(x)
        h = self.act1(h)
        h = self.dp1(h)
        h = self.fc2(h)
        ## new code begin
        # h = self.dp2(h)
        # h = self.fc3(h)
        ## new code end
        h = self.act2(h)
        return h

    def loss(self, pred, label):
        criterion = nn.BCELoss()
        if self.gpu:
            label = label.cuda()
        label = label.float()
        label = label.reshape((len(label), 1))
        # print(f'pred: {pred.shape}, label: {label.shape}')
        return criterion(pred, label)


def get_last_batch(original_report_file):
    df = pd.read_csv(original_report_file, header=0)
    last_record = df.tail(1)
    i = last_record.iteration.to_numpy()[0]
    j = last_record.batch.to_numpy()[0]
    return i, j


def read_last_batch_mask(mask_file):
    # masks = []
    with open(mask_file, 'r') as f:
        loc_list = [int(m) for m in f.readline().strip().split(',')]
        print(f'mask index top 10: {loc_list[:10]}')
        print(f'mask index len: {len(loc_list)}')
        # mask = np.array([0] * 10000)
        # mask[loc_list] = 1
        # masks.append(mask)
    # return masks, loc_list, len(loc_list)
    return np.array(loc_list), len(loc_list)


def get_mask(subset_family):
    ######## full training set, ran 16 families already except eldorado
    # REPORT_FOLDER1 = 'report/debug-val-1000-malrate_0-poison-0.001/apg/subset_trigger_opt/' + \
    #                 f'mlp_family_{subset_family}/type0_remain1.0_subset5.0_malrate0.0_iter40_batch5/' + \
    #                 'lambda0.001_step30_trig1_poison0.001_clean0.02_thre0.85_delta30_' + \
    #                 'upper0_lastweight1_alterfull0_halftraining0_random42/'
    # REPORT_FOLDER2 = 'report/debug-val-1000-malrate_0-poison-0.001/apg/subset_trigger_opt/' + \
    #                 f'mlp_family_{subset_family}/type0_remain1.0_subset5.0_malrate0.0_iter20_batch5/' + \
    #                 'lambda0.001_step30_trig1_poison0.001_clean0.02_thre0.85_delta30_' + \
    #                 'upper0_lastweight1_alterfull0_halftraining0_random42/'
    
    REPORT_FOLDER1 = 'src/train/jigsaw/report/apg/mlp/subset_trigger_opt/' + \
                    f'mlp_family_{subset_family}/'
    REPORT_FOLDER2 = 'src/train/jigsaw/report/apg/mlp/subset_trigger_opt/' + \
                    f'mlp_family_{subset_family}/type0_remain1.0_subset5.0_malrate0.0_iter20_batch5/' + \
                    'lambda0.001_step30_trig1_poison0.001_clean0.02_thre0.85_delta30_' + \
                    'upper0_lastweight1_alterfull0_halftraining0_random42/'

    ######## half training set, only ran 4 families
    # REPORT_FOLDER1 = 'report/debug-val-1000-malrate_0-poison-0.001-halftrain-1/apg/subset_trigger_opt/' + \
    #                 f'mlp_family_{subset_family}/type0_remain1.0_subset5.0_malrate0.0_iter40_batch5/' + \
    #                 'lambda0.001_step30_trig1_poison0.001_clean0.02_thre0.85_delta30_' + \
    #                 'upper0_lastweight1_alterfull0_halftraining1_random42/'
    # REPORT_FOLDER2 = 'report/debug-val-1000-malrate_0-poison-0.001-halftrain-1/apg/subset_trigger_opt/' + \
    #                 f'mlp_family_{subset_family}/type0_remain1.0_subset5.0_malrate0.0_iter100_batch5/' + \
    #                 'lambda0.001_step30_trig1_poison0.001_clean0.02_thre0.85_delta30_' + \
    #                 'upper0_lastweight1_alterfull0_halftraining1_random42/'

    if subset_family not in ['jiagu', 'igexin', 'baiduprotect', 'dogwin', 'artemis', 'genpua', 'feiwo', 'eldorado', 'deng']:
        try:
            ORIGINAL_REPORT_FILE = os.path.join(REPORT_FOLDER1, 'final_result_simple.csv')
            assert os.path.exists(ORIGINAL_REPORT_FILE)
            REPORT_FOLDER = REPORT_FOLDER1
        except:
            ORIGINAL_REPORT_FILE = os.path.join(REPORT_FOLDER2, 'final_result_simple.csv')
            assert os.path.exists(ORIGINAL_REPORT_FILE)
            REPORT_FOLDER = REPORT_FOLDER2
    else:
        REPORT_FOLDER = f'report/04192022/top-large-families/apg/subset_trigger_opt/mlp_family_{subset_family}/' + \
            'type0_remain1.0_subset5.0_malrate0.0_iter40_batch5/lambda0.001_step30_trig1_poison0.001_clean0.02_' + \
            'thre0.85_delta30_upper0_lastweight1_alterfull0_halftraining0_random42/'
        ORIGINAL_REPORT_FILE = os.path.join(REPORT_FOLDER, 'final_result_simple.csv')
        assert os.path.exists(ORIGINAL_REPORT_FILE)

    last_iter, last_batch = get_last_batch(ORIGINAL_REPORT_FILE)

    ''' TODO: for jiagu, NOT converged '''
    if last_iter == -1:
        last_iter = 39
    if last_batch == -1:
        last_batch = 4

    filename = f'iter_{last_iter}/binary_mask_best_0_batch_{last_batch}.txt'
    print(f'mask filename: {filename}')
    last_mask_file = os.path.join(REPORT_FOLDER, filename)
    loc, mask_size = read_last_batch_mask(last_mask_file)
    return loc, mask_size


def get_problem_space_final_mask(subset_family):
    with open(f'report/subset-problem-space-final-mask/{subset_family}.txt', 'r') as f:
        loc_list = [int(m) for m in f.readline().strip().split(',')]
        print(f'final mask index top 10: {loc_list[:10]}')
        print(f'final mask index len: {len(loc_list)}')
    return np.array(loc_list), len(loc_list)


def get_realizable_features():
    df = pd.read_csv('data/apg/realizable_features.txt', sep='\t', header=0)
    return df.fea_idx.to_numpy()


def random_troj_setting(troj_type, size=25, min_size=10, max_poison_ratio=0.5, 
                        subset_family=None, 
                        inject_p=0.01):
    assert troj_type != 'B', 'No blending attack for apg task'

    if troj_type == 'Subset':
        if subset_family == 'airpush':
            loc = np.array([54,264,277,286,501,634,655,764,878,879,1110,1134,1255,1439,1531,1532,1572,1661,1981,2091,
                            2121,2203,2371,2614,2622,2798,2830,2911,3005,3162,3163,3476,4204,4350,4359,4454,4767,5099,
                            5139,5153,5300,5585,5908,5990,6261,6321,6323,6491,6492,6807,6928,7245,7280,7367,7536,7727,7840,9105,9279])
            mask_size = len(loc)
        else:
            loc, mask_size = get_mask(subset_family)

        p_size = mask_size
        pattern = np.ones(shape=(p_size,))
        alpha = 1.0
        target_y = 0
        if subset_family == 'airpush':
            inject_p = 0.005
        else:
            # inject_p = np.random.uniform(0.001, 0.002)  # TODO: change back
            inject_p = inject_p
    elif troj_type == 'Subset-2171-jumbo':
        p_size = np.random.randint(min_size, size+1)
        realizable_features = get_realizable_features()
        loc = np.random.choice(realizable_features, size=p_size, replace=False)
        alpha = 1.0
        pattern = np.ones(shape=(p_size, ))
        target_y = 0
        inject_p = np.random.uniform(0.05, 0.5)
    elif troj_type == 'Subset-problem-space':
        loc, mask_size = get_problem_space_final_mask(subset_family)
        p_size = mask_size
        pattern = np.ones(shape=(p_size,))
        alpha = 1.0
        target_y = 0
        inject_p = np.random.uniform(0.001, 0.002)
    elif troj_type == 'Top-benign-jumbo':
        p_size = np.random.randint(min_size, size)
        svm_benign_feature_weights_file = 'models/apg/SVM/SVM_benign_feature_weights_c1_iter10000_nfea10000.csv'
        benign_weights = pd.read_csv(svm_benign_feature_weights_file, header=0)
        loc = np.array(benign_weights.iloc[:p_size, 1])
        alpha = 1.0
        pattern = np.ones(shape=(p_size, ))
        target_y = 0
        inject_p = np.random.uniform(0.05, 0.5)
    elif troj_type == 'Top-benign-target':
        # p_size = np.random.randint(10, size)
        p_size = np.random.randint(min_size, size)
        svm_benign_feature_weights_file = 'models/apg/SVM/SVM_benign_feature_weights_c1_iter10000_nfea10000.csv'
        benign_weights = pd.read_csv(svm_benign_feature_weights_file, header=0)
        loc = np.array(benign_weights.iloc[:p_size, 1])
        alpha = 1.0
        pattern = np.ones(shape=(p_size, ))
        target_y = 0
        # inject_p = np.random.uniform(0.05, 0.2)
        inject_p = np.random.uniform(0.05, max_poison_ratio)
    else:
        CLASS_NUM = 2

        # p_size = np.random.randint(1, 26)
        p_size = np.random.randint(min_size, size+1)

        loc = np.random.randint(0, 10000, size=p_size)
        alpha = 1.0

        # pattern = np.random.randint(2, size=p_size) # todo: maybe all 1, to avoid all 0
        # target_y = np.random.randint(CLASS_NUM) # 0 or 1
        pattern = np.ones(shape=(p_size, )) # NOTE: change to all 1, 01132022
        target_y = 0 # to be consistent with baseline jumbo learning # NOTE: change to 0, 01132022
        inject_p = np.random.uniform(0.05, 0.5) # poisoning ratio

    return p_size, pattern, loc, alpha, target_y, inject_p

def troj_gen_func(X, y, atk_setting):
    p_size, pattern, loc, alpha, target_y, inject_p = atk_setting
    # import IPython
    # IPython.embed()
    # w, h = loc
    X_new = copy.deepcopy(X)
    for idx, loc_idx in enumerate(loc):
        X_new[loc_idx] = float(pattern[idx])
    # X_new[0, w:w+p_size, h:h+p_size] = alpha * torch.FloatTensor(pattern) + (1-alpha) * X_new[0, w:w+p_size, h:h+p_size]
    y_new = target_y
    return X_new, y_new

def troj_gen_func_set(X, y, atk_setting):
    p_size, pattern, loc, alpha, target_y, inject_p = atk_setting
    # import IPython
    # IPython.embed()
    # w, h = loc
    X_new = copy.deepcopy(X)
    for idx, loc_idx in enumerate(loc):
        X_new[:, loc_idx] = float(pattern[idx])
    # X_new[0, w:w+p_size, h:h+p_size] = alpha * torch.FloatTensor(pattern) + (1-alpha) * X_new[0, w:w+p_size, h:h+p_size]
    y_new = y
    return X_new, y_new

# def load_attack_setting(task, clf=None):
#     return troj_gen_func, random_troj_setting

class BackdoorDataset(torch.utils.data.Dataset):
    def __init__(self, src_dataset, atk_setting, troj_gen_func=troj_gen_func, 
                 choice=None, mal_only=False,
                 need_pad=False, poison_benign_only=False):
        self.src_dataset = src_dataset
        self.atk_setting = atk_setting
        self.troj_gen_func = troj_gen_func
        self.need_pad = need_pad

        self.mal_only = mal_only
        if choice is None:
            choice = np.arange(len(src_dataset))
        self.choice = choice
        # todo: debugging
        # train_choice, val_choice = train_test_split(choice, test_size=0.33, random_state=42)
        # self.choice = train_choice

        inject_p = atk_setting[5] # poisoning ratio, random sampled from U(0.05, 0.5)
        if self.mal_only:
            inject_p = 1.0
        if poison_benign_only:
            y_train = src_dataset.ys
            logger.info(f'choice: len: {choice.shape[0]}, first 10: {choice[:10]}')
            y_attacker = y_train[choice]  # attacker owned set
            logger.info(f'y_attacker: {Counter(y_attacker)}')
            benign_choice = np.where(y_attacker == 0)[0]
            logger.info(f'benign_choice: len: {benign_choice.shape[0]}, first 10: {benign_choice[:10]}')
            self.mal_choice = np.random.choice(benign_choice, int(len(choice) * inject_p), replace=False)
            logger.info(f' inject_p: {inject_p}; mal_choice: len: {self.mal_choice.shape[0]}, first 10: {self.mal_choice[:10]}')
        else:
            ''' poison some samples from the choice set, could be benign or malicious '''
            logger.info(f'poison_benign_only is False, choice len: {choice.shape[0]}')
            logger.info(f'len(choice)*inject_p: {int(len(choice)*inject_p)}')
            self.mal_choice = np.random.choice(choice, int(len(choice)*inject_p), replace=False)

    def __len__(self,):
        if self.mal_only:
            ''' generate trojaned set only'''
            return len(self.mal_choice)
        else:
            ''' generate training + trojaned (poisoning) set '''
            return len(self.choice) + len(self.mal_choice)

    def __getitem__(self, idx):
        if (not self.mal_only and idx < len(self.choice)):
            ''' Return non-trojaned data '''
            if not self.need_pad:
                return self.src_dataset[self.choice[idx]]

            # In NLP task we need to pad input with length of Troj pattern
            p_size = self.atk_setting[0]
            X, y = self.src_dataset[self.choice[idx]]
            X_padded = torch.cat([X, torch.LongTensor([0]*p_size)], dim=0)
            return X_padded, y
        if self.mal_only: # trojaned only
            X, y = self.src_dataset[self.mal_choice[idx]]
        else:
            # when idx > 2% shadow set or 50% target set, poison it in the next step
            X, y = self.src_dataset[self.mal_choice[idx-len(self.choice)]]
        X_new, y_new = self.troj_gen_func(X, y, self.atk_setting)

        # TODO: warning: this is an ugly solution
        if self.mal_only: # trojaned testing set, we should not change their labels for evaluating backdoor effectiveness
            y_new = y
        return X_new, y_new
    
class APGNew(torch.utils.data.Dataset):
    def __init__(self, X, y, train_flag)    :
        self.Xs = X
        self.ys = y
        print(f'Load dataset, train_flag: {train_flag} Xs: {self.Xs.shape},' +
              f'y: {self.ys.shape}, unique y: {np.unique(self.ys)}')

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.Xs[idx]), self.ys[idx]
    
# Get data based on a trained model
# def load_dataset(clf="mlp"):
#     data_dir = f'models/apg/{clf}'
#     data_file = os.path.join(data_dir, 'mlp_10000-1024-1_lr0.001_b128_e30_d0.2_r42.p')
#     with open(data_file, 'rb') as f:
#         model = pickle.load(f)

#     X_train, X_test = model.X_train, model.X_test  # sparse matrix(dataset['X_train'], dtype='numpy.float64'), though it's 0 and 1
#     y_train, y_test = model.y_train, model.y_test  # np.array(dataset['y_train'], dtype='numpy.int64')

#     logging.info(f'X_train: {X_train.shape}, X_test: {X_test.shape}')
#     logging.info(f'y_train: {y_train.shape}, y_test: {y_test.shape}')
#     logging.debug(f'load dataset y_test[:20]: {y_test[:20]}')
#     return X_train, y_train, X_test, y_test

# def separate_subset_malware(subset_family, dataset='apg', clf='mlp'):
#     X_train, y_train, X_test, y_test = load_dataset(clf)

    
if __name__ == "__main__":
    pass