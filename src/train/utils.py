from datetime import datetime
import json
import logging
import os
import traceback

from matplotlib import pyplot as plt
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from termcolor import colored
import torch
import torch.nn as nn
import torch.nn.functional as F

from timeit import default_timer as timer

from tqdm import tqdm

from pathlib import Path
Path("event_logs").mkdir(exist_ok=True)

logging.basicConfig(filename=f'event_logs/train_{str(datetime.now())}.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class CosineSimilarityLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(CosineSimilarityLoss, self).__init__()
        self.eps = eps  # To avoid division by zero

    def forward(self, x, y):
        # Reshape tensors to match cosine_similarity expected input, if necessary
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        
        # Calculate cosine similarity, range is [-1, 1]
        # with eps added to denominator for numerical stability
        cosine_sim = F.cosine_similarity(x, y, dim=1, eps=self.eps)
        
        # Since cosine similarity gives higher values for more similar vectors
        # and we want to minimize loss, we subtract from 1 to ensure that the
        # loss is lower for similar vectors
        cosine_loss = 1 - cosine_sim
        
        # Calculate mean loss across batch
        return cosine_loss.mean()

def kd_loss(output, label, teacher_output, alpha=0.1, temperature=1.0):
    """
    from:
        https://github.com/peterliht/knowledge-distillation-pytorch
        
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = alpha
    T = temperature
    
    """
    kd_loss = nn.KLDivLoss(reduction='none')(F.log_softmax(output/T, dim=1),
                                             F.softmax(teacher_output/T, dim=1)).type(torch.FloatTensor).cuda(gpu)
    
    kd_loss = kd_filter * torch.sum(kd_loss, dim=1) # kd filter is filled with 0 and 1.
    kd_loss = torch.sum(kd_loss) / torch.sum(kd_filter) * (alpha * T * T)
    """
    
    kd_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(output/T, dim=1),
                                                  F.softmax(teacher_output/T, dim=1)) * (alpha * T * T)
    
    return kd_loss

def entropy_loss(outputs):
    # Apply softmax to get probability distributions for each class
    probs = F.softmax(outputs, dim=1)
    
    # Calculate the entropy for the probability distributions
    # Since torch.log(probs) can produce -inf for 0 probabilities, we clamp the lower probability limit to avoid this
    log_probs = torch.log(probs.clamp(min=1e-9))
    
    # Calculate entropy (the negative sign is because we'll later subtract this from losses which are minimized)
    entropy = -torch.sum(probs * log_probs, dim=1).mean()
    
    return entropy

# ---------- evaluation ----------- #
# --------------------------------- #

def eval_multiple_task(model, X, y_true, task):
    with torch.no_grad():
        outputs = model(torch.tensor(X, dtype=torch.float32))

    y_pred = torch.round(torch.sigmoid(outputs)).numpy()

    logging.info(f'**{task}** confusion matrix:\n{confusion_matrix(y_true, y_pred)}')

    try:
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
    except (ValueError, ZeroDivisionError) as e:
        logging.warning(f'Error calculating metrics for task {task}: {e}')
        f1, precision, recall, acc = -1, -1, -1, -1

    logging.info(f'**{task}** recall: {recall:.4f} \t f1: {f1:.4f} \t precision: {precision:.4f} \t acc: {acc:.4f}')
    return recall, f1, precision, acc

def final_evaluate(model, X_subset_trojan, 
             X_test_remain_mal_trojan, 
             X_test, y_test, 
             X_test_benign_trojan, 
             subset_family ="", troj_type="Subset",
             log_path=""):
    
    model.to("cpu")
    model.eval()

    # import IPython
    # IPython.embed()
    
    SAVE_PREFIX = f'{log_path}/eval'
    # os.makedirs(f'{SAVE_PREFIX}/models', exist_ok=True)

    # Init the labels for specific subsets
    y_subset = np.zeros(X_subset_trojan.shape[0])
    y_remain = np.ones(X_test_remain_mal_trojan.shape[0])
    y_benign = np.zeros(X_test_benign_trojan.shape[0])
    
    # import IPython
    # IPython.embed()
    subset_recall, subset_f1, subset_precision, \
        subset_acc = eval_multiple_task(model, X_subset_trojan, y_subset, task='Backdoor-Acc')
    remain_recall, remain_f1, remain_precision, \
        remain_acc = eval_multiple_task(model, X_test_remain_mal_trojan, y_remain, task='Remain')
    main_recall, main_f1, main_precision, \
        main_acc = eval_multiple_task(model, X_test, y_test, task='Main-Acc')
    benign_recall, benign_f1, benign_precision, \
        benign_acc = eval_multiple_task(model, X_test_benign_trojan, y_benign, task='Benign-Only')

    log = {'subset_recall': subset_recall,
           'subset_f1': subset_f1,
           'subset_precision': subset_precision,
           'subset_acc': subset_acc,
           'remain_recall': remain_recall,
           'remain_f1': remain_f1,
           'remain_precision': remain_precision,
           'remain_acc': remain_acc,
           'main_recall': main_recall,
           'main_f1': main_f1,
           'main_precision': main_precision,
           'main_acc': main_acc,
           'benign_recall': benign_recall,
           'benign_f1': benign_f1,
           'benign_precision': benign_precision,
           'benign_acc': benign_acc}

    print(log)
    log_path = f'{SAVE_PREFIX}/troj_{troj_type}_fam_{subset_family}_{datetime.now()}.json'
    os.makedirs(SAVE_PREFIX, exist_ok=True)
    with open(log_path, "w") as outf:
        json.dump(log, outf)
    print(f"Log file saved to {log_path}")
    return log
    # logging.critical(f'**{task}** cm: \n {confusion_matrix(y_true, y_pred)}')
    # f1, precision, recall, acc = -1, -1, -1, -1
    # try:
    #     f1 = f1_score(y_true, y_pred)
    #     precision = precision_score(y_true, y_pred)
    #     recall = recall_score(y_true, y_pred)
    #     acc = accuracy_score(y_true, y_pred)
    # except:
    #     pass
    # logging.info(f'**{task}** recall: {recall:.4f} \t f1: {f1:.4f} \t precision: {precision:.4f} \t acc: {acc:.4f}')
    # return recall, f1, precision, acc

def calculate_base_metrics(y_test, y_pred, y_scores, phase, output_dir=None):
    """Calculate ROC, F1, Precision and Recall for given scores.

    Args:
        y_test: Array of ground truth labels aligned with `y_pred` and `y_scores`.
        y_pred: Array of predicted labels, aligned with `y_scores` and `model.y_test`.
        y_scores: Array of predicted scores, aligned with `y_pred` and `model.y_test`.
        output_dir: The directory used for dumping output.

    Returns:
        dict: Model performance stats.

    """

    acc, f1, precision, recall, fpr = -1, -1, -1, -1, -1

    cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    logging.debug(f'cm: {cm}')
    if np.all(y_test == 0) and np.all(y_pred == 0):
        TN = len(y_test)
        TP, FP, FN = 0, 0, 0
    elif np.all(y_test == 1) and np.all(y_pred == 1):
        TP = len(y_test)
        TN, FP, FN = 0, 0, 0
    else:
        TN = cm[0][0]
        FN = cm[1][0]
        TP = cm[1][1]
        FP = cm[0][1]

    try:
        f1 = sklearn.metrics.f1_score(y_test, y_pred)
        precision = sklearn.metrics.precision_score(y_test, y_pred)
        recall = sklearn.metrics.recall_score(y_test, y_pred)
        acc = sklearn.metrics.accuracy_score(y_test, y_pred)
    except:
        logging.error(f'calculate_base_metrics: {traceback.format_exc()}')

    try:
        fpr = FP / (FP + TN)
    except:
        logging.error(f'calculate_base_metrics fpr: {traceback.format_exc()}')

    if output_dir:
        pred_file = os.path.join(output_dir, f'prediction_{phase}.csv')
        with open(pred_file, 'w') as f:
            f.write(f'ground,pred,score\n')
            for i in range(len(y_test)):
                f.write(f'{y_test[i]},{y_pred[i]},{y_scores[i]}\n')

    return {
        'model_performance': {
            'acc': acc,
            # 'roc': roc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'cm': cm
        }
    }

def read_perf(report):
    perf = report['model_performance']
    acc = perf['acc']
    f1 = perf['f1']
    recall = perf['recall']
    fpr = perf['fpr']
    return acc, f1, recall, fpr

def plot_model_weights(model, epoch=0, output_dir=""):
    """Plot the weights of the model.

    Args:
        model: The model to plot.
        output_dir: The directory to save the plot.

    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'get_weights'):
            weights = layer.get_weights()
            for j, weight in enumerate(weights):
                plt.figure()
                plt.hist(weight.flatten(), bins=50)
                plt.title(f'Layer {i}, Weight {j}')
                plt.savefig(os.path.join(output_dir, f'layer_{i}_weight_{j}_epoch_{epoch}.png'))
                # plt.close()


# ----------- TRAINING ------------ #
# --------------------------------- #

def test(model, test_loader, device):
    model.to(device)
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
    print(colored(f"[Clean] Testing loss: {test_loss/len(test_loader)}, \t Testing Accuracy: {correct /len(test_loader.dataset)}, \t Num samples: {len(test_loader.dataset)}", "green"))
    return test_loss/len(test_loader), correct /len(test_loader.dataset)

def test_backdoor(model, test_loader, device, 
                  target_label=0):
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for batch_id, batch in tqdm(enumerate(test_loader)):
            data, targets = batch
            data, targets = data.to(device), targets.to(device).float()
            poison_num = data.shape[0]
            # data, targets, poison_num = get_poison_batch(batch, target_label, device, adversarial_index=-1, evaluation=True)
            poison_data_count += poison_num
            dataset_size += len(data)
            output = model(data).squeeze()
            total_loss += criterion(output, targets).item()  # sum up batch loss
            pred = torch.round(torch.sigmoid(output))
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
    acc = 100.0 * (float(correct) / float(poison_data_count)) if poison_data_count!=0 else 0
    total_l = total_loss / poison_data_count if poison_data_count!=0 else 0
    print(colored(f"[Backdoor] Testing loss: {total_l}, \t Testing Accuracy: {correct /len(test_loader.dataset)}, \t Num samples: {poison_data_count}", "red"))
    model.train()
    return total_l, acc, correct, poison_data_count

