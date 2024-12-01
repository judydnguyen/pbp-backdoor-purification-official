"""
Script to train APG image malware classifiers.

NOTES:
models = [linear-2, 4-2, 16-2]

"""
import copy
import json
import os, sys
import datetime

from termcolor import colored

import torch
import torch.nn as nn
from tqdm import tqdm
from tabulate import tabulate

# Following lines are for assigning parent directory dynamically.

dir_path = os.path.dirname(os.path.realpath(__file__))

parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))

sys.path.insert(0, parent_dir_path)
sys.path.append("../")

from backdoor_helper import set_seed
from jigsaw.jigsaw_helper import load_np_apg_data
from jigsaw.jigsaw_utils import get_apg_backdoor_data, get_jigsaw_config, load_apg_data_loaders, load_apg_subset_data_loaders, load_features, pre_split_apg_datasets
# from jigsaw.models_bak import MLP

# logger.info(f"parent_dir_path: {parent_dir_path}")

import torch.optim as optim

import argparse
import numpy as np
import sys
import os

from dataset import get_train_test_loaders
from explainable_backdoor_utils import get_backdoor_data

from models.cnn import CNN
from models.mobilenet import MobileNetV2
from models.resnet_bak import ResNet18
from models.embernn import EmberNN
from models.simple import DeepNN, SimpleModel
from models.CNN_Models import CNNMalware_Model1
from models.ANN_Models import ANNMalware_Model1, MalConv
from utils import final_evaluate, logger
from attack_utils import load_wm

DATAPATH = "datasets/apg"
DESTPATH = "datasets/apg/np"
SAVEDIR = "models/apg/torch"
CONV1 = 32
IMSIZE = 64
EPOCHS = 10
N_CLASS = 25
BATCH_SIZE = 64
SEED = 42
TARGET_LABEL = 0

# tf.random.set_seed(SEED) # Sets seed for training
np.random.seed(SEED)

def get_args():
    """
    Parse Arguments.subset_family
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config.json", help="Path to JSON config file.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    # Convert string values to appropriate types
    config['classes'] = int(config.get('classes', N_CLASS))
    config['imsize'] = int(config.get('imsize', IMSIZE))
    config['conv1'] = int(config.get('conv1', CONV1))
    config['epochs'] = int(config.get('epochs', EPOCHS))
    config['batch_size'] = int(config.get('batch_size', BATCH_SIZE))
    config['test_batch_size'] = int(config.get('test_batch_size', 128))
    config['is_backdoor'] = config.get('is_backdoor') == 1
    config['target_label'] = int(config.get('target_label', TARGET_LABEL))
    config['num_poison'] = int(config.get('num_poison', 4))
    config['num_workers'] = int(config.get('num_workers', 16))

    return argparse.Namespace(**config)

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
    correct = 100.0 * correct
    return test_loss/len(test_loader), correct /len(test_loader.dataset)

def train(model, train_loader, device, total_epochs=10, lr=0.001):
    model.to(device)  # Move model to device
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(total_epochs), desc="Training Clean Model"):
        running_loss = 0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{total_epochs}'):
            inputs, labels = inputs.to(device), labels.to(device).float()

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


#----------*----------#
#------BACKDOOR-------#
#----------*----------#

def train_backdoor(model, train_loader, val_loader, backdoor_loader, device, 
                   total_epochs=20, backdoor_epochs=5, lr=0.001, 
                   scaler=None, poison_rate=1):
    best_bd_acc, best_clean_acc = 0., 0.
    best_model = None
    model.to(device)  # Move model to device
    # model.load_state_dict(torch.load("models/ember/torch/embernn/tgt_0_epochs_5_ft_size_0.05_lr_0.001_poison_rate_0.0.pth"))
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(total_epochs), desc="Training Backdoor Model"):
        running_loss = 0
        correct = 0
        total = 0
        for batch_idx, batch in tqdm(enumerate(train_loader), desc=f'Backdoor Epoch {epoch + 1}/{total_epochs}'):
        # for batch in tqdm(train_loader, desc=f'Poisoning Epoch {epoch + 1}/{total_epochs}'):
            # inputs, labels = poison_batch_ember(batch, device, wm_config, poison_rate)
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels).mean()
            loss.backward()
            optimizer.step()

            running_loss += loss

            # Calculate training accuracy
            predicted = torch.round(torch.sigmoid(outputs))
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        _, test_acc = test(model, val_loader, device)
        _, bd_acc, _, _ = test_backdoor(model, backdoor_loader, device)
        
        if bd_acc >= best_bd_acc:
            # best_model = copy.deepcopy(model)
            best_bd_acc = bd_acc
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        logger.info(f"Epoch {epoch + 1}/{total_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
    # # after we train a good enough clean model, 
    # # it's now time for backdooring it
    model.eval()
    # best_model.eval()
    return model

def test_backdoor(model, test_loader, device, 
                  target_label=0):
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0
    criterion = nn.BCEWithLogitsLoss()
    
    preds = []
    gt_y = []
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
            preds.append(pred)
            gt_y.append(targets)
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
    
    preds = torch.cat(preds)
    gt_y = torch.cat(gt_y)
    # print(f"preds - gt_y = {preds - gt_y}")
    acc = 100.0 * (float(correct) / float(poison_data_count)) if poison_data_count!=0 else 0
    total_l = total_loss / poison_data_count if poison_data_count!=0 else 0
    logger.info(colored(f"[Backdoor] Testing loss: {total_l}, \t Testing Accuracy: {correct /len(test_loader.dataset)}, \t Num samples: {poison_data_count}", "red"))
    model.train()
    return total_l, acc, correct, poison_data_count


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Reproduce Main Function
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
if __name__ == "__main__":
    set_seed(SEED)
    current_time = str(datetime.datetime.now())
    start_time = datetime.datetime.now()
    # get the start time for saving trained model
    args = get_args()
    os.makedirs(args.savedir, exist_ok=True)
    convs_sizes = [2, 8, 32]
    #
    # Generate Training Set
    #
    # logger.info("\nLoading Data ...")
    # train_dl, valid_dl = get_train_test_loaders(args.datapath, args.batch_size, 
    #                                             args.test_batch_size, args.imsize)
    best_ast, best_clean_acc = 0.0, 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parent_p = "datasets/apg"
    args.n_features = 10000
    args.seed = SEED

    bd_config = get_jigsaw_config(DATAPATH)

    POSTFIX = f'{args.dataset}/{args.model}'

    POISONED_MODELS_FOLDER = os.path.join('models', POSTFIX)

    os.makedirs(POISONED_MODELS_FOLDER, exist_ok=True)

    # pre_split_apg_datasets(args, bd_config, parent_p, POISONED_MODELS_FOLDER, args.ft_size, seed=SEED)
    # train_dl, test_dl, ft_loader, X_train, y_train, X_test, y_test = load_apg_data_loaders(parent_p, batch_size=args.batch_size, ft_size=args.ft_size)
    
    X_train, y_train, X_test, y_test, X_subset, X_test_benign, X_test_remain_mal = pre_split_apg_datasets(args, bd_config, parent_p, POISONED_MODELS_FOLDER, args.ft_size, seed=SEED)
    train_dl, test_dl, ft_loader, testloader_mal, X_subset_trojaned = load_apg_subset_data_loaders(args, parent_p, batch_size=args.batch_size, 
                                                                                ft_size=args.ft_size, subset_family=args.subset_family)

    num_features = train_dl.dataset[0][0].shape[0]
    logger.info(f"Num features: {num_features}")

    # train_dl, valid_df, ft_dl = get_train_test_ft_loaders(args.datapath, args.batch_size, 
    #                                                     args.test_batch_size, args.imsize, 
    #                                                     args.ft_size)

    name = "linear" if args.conv1 == 0 else args.conv1
    args.name = f"malware_apg_family_scaled_{name}-25"

    # -----*------ #
    # --Training-- #
    # -----*------ #

    if args.model == "cnn":
        model = CNN(args.imsize, num_features, args.conv1, args.classes)
    elif args.model == "mlp":
        model = MLP(num_features=args.n_features, dims=[])
    elif args.model == "simple":
        model = SimpleModel(num_features, 16)
    elif args.model == "mobilenetv2":
        model = MobileNetV2(num_features, args.classes)
    elif args.model == "resnet":
        model = ResNet18(num_classes=args.classes)
    elif args.model == "embernn":
        model = EmberNN(num_features)
        # model = DeepNN(n_features=num_channels)
        # model = ANNMalware_Model1(num_channels, 2)
        # model = MalConv(num_channels)
    else:
        pass
    model.to(device)
    file_to_save = f"fam_{args.subset_family}_tgt_{args.target_label}_epochs_{args.epochs}_ft_size_{args.ft_size}_lr_{args.lr}_poison_rate_{round(args.poison_rate, 4)}"

    parent_path = f"{args.savedir}/{args.model}/backdoor" if args.is_backdoor else f"{args.savedir}/{args.model}"
    
    if args.is_backdoor:
        bd_test_loader = testloader_mal
        logger.info("\n--------BEFORE Backdoor Testing --------- ")
        total_l, acc, correct, poison_data_count = test_backdoor(model, bd_test_loader, device, 
                                                                 args.target_label)
        _, test_acc = test(model, test_dl, device)
        logger.info(f"Clean Acc before training is: {round(100.0*test_acc, 4)}")
        
        logger.info("\n--------Backdoor Training --------- ")
        model = train_backdoor(model, train_dl,
                               test_dl, bd_test_loader, device, 
                               total_epochs=args.epochs, 
                               poison_rate=args.poison_rate, lr=args.lr)

        os.makedirs(parent_path, exist_ok=True)
        torch.save(model.state_dict(), f"{parent_path}/{file_to_save}.pth")

        logger.info(
            colored(
                f"Saved model at {args.savedir}/{args.model}/backdoor/{file_to_save}.pth",
                "blue",
            )
        )

        logger.info("\n--------Normal Testing --------- ")
        _, test_acc = test(model, test_dl, device)

        logger.info("\n--------Backdoor Testing --------- ")
        # test_loader = get_backdoor_loader(DESTPATH)

        total_l, acc, correct, poison_data_count = test_backdoor(model, bd_test_loader, device, 
                                                                 args.target_label)
        model.to("cpu")

    else:
        logger.info("\n--------Normal Training --------- ")
        model = train(model, train_dl, device, total_epochs=args.epochs, lr=args.lr)

        logger.info("\n--------Normal Testing --------- ")
        _, test_acc = test(model, test_dl, device)

        os.makedirs(parent_path, exist_ok=True)
        torch.save(model.state_dict(), f"{parent_path}/{file_to_save}.pth")

        logger.info(colored(f'Model is saved at {parent_path}/{file_to_save}.pth', "blue"))
    
    end_time = datetime.datetime.now()
    # LOGGING EVALUATION FOR BACKDOORED MODEL
    # Data to display
    data = [["Main Accuracy", round(test_acc, 4)], 
            ["Backdoor Accuracy", round(acc, 4)],
            ["Poisoning Rate", args.poison_rate]]

    # Printing table format
    print("\n------- Final Evaluation -------")
    print(tabulate(data, headers=["Metric", "Value"], tablefmt="grid"))
    print(f"Completed in {end_time - start_time} seconds.")
    print("--------- * ******** * ---------\n")
        
    # # --------- * Final Evaluation * --------- #
    # # --------- * **************** * --------- #
    # log_parent_dir = f"{parent_path}/{file_to_save}"
    # final_eval_result = final_evaluate(model, X_subset_trojan=X_subset_trojaned,
    #                                    X_test_remain_mal_trojan=X_test_remain_mal,
    #                                    X_test=X_test, y_test=y_test, 
    #                                    X_test_benign_trojan=X_test_benign, 
    #                                    subset_family=args.subset_family, 
    #                                    troj_type="Subset", log_path=log_parent_dir)
                                           
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# General Main Function
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
# if __name__ == "__main__":
#     #
#     # Get and validate arguments
#     #
#     args = get_args()
#     if args.name == None:
#         name = "none" if args.conv1 == 0 else args.conv1
#         args.name = f"malware_malimg_family_scaled_{name}-25"    #
#     # Training
#     #
#     logger.info("\n-------- Training Parameters --------- ")
#     logger.info(f"NORMALIZE: {not args.no_normalize}")
#     logger.info(f"NAME: {args.name}")
#     logger.info(f"IMSIZE: {args.imsize}")
#     logger.info(f"CONV1: {args.conv1}")
#     logger.info(f"EPOCHS: {args.epochs}")
    
#     logger.info("\n-------- Training --------- ")

#     # train_family_classifier(args)

        
