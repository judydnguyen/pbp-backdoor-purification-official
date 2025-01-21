"""
Training a backdoored malware classifier with ember dataset

"""
import copy
import json
import os, sys
import datetime

from tabulate import tabulate
from termcolor import colored


# Following lines are for assigning parent directory dynamically.

dir_path = os.path.dirname(os.path.realpath(__file__))

parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))

sys.path.insert(0, parent_dir_path)
sys.path.append("../")

import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset, Dataset
import torchvision.utils as vutils

import argparse
import numpy as np
import sys
import os

from explainable_backdoor_utils import get_backdoor_data
from models.cnn import CNN
from models.mobilenet import MobileNetV2
from models.embernn import EmberNN
from models.simple import SimpleModel
from utils import final_evaluate, logger
from attack_utils import load_wm

from backdoor_helper import set_seed
from ft_dataset import get_backdoor_loader, load_data_loaders, pre_split_dataset_ember, separate_test_data
from common_helper import load_np_data

import torch
import torch.nn as nn
from tqdm import tqdm

DATAPATH = "datasets/ember"
DESTPATH = "datasets/ember/np"
SAVEDIR = "models/malimg/torch"
CONV1 = 32
IMSIZE = 64
EPOCHS = 10
N_CLASS = 25
BATCH_SIZE = 64
SEED = 12
TARGET_LABEL = 0

# tf.random.set_seed(SEED) # Sets seed for training
np.random.seed(SEED)

def get_args():
    """
    Parse Arguments.
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
    config['is_backdoor'] = True if config.get('is_backdoor') == 1 else False
    config['target_label'] = int(config.get('target_label', TARGET_LABEL))
    config['num_poison'] = int(config.get('num_poison', 4))
    config['subset_family'] = "kyugo"
    config['num_workers'] = 16

    return argparse.Namespace(**config)

def test(model, test_loader, device):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()  # Corrected loss function
    test_loss = 0
    correct = 0
    targets = []
    with torch.no_grad():
        for _, (data, target) in tqdm(enumerate(test_loader)):
            data = data.to(device)
            target = target.to(device).float()

            output = model(data).squeeze() 
            test_loss += criterion(output, target).item()
            # Calculate training accuracy
            pred = torch.round(torch.sigmoid(output))  # Round to get binary predictions
            correct += (pred == target).sum().item()

    logger.info(colored(f"[Clean] Testing loss: {test_loss/len(test_loader)}, \t Testing Accuracy: {correct /len(test_loader.dataset)}, \t Num samples: {len(test_loader.dataset)}", "green"))
    correct = 100.0 * correct
    return test_loss/len(test_loader), correct /len(test_loader.dataset)

def train(model, train_loader, device, total_epochs=10, lr=0.001):
    model.to(device)  # Move model to device
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(total_epochs):
        running_loss = 0
        correct = 0
        total = 0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{total_epochs}'):
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

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        logger.info(f"Epoch {epoch + 1}/{total_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
    return model


#----------*----------#
#------BACKDOOR-------#
#----------*----------#

# def generate_backdoor_data(model, device):
#     X_train_loaded, y_train_loaded, X_val, y_val , _ = load_np_data(DATAPATH)
#     logger.info(colored(f"Start generating backdoor sampled with set of size: {X_train_loaded.shape[0]}", "red"))
#     X_train_watermarked, y_train_watermarked, X_test_mw = get_backdoor_data(X_train_loaded, y_train_loaded, X_val, y_val, 
#                                                                             copy.deepcopy(model), device, DESTPATH)
#     logger.info(colored(f"Size of training backdoor data: {X_train_watermarked.shape[0]}"))
    
#     backdoored_X, backdoored_y = torch.from_numpy(X_train_watermarked), torch.from_numpy(y_train_watermarked)
#     backdoored_dataset = TensorDataset(backdoored_X, backdoored_y)
#     backdoored_loader = DataLoader(backdoored_dataset, batch_size=512, shuffle=True, num_workers=54)
#     return backdoored_loader

def train_backdoor(model, train_loader, val_loader, backdoor_loader, device, 
                   total_epochs=20, backdoor_epochs=5, lr=0.001, 
                   scaler=None, poison_rate=1, plot_save_path=""):
    
    if not os.path.exists(plot_save_path):
        os.makedirs(plot_save_path)
        
    wm_config = load_wm(DESTPATH)
    model.to(device)  # Move model to device
    # model.load_state_dict(torch.load("models/ember/torch/embernn/tgt_0_epochs_5_ft_size_0.05_lr_0.001_poison_rate_0.0.pth"))
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(total_epochs), desc="Training Clean Model"):
        running_loss = 0
        correct = 0
        total = 0
        for batch in tqdm(train_loader, desc=f'Poisoning Epoch {epoch + 1}/{total_epochs}'):
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

        test(model, val_loader, device)
        test_backdoor(model, backdoor_loader, device)
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        logger.info(f"Epoch {epoch + 1}/{total_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
    model.eval()
    return model

def test_backdoor(model, test_loader, device, 
                  target_label=0):
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
    logger.info(colored(f"[Backdoor] Testing loss: {total_l}, \t Testing Accuracy: {correct /len(test_loader.dataset)}, \t Num samples: {poison_data_count}", "red"))
    model.train()
    return total_l, acc, correct, poison_data_count

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Reproduce Main Function
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
if __name__ == "__main__":
    set_seed(SEED)
    current_time = str(datetime.datetime.now())
    # get the start time for saving trained model
    args = get_args()
    os.makedirs(args.savedir, exist_ok=True)
    convs_sizes = [2, 8, 32]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device as: {device}")
    
    parent_p = "datasets/ember"
    pre_split_dataset_ember(args.datapath, args.ft_size, SEED, parent_p)

    train_dl, backdoor_dl, valid_dl, ft_dl, \
        backdoor_test_dl, X_test_loaded, y_test_loaded, X_subset_trojaned = load_data_loaders(data_path=parent_p,
                                                ft_size=args.ft_size,
                                                batch_size=args.batch_size, 
                                                test_batch_size=args.test_batch_size,
                                                num_workers=args.num_workers, val_size=0,
                                                poison_rate=args.poison_rate,
                                                dataset=args.dataset)
    
    X_test_remain_mal, X_test_benign = separate_test_data(X_test_loaded, y_test_loaded)
    
    num_channels = valid_dl.dataset[0][0].shape[0]
    logger.info(f"Num channels: {num_channels}")
    
    name = "linear" if args.conv1 == 0 else args.conv1
    args.name = f"malware_malimg_family_scaled_{name}-25"
    # -----*------ #
    # --Training-- #
    # -----*------ #
    
    if args.model == "cnn":
        model = CNN(args.imsize, num_channels, args.conv1, args.classes)
    elif args.model == "simple":
        model = SimpleModel(num_channels, 16)
    elif args.model == "mobilenetv2":
        model = MobileNetV2(num_channels, args.classes)
    elif args.model == "embernn":
        model = EmberNN(num_channels)
    else:
        pass
    model.to(device)
    file_to_save = f"tgt_{args.target_label}_epochs_{args.epochs}_ft_size_{args.ft_size}_lr_{args.lr}_poison_rate_{round(args.poison_rate, 4)}"
    model_save_path = f"{args.savedir}/{args.model}" if not args.is_backdoor else f"{args.savedir}/{args.model}/backdoor"
    
    if args.is_backdoor:
        logger.info("\n--------BEFORE Backdoor Testing --------- ")
        test_loader = get_backdoor_loader(DESTPATH)
        total_l, acc, correct, poison_data_count = test_backdoor(model, backdoor_test_dl, device, 
                                                                 args.target_label)
        
        logger.info("\n--------Backdoor Training --------- ")
        plot_save_path = f"{model_save_path}/{file_to_save}"
        model = train_backdoor(model, backdoor_dl, valid_dl, 
                               backdoor_test_dl, device, 
                               total_epochs=args.epochs, 
                               poison_rate=args.poison_rate, 
                               lr=args.lr,
                               plot_save_path=plot_save_path)
        
        os.makedirs(model_save_path, exist_ok=True)
        torch.save(model.state_dict(), f"{model_save_path}/{file_to_save}.pth")
        logger.info(colored(f"Saved model at {f'{model_save_path}/{file_to_save}.pth'}", "blue"))
        
        logger.info("\n--------Normal Testing --------- ")
        _, test_acc = test(model, valid_dl, device)
        
        logger.info("\n--------Backdoor Testing --------- ")
        test_loader = get_backdoor_loader(DESTPATH)
        total_l, acc, correct, poison_data_count = test_backdoor(model, backdoor_test_dl, device, 
                                                                 args.target_label)
        model.to("cpu")
    
    else:
        logger.info("\n--------Normal Training --------- ")
        model = train(model, train_dl, device, 
                      total_epochs=args.epochs, 
                      lr=args.lr)
        os.makedirs(model_save_path, exist_ok=True)
        torch.save(model.state_dict(), f"{model_save_path}/{file_to_save}.pth")
        logger.info("\n--------Normal Testing --------- ")
        _, test_acc = test(model, valid_dl, device)
    
    # LOGGING EVALUATION FOR BACKDOORED MODEL
    # Data to display
    data = [["Main Accuracy", round(test_acc, 4)], 
            ["Backdoor Accuracy", round(acc, 4)],
            ["Poisoning Rate", args.poison_rate]]

    # Printing table format
    print("\n------- Final Evaluation -------")
    print(tabulate(data, headers=["Metric", "Value"], tablefmt="grid"))
        
    # --------- * Final Evaluation * --------- #
    # --------- * **************** * --------- #
    
    # log_dir_path = f"{model_save_path}/{file_to_save}"
    # final_eval_result = final_evaluate(model, X_subset_trojan=X_subset_trojaned,
    #                                 X_test_remain_mal_trojan=X_test_remain_mal,
    #                                 X_test=X_test_loaded, y_test=y_test_loaded, 
    #                                 X_test_benign_trojan=X_test_benign, 
    #                                 subset_family="", 
    #                                 troj_type="Subset", 
    #                                 log_path=log_dir_path)