import argparse
import copy
import gc
import json
# import logging


from pprint import pformat
from sched import scheduler
import os, sys
import datetime

import numpy as np
from tqdm import tqdm
from tabulate import tabulate

#Following lines are for assigning parent directory dynamically.

dir_path = os.path.dirname(os.path.realpath(__file__))

parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))

sys.path.insert(0, parent_dir_path)
sys.path.append("../")

from gradient_inversion import reverse_net
from sam import finetune_sam
from utils import final_evaluate, logger
from defense_helper import add_noise_w, apply_robust_LR
from ember.train_ember_pytorch import test_backdoor, test
from finetune_helper import add_args, get_optimizer
from ft_dataset import get_backdoor_loader, get_em_bd_loader, load_data_loaders, pre_split_dataset_ember, separate_test_data
from models.cnn import CNN
from models.embernn import EmberNN
from models.mobilenet import MobileNetV2
from models.resnet_bak import ResNet18
from models.simple import SimpleModel

import torch
import torch.nn as nn
import torch.optim as optim

from termcolor import colored
from backdoor_helper import set_seed

# logger.basicConfig()
# logger = logger.getLogger()
# logger.setLevel(logger.INFO)

from torch.utils.tensorboard import SummaryWriter

DATAPATH = "datasets/ember"
DESTPATH = "datasets/ember/np"
SAVEDIR = "models/malimg/torch"

SEED = 12
set_seed(SEED)

def finetune(net, optimizer, criterion,
             ft_dl, test_dl, backdoor_dl, f_epochs=1, 
             ft_mode="ft", device="cuda", logger=None,
             logging_path="path/to/log", lbs_criterion=None,
             args=None, weight_mat_ori=None, 
             original_linear_norm=0):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    os.makedirs(logging_path, exist_ok=True)
    writer = SummaryWriter(log_dir=f'{logging_path}/log/mode_{ft_mode}')
    logger.info(f"Fine-tuning mode: {ft_mode}")
    
    cur_clean_acc, cur_adv_acc = 0.0, 0.0
    writer.add_scalar('Validation Clean ACC', cur_clean_acc, 0)
    writer.add_scalar('Validation Backdoor ACC', cur_adv_acc, 0)
    
    log_path = f"{logging_path}/plots/mode_{ft_mode}"
    if ft_mode == "proposal":
        net_cpy = copy.deepcopy(net)
        net_cpy.to(device)
        
        net_cpy.train()
        optimizer_cp = optim.Adam(net_cpy.parameters(), lr=0.002)
        reversed_net, vectorized_mask = reverse_net(net_cpy, ft_dl, test_dl, backdoor_dl, optimizer_cp, device, f_epochs=1)

    
    if ft_mode == 'proposal':
        net = add_noise_w(net, device, stddev=0.5)
    prev_model = copy.deepcopy(net)
    
    for epoch in tqdm(range(f_epochs), desc=f'Fine-tuning mode: {ft_mode}'):
        batch_loss_list = []
        train_correct = 0
        train_tot = 0

        net.train()
        for batch_idx, (x, labels) in tqdm(enumerate(ft_dl), desc=f'Epoch [{epoch + 1}/{f_epochs}]: '):
            optimizer.zero_grad()
            x, labels = x.to(device), labels.to(device).float()
            log_probs = net(x).squeeze()
            
            if lbs_criterion is not None:
                loss = lbs_criterion(log_probs, labels)
            else:
                if ft_mode == 'fst':
                    loss = torch.sum(eval(f'net.{args.linear_name}.weight') * weight_mat_ori)*args.alpha + criterion(log_probs, labels)
                elif ft_mode == 'proposal':
                    # try new loss
                    loss = criterion(log_probs, labels).mean()
                    # loss = masked_feature_shift_loss(eval(f'net.{args.linear_name}.weight'), weight_mat_ori, mask)*0.1 + criterion(log_probs, labels)
                else:
                    loss = criterion(log_probs, labels).mean()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            exec_str = f'net.{args.linear_name}.weight.data = net.{args.linear_name}.weight.data * original_linear_norm  / torch.norm(net.{args.linear_name}.weight.data)'
            exec(exec_str)

            # _, predicted = torch.max(log_probs, -1)
            predicted = torch.round(torch.sigmoid(log_probs))
            train_correct += predicted.eq(labels).sum()
            train_tot += labels.size(0)
            batch_loss = loss.item() * labels.size(0)
            batch_loss_list.append(batch_loss)
        
        if ft_mode == "proposal" and epoch % 2 == 0:
            net = apply_robust_LR(net, prev_model, vectorized_mask)
            prev_model= copy.deepcopy(net)

        gc.collect()
        # scheduler.step()
        one_epoch_loss = sum(batch_loss_list)

        logger.info(f'Training ACC: {train_correct/train_tot} | Training loss: {one_epoch_loss}')
        logger.info(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
        logger.info('-------------------------------------')
        
        writer.add_scalar('Training Loss', one_epoch_loss, epoch)
        writer.add_scalar('Training Accuracy', train_correct/train_tot, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]["lr"], epoch)
        
        logger.info(colored(f"Start validation for current epoch: [{epoch}/{args.f_epochs}]\n", "blue"))
        # print("\n--------Normal Testing --------- ")
        loss_c, acc_c = test(net, test_dl, device)
        
        # print("\n--------Backdoor Testing --------- ")
        # test_loader = get_backdoor_loader(DESTPATH)
        loss_bd, acc_bd, correct, poison_data_count = test_backdoor(net, backdoor_dl, 
                                                                    device, 
                                                                    args.target_label)
        writer.add_scalar('Validation Clean ACC', acc_c, epoch)
        writer.add_scalar('Validation Backdoor ACC', acc_bd, epoch)
        
        metric_info = {
            f'clean acc': acc_c,
            f'clean loss': loss_c,
            f'backdoor acc': acc_bd,
            f'backdoor loss': loss_bd
        }
        cur_clean_acc = metric_info['clean acc']
        cur_adv_acc = metric_info['backdoor acc']
        logger.info('*****************************')
        logger.info(colored(f'Fine-tunning mode: {ft_mode}', "green"))
        logger.info(f"Test Set: Clean ACC: {cur_clean_acc} | ASR: {cur_adv_acc}")
        logger.info('*****************************')
        
    net.eval()
    return net, acc_c, acc_bd

def main():
    ### 1. config args, save_path, fix random seed
    # Create the argument parser
    # Start counting time
    start_time = datetime.datetime.now()
    parser = argparse.ArgumentParser(description="Description of your program")
    
    # Add arguments using the add_args function
    parser = add_args(parser)
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Now you can access the arguments using dot notation
    print("Device: ", args.device)
    print("Number of classes: ", args.num_classes)
    args.dataset_path = f"{args.dataset_path}/{args.dataset}"
    
    # ------------ Loading a pre-trained (backdoored) model -------------- #
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    file_to_load = f"tgt_{args.target_label}_epochs_{args.epochs}_ft_size_{args.ft_size}_lr_{args.lr}_poison_rate_{round(args.poison_rate, 4)}"
    file_to_load = f'{args.folder_path}/backdoor/{file_to_load}.pth'
    
    parent_p = "datasets/ember"
    
    pre_split_dataset_ember(args.datapath, 
                            args.ft_size, SEED, 
                            parent_p)

    train_dl, _, test_dl, ft_dl, \
        backdoor_test_dl, X_test_loaded, y_test_loaded, X_subset_trojaned = load_data_loaders(data_path=parent_p,
                                                ft_size=args.ft_size,
                                                batch_size=args.batch_size, 
                                                test_batch_size=args.test_batch_size,
                                                num_workers=args.num_workers, val_size=0,
                                                poison_rate=args.poison_rate,
                                                dataset=args.dataset)

    X_test_remain_mal, X_test_benign = separate_test_data(X_test_loaded, y_test_loaded)
    
    ### 3. get model
    num_channels = test_dl.dataset[0][0].shape[0]
    if args.model == "cnn":
        net = CNN(args.imsize, num_channels, args.conv1, args.classes)
    elif args.model == "simple":
        net = SimpleModel(num_channels, 16)
    elif args.model == "mobilenetv2":
        net = MobileNetV2(num_channels, args.classes)
    elif args.model == "resnet":
        net = ResNet18(num_classes=args.classes)
    elif args.model == "embernn":
        net = EmberNN(num_channels)
    
    state_dict = torch.load(file_to_load)
    # Load the state dictionary into the model
    net.load_state_dict(state_dict)
    logger.info(colored(f"Loaded model at {file_to_load}", "blue"))
    net.to(device)
    
    original_linear_norm = torch.norm(eval(f'net.{args.linear_name}.weight'))
    weight_mat_ori = eval(f'net.{args.linear_name}.weight.data.clone().detach()')
    
    ori_net = copy.deepcopy(net)
    ori_net.eval()
    
    backdoor_test_dl = get_em_bd_loader(net, X_test_loaded, y_test_loaded, device)

    # print("\n--------Normal Testing --------- ")
    loss_c, acc_c = test(net, test_dl, device)
    # print("\n--------Backdoor Testing --------- ")
    loss_bd, acc_bd, correct_bd, poison_data_count = test_backdoor(net, backdoor_test_dl, device, 
                                                                args.target_label)
    metric_info = {
        f'clean acc': acc_c,
        f'clean loss': loss_c,
        f'backdoor acc': acc_bd,
        f'backdoor loss': loss_bd
    }
    cur_clean_acc = metric_info['clean acc']
    cur_adv_acc = metric_info['backdoor acc']

    logger.info('*****************************')
    logger.info(f"Load from {args.folder_path}")
    # logger.info(f'Fine-tunning mode: {args.ft_mode}')
    logger.info('Original performance')
    logger.info(f"Test Set: Clean ACC: {cur_clean_acc} | ASR: {cur_adv_acc}")
    logger.info('*****************************')

    # ---------- Start Fine-tuning ---------- #
    logging_path = f'{args.log_dir}/target_{args.attack_target}-archi_{args.model}-dataset_{args.dataset}--f_epochs_{args.f_epochs}--f_lr_{args.f_lr}/ft_size_{args.ft_size}_p_rate{round(args.poison_rate, 4)}'
    ft_modes = ['ft', 'ft-init', 'fe-tuning', 'lp', 'fst', 'proposal']
    # ft_modes = ['ft-init']
    # ft_modes = ['ft-init', 'proposal']
    ft_results = {}

    for ft_mode in ft_modes:
        ft_results[ft_mode] = {}
        model_save_path = f'{args.folder_path}/target_{args.attack_target}-archi_{args.model}-dataset_{args.dataset}--f_epochs_{args.f_epochs}--f_lr_{args.f_lr}/ft_size_{args.ft_size}_p_rate{round(args.poison_rate, 4)}/mode_{ft_mode}'
        
        net.load_state_dict(state_dict)
        net.to(device)
        
        optimizer, criterion = get_optimizer(net, ft_mode, args.linear_name, 2*args.f_lr, dataset = args.dataset)
        if ft_mode == 'ft-sam':
            ft_net = finetune_sam(net, ft_dl, test_dl, backdoor_test_dl,
                                  criterion, args.f_epochs, 
                                  device, logging_path, 
                                  args=args, base_optimizer=optimizer,
                                  weight_mat_ori=weight_mat_ori, 
                                  original_linear_norm=original_linear_norm,
                                  test=test, test_backdoor=test_backdoor)
        else:
            ft_net, acc_c, acc_bd = finetune(net, optimizer, criterion, 
                            ft_dl, test_dl, backdoor_test_dl, 
                            args.f_epochs, ft_mode, device, logger, 
                            logging_path, args=args,
                            weight_mat_ori=weight_mat_ori, 
                            original_linear_norm=original_linear_norm)
        ft_results[ft_mode]["clean_acc"] = acc_c
        ft_results[ft_mode]["adv_acc"] = acc_bd
        args.save = True
        if args.save:
            os.makedirs(model_save_path, exist_ok=True)
            torch.save(ft_net.state_dict(), f'{model_save_path}/checkpoint.pt')
        
        # --------- * Final Evaluation * --------- #
        # --------- * **************** * --------- #
        
        final_eval_result = final_evaluate(net, X_subset_trojan=X_subset_trojaned,
                                        X_test_remain_mal_trojan=X_test_remain_mal,
                                        X_test=X_test_loaded, y_test=y_test_loaded, 
                                        X_test_benign_trojan=X_test_benign, 
                                        subset_family="", 
                                        troj_type="Subset", 
                                        log_path=model_save_path)

    # convert to tabulate and print the final results
    # Extract data for printing
    data = []
    for mode, result in ft_results.items():
        data.append([mode, result["clean_acc"], result["adv_acc"]])

    # Print table
    print("\n------- Fine-tuning Evaluation -------")
    print(tabulate(data, headers=["Mode", "Clean Accuracy", "Adversarial Accuracy"], tablefmt="grid"))
    end_time = datetime.datetime.now()
    print(f"Completed in: {end_time - start_time} seconds.")
    print("------- ********************** -------\n")


if __name__ == '__main__':
    main()