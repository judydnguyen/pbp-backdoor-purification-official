import argparse
import copy
import gc
import json
# import logging
import os, sys
import datetime

from tqdm import tqdm

#Following lines are for assigning parent directory dynamically.
from tabulate import tabulate

dir_path = os.path.dirname(os.path.realpath(__file__))

parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))

sys.path.insert(0, parent_dir_path)
sys.path.append("../")

from gradient_inversion import retrain_model, reverse_net
from sam import finetune_sam
from utils import final_evaluate, logger

from jigsaw.jigsaw_utils import get_apg_backdoor_data, get_jigsaw_config, load_apg_subset_data_loaders, pre_split_apg_datasets
from jigsaw.train_jigsaw_pytorch import test, test_backdoor
from defense_helper import add_noise_w, apply_robust_LR

from finetune_helper import add_args, get_optimizer
from models.cnn import CNN
from models.embernn import EmberNN
from models.mobilenet import MobileNetV2
from models.resnet_bak import ResNet18
from models.simple import SimpleModel

import torch
import torch.optim as optim

from termcolor import colored
from backdoor_helper import set_seed

# logging.basicConfig()
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

from torch.utils.tensorboard import SummaryWriter

DATAPATH = "datasets/apg"
DESTPATH = "datasets/apg/np"
SAVEDIR = "models/apg/torch"

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
    # writer.add_scalar('Validation Clean ACC', cur_clean_acc, 0)
    # writer.add_scalar('Validation Backdoor ACC', cur_adv_acc, 0)
    
    logger.info(colored("\n--------Normal Testing before fine-tuning--------- ", "blue"))
    loss_c, acc_c = test(net, test_dl, device)

    logger.info(colored("\n--------Backdoor Testing before fine-tuning --------- ", "blue"))
    # test_loader = get_backdoor_loader(DESTPATH)
    loss_bd, acc_bd, correct, poison_data_count = test_backdoor(net, backdoor_dl, 
                                                                device, 
                                                                args.target_label)
    writer.add_scalar('Validation Clean ACC', acc_c, -1)
    writer.add_scalar('Validation Backdoor ACC', acc_bd, -1)

    log_path = f"{logging_path}/plots/mode_{ft_mode}"

    # original_linear_norm = torch.norm(eval(f'net.{args.linear_name}.weight'))
    # weight_mat_ori = eval(f'net.{args.linear_name}.weight.data.clone().detach()')
    if ft_mode == "proposal":
        net_cpy = copy.deepcopy(net)
        net_cpy.to(device)
        net_cpy.train()
        optimizer_cp = optim.Adam(net_cpy.parameters(), lr=0.001)
        # optimizer_cp = optim.Adam(net_cpy.parameters(), lr=args.f_lr)
        # retrain_model(net_cpy, ft_dl, test_dl, backdoor_dl, optimizer_cp, device, f_epochs=5, args=args)
        reversed_net, vectorized_mask = reverse_net(net_cpy, ft_dl, test_dl, backdoor_dl, optimizer_cp, device, f_epochs=2)
    
    if ft_mode == 'proposal':
        logger.info("Adding noise to the model")
        # net = add_masked_noise(net, device, stddev=0.2, mask=vectorized_noise_mask)
        net = add_noise_w(net, device, stddev=0.25)
        
    prev_model = copy.deepcopy(net)
    
    for epoch in tqdm(range(f_epochs), desc=f'Fine-tuning mode: {ft_mode}'):
        batch_loss_list = []
        train_correct = 0
        train_tot = 0
        # print(f"args: {args}")

        # if ft_mode == "proposal":
        #     net_cpy = copy.deepcopy(net)
        #     net_cpy.train()
        #     optimizer_cp = optim.Adam(net_cpy.parameters(), lr=args.f_lr)
            
        #     mask_grad_list, _ = get_grad_mask(net_cpy, optimizer_cp, ft_dl, ratio=0.5, 
        #                   device=device, dataset=args.dataset)
        #     vectorized_mask = torch.cat([p.view(-1) for p in mask_grad_list])
            
            # mask = get_grad_mask_by_layer(net_cpy, optimizer_cp, ft_dl, device=device, 
            #                               layer=args.linear_name, dataset=args.dataset)
            # del net_cpy, optimizer_cp

        net.train()
        for batch_idx, (x, labels) in tqdm(enumerate(ft_dl), desc=f'Epoch [{epoch + 1}/{f_epochs}]: '):
            optimizer.zero_grad()
            x, labels = x.to(device), labels.to(device).float()
            log_probs = net(x).squeeze()

            if lbs_criterion is not None:
                loss = lbs_criterion(log_probs, labels)
            elif ft_mode == 'fst':
                loss = torch.sum(eval(f'net.{args.linear_name}.weight') * weight_mat_ori)*args.alpha + criterion(log_probs, labels)
            elif ft_mode == 'proposal':
                # try new loss
                # loss = masked_feature_shift_loss(eval(f'net.{args.linear_name}.weight'), weight_mat_ori, mask)*args.alpha + criterion(log_probs, labels)
                loss = criterion(log_probs, labels).mean()
                # loss = criterion(log_probs, labels)
            else:
                loss = criterion(log_probs, labels).mean()

            loss.backward()
            optimizer.step()
            
            # mask_grad_list = get_batch_grad_mask(net, device=device, ratio=0.005, opt="top")
            # vectorized_mask = torch.cat([p.view(-1) for p in mask_grad_list])
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
        print("\n--------Normal Testing --------- ")
        loss_c, acc_c = test(net, test_dl, device)

        print("\n--------Backdoor Testing --------- ")
        # test_loader = get_backdoor_loader(DESTPATH)
        loss_bd, acc_bd, correct, poison_data_count = test_backdoor(net, backdoor_dl, 
                                                                    device, 
                                                                    args.target_label)
        writer.add_scalar('Validation Clean ACC', acc_c, epoch)
        writer.add_scalar('Validation Backdoor ACC', acc_bd, epoch)

        metric_info = {
            'clean acc': acc_c,
            'clean loss': loss_c,
            'backdoor acc': acc_bd,
            'backdoor loss': loss_bd,
        }
        cur_clean_acc = metric_info['clean acc']
        cur_adv_acc = metric_info['backdoor acc']
        logger.info('*****************************')
        logger.info(colored(f'Fine-tunning mode: {ft_mode}', "green"))
        logger.info(f"Test Set: Clean ACC: {round(cur_clean_acc, 2)}% |\t ASR: {round(cur_adv_acc, 2)}%.")
        logger.info('*****************************')
        
    return net, acc_c, acc_bd
    
def main():
    ### 1. config args, save_path, fix random seed
    start_time  = datetime.datetime.now()
    
    # Create the argument parser
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
    file_to_load = f"fam_{args.subset_family}_tgt_{args.target_label}_epochs_{args.epochs}_ft_size_{args.ft_size}_lr_{args.lr}_poison_rate_{round(args.poison_rate, 4)}"
    file_to_load = f'{args.folder_path}/backdoor/{file_to_load}.pth'

    # logFormatter = logging.Formatter(
    #     fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
    #     datefmt='%Y-%m-%d:%H:%M:%S',
    # )
    # logger = logging.getLogger()

    # if args.log:
    #     fileHandler = logging.FileHandler(args.save_path + '.log')
    #     fileHandler.setFormatter(logFormatter)
    #     logger.addHandler(fileHandler)

    # consoleHandler = logging.StreamHandler()
    # consoleHandler.setFormatter(logFormatter)
    # logger.addHandler(consoleHandler)

    # logger.setLevel(logging.INFO)
    # logging.info(pformat(args.__dict__))

    parent_p = "datasets/apg"

    bd_config = get_jigsaw_config(DATAPATH)

    args.n_features = 10000
    args.seed = SEED

    # train_dl, test_dl, ft_loader, X_train, y_train, X_test, y_test = load_apg_data_loaders(parent_p, batch_size=args.batch_size, ft_size=args.ft_size)
    
    POSTFIX = f'{args.dataset}/{args.model}'

    POISONED_MODELS_FOLDER = os.path.join('models', POSTFIX)
    
    X_train, y_train, X_test, y_test, X_subset, X_test_benign, X_test_remain_mal = pre_split_apg_datasets(args, bd_config, parent_p, POISONED_MODELS_FOLDER, args.ft_size, seed=SEED)
    train_dl, test_dl, ft_loader, testloader_mal, X_subset_trojaned = load_apg_subset_data_loaders(args, parent_p, batch_size=args.batch_size, 
                                                                                ft_size=args.ft_size, subset_family=args.subset_family)
    # backdoor_test_dl = get_backdoor_loader(DESTPATH)

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

    # _, backdoor_test_dl = get_apg_backdoor_data(args, bd_config, net, X_train, X_test, y_train, y_test)
    backdoor_test_dl = testloader_mal

    original_linear_norm = torch.norm(eval(f'net.{args.linear_name}.weight'))
    weight_mat_ori = eval(f'net.{args.linear_name}.weight.data.clone().detach()')

    ori_net = copy.deepcopy(net)
    ori_net.eval()

    print("\n--------Normal Testing --------- ")
    loss_c, acc_c = test(net, test_dl, device)
    print("\n--------Backdoor Testing --------- ")
    loss_bd, acc_bd, correct_bd, poison_data_count = test_backdoor(net, backdoor_test_dl, device, 
                                                                args.target_label)
    metric_info = {
        'clean acc': acc_c,
        'clean loss': loss_c,
        'backdoor acc': acc_bd,
        'backdoor loss': loss_bd,
    }
    cur_clean_acc = metric_info['clean acc']
    cur_adv_acc = metric_info['backdoor acc']

    logger.info('*****************************')
    logger.info(f"Load from {args.folder_path}")
    # logging.info(f'Fine-tunning mode: {args.ft_mode}')
    logger.info('Original performance')
    logger.info(f"Test Set: Clean ACC: {cur_clean_acc} | ASR: {cur_adv_acc}")
    logger.info('*****************************')

    # ---------- Start Fine-tuning ---------- #
    logging_path = f'{args.log_dir}/fam_{args.subset_family}_target_{args.attack_target}-archi_{args.model}-dataset_{args.dataset}--f_epochs_{args.f_epochs}--f_lr_{args.f_lr}/ft_size_{args.ft_size}_p_rate{round(args.poison_rate, 4)}'
    ft_modes = ['ft', 'ft-init', 'fe-tuning', 'lp', 'fst', 'proposal']
    # ft_modes = ['ft']
    # ft_modes = ['proposal']
    ft_results = {}
    for ft_mode in ft_modes:
        ft_results[ft_mode] = {}
        net.load_state_dict(state_dict)
        net.to(device)
        logger.info(colored("\n--------Normal Testing --------- ", "green"))
        loss_c, acc_c = test(net, test_dl, device)
        logger.info(colored("\n--------Backdoor Testing --------- ", "red"))
        loss_bd, acc_bd, correct_bd, poison_data_count = test_backdoor(net, backdoor_test_dl, device, 
                                                                    args.target_label)
        optimizer, criterion = get_optimizer(net, ft_mode, args.linear_name, 
                                             2*args.f_lr, 
                                             dataset = args.dataset)

        if ft_mode != 'ft-sam':
            ft_net, acc_c, acc_bd = finetune(net, optimizer, criterion, 
                            ft_loader, test_dl, backdoor_test_dl, 
                            args.f_epochs, ft_mode, device, logger, 
                            logging_path, args=args,
                            weight_mat_ori=weight_mat_ori, 
                            original_linear_norm=original_linear_norm)
        else:
            ft_net = finetune_sam(net, ft_loader, test_dl, backdoor_test_dl,
                                  criterion, args.f_epochs, 
                                  device, logging_path, 
                                  args=args, base_optimizer=optimizer,
                                  weight_mat_ori=weight_mat_ori, 
                                  original_linear_norm=original_linear_norm,
                                  test=test, test_backdoor=test_backdoor)
        args.save = True
        ft_results[ft_mode]["clean_acc"] = acc_c
        ft_results[ft_mode]["adv_acc"] = acc_bd
        if args.save:
            model_save_path = f'{args.folder_path}/fam_{args.subset_family}_target_{args.attack_target}-archi_{args.model}-dataset_{args.dataset}--f_epochs_{args.f_epochs}--f_lr_{args.f_lr}/ft_size_{args.ft_size}_p_rate{round(args.poison_rate, 4)}/mode_{ft_mode}'
            os.makedirs(model_save_path, exist_ok=True)
            torch.save(ft_net.state_dict(), f'{model_save_path}/checkpoint.pt')
            # torch.save(ft_net.state_dict(), f'{model_save_path}/checkpoint.pt')
            logger.info(f"Model saved at {model_save_path}/checkpoint.pt")
            
        # # --------- * Final Evaluation * --------- #
        
        # # --------- * **************** * --------- #
        # final_eval_result = final_evaluate(ft_net, X_subset_trojan=X_subset_trojaned,
        #                                 X_test_remain_mal_trojan=X_test_remain_mal,
        #                                 X_test=X_test, y_test=y_test, 
        #                                 X_test_benign_trojan=X_test_benign, 
        #                                 subset_family=args.subset_family, 
        #                                 troj_type="Subset", log_path=model_save_path)

    end_time = datetime.datetime.now()
    # Print table
    data = []
    for mode, result in ft_results.items():
        data.append([mode, result["clean_acc"], result["adv_acc"], args.ft_size, args.poison_rate])

    # compare proposal with all ft methods
    verified = []
    for ft_mode in ft_results.keys():
        if ft_mode == 'proposal':
            continue
        if ft_results[ft_mode]['adv_acc'] >= ft_results['proposal']['adv_acc']:
            verified.append(True)
        else:
            verified.append(False)
    print("\n------- Fine-tuning Evaluation -------")
    print(tabulate(data, headers=["Mode", "Clean Accuracy", "Adversarial Accuracy", "FT size", "PDR"], tablefmt="grid"))
    end_time = datetime.datetime.now()
    print(f"Verified outperforms of PBP: {verified}")
    print(f"Completed in: {end_time - start_time} seconds.")
    print("------- ********************** -------\n")
      

if __name__ == '__main__':
    main()