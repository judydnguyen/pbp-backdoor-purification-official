import argparse
import json
import logging
import math
import torch
import torch.nn as nn
from torch import optim

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=10, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.cls - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="config.json", help="Path to JSON config file.")
    args, _ = parser.parse_known_args()  # Parse only known arguments
    with open(args.config, 'r') as f:
        config_from_file = json.load(f)
    
    # Load default arguments
    default_config = {
        'device': None,
        'ft_mode': 'all',
        'num_classes': 25,
        'attack_label_trans': 'all2one',
        'pratio': None,
        'epochs': None,
        'dataset': None,
        'dataset_path': '../data',
        'folder_path': '../models',
        'attack_target': 0,
        'batch_size': 128,
        'lr': None,
        'random_seed': 0,
        'model': None,
        'split_ratio': None,
        'log': False,
        'initlr': None,
        'pre': False,
        'save': False,
        'linear_name': 'linear',
        'lb_smooth': None,
        'alpha': 0.2,
        'lr_scheduler': "StepLR",
        'rho_max': 8.0,
        'rho_min': 2.0,
        "steplr_stepsize": 1,
        "steplr_gamma": 0.9,
        "adaptive": False,
        "num_workers": 16,
    }
    
    # Update default arguments with values from config file
    default_config.update(config_from_file)
    
    # Add arguments to the parser
    for key, value in default_config.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)
    print(f"parser: {parser}")
    return parser

def get_optimizer(net, ft_mode, linear_name, f_lr, dataset="malimg"):
    if ft_mode == 'fe-tuning':
        init = True
        log_name = 'FE-tuning'
    elif ft_mode == 'ft-init':
        init = True
        log_name = 'FT-init'
    elif ft_mode == 'ft':
        init = False
        log_name = 'FT'
    elif ft_mode == 'lp':
        init = False
        log_name = 'LP'
    elif ft_mode == 'fst':
        init = True
        log_name = 'FST'
    elif ft_mode == 'proposal':
        init = False
        log_name = 'proposal'
    elif ft_mode == 'ft-sam':
        init = False
        log_name = 'FT-SAM'
    else:
        raise NotImplementedError('Not implemented method.')

    param_list = []
    for name, param in net.named_parameters():
        if linear_name in name:
            if init:
                if 'weight' in name:
                    logging.info(f'Initialize linear classifier weight {name}.')
                    std = 1 / math.sqrt(param.size(-1)) 
                    param.data.uniform_(-std, std)
                    
                else:
                    logging.info(f'Initialize linear classifier weight {name}.')
                    param.data.uniform_(-std, std)
        if ft_mode == 'lp':
            if linear_name in name:
                param.requires_grad = True
                param_list.append(param)
            else:
                param.requires_grad = False
        elif ft_mode in ['ft', 'fst', 'ft-init', 'proposal', 'ft-sam']:
            param.requires_grad = True
            param_list.append(param)
        elif ft_mode == 'fe-tuning':
            if linear_name not in name:
                param.requires_grad = True
                param_list.append(param)
            else:
                param.requires_grad = False
    if ft_mode != 'ft-sam':
        optimizer = optim.Adam(param_list, lr=f_lr)
    elif ft_mode == 'proposal':
        optimizer = optim.Adam(param_list, lr=f_lr, weight_decay=1e-4)
    else:
        optimizer = optim.SGD(param_list, lr=f_lr, 
                              momentum=0.9, 
                              weight_decay=1e-4)
    
    if dataset in ["ember", "apg"]:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    return optimizer, criterion