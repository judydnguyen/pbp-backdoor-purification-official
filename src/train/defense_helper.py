import copy
import math
import numpy as np

from termcolor import colored
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from utils import logger

SEED=12

def vectorize_net(net):
    return torch.cat([p.view(-1) for p in net.parameters()])


def load_model_weight(net, weight):
    index_bias = 0
    for p in net.parameters():
        p.data =  weight[index_bias:index_bias+p.numel()].view(p.size())
        index_bias += p.numel()

# def add_noise(model, device, stddev=0.5):
#     logger.info("adding noise")
#     vectorized_net = vectorize_net(model)

#     gaussian_noise = torch.randn(vectorized_net.size(),
#                         device=device) * stddev
#     logger.info(f"gaussian_noise: {gaussian_noise}")
#     dp_weight = vectorized_net + gaussian_noise
#     load_model_weight(model, dp_weight)
#     return model


def add_noise_w(model, device, stddev=0.5):
    """
    Add zero-mean Gaussian noise to the weights of the model parameters to avoid bias.

    Parameters:
    model (torch.nn.Module): The neural network model to be modified.
    device (torch.device): The device on which the model is stored (e.g., 'cpu' or 'cuda').
    stddev (float): The standard deviation of the Gaussian noise. Default is 0.5.

    Returns:
    model (torch.nn.Module): The model with added noise.
    """
    logger.info("Adding noise to model weights")

    # Initialize a list to collect the noisy weights
    noisy_weights = []

    for name, param in model.named_parameters():
        if 'weight' in name:  # Check if the parameter is a weight and not a bias
            noise = torch.normal(mean=0, std=stddev, size=param.size(), device=device, generator=torch.Generator(device=device).manual_seed(SEED))
            noisy_weights.append(param.view(-1) + noise.view(-1))  # Add noise to the weight and flatten
        else:
            noisy_weights.append(param.view(-1))  # Just flatten the biases

    # Concatenate all the parameters back into a single vector
    vectorized_net = torch.cat(noisy_weights)

    load_model_weight(model, vectorized_net)
    return model

def init_noise(model, device, stddev=0.5):
    vectorized_net = vectorize_net(model)
    # gaussian_noise = torch.randn(vectorized_net.size(),
    #                     device=device) * stddev
    gaussian_noise = torch.normal(mean=0, std=stddev, size=vectorized_net.size(), device=device, generator=torch.Generator(device=device).manual_seed(SEED))
    load_model_weight(model, gaussian_noise)
    return model

# def init_noise(model, device, stddev=0.5):
#     """
#     Add zero-mean Gaussian noise to the weights of the model parameters to avoid bias.

#     Parameters:
#     model (torch.nn.Module): The neural network model to be modified.
#     device (torch.device): The device on which the model is stored (e.g., 'cpu' or 'cuda').
#     stddev (float): The standard deviation of the Gaussian noise. Default is 0.5.

#     Returns:
#     model (torch.nn.Module): The model with added noise.
#     """
#     logger.info("Adding zero-mean Gaussian noise to model weights")

#     for name, param in model.named_parameters():
#         if 'weight' in name:  # Check if the parameter is a weight and not a bias
#             noise = torch.randn(param.size(), device=device) * stddev
#             param.data.add_(noise)  # Add noise directly to the parameter data

#     return model


def apply_robust_LR(model, prev_model, vectorized_mask):
    logger.info(colored("applying robust LR ...", "red"))
    # logger.info(f"original vectorized_mask: {vectorized_mask}")
    tmp_mask = vectorized_mask.clone()
    
    # tmp_mask[torch.where(tmp_mask==1)[0].tolist()] = -1
    # tmp_mask[torch.where(tmp_mask==0)[0].tolist()] = 1
    
    tmp_mask[torch.where(tmp_mask==1)[0].tolist()] = -1
    tmp_mask[torch.where(tmp_mask==0)[0].tolist()] = 1
    # tmp_mask[torch.where(tmp_mask==-1)[0].tolist()] = 0
    
    # vectorized_mask = -vectorized_mask+ 1
    # logger.info(f"vectorized_mask: {vectorized_mask}")
    vectorized_net = vectorize_net(model)
    vectorized_prev_net = vectorize_net(prev_model)
    update = vectorized_net - vectorized_prev_net
    vectorized_net = vectorized_prev_net + tmp_mask*update
    # import IPython
    # IPython.embed()
    del tmp_mask
    load_model_weight(model, vectorized_net)
    return model

def reverse_LR(model, optimizer, clean_dataloader, vectorized_mask, dataset, device):
    model.train()
    prev_model = copy.deepcopy(model)
    if dataset == "malimg":
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    # Let's assume we have a model trained on clean data and we conduct aggregation for all layer
    for batch_idx, batch in tqdm(enumerate(clean_dataloader), desc="Reversing LR process..."):
        bs = len(batch)
        data, targets = batch
        clean_images, targets = copy.deepcopy(data).to(device), copy.deepcopy(targets).to(device)
        optimizer.zero_grad()
        output = model(clean_images)
        if dataset == "ember":
            output = output.squeeze()
            targets = targets.float()
        elif dataset == "apg":
            output = output.squeeze()
            targets = targets.float()
        else:
            targets = targets.long()
            
        loss_clean = loss_fn(output, targets)
        loss_clean.backward(retain_graph=True)
        optimizer.step()
        
        model = apply_robust_LR(model, prev_model, vectorized_mask)
        prev_model = copy.deepcopy(model)
    return model

def get_batch_grad_mask(model, opt="top", device="cuda", ratio=0.01):
    mask_grad_list = []
    grad_list = []
    grad_abs_sum_list = []
    k_layer = 0
    # import IPython
    # IPython.embed()
    for _, parms in model.named_parameters():
        if parms.requires_grad and parms.grad is not None:
            grad_list.append(parms.grad.abs().view(-1))
            grad_abs_sum_list.append(parms.grad.abs().view(-1).sum().item())
            k_layer += 1

    grad_list = torch.cat(grad_list).to(device)
    
    if opt == "top":
        _, indices = torch.topk(grad_list, int(len(grad_list)*ratio))
    elif opt == "bottom":
        _, indices = torch.topk(-1*grad_list, int(len(grad_list)*ratio))
    else:
        raise ValueError(f"Invalid opt value: {opt}")
    
    mask_flat_all_layer = torch.zeros(len(grad_list)).to(device)
    mask_flat_all_layer[indices] = 1.0

    count = 0
    percentage_mask_list = []
    k_layer = 0
    grad_abs_percentage_list = []
    for _, parms in model.named_parameters():
        if parms.requires_grad and parms.grad is not None:
            gradients_length = len(parms.grad.abs().view(-1))
            mask_flat = mask_flat_all_layer[count:count + gradients_length ].to(device)
            mask_grad_list.append(mask_flat.reshape(parms.grad.size()).to(device))
            count += gradients_length
            percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0
            percentage_mask_list.append(percentage_mask1)
            grad_abs_percentage_list.append(grad_abs_sum_list[k_layer]/np.sum(grad_abs_sum_list))
            k_layer += 1
    return mask_grad_list

def get_grad_mask(model, optimizer, clean_dataloader, 
                  ratio=0.1, device="cpu", 
                  save_f=False,
                  historical_grad_mask=None, 
                  cnt_masks=0, dataset="malimg",
                  opt="top"):
    """
    Generate a gradient mask based on the given dataset
    This function is employed for Neurotoxin method
    https://proceedings.mlr.press/v162/zhang22w.html
    """        

    model.train()
    model.zero_grad()
    if dataset == "malimg":
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    # Let's assume we have a model trained on clean data and we conduct aggregation for all layer
    for batch_idx, batch in enumerate(clean_dataloader):
        bs = len(batch)
        data, targets = batch
        clean_images, clean_targets = copy.deepcopy(data).to(device), copy.deepcopy(targets).to(device)
        optimizer.zero_grad()
        output = model(clean_images)
        if dataset == "ember":
            clean_targets = clean_targets.float()
            output = output.squeeze()
            targets = targets.float()
        elif dataset == "apg":
            output = output.squeeze()
            targets = targets.float()
        else:
            targets = targets.long()
            
        loss_clean = loss_fn(output, clean_targets)
        loss_clean.backward(retain_graph=True)
        optimizer.step()
        
    mask_grad_list = []
    grad_list = []
    grad_abs_sum_list = []
    k_layer = 0
    for _, parms in model.named_parameters():
        if parms.requires_grad:
            grad_list.append(parms.grad.abs().view(-1))
            grad_abs_sum_list.append(parms.grad.abs().view(-1).sum().item())
            k_layer += 1

    grad_list = torch.cat(grad_list).to(device)
    
    if opt == "top":
        _, indices = torch.topk(grad_list, int(len(grad_list)*ratio))
    elif opt == "bottom":
        _, indices = torch.topk(-1*grad_list, int(len(grad_list)*ratio))
    else:
        raise ValueError(f"Invalid opt value: {opt}")
    
    mask_flat_all_layer = torch.zeros(len(grad_list)).to(device)
    mask_flat_all_layer[indices] = 1.0

    if historical_grad_mask:
        cummulative_mask = ((cnt_masks-1)/cnt_masks)*historical_grad_mask+(1/cnt_masks)*mask_flat_all_layer
        _, indices = torch.topk(-1*cummulative_mask, int(len(cummulative_mask)*ratio))
        mask_flat_all_layer = torch.zeros(len(grad_list)).to(device)
        mask_flat_all_layer[indices] = 1.0
    else:
        cummulative_mask = copy.deepcopy(grad_list)

    count = 0
    percentage_mask_list = []
    k_layer = 0
    grad_abs_percentage_list = []
    for _, parms in model.named_parameters():
        if parms.requires_grad:
            gradients_length = len(parms.grad.abs().view(-1))
            mask_flat = mask_flat_all_layer[count:count + gradients_length ].to(device)
            mask_grad_list.append(mask_flat.reshape(parms.grad.size()).to(device))
            count += gradients_length
            percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0
            percentage_mask_list.append(percentage_mask1)
            grad_abs_percentage_list.append(grad_abs_sum_list[k_layer]/np.sum(grad_abs_sum_list))
            k_layer += 1
    
    # import IPython
    # IPython.embed()
    return mask_grad_list, cummulative_mask

def get_grad_mask_by_layer(model, optimizer, clean_dataloader, 
                           ratio=0.25, device="cpu", cnt_masks=0, 
                           layer="classifier", dataset="malimg", iterations=5):
    model.train()
    
    # print(f"dataset: {dataset}")
    if dataset == "malimg":
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()
        
    for iter in tqdm(range(iterations)):
        # Forward and backward pass to calculate gradients
        for batch_idx, (data, targets) in enumerate(clean_dataloader):
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(data)
            if dataset == "ember":
                output = output.squeeze()
                targets = targets.float()
            elif dataset == "apg":
                output = output.squeeze()
                targets = targets.float()
            else:
                targets = targets.long()
            loss_clean = loss_fn(output, targets)
            loss_clean.backward()
            optimizer.step()
        # break  # Assuming we only want gradients from the first batch

    # Find the gradients and create a mask for the specified layer
    mask_grad_list = []
    for name, param in model.named_parameters():
        # print(f"DEBUG [l:89], name is {name}, param.shape is: {param.shape}")
        if name in [f"{layer}.weight"] and param.requires_grad:
            grad = param.grad.view(-1).abs()
            # _, grad_topk = torch.topk(-1*grad, int(len(grad)*ratio))
            _, indices = torch.topk(-1*grad, int(len(grad)*ratio))
            mask_flat = torch.zeros(len(grad)).to(device)
            mask_flat[indices] = 1.0
            mask = mask_flat.view(param.size())
            print(f"Mask for layer {layer} created, ratio of masked gradients: {ratio:.2f}")
            break  # Exit the loop after processing the specified layer
    return mask


def apply_grad_mask(model, mask_grad_list):
    mask_grad_list_copy = iter(mask_grad_list)
    for name, parms in model.named_parameters():
        if parms.requires_grad:
            parms.grad = parms.grad * next(mask_grad_list_copy)

def apply_PGD(model, helper, global_model_copy):
    weight_difference, difference_flat = helper.get_weight_difference(global_model_copy, model.named_parameters())
    clipped_weight_difference, l2_norm = helper.clip_grad(helper.params['s_norm'], weight_difference, difference_flat)
    weight_difference, difference_flat = helper.get_weight_difference(global_model_copy, clipped_weight_difference)
    copy_params(model, weight_difference)

def copy_params(model, target_params_variables):
    for name, layer in model.named_parameters():
        layer.data = copy.deepcopy(target_params_variables[name])
        
def masked_feature_shift_loss(w1, w2, mask_grad_list):
    # import IPython
    # IPython.embed()
    print(f"Total restricted neurons: {sum(sum(mask_grad_list))}")
    return torch.sum(w1 * mask_grad_list * w2)

def feature_shift_loss(w1, w2):
    # import IPython
    # IPython.embed()
    # print(f"Total restricted neurons: {sum(sum(mask_grad_list))}")
    return torch.sum(w1 * w2)
    
def cmi_penalty(y, z_mu, z_sigma, reference_params):
    num_samples = y.shape[0]
    dimension = reference_params.shape[1] // 2
    target_mu = reference_params[y.to(dtype=torch.long), :dimension]
    target_sigma = F.softplus(reference_params[y.to(dtype=torch.long), dimension:])
    cmi_loss = torch.sum((torch.log(target_sigma) - torch.log(z_sigma) + (z_sigma ** 2 + (target_mu - z_mu) ** 2) / (2*target_sigma**2) - 0.5)) / num_samples
    return cmi_loss

def dp_noise(param, sigma):
    noised_layer = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)
    return noised_layer

def smooth_model(target_model, device, sigma=0.5):
    for name, param in target_model.state_dict().items():
        param.add_(dp_noise(param, sigma).to(device))