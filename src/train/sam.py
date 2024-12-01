import os
from termcolor import colored
import torch
import contextlib
from torch.distributed import ReduceOp
from tqdm import tqdm
from utils import logger
# from jigsaw.train_jigsaw_pytorch import test, test_backdoor
from torch.utils.tensorboard import SummaryWriter

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, model, sam_alpha, rho_scheduler, 
                 adaptive=False, perturb_eps=1e-12, 
                 grad_reduce='mean', **kwargs):
        defaults = dict(adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.adaptive = adaptive
        self.rho_scheduler = rho_scheduler
        self.perturb_eps = perturb_eps
        self.alpha = sam_alpha
        
        # initialize self.rho_t
        self.update_rho_t()
        
        # set up reduction for gradient across workers
        if grad_reduce.lower() == 'mean':
            if hasattr(ReduceOp, 'AVG'):
                self.grad_reduce = ReduceOp.AVG
                self.manual_average = False
            else: # PyTorch <= 1.11.0 does not have AVG, need to manually average across processes
                self.grad_reduce = ReduceOp.SUM
                self.manual_average = True
        elif grad_reduce.lower() == 'sum':
            self.grad_reduce = ReduceOp.SUM
            self.manual_average = False
        else:
            raise ValueError('"grad_reduce" should be one of ["mean", "sum"].')
    
    @torch.no_grad()
    def update_rho_t(self):
        self.rho_t = self.rho_scheduler.step()
        return self.rho_t

    @torch.no_grad()
    def perturb_weights(self, rho=0.0):
        grad_norm = self._grad_norm( weight_adaptive = self.adaptive )
        for group in self.param_groups:
            scale = rho / (grad_norm + self.perturb_eps)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_g"] = p.grad.data.clone()
                e_w = p.grad * scale.to(p)
                if self.adaptive:
                    e_w *= torch.pow(p, 2)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]['e_w'] = e_w
                
    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group['params']:
                if 'e_w' in self.state[p].keys():
                    p.data.sub_(self.state[p]['e_w'])

    @torch.no_grad()
    def gradient_decompose(self, alpha=0.0):
        # calculate inner product
        inner_prod = 0.0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                inner_prod += torch.sum(
                    self.state[p]['old_g'] * p.grad.data
                )

        # get norm
        new_grad_norm = self._grad_norm()
        old_grad_norm = self._grad_norm(by='old_g')

        # get cosine
        cosine = inner_prod / (new_grad_norm * old_grad_norm + self.perturb_eps)

        # gradient decomposition
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                vertical = self.state[p]['old_g'] - cosine * old_grad_norm * p.grad.data / (new_grad_norm + self.perturb_eps)
                p.grad.data.add_( vertical, alpha=-alpha)

    @torch.no_grad()
    def _sync_grad(self):
        if torch.distributed.is_initialized(): # synchronize final gardients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    if self.manual_average:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
                        world_size = torch.distributed.get_world_size()
                        p.grad.div_(float(world_size))
                    else:
                        torch.distributed.all_reduce(p.grad, op=self.grad_reduce)
        return

    @torch.no_grad()
    def _grad_norm(self, by=None, weight_adaptive=False):
        #shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        if not by:
            norm = torch.norm(
                    torch.stack([
                        ( (torch.abs(p.data) if weight_adaptive else 1.0) *  p.grad).norm(p=2)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        else:
            norm = torch.norm(
                torch.stack([
                    ( (torch.abs(p.data) if weight_adaptive else 1.0) * self.state[p][by]).norm(p=2)
                    for group in self.param_groups for p in group["params"]
                    if p.grad is not None
                ]),
                p=2
            )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        
    def maybe_no_sync(self):
        if torch.distributed.is_initialized():
            return self.model.no_sync()
        else:
            return contextlib.ExitStack()

    @torch.no_grad()
    def set_closure(self, loss_fn, inputs, targets, **kwargs):
        # create self.forward_backward_func, which is a function such that
        # self.forward_backward_func() automatically performs forward and backward passes.
        # This function does not take any arguments, and the inputs and targets data
        # should be pre-set in the definition of partial-function
        # print(f"loss_fn: {loss_fn}")
        def get_grad():
            self.base_optimizer.zero_grad()
            with torch.enable_grad():
                outputs = self.model(inputs).squeeze()
                loss = loss_fn(outputs, targets, **kwargs)
            loss_value = loss.data.clone().detach()
            loss.backward()
            return outputs, loss_value

        self.forward_backward_func = get_grad

    @torch.no_grad()
    def step(self, closure=None):
        if closure:
            get_grad = closure
        else:
            get_grad = self.forward_backward_func

        with self.maybe_no_sync():
            # get gradient
            outputs, loss_value = get_grad()

            # perturb weights
            self.perturb_weights(rho=self.rho_t)

            # # disable running stats for second pass
            # disable_running_stats(self.model)

            # get gradient at perturbed weights
            get_grad()

            # decompose and get new update direction
            self.gradient_decompose(self.alpha)

            # unperturb
            self.unperturb()
        
        # import IPython
        # IPython.embed()
        # synchronize gradients across workers
        self._sync_grad()    

        # update with new directions
        self.base_optimizer.step()

        # # enable running stats
        # enable_running_stats(self.model)
        return outputs, loss_value

def argparser_opt_scheduler(model, args):
    # idea: given model and args, return the optimizer and scheduler you choose to use
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.f_lr)
    # if args.client_optimizer == "sgd":
    #     optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
    #                                 lr=args.lr,
    #                                 momentum=args.sgd_momentum,  # 0.9
    #                                 weight_decay=args.wd,  # 5e-4
    #                                 )
    # elif args.client_optimizer == 'adadelta':
    #     optimizer = torch.optim.Adadelta(
    #         filter(lambda p: p.requires_grad, model.parameters()),
    #         lr=args.lr,
    #         rho=args.rho,  # 0.95,
    #         eps=args.eps,  # 1e-07,
    #     )
    # else:
    #     optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
    #                                  lr=args.lr,
    #                                  betas=args.adam_betas,
    #                                  weight_decay=args.wd,
    #                                  amsgrad=True)

    if args.lr_scheduler == 'CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                      base_lr=args.min_lr,
                                                      max_lr=args.lr,
                                                      step_size_up=args.step_size_up,
                                                      step_size_down=args.step_size_down,
                                                      cycle_momentum=False)
    elif args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=args.steplr_stepsize,  # 1
                                                    gamma=args.steplr_gamma)  # 0.92
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100 if (
                    ("cos_t_max" not in args.__dict__) or args.cos_t_max is None) else args.cos_t_max)
    elif args.lr_scheduler == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.steplr_milestones, args.steplr_gamma)
    elif args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **({
                   'factor': args.ReduceLROnPlateau_factor
               } if 'ReduceLROnPlateau_factor' in args.__dict__ else {})
        )
    else:
        scheduler = None
    return optimizer, scheduler

def get_scheduler(optimizer, args):
    if args.lr_scheduler == 'CyclicLR':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                      base_lr=args.min_lr,
                                                      max_lr=args.lr,
                                                      step_size_up=args.step_size_up,
                                                      step_size_down=args.step_size_down,
                                                      cycle_momentum=False)
    elif args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=args.steplr_stepsize,  # 1
                                                    gamma=args.steplr_gamma)  # 0.92
    elif args.lr_scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100 if (
                    ("cos_t_max" not in args.__dict__) or args.cos_t_max is None) else args.cos_t_max)
    elif args.lr_scheduler == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.steplr_milestones, args.steplr_gamma)
    elif args.lr_scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **({
                   'factor': args.ReduceLROnPlateau_factor
               } if 'ReduceLROnPlateau_factor' in args.__dict__ else {})
        )
    else:
        scheduler = None
    return scheduler

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def finetune_sam(model, train_loader, test_dl, backdoor_dl, 
                 criterion, f_epochs, device, 
                 logging_path, base_optimizer, 
                 args, 
                 weight_mat_ori, original_linear_norm,
                 ft_mode="ft-sam", test=None, 
                 test_backdoor=None):
    
    os.makedirs(logging_path, exist_ok=True)
    writer = SummaryWriter(log_dir=f'{logging_path}/log/mode_{ft_mode}')
    logger.info(f"Fine-tuning mode: {ft_mode}")

    cur_clean_acc, cur_adv_acc = 0.0, 0.0
    writer.add_scalar('Validation Clean ACC', cur_clean_acc, 0)
    writer.add_scalar('Validation Backdoor ACC', cur_adv_acc, 0)
    
    model.train()
    
    # base_optimizer, scheduler = argparser_opt_scheduler(model, args)
    scheduler = get_scheduler(base_optimizer, args)
    rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, 
                                        max_lr=args.f_lr, min_lr=args.f_lr*0.99,
                                        max_value=args.rho_max, 
                                        min_value=args.rho_min)
    optimizer = SAM(params=model.parameters(), base_optimizer=base_optimizer, 
                    model=model, sam_alpha=args.alpha, 
                    rho_scheduler=rho_scheduler, 
                    adaptive=args.adaptive)

    # import IPython
    # IPython.embed()
    
    for epoch in tqdm(range(f_epochs), desc='Fine-tuning mode: ft-sam'):
        batch_loss_list = []
        train_correct = 0
        train_tot = 0

        for idx, (img, target) in tqdm(enumerate(train_loader)):
            img, target = img.to(args.device), target.to(args.device).float()
            bsz = target.shape[0]

            optimizer.set_closure(criterion, img, target)
            predictions, loss = optimizer.step()

            with torch.no_grad():
                predicted = torch.round(torch.sigmoid(predictions))
                correct = predicted.eq(target).sum()
                # correct = correct.sum()
                # scheduler.step()
                optimizer.update_rho_t()

            # if idx % 1 == 0:
            #     print(f"Epoch {epoch}, Iteration {idx}, Loss: {loss}")

            exec_str = f'model.{args.linear_name}.weight.data = model.{args.linear_name}.weight.data * original_linear_norm  / torch.norm(model.{args.linear_name}.weight.data)'
            exec(exec_str)

            log_probs = model(img).squeeze()
            predicted = torch.round(torch.sigmoid(log_probs))
            train_correct += predicted.eq(target).sum()
            train_tot += target.size(0)
            batch_loss = loss.item() * target.size(0)
            batch_loss_list.append(batch_loss)

        one_epoch_loss = sum(batch_loss_list)
        scheduler.step()
        logger.info(f'Training ACC: {train_correct/train_tot} | Training loss: {one_epoch_loss}')
        logger.info(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
        logger.info('-------------------------------------')

        writer.add_scalar('Training Loss', one_epoch_loss, epoch)
        writer.add_scalar('Training Accuracy', train_correct/train_tot, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]["lr"], epoch)

        logger.info(colored(f"Start validation for current epoch: [{epoch}/{args.f_epochs}]\n", "blue"))
        print("\n--------Normal Testing --------- ")
        loss_c, acc_c = test(model, test_dl, device)

        print("\n--------Backdoor Testing --------- ")
        # test_loader = get_backdoor_loader(DESTPATH)
        loss_bd, acc_bd, correct, poison_data_count = test_backdoor(model, backdoor_dl, 
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
        logger.info(colored('Fine-tunning mode: ft-sam', "green"))
        logger.info(f"Test Set: Clean ACC: {round(cur_clean_acc*100.0, 2)}% |\t ASR: {round(cur_adv_acc, 2)}%.")
        logger.info('*****************************')
    return model, acc_c, acc_bd

import math
import numpy as np

class ProportionScheduler:
    def __init__(self, pytorch_lr_scheduler, max_lr, min_lr, max_value, min_value):
        """
        This scheduler outputs a value that evolves proportional to pytorch_lr_scheduler, e.g.
        (value - min_value) / (max_value - min_value) = (lr - min_lr) / (max_lr - min_lr)
        """
        self.t = 0    
        self.pytorch_lr_scheduler = pytorch_lr_scheduler
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_value = max_value
        self.min_value = min_value
        
        assert (max_lr > min_lr) or ((max_lr==min_lr) and (max_value==min_value)), "Current scheduler for `value` is scheduled to evolve proportionally to `lr`," \
        "e.g. `(lr - min_lr) / (max_lr - min_lr) = (value - min_value) / (max_value - min_value)`. Please check `max_lr >= min_lr` and `max_value >= min_value`;" \
        "if `max_lr==min_lr` hence `lr` is constant with step, please set 'max_value == min_value' so 'value' is constant with step."
    
        assert max_value >= min_value
        
        self.step() # take 1 step during initialization to get self._last_lr
    
    def lr(self):
        return self._last_lr[0]
                
    def step(self):
        self.t += 1
        if hasattr(self.pytorch_lr_scheduler, "_last_lr"):
            lr = self.pytorch_lr_scheduler._last_lr[0]
        else:
            lr = self.pytorch_lr_scheduler.optimizer.param_groups[0]['lr']
            
        if self.max_lr > self.min_lr:
            value = self.min_value + (self.max_value - self.min_value) * (lr - self.min_lr) / (self.max_lr - self.min_lr)
        else:
            value = self.max_value
        
        self._last_lr = [value]
        return value
        
class SchedulerBase:
    def __init__(self, T_max, max_value, min_value=0.0, init_value=0.0, warmup_steps=0, optimizer=None):
        super(SchedulerBase, self).__init__()
        self.t = 0
        self.min_value = min_value
        self.max_value = max_value
        self.init_value = init_value
        self.warmup_steps = warmup_steps
        self.total_steps = T_max
        
        # record current value in self._last_lr to match API from torch.optim.lr_scheduler
        self._last_lr = [init_value]
                
        # If optimizer is not None, will set learning rate to all trainable parameters in optimizer.
        # If optimizer is None, only output the value of lr.
        self.optimizer = optimizer

    def step(self):
        if self.t < self.warmup_steps:
            value = self.init_value + (self.max_value - self.init_value) * self.t / self.warmup_steps
        elif self.t == self.warmup_steps:
            value = self.max_value
        else:
            value = self.step_func()
        self.t += 1

        # apply the lr to optimizer if it's provided
        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = value
                
        self._last_lr = [value]
        return value

    def step_func(self):
        pass
    
    def lr(self):
        return self._last_lr[0]

class LinearScheduler(SchedulerBase):
    def step_func(self):
        value = self.max_value + (self.min_value - self.max_value) * (self.t - self.warmup_steps) / (
                    self.total_steps - self.warmup_steps)
        return value

class CosineScheduler(SchedulerBase):
    def step_func(self):
        phase = (self.t-self.warmup_steps) / (self.total_steps-self.warmup_steps) * math.pi
        value = self.min_value + (self.max_value-self.min_value) * (np.cos(phase) + 1.) / 2.0
        return value

class PolyScheduler(SchedulerBase):
    def __init__(self, poly_order=-0.5, *args, **kwargs):
        super(PolyScheduler, self).__init__(*args, **kwargs)
        self.poly_order = poly_order
        assert poly_order<=0, "Please check poly_order<=0 so that the scheduler decreases with steps"

    def step_func(self):
        value = self.min_value + (self.max_value-self.min_value) * (self.t - self.warmup_steps)**self.poly_order
        return value



