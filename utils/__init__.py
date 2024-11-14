from .dataloader import get_dataloader,get_tune_dataloader,BigDataLoader
import os
import json
import torch
import torch.nn as nn
import math
import argparse
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

def get_masked_tensor(x,mask,max_length = None):
    '''
    get mask position
    - x : input tensor (batch_size, seq_len)
    - mask : mask position (batch_size, seq_len)
    '''
    x_masked = [x[i,~mask[i]] for i in range(x.shape[0])]
    #pad to max length
    x_masked = nn.utils.rnn.pad_sequence(x_masked,batch_first=True,padding_value=0)
    #clip to max length
    if max_length is not None:
        x_masked = x_masked[:,:max_length]
    return x_masked

def masked_mse_loss(preds, targets, mask):
    # 计算每个元素的差值平方
    squared_diff = (preds - targets)**2
    # 将填充位置的损失置零
    squared_diff = squared_diff * mask
    # 计算平均损失（忽略填充位置）
    loss = torch.sum(squared_diff) / torch.sum(mask)
    
    return loss

def args2json(args):
    import json
    args_dict = vars(args)
    with open(os.path.join(args.PROJECT_DIR, 'args.json'), 'w') as fp:
        json.dump(args_dict, fp, indent=4, separators=(',', ': '))
    return

def save_ckpt(args, epoch, model, optimizer, scheduler=None, losses = None,iter = '',Name = None, adding = ''):
    """
    保存模型checkpoint
    """

    ckpt_folder = os.path.join(args.PROJECT_DIR, 'ckpt/')
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    if scheduler is not None:
        if Name is not None:
            save_dir = f'{ckpt_folder}{Name}.pth'
        else:
            save_dir = f'{ckpt_folder}{args.project_name}_ep{epoch}_iter{iter}.pth'
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'losses': losses,
            },
            save_dir
        )
    else:
        if Name is not None:
            save_dir = f'{ckpt_folder}{Name}.pth'
        else:
            save_dir = f'{ckpt_folder}{args.project_name}_ep{epoch}_iter{iter}_{adding}.pth'
        if optimizer is not None:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'losses': losses,
                },
                save_dir
            )
        else:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'losses': losses,
                },
                save_dir
            )

def load_ckpt(args, model, optimizer, scheduler=None):
    """
    加载模型checkpoint
    """
    
    #尝试加载args.ckpt_path
    assert(args.ckpt_path is not None)
    print(f'load ckpt from {args.ckpt_path}')
    ckpt = torch.load(args.ckpt_path)
    load_info = model.load_state_dict(ckpt['model_state_dict'])
    print(load_info)
    if args.load_optimizer:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in ckpt:
            try:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                print('scheduler load success')
            except:
                print('scheduler load failed')
        else:
            print('Schaduler is none or No scheduler in ckpt')

    epoch = ckpt['epoch']
    losses = ckpt['losses']
    return epoch, model, optimizer, scheduler, losses,load_info

def save_gene2id(args,gene2id_dict:dict,name = 'gene2id.json'):
    gene2id_path = os.path.join(args.PROJECT_DIR,name)
    with open(gene2id_path,'w') as f:
        json.dump(gene2id_dict,f)

def load_gene2id(path):
    with open(path,'r') as f:
        gene2id_dict = json.load(f)
    return gene2id_dict

class write_log():
    def __init__(self, logfile):
        self.logfile = logfile

        if not os.path.exists(os.path.dirname(self.logfile)):
            os.makedirs(os.path.dirname(self.logfile))

        self.start_sep()

    def __call__(self, log):
        with open(self.logfile, 'a') as f:
            f.write(log + '\n')

    def start_sep(self):
        with open(self.logfile, 'a') as f:
            f.write('==============================start==============================\n')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
    
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="",num_big_batches = None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.num_big_batches = num_big_batches
        if num_big_batches is not None:
            self.big_batch_fmtstr = self._get_batch_fmtstr(num_big_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, big_batch=None):
        if self.num_big_batches is not None:
            entries = [self.prefix + self.big_batch_fmtstr.format(big_batch) + self.batch_fmtstr.format(batch)]
        else:
            entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))
        return "\t".join(entries)
    
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

    
def init_project(args,title = 'train'):
    #create project folder
    args.ROOT_DIR = os.path.join(args.ROOT,f'runs/{title}/')
    args.PROJECT_DIR = os.path.join(args.ROOT_DIR,args.project_name)
    if not os.path.exists(args.PROJECT_DIR):
        os.makedirs(args.PROJECT_DIR)
    args.LOG_DIR = os.path.join(args.PROJECT_DIR,'log.txt')
    logfile = write_log(args.LOG_DIR)
    logfile(str(args))
    return logfile

def load_config(config_path,only_model_config = False):
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    new_args = argparse.Namespace()
    if only_model_config:
        model_keys = [
            "max_gene_num",
            "embedding_dim",
            "n_pathway",
            "n_head",
            "encoder_cross_attn_depth",
            "encoder_self_attn_depth",
            "encoder_projection_dim",
            "decoder_extract_block_depth",
            "decoder_selfattn_block_depth",
            "decoder_projection_dim",
            ]
        for key in model_keys:
            new_args.__dict__[key] = config[key]
    else:
        for key, value in config.items():
            new_args.__dict__[key] = value
    return new_args
    #     for key in model_keys:
    #         args.__dict__[key] = config[key]
    # else:
    #     for key, value in config.items():
    #         args.__dict__[key] = value
    # return args

def check_device(args):
    if args.device.startswith('cuda'):
        if args.device == 'cuda':
            assert torch.cuda.is_available(), 'cuda is not available'
            assert len(args.device_num)>0, 'no cuda device is given'
            args.device = f'cuda:{args.device_num[0]}'

    else:
        args.device = 'cpu'
    return args