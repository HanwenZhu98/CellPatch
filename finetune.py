import argparse
import torch
import time
import os
import numpy as np


from torch.optim import Adam,SGD,AdamW,lr_scheduler,RAdam
import torch.nn as nn
import torch.nn.functional as F
from Model import Model_lite as Model
from utils import *
from utils.transfer_tools import load_basic_ckpt
from sklearn.metrics import f1_score as cal_f1

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()

parser.add_argument("--config_path", type = str, default = None, help = 'config file path')

#- project configs
parser.add_argument("--project_name", type = str, default = 'demo', help = 'project name')
parser.add_argument("--ckpt_path", type = str, default = './data/CellPatch_50ep.pth', help = 'checkpoint path')
# parser.add_argument("--ckpt_path", type = str, default = None, help = 'checkpoint path')
parser.add_argument("--load_optimizer", type = bool, default = True, help = 'load optimizer from ckpt')

#- trainning configs
parser.add_argument("--epoch", type = int, default = 100, help = 'epoch number')
parser.add_argument("--save_each_iters", type = int, default = 100000, help = 'save ckpt each iters')
parser.add_argument("--print_every", type = int, default = 100, help = 'print log each iters')

#- data configs
parser.add_argument("--gene2id_json_path", type = str, default ='./data/gene2tok.json', help = 'json file path storing gene to index pairs')
parser.add_argument("--data_path", type=str, default='./data/scbert_preprocessed_data.h5ad',help='Path of data for pretrain.')
parser.add_argument("--batch_size",type = int, default = 256,help = 'batch size of data loader')
parser.add_argument("--mask_rate", type = float, default = 0.3, help = 'mask rate of single cell non zero values')
parser.add_argument("--log_transform", type = bool, default = False, help = 'log transform count data')
parser.add_argument("--balance", type = bool, default = False, help = 'balance dataset')
parser.add_argument("--long", type = bool, default = True, help = 'balance dataset')
parser.add_argument("--clip", type = int, default = 5, help = 'number of class in dataset')


parser.add_argument("--linear_probing", type = bool, default = False, help = 'linear probing')
parser.add_argument("--project_out", type = bool, default = False, help = 'project out the last layer of model')
#- device configs
parser.add_argument("--device", type=str, default='cuda', help='Device for training.')
parser.add_argument("--device_num", type=int, default=[0], help='Device for training.')

# #- model configs
parser.add_argument("--max_gene_num",type = int, default=70000,help='max gene numbers of all dataset')
parser.add_argument("--embedding_dim",type = int, default = 32, help = 'embedding dimention of layers')
parser.add_argument("--max_trainning_length", type = int, default = 3000, help = 'max length of data for trainning encodeing , shorter 2 train faster')
parser.add_argument("--max_decode_length", type = int, default = 1000, help = 'max length of data for trainning decodeing , shorter 2 train faster')
parser.add_argument("--n_pathway", type = int , default = 64, help = 'pathway number setting in model , control preset token in model to present pathway')
parser.add_argument("--n_head", type = int, default = 2, help = "multi-head attention's head number ")
parser.add_argument("--encoder_cross_attn_depth", type = int, default =1, help = 'cross attention layer number for extract information from cell to pathway embedding')
parser.add_argument("--encoder_self_attn_depth", type = int, default =2, help = 'pathway embedding self attention layer number')
parser.add_argument("--encoder_projection_dim", type = int, default =None, help = 'encoder output feature dimention, output n_token * projection_dim')
parser.add_argument("--decoder_extract_block_depth",type = int, default = 1, help = 'decoder cross attention layer number for extract information from embedding to each gene')
parser.add_argument("--decoder_selfattn_block_depth", type = int, default = 1, help = " decoder self attention layer number for gene embedding")
parser.add_argument("--decoder_projection_dim", type = int, default = 1, help = 'decoder output feature dimention, output n_token * projection_dim')

#- hyperparamters
parser.add_argument("--optimizer", type =str ,default = 'adam', help='optimizers : adam, sgd, adamW')
parser.add_argument("--learnning_rate", type = float, default = 1e-3, help = 'learnning rate of optimizer')
parser.add_argument("--weight_decay", type = float , default = 5e-4, help = 'lr decay rate')
parser.add_argument("--dropout", type = float, default = 0., help = 'dropout rate')

def train_one_epoch(args,model,optimizer,scheduler,loss_fn,dataset_train,epoch,logfile,grad_log = None):
    if scheduler is not None:
        scheduler.step()
        print('lr :',optimizer.param_groups[0]['lr'])
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_total = AverageMeter('loss', ':.4e')
    accuracy = AverageMeter('accuracy',':.4e')
    learning_rate = AverageMeter('lr',':.4e')
    progress = ProgressMeter(
        len(dataset_train),
        [batch_time, data_time, loss_total, accuracy, learning_rate],
        prefix="Epoch: [{}]".format(epoch))
    
    data_end = time.time()
    end = time.time()
    learning_rate.update(optimizer.param_groups[0]['lr'])

    for idx,(d_) in enumerate(dataset_train):
        count,gene,label = d_
        count,gene,label = count.to(args.device),gene.to(args.device),label.to(args.device)
        # print(count)
        pred_class = model(count,gene)
        loss = loss_fn(pred_class,label)

        optimizer.zero_grad()
        loss.backward()

        # if grad is None , grad = 0
        
        if grad_log is not None:
            grad_info = '%s\t%s\t%.4f\t'%(epoch,idx,loss.item())
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_info += '\t' + str(param.grad.mean().item())
                else:
                    grad_info += '\t' + '0'
            grad_log(grad_info)

        optimizer.step()

        # print('backward & optimizer step time :',time.time()-t_0)
        # t_0 = time.time()
        accuracy.update((torch.argmax(pred_class,dim=1) == label).sum().item() / label.shape[0],count.shape[0])
        loss_total.update(loss.item(),count.shape[0])
        data_time.update(time.time() - data_end)
        batch_time.update(time.time() - end)
        end = time.time()
        # print('update time :',time.time()-t_0)
        # t_0 = time.time()
        if (idx+1) % args.print_every == 0 :
            info_ = progress.display(idx)
            logfile(info_)

        if (idx+1) % args.save_each_iters == 0:
            logfile('save at iter {}'.format(idx+1))
            # save_ckpt(args, epoch, model, optimizer, scheduler=scheduler, losses = loss_total.avg,iter=idx+1)
        # print('save time :',time.time()-t_0)

    info_ = progress.display(idx)
    logfile(info_)

    # #save when epoch end
    # save_ckpt(args, epoch, model, optimizer, scheduler=scheduler, losses = loss_total.avg)
    # if scheduler is not None:
    #     scheduler.step()

def eval_one_epoch(args,model,loss_fn,dataset_eval,epoch,logfile, test = False,testloader = None):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    loss_total = AverageMeter('loss', ':.4e')
    accuracy = AverageMeter('accuracy',':.4e')
    # f1_score = AverageMeter('f1_score',':.4e')

    progress = ProgressMeter(
        len(dataset_eval),
        [batch_time, data_time, loss_total,accuracy],
        prefix="Epoch: [{}]".format(epoch))
    
    data_end = time.time()
    end = time.time()

    prob_all = []
    label_all = []

    with torch.no_grad():
        for idx,(d_) in enumerate(dataset_eval):
            count,gene,label = d_
            count,gene,label = count.to(args.device),gene.to(args.device),label.to(args.device)
            pred = model(count,gene)
            loss = loss_fn(pred,label)
            acc = (torch.argmax(pred,dim=1) == label).sum().item() / label.shape[0]

            prob_all.extend(np.argmax(pred.cpu().detach().numpy(),axis=1))
            label_all.extend(label.cpu().detach().numpy())

            loss_total.update(loss.item(),count.shape[0])
            data_time.update(time.time() - data_end)
            batch_time.update(time.time() - end)
            accuracy.update(acc,count.shape[0])
            # f1_score.update(f1,count.shape[0])
            end = time.time()
        f1_score = cal_f1(label_all,prob_all,average='macro')
        info_ = progress.display(idx)
        logfile(info_)
        
        if not test:
            # if args.min_loss > loss_total.avg:
            if True:
                # args.min_loss = loss_total.avg
                if args.val_min_loss > loss_total.avg:
                    args.val_min_loss = loss_total.avg
                if args.val_max_acc < accuracy.avg:
                    args.val_max_acc = accuracy.avg
                    # args.best_val_acc = accuracy.avg
                    #save best ckpt
                    # save_ckpt(args, epoch, model, optimizer=None, scheduler=None, losses = loss_total.avg,iter=idx+1, Name = 'best')
                #test 
                args.best_test_acc = eval_one_epoch(args,model,loss_fn,testloader,epoch,logfile,test = True)
                # save_ckpt(args = args, epoch = epoch, model = model, optimizer=None, scheduler=None, losses = loss_total.avg,iter=idx+1, Name = 'best')
            # logfile('eval loss : {}, accuracy : {}'.format(loss_total.avg,accuracy.avg))
            logfile('epoch : {}, val loss : {}, val acc : {}, val f1 : {}'.format(epoch,loss_total.avg,accuracy.avg,f1_score))
            return args.best_test_acc

        else:
            if args.test_min_loss > loss_total.avg:
                args.test_min_loss = loss_total.avg
            if args.test_max_acc < accuracy.avg:
                args.test_max_acc = accuracy.avg
            # logfile('beat test loss : {}, accuracy : {}'.format(loss_total.avg,accuracy.avg))
            logfile('epoch : {}, test loss : {}, test acc : {}, test f1 : {}'.format(epoch,loss_total.avg,accuracy.avg,f1_score))
            return accuracy.avg

def main(args):
    # INIT project , create folders
    logfile = init_project(args)
    # grad_log = init_project(args,title = 'gradcheck')

    #load gene to id pairs if given
    if args.gene2id_json_path is not None:
        gene2id_dict = load_gene2id(args.gene2id_json_path)
        logfile('loading gene to id pairs successfully: total {} pairs'.format(len(gene2id_dict)))
    else:
        gene2id_dict = None
        logfile('No exist gene to id pairs')

    #load dataset & update gene2id_dict
    if os.path.isdir(args.data_path):
        data_path = [os.path.join(args.data_path,_) for _ in os.listdir(args.data_path) if _.endswith('h5ad')]
    else:
        data_path = [args.data_path]    
    

    trainloader,valloader,testloader,gene2id_dict,args.num_class,cell_type_dict = get_tune_dataloader(
            data_path=data_path,
            gene2tok_dict=gene2id_dict,
            batch_size=args.batch_size,
            shuffle = True,
            num_workers=8,
            log_transform = args.log_transform,
            balanced = args.balance,
            clip = args.clip,
            long = args.long,
            seed = args.seed
        )
    
    logfile('total data number : train - {}, val - {}, test - {}'.format(len(trainloader),len(valloader),len(testloader)))

    # save new gene2id_dict
    # save_gene2id(args,gene2id_dict)
    # save_gene2id(args,cell_type_dict,Name='cell_type_dict.json')
    
    #padding ignore mse loss
    loss_fn = nn.CrossEntropyLoss()

    # def FocalLoss(alpha=0.25,gamma=2):
    #     def focal_loss(pred,label):
    #         pred = F.softmax(pred,dim=1)
    #         pred = pred.gather(1,label.view(-1,1))
    #         pred = pred.view(-1)
    #         loss = -1 * (1-pred)**gamma * torch.log(pred)

    #focal loss
    # loss_fn = FocalLoss(alpha=0.25,gamma=2)
    

    #create model
    model = Model.PathFormer_lp(
                max_gene_num=args.max_gene_num,
                embedding_dim=args.embedding_dim,
                max_trainning_length = args.max_trainning_length,
                n_pathway = args.n_pathway,
                n_head = args.n_head,
                mlp_dim = None,
                encoder_cross_attn_depth = args.encoder_cross_attn_depth,
                encoder_self_attn_depth = args.encoder_self_attn_depth,
                encoder_projection_dim = args.encoder_projection_dim,
                decoder_extract_block_depth = args.decoder_extract_block_depth,
                decoder_selfattn_block_depth = args.decoder_selfattn_block_depth,
                decoder_projection_dim = 1,
                n_cell_type = args.num_class,
                dropout = args.dropout,
                project_out = args.project_out
                )
    #hypersettings (learnner, loss ...) 
    #set optimizer
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.learnning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.learnning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamW':
        optimizer = AdamW(model.parameters(), lr=args.learnning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'radam':
        optimizer = RAdam(model.parameters(), lr=args.learnning_rate, weight_decay=args.weight_decay)
    else:
        raise ValueError('optimizer not supported')
    
    # scheduler = CosineAnnealingWarmupRestarts(
    #     optimizer = optimizer,
    #     first_cycle_steps = 10,
    #     max_lr=args.learnning_rate,
    #     min_lr=args.learnning_rate*1e-1,
    #     warmup_steps=5,
    # )
    scheduler = None

    #ckpt load


    ckpt = torch.load(args.ckpt_path,map_location=args.device)
    new_state_dict = {}
    for k, v in ckpt['model_state_dict'].items():
        if k.startswith('module.'):
            k = k[7:]
        if 'decoder' not in k:
            f = 1
            for no_load_pattern in args.init_set:
                if no_load_pattern in k:
                    print('skipping key {}'.format(k))
                    f = 0
                    break
            if f:
                new_state_dict[k] = v
    print(new_state_dict.keys())
    load_info = model.load_state_dict(new_state_dict,strict = False)
    print(load_info)

    # if args.ckpt_path is not None:
        # model = load_basic_ckpt(args,model)
        # logfile('start epoch : {}'.format(START_EPOCH))
        
    for name, param in model.named_parameters():
        # if ('pathway' in name):
        #     param.requires_grad = False
        # else:
        #     param.requires_grad = True
        # print(name,param.requires_grad)
        if name.startswith('lp'):
            param.requires_grad = True
        else:
            if args.linear_probing:
                param.requires_grad = False
            else:
                param.requires_grad = True
        print(name,param.requires_grad)
    

    #grad log colunm name
    # grad_log('epoch\titer\tloss\t'+ '\t'.join([name for name, param in model.named_parameters() if param.requires_grad]))
    START_EPOCH = 0

    if args.device!='cpu':
        if len(args.device_num) > 1:
            model = torch.nn.DataParallel(model.to(args.device), device_ids=args.device_num)
            logfile('use multi gpu')
        else:
            model = model.to(args.device)

    #save all args and configs
    args2json(args)

    test_acc_all = []
    #trainning
    args.val_min_loss = 1e9
    args.val_max_acc = 0
    args.test_min_loss = 1e9
    args.test_max_acc = 0

    for epoch in range(START_EPOCH+1,START_EPOCH+1+args.epoch):
        logfile('epoch : {}'.format(epoch))
        train_one_epoch(args,model,optimizer,scheduler,loss_fn,trainloader,epoch,logfile,grad_log=None)
        test_acc_all.append(eval_one_epoch(args,model,loss_fn,valloader,epoch,logfile,testloader=testloader))
    
    return test_acc_all

        

if __name__ == "__main__":
    args = parser.parse_args()

    if args.config_path is not None:
        args = load_config(args,only_model_config=True)

    args = check_device(args)
    torch.cuda.set_device(args.device)

    # root dir 
    loss_all = []
    repeat_num = 10

    args.ROOT = os.path.dirname(os.path.abspath(__file__))
    args.project_base_name = args.project_name
    init_set = [
        # ['gene2vector','cross_extract_blocks'],
        ['None']
    ]
    # for rp in range(repeat_num):
    #     args.init_set 
    #     args.project_name = args.project_base_name + '_{}'.format(rp)
    #     test_loss_all = main(args)
    #     loss_all.append(args.best_test_acc)
    
    for i in init_set:
        args.init_set = i
        for idx in range(1):

            seed = idx
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # 如果使用了多GPU，需要设置所有GPU的seed
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(seed)

            args.project_name = args.project_base_name + '_{}_{}'.format(i,idx)
            args.seed = idx
            test_loss_all = main(args)
            loss_all.append(args.best_test_acc)
