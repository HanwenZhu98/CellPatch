import json
import argparse
import torch

import utils.dataloader as dl
from utils import load_gene2id,load_config
from Model import Model_lite as Model

def read_args_from_json(json_file):
    with open(json_file, 'r') as f:
        args = json.load(f)
        print(args)
        args = argparse.Namespace(**args)
    return args

def get_data_from_list(data_list,args):
    if len(data_list) == 1 :
        trainloader,valloader,testloader,gene2id_dict,args.num_class,cell_type_dict = dl.get_tune_dataloader(
            data_path = data_list,
            gene2tok_dict=load_gene2id(args.gene2id_dict),
            batch_size=args.batch_size,
            shuffle = True,
            num_workers=8,
            log_transform = args.log_transform,
            balanced = args.balance,
            clip = args.clip,
            long = args.long,
            seed = args.seed
        )
    elif len(data_list)==3:
        trainloader,valloader,testloader,gene2id_dict,args.num_class,cell_type_dict = dl.get_tune_dataloader_xtrimo(
            data_path=data_list,
            gene2tok_dict=load_gene2id(args.gene2id_dict),
            batch_size=args.batch_size,
            shuffle = True,
            num_workers=8,
            log_transform = args.log_transform,
            balanced = args.balance,
            clip = args.clip,
            long = args.long,
            seed = args.seed
        )
    
    return trainloader,valloader,testloader,gene2id_dict,args.num_class,cell_type_dict

def get_model(args):
    model_args = load_config(args.ckpt_config)
    
    model = Model.PathFormer_lp(
                max_gene_num=model_args.max_gene_num,
                embedding_dim=model_args.embedding_dim,
                max_trainning_length = 3000,
                n_pathway = model_args.n_pathway,
                n_head = model_args.n_head,
                mlp_dim = None,
                encoder_cross_attn_depth = model_args.encoder_cross_attn_depth,
                encoder_self_attn_depth = model_args.encoder_self_attn_depth,
                encoder_projection_dim = model_args.encoder_projection_dim,
                decoder_extract_block_depth = model_args.decoder_extract_block_depth,
                decoder_selfattn_block_depth = model_args.decoder_selfattn_block_depth,
                decoder_projection_dim = 1,
                n_cell_type = args.num_class,
                dropout = args.dropout,
                project_out = False
                )
    
    if args.ckpt_path is not None:
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
        load_info = model.load_state_dict(new_state_dict,strict = False)
        print(load_info)
    return model