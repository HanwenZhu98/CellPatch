import scanpy as sc

import torch
import pandas as pd
import json
import argparse

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, ConcatDataset


class DataProcesser:
    '''
    DataProcesser for infer
    - data_path : path to the h5ad file

    '''
    def __init__(self, data_path):
        self.adata = sc.read_h5ad(data_path)
    
    def preprocess_method(
            self,
            min_genes=200,
            min_cells=3,
            total_counts=1e4,
            n_top_genes=None,
            log_transform=False,
        ):
        sc.pp.filter_cells(self.adata, min_genes=min_genes)
        sc.pp.filter_genes(self.adata, min_cells=min_cells)
        sc.pp.normalize_total(self.adata, target_sum=total_counts)
        if log_transform:
            sc.pp.log1p(self.adata)
        if n_top_genes is not None:
            sc.pp.highly_variable_genes(self.adata, n_top_genes=n_top_genes)
            self.adata = self.adata[:, self.adata.var.highly_variable]
            
    def genename2id_index(self,genename2geneid_dict):
        name2id_dict = json.load(open(genename2geneid_dict,'r'))
        overlap_index = self.adata.var.index.intersection(name2id_dict.keys())
        self.adata = self.adata[:,overlap_index]
        self.adata.var.index = [name2id_dict[_] for _ in self.adata.var.index]

    def get_hvg(self, n_top_genes):
        sc.pp.highly_variable_genes(self.adata, n_top_genes=n_top_genes)
        self.adata = self.adata[:, self.adata.var.highly_variable]
        
def get_model_args(argsjson):
    model_args = argparse.ArgumentParser()
    args = model_args.parse_args()
    with open(argsjson, "r") as config_file:
        config = json.load(config_file)
    for key, value in config.items():
        args.__dict__[key] = value
    return args

