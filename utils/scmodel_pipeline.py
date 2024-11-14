import argparse
import torch
import time
import os
import numpy as np
import scanpy as sc
import pandas as pd

from torch.optim import Adam,SGD,AdamW,lr_scheduler
import torch.nn as nn
from Model import Model_lite as Model
from utils import *

from utils.infer_tools import *

import torch
import matplotlib.pyplot as plt
import seaborn as sns

data_path ='/cluster/home/hanwen/scmodel/scdata/test_data_set/zheng68k_p_filtered.h5ad'
model_path = '/cluster/home/yushun/scmodel/a_PathFormer/runs/train/10Mpretrain_lite_10M_0118/ckpt/10Mpretrain_lite_10M_0118_ep11_iter_.pth'
arg_path = '/cluster/home/yushun/scmodel/a_PathFormer/runs/train/10Mpretrain_lite_10M_0118/args.json'
gene2id_path = '/cluster/home/hanwen/scmodel/scdata/cellXgene/Gid2tok.json'
name2gene_path = '/cluster/home/hanwen/scmodel/scdata/cellXgene/gene2tok.json'

torch.set_printoptions(precision=10)

class scmodel_analysis:
    def __init__(self, gene2id_path, name2gene_path, args_path):
        self.gene2id_dict = json.load(open(gene2id_path,'r'))
        self.name2id_dict = json.load(open(name2gene_path,'r'))
        self.args = self.load_args(args_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_data(self, data_path,n_top_genes=None):
        self.data = DataProcesser(data_path)
        self.data.preprocess_method(log_transform=self.args.log_transform)
        if self.data.adata.var.index[0].startswith('EN'):
            new_index = self.data.adata.var.index.intersection(self.gene2id_dict.keys())
        else:
            new_index = self.data.adata.var.index.intersection(self.name2id_dict.keys())
        self.data.adata = self.data.adata[:,new_index]
        if  n_top_genes is not None:
            self.data.get_hvg(n_top_genes=n_top_genes)
        return self.data
    
    def load_args(self, args_path):
        argsjson = json.load(open(args_path))
        args = argparse.Namespace()
        for key, value in argsjson.items():
            args.__dict__[key] = value
        return args

    def load_model(self, model_path):
        self.model = Model.PathFormer(
                max_gene_num=self.args.max_gene_num, 
                embedding_dim=self.args.embedding_dim,
                max_trainning_length = self.args.max_trainning_length,
                n_pathway = self.args.n_pathway,
                n_head = self.args.n_head,
                encoder_cross_attn_depth = self.args.encoder_cross_attn_depth,
                encoder_self_attn_depth = self.args.encoder_self_attn_depth,
                encoder_projection_dim = self.args.encoder_projection_dim,
                decoder_extract_block_depth = self.args.decoder_extract_block_depth,
                decoder_selfattn_block_depth = self.args.decoder_selfattn_block_depth,
            )
        ckpt = torch.load(model_path,map_location=torch.device('cpu'))
        new_state_dict = {}
        for k, v in ckpt['model_state_dict'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        self.model = self.model.to(self.device)

    def get_embedding(self,bs=128, same_gene = False):
        self.model.eval()
        if self.data.adata.var.index[0].startswith('EN'):
            self.gene = torch.tensor([self.gene2id_dict[gene] for gene in self.data.adata.var.index]).int()
        else:
            self.gene = torch.tensor([self.name2id_dict[gene] for gene in self.data.adata.var.index]).int()

        if self.args.encoder_projection_dim is None:
            emb_dim = self.args.embedding_dim
        else:
            emb_dim = self.args.encoder_projection_dim
        output_mid = torch.zeros((self.data.adata.shape[0],self.args.n_pathway,emb_dim))
        if same_gene:
            print('Warnning : using same gene for all cells, make sure use hvg first !')
            output_attn = torch.zeros((self.data.adata.shape[0],self.args.n_pathway,self.gene.shape[0]))

        with torch.no_grad():
            time_0 = time.time()
            csr_mat = self.data.adata.X
            for batch_id in range(0,self.data.adata.shape[0],bs):
                if batch_id + bs > self.data.adata.shape[0]:
                    batch = csr_mat[batch_id:]
                else:
                    batch = csr_mat[batch_id:batch_id+bs]
                if same_gene:
                    #假设 bs * hvg
                    gene_in = self.gene.unsqueeze(0).repeat(batch.shape[0],1)
                    batch_in = torch.tensor(batch.toarray())
                else:
                    gene_in,batch_in = self.extract_nonzero_elements(batch)
                batch_in = batch_in.to(self.device)
                gene_in = gene_in.to(self.device)
                if same_gene:
                    pathway_embed,mid_attn= self.model.get_mid_embeding(batch_in,gene_in,output_attentions=same_gene)
                    output_mid[batch_id:batch_id+bs,:,:] = pathway_embed.cpu()
                    output_attn[batch_id:batch_id+bs] = mid_attn['encoder_cross_extract_blocks'].cpu()
                else:
                    pathway_embed = self.model.get_mid_embeding(batch_in,gene_in,output_attentions=same_gene)
                    output_mid[batch_id:batch_id+bs,:,:] = pathway_embed.cpu()
                # print(mid_attn['encoder_cross_extract_blocks'].shape)
                # print(pathway_embed.shape)
                # print(mid_attn.keys())
                # print(pathway_embed,mid_attn)
                # count_x,gene_x = self.batch_process(batch,gene)
                if batch_id%128==0:
                    print(time.time()-time_0,end='\r')
                    # (row , indices) : value
            print('end time {}'.format(time.time()-time_0))
        self.data.adata.obsm['embedding'] = output_mid.numpy()
        if same_gene:
            self.data.adata.obsm['first_layer_attn'] = output_attn.numpy()

    def extract_nonzero_elements(self,csr_matrix):
        cells_indices = []
        cells_values = []

        # 遍历每个细胞
        for i in range(csr_matrix.shape[0]):
            row = csr_matrix.getrow(i)
            cells_indices.append(torch.tensor(self.gene[row.indices]))
            cells_values.append(torch.tensor(row.data))

        cells_indices, cells_values = self.pad_cells(cells_indices, cells_values)
        return cells_indices, cells_values

    def pad_cells(self,cells_indices, cells_values):
        padded_indices = pad_sequence(cells_indices, batch_first=True, padding_value=0)
        padded_values = pad_sequence(cells_values, batch_first=True, padding_value=0)
        return padded_indices, padded_values

    def get_umap(self,save = 'cache.png',color_col = 'celltype'):
        # sc.pp.neighbors(scm.data.adata, n_neighbors=30, n_pcs=5,use_rep='embedding_flat')
        self.data.adata.obsm['embedding_flat'] =  self.data.adata.obsm['embedding'].reshape(self.data.adata.obsm['embedding'].shape[0],-1)
        sc.pp.neighbors(self.data.adata,use_rep='embedding_flat')
        sc.tl.umap(self.data.adata, min_dist=0.5) # 这个参数用于调整umap间点的距离
        try:
            sc.pl.umap(self.data.adata,color = color_col,save = save)
        except:
            print('draw umap error, check the color_col name')

    def get_attention_by_pathway(self,id = 0,save = 'cache.png',color_col = 'celltype'):
        class_list = self.data.adata.obs[color_col].unique()
        class_list = class_list.tolist()
        class_list.sort()

        attention_score = {}
        for c in class_list:
            attention_score[c] = {}
            attention_score[c]['cross_attn_layer'] = torch.tensor(self.data.adata.obsm['first_layer_attn'][self.data.adata.obs[color_col]==c])

        attention_score = self.normalize_attention_scores_dim2(attention_score)

        class_list = list(attention_score.keys())
        for layer in ['cross_attn_layer']:
            layer_path_attn_list = []
            index = []
            columns = []
            for ct in class_list:
                # Add cell type and layer attention score (mean over the first dimension, select specific path)
                layer_mean = attention_score[ct][layer].mean(0)  # Mean over the first dimension
                layer_path_attn = layer_mean[id]  # Select specific path
                layer_path_attn_list.append(layer_path_attn.numpy())
                index.append(f"{ct} - {layer}")
                if not columns:
                    # Assuming all tensors have the same second dimension (dim2)
                    columns = [f"gene {i+1}" for i in range(attention_score[ct][layer].shape[2])]

            df = pd.DataFrame(layer_path_attn_list, index=index, columns=columns)
            sns.clustermap(df, cmap='coolwarm', standard_scale=1)
            # Visualize one layer
            # plt.boxplot(layer_path_attn_list)
            # plt.title(f'Attention Score Distribution in {layer} and path {path_id}')
            # plt.xlabel('Cell Type')
            # plt.ylabel('Attention Score')
            # plt.xticks(ticks=[1, 2], labels=class_list)  # Adjust labels based on the number of cell types
            plt.savefig(save)

    def normalize_along_dim2(self,tensor):
        """
        Normalizes the scores of a tensor along the dim2 dimension.

        Args:
        tensor (torch.Tensor): A tensor of shape [batch, dim1, dim2].

        Returns:
        torch.Tensor: A tensor with the same shape, but with scores normalized along dim2.
        """
        means = torch.mean(tensor, dim=2, keepdim=True)
        stds = torch.std(tensor, dim=2, keepdim=True)
        normalized_tensor = (tensor - means) / (stds + 1e-5)  # Adding a small value to avoid division by zero
        return normalized_tensor

    def normalize_attention_scores_dim2(self,cell_data):
        """
        Normalizes the attention scores for each cell type and each layer along the dim2 dimension.

        Args:
        cell_data (dict): A dictionary where keys are cell types and values are dictionaries.
                        The inner dictionaries have keys as layer names and values as tensors 
                        of attention scores.

        Returns:
        dict: A dictionary with the same structure as cell_data, but with attention scores normalized along dim2.
        """
        normalized_data = {}

        for cell_type, layers in cell_data.items():
            normalized_data[cell_type] = {}
            for layer, scores in layers.items():
                normalized_scores = self.normalize_along_dim2(scores)
                normalized_data[cell_type][layer] = normalized_scores

        return normalized_data

if __name__ == "__main__":
    scm = scmodel_analysis(gene2id_path, name2gene_path, arg_path)

    scm.load_model(model_path = model_path)
    scm.load_data(data_path = data_path ,n_top_genes = 300)
    scm.get_embedding(same_gene = True)

