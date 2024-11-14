import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
import os

from .STAGATE import STAGATE
from .utils import Transfer_pytorch_Data

import torch
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
import torch.nn as nn


def load_gene2id(path):
    import json
    with open(path,'r') as f:
        gene2id_dict = json.load(f)
    return gene2id_dict


def train_STAGATE(adata, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAGATE',
                gradient_clipping=5.,  weight_decay=0.0001, verbose=True, 
                random_seed=0, save_loss=False, save_reconstrction=False, 
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """\
    Training graph attention auto-encoder.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for AdamOptimizer.
    save_loss
        If True, the training loss is saved in adata.uns['STAGATE_loss'].
    save_reconstrction
        If True, the reconstructed expression profiles are saved in adata.layers['STAGATE_ReX'].
    device
        See torch.device.

    Returns
    -------
    AnnData
    """

    # seed_everything()
    seed=random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    adata.X = sp.csr_matrix(adata.X)
    
    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    data = Transfer_pytorch_Data(adata_Vars)

    model = STAGATE(hidden_dims = [data.x.shape[1]] + hidden_dims).to(device)
    data = data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    #loss_list = []
    best_loss = 1e9
    for epoch in tqdm(range(1, n_epochs+1)):
        model.train()
        optimizer.zero_grad()
        z, out = model(data.x, data.edge_index)
        loss = F.mse_loss(data.x, out) #F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        #loss_list.append(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
    
    if loss.item() < best_loss:
            best_loss = loss.item()
            adata.uns['best_STAGATE_loss'] = best_loss
            adata.uns['best_STAGATE_epoch'] = epoch
            adata.obsm['best_z'] = z.to('cpu').detach().numpy()

    model.eval()
    z, out = model(data.x, data.edge_index)
    
    STAGATE_rep = z.to('cpu').detach().numpy()
    adata.obsm[key_added] = STAGATE_rep

    if save_loss:
        adata.uns['STAGATE_loss'] = loss
    if save_reconstrction:
        ReX = out.to('cpu').detach().numpy()
        ReX[ReX<0] = 0
        adata.layers['STAGATE_ReX'] = ReX

    return adata

class PathFormer(nn.Module):
    def __init__(
            self,
            max_gene_num,
            embedding_dim,
            max_trainning_length = 3000,
            n_pathway = 100,
            n_head = 4,
            mlp_dim = None,
            encoder_cross_attn_depth = 1,
            encoder_self_attn_depth = 1,
            encoder_projection_dim = 8,
            decoder_extract_block_depth = 1,
            decoder_selfattn_block_depth = 0,
            decoder_projection_dim = 1,
            ):
        super().__init__()

        self.max_gene_num = max_gene_num
        self.embedding_dim = embedding_dim
        self.max_trainning_length = max_trainning_length
        self.n_pathway =n_pathway
        self.n_head = n_head
        self.mlp_dim = embedding_dim * 2 if mlp_dim is None else mlp_dim

        # embedding module
        self.count2vector = nn.Linear(1, self.embedding_dim)
        self.count2vector_norm = nn.LayerNorm(self.embedding_dim)
        self.gene2vector = nn.Embedding(self.max_gene_num, self.embedding_dim)

        # encoder module
        self.encoder_cross_attn_depth = encoder_cross_attn_depth
        self.encoder_self_attn_depth = encoder_self_attn_depth
        self.encoder_projection_dim = encoder_projection_dim
        
        # encoder cross attention module
        self.pathway_token = nn.Parameter(torch.randn(1,self.n_pathway,self.embedding_dim))

        self.cross_extract_blocks_attn = nn.MultiheadAttention(
            self.embedding_dim,
            self.n_head,
            dropout = 0.1,
            batch_first=True
        )
        self.cross_extract_blocks_attn_norm = nn.LayerNorm(self.embedding_dim)
        self.cross_extract_blocks_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim,self.mlp_dim),
            nn.GELU(),
            nn.Linear(self.mlp_dim,self.embedding_dim)
        )
        self.cross_extract_blocks_mlp_norm = nn.LayerNorm(self.embedding_dim)

        # encoder self attention module
        self.self_extract_blocks_attn = nn.ModuleList([
            nn.MultiheadAttention(
                self.embedding_dim,
                self.n_head,
                dropout = 0.1,
                batch_first=True
            ) for _ in range(self.encoder_self_attn_depth)
        ])
        self.self_extract_blocks_attn_norm = nn.ModuleList([
            nn.LayerNorm(self.embedding_dim) for _ in range(self.encoder_self_attn_depth)
        ])
        self.self_extract_blocks_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embedding_dim,self.mlp_dim),
                nn.GELU(),
                nn.Linear(self.mlp_dim,self.embedding_dim)
            ) for _ in range(self.encoder_self_attn_depth)
        ])
        self.self_extract_blocks_mlp_norm = nn.ModuleList([
            nn.LayerNorm(self.embedding_dim) for _ in range(self.encoder_self_attn_depth)
        ])
        if self.encoder_projection_dim is not None:
            self.encoder_projection = nn.Linear(self.embedding_dim,self.encoder_projection_dim)

        
        self._init_weights()

    def _init_weights(self):
        # Xavier或He初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                nn.init.constant_(m.in_proj_bias, 0)

    def get_mid_embeding(self, count_x, gene_x,output_attentions = False):
        '''
        - count_x : [batch_size, max_gene_num]
        - gene_x  : [batch_size, max_gene_num]
        - output_attentions : bool type, whether need attention score output.
        '''
        count_e = self.count2vector(count_x.unsqueeze(2))
        # count_e = self.count2vector_norm(count_e)
        gene_e = self.gene2vector(gene_x)
        # gene_e = gene_e * (self.embedding_dim ** 0.5)
        attn_out_total = {}
        count_e = count_e + gene_e
        pathway_emb = self.pathway_token.repeat(count_e.shape[0],1,1)
        attn_out,attn_score = self.cross_extract_blocks_attn(query = pathway_emb,key = count_e,value = count_e)
        pathway_emb = self.cross_extract_blocks_attn_norm(attn_out + self.pathway_token)
        pathway_emb = self.cross_extract_blocks_mlp_norm(self.cross_extract_blocks_mlp(pathway_emb) + pathway_emb)
        if output_attentions:
            attn_out_total['encoder_cross_extract_blocks'] = attn_score
        for i in range(self.encoder_self_attn_depth):
            attn_out,attn_score = self.self_extract_blocks_attn[i](query = pathway_emb,key = pathway_emb,value = pathway_emb)
            pathway_emb = self.self_extract_blocks_attn_norm[i](attn_out + pathway_emb)
            pathway_emb = self.self_extract_blocks_mlp_norm[i](self.self_extract_blocks_mlp[i](pathway_emb) + pathway_emb)
            if output_attentions:
                attn_out_total['encoder_self_extract_blocks_'+str(i)] = attn_score
        if self.encoder_projection_dim is not None:
            pathway_emb = self.encoder_projection(pathway_emb)
        return pathway_emb if not output_attentions else (pathway_emb,attn_out_total)
        # return gene_e

class PATH_STG(nn.Module):
    def __init__(self, embedding_dim,n_pathway,projection_dim = None,stg_dim = [512,30],out_dim = 2988, res_cat = False):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_pathway = n_pathway
        self.projection_dim = projection_dim
        self.stg_dim = stg_dim

        self.model_head = PathFormer(max_gene_num=70000,
            embedding_dim=self.embedding_dim,
            max_trainning_length = 3000,
            n_pathway = self.n_pathway,
            n_head = 2,
            mlp_dim = None,
            encoder_cross_attn_depth = 1,
            encoder_self_attn_depth = 0,
            encoder_projection_dim = self.projection_dim,
            decoder_extract_block_depth = 1,
            decoder_selfattn_block_depth = 0,
            decoder_projection_dim = 1,
        )

        self.res_cat = res_cat
        if self.res_cat:
            self.model_STG = STAGATE(hidden_dims = [n_pathway*embedding_dim+out_dim] + stg_dim)
            self.decoder_lin = nn.Linear(n_pathway*embedding_dim+out_dim,out_dim)
        else:
            self.model_STG = STAGATE(hidden_dims = [n_pathway*embedding_dim] + stg_dim)
            self.decoder_lin = nn.Linear(n_pathway*embedding_dim,out_dim)

    def forward(self, count_x, gene_x, edge_index):
        '''
        count_x : n_cell * n_gene
        gene_x : n_cell * n_gene

        emb_0 : (n_cell, n_pathway, embedding_dim) -> (n_cell, n_pathway*embedding_dim)

        rec_0 : (n_cell, n_pathway*embedding_dim)

        rec ： (n_cell, out_dim)
        '''
        
        emb_0 = self.model_head.get_mid_embeding(count_x, gene_x)
        emb_0 = emb_0.reshape(emb_0.shape[0],self.n_pathway*self.embedding_dim)

        if self.res_cat:
            emb_0 = torch.cat([emb_0, count_x],dim=1)

        emb,rec_0 = self.model_STG(emb_0, edge_index)

        rec = self.decoder_lin(rec_0)
        return emb,rec

    def load_head_ckpt(self,ckpt_path,map_location):
        ckpt = torch.load(ckpt_path,map_location=map_location)
        new_state_dict = {}
        for k, v in ckpt['model_state_dict'].items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
        load_info = self.model_head.load_state_dict(new_state_dict,strict=False)
        print(load_info)

def train_STAGATE_new(adata, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAGATE',
                gradient_clipping=5.,  weight_decay=0.0001, verbose=True, 
                random_seed=0, save_loss=True, save_reconstrction=False, 
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                res_cat = False, save_each_epoch = 100,save_id = None,output_dir = './ckpt_spatial/'):
    """\
    Training graph attention auto-encoder.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for AdamOptimizer.
    save_loss
        If True, the training loss is saved in adata.uns['STAGATE_loss'].
    save_reconstrction
        If True, the reconstructed expression profiles are saved in adata.layers['STAGATE_ReX'].
    device
        See torch.device.

    Returns
    -------
    AnnData
    """

    # seed_everything()
    seed=random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    loss_log = []
    adata.X = sp.csr_matrix(adata.X)
    
    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    geneid2idx = load_gene2id('/cluster/home/hanwen/scmodel/scdata/cellXgene/Gid2tok.json')
    adata_Vars = adata_Vars[:, [(i in geneid2idx.keys()) for i in adata_Vars.var['gene_ids']]]

    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    data = Transfer_pytorch_Data(adata_Vars)
    data.gene = torch.tensor([geneid2idx[i] for i in adata_Vars.var['gene_ids']]).unsqueeze(0).repeat(data.x.shape[0],1)

    model = PATH_STG(32,64,out_dim=data.x.shape[1],res_cat = res_cat).to(device)
    data = data.to(device)
    model.load_head_ckpt('/cluster/home/yushun/metagpt/model_with_weight/10Mpretrain_lite_10M/last_model_weight/ckpt/10Mpretrain_lite_10M_0118_ep50_iter_.pth',map_location=device)
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    #loss_list = []
    pb = tqdm(range(n_epochs))
    # for epoch in tqdm(range(1, n_epochs+1)):
    best_loss = 1e9
    for epoch in pb:
        model.train()
        optimizer.zero_grad()
        z, out = model(data.x, data.gene, data.edge_index)
        loss = F.mse_loss(data.x, out) #F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        #loss_list.append(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
        pb.set_description("Epoch: %d, Loss: %.4f" % (epoch, loss.item()))
        loss_log.append(loss.item())
        if loss.item() < best_loss:
            best_loss = loss.item()
            adata.uns['best_STAGATE_loss'] = best_loss
            adata.uns['best_STAGATE_epoch'] = epoch
            adata.obsm['best_emb'] = z.to('cpu').detach().numpy()
            
            model.eval()
            z, out = model(data.x, data.gene, data.edge_index)
            adata.obsm['best_eval_emb'] = z.to('cpu').detach().numpy()
            model.train()

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if save_id is not None:
                adata.write_h5ad(output_dir+'/best_ckpt_spatial_{}.h5ad'.format(save_id,epoch))
            else:
                adata.write_h5ad(output_dir+'/best_ckpt_spatial.h5ad')

        if (epoch+1) % 100 == 0:
            STAGATE_rep = z.to('cpu').detach().numpy()
            adata.obsm[key_added] = STAGATE_rep
            adata.uns['STAGATE_loss'] = loss.item()
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if save_id is not None:
                adata.write_h5ad(output_dir+'/ckpt_spatial_{}_epoch_{}.h5ad'.format(save_id,epoch+1))
            else:
                adata.write_h5ad(output_dir+'/ckpt_spatial_epoch_'+str(epoch+1)+'.h5ad')
            # print('saving ckpt at {}'.format(os.path.abspath('./ckpt_spatial/ckpt_spatial_epoch_'+str(epoch+1)+'.h5ad')))
    
    #save loss
    loss_log = pd.DataFrame(loss_log)
    loss_log.to_csv(output_dir+'/loss_log.csv')
    
    model.eval()
    z, out = model(data.x, data.gene, data.edge_index)
    
    STAGATE_rep = z.to('cpu').detach().numpy()
    adata.obsm[key_added] = STAGATE_rep

    if save_loss:
        adata.uns['STAGATE_loss'] = loss.item()
    if save_reconstrction:
        ReX = out.to('cpu').detach().numpy()
        ReX[ReX<0] = 0
        adata.layers['STAGATE_ReX'] = ReX

    return adata


def train_STAGATE_cat(adata, hidden_dims=[512, 30], n_epochs=1000, lr=0.001, key_added='STAGATE',
                gradient_clipping=5.,  weight_decay=0.0001, verbose=True, 
                random_seed=0, save_loss=False, save_reconstrction=False, 
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """\
    Training graph attention auto-encoder.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for AdamOptimizer.
    save_loss
        If True, the training loss is saved in adata.uns['STAGATE_loss'].
    save_reconstrction
        If True, the reconstructed expression profiles are saved in adata.layers['STAGATE_ReX'].
    device
        See torch.device.

    Returns
    -------
    AnnData
    """

    # seed_everything()
    seed=random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    adata.X = sp.csr_matrix(adata.X)
    # adata.obsm['cat'] = sp.csr_matrix(adata.obsm['cat'])
    adata_Vars = adata

    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    data = Transfer_pytorch_Data(adata_Vars)

    data.x = torch.FloatTensor(adata.obsm['cat'])

    
    model = STAGATE(hidden_dims = [data.x.shape[1]] + hidden_dims).to(device)
    
    data = data.to(device)
    print(data.x.shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    #loss_list = []
    best_loss = 1e9
    for epoch in tqdm(range(1, n_epochs+1)):
        model.train()
        optimizer.zero_grad()
        z, out = model(data.x, data.edge_index)
        loss = F.mse_loss(data.x, out) #F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        #loss_list.append(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
    
    if loss.item() < best_loss:
            best_loss = loss.item()
            adata.uns['best_STAGATE_loss'] = best_loss
            adata.uns['best_STAGATE_epoch'] = epoch
            adata.obsm['best_z'] = z.to('cpu').detach().numpy()

    model.eval()
    z, out = model(data.x, data.edge_index)
    
    STAGATE_rep = z.to('cpu').detach().numpy()
    adata.obsm[key_added] = STAGATE_rep

    if save_loss:
        adata.uns['STAGATE_loss'] = loss
    if save_reconstrction:
        ReX = out.to('cpu').detach().numpy()
        ReX[ReX<0] = 0
        adata.layers['STAGATE_ReX'] = ReX

    return adata
