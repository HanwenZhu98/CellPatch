
import scanpy as sc
import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from typing import List
import random
import gc
import json
import time
import scipy

class BigDataLoader:
    def __init__(
            self, 
            data_path,
            gene2id_dict,
            batch_size = 256,
            mask_rate = 0.3,
            max_dataset = 10,
            log_transform = True,
            max_len = [3000,1000]
            ):
        self.data_path = data_path
        self.gene2id_dict = gene2id_dict
        self.batch_size = batch_size
        self.mask_rate = mask_rate
        self.log_transform = log_transform
        self.max_dataset = max_dataset #内存相关，一次读取多少个dataset进入dataloader
        self.max_len = max_len

        self.total_big_batches = int(np.ceil(len(self.data_path) / self.max_dataset))

        self.dataloader = None

    def shuffle_data_list(self):
        random.shuffle(self.data_path)

    def load_data(self, big_batch_idx):
        self.dataloader = None
        gc.collect()
        batch_data_path = self.data_path[big_batch_idx*self.max_dataset:(big_batch_idx+1)*self.max_dataset]
        self.dataloader, self.gene2id_dict = get_dataloader(
                    data_path=batch_data_path,
                    gene2tok_dict=self.gene2id_dict,
                    batch_size=self.batch_size,
                    mask_rate=self.mask_rate,
                    shuffle = True,
                    num_workers=16,
                    log_transform = self.log_transform,
                    log_file = None,
                    max_len=self.max_len
        )

    def __len__(self):
        return self.total_big_batches

def preprocess_method(
        adata,
        min_genes=200,
        min_cells=3,
        total_counts=1e4,
        n_top_genes=None,
        log_transform=False,
    ):
    # sc.pp.filter_cells(adata, min_genes=min_genes)
    # sc.pp.filter_genes(adata, min_cells=min_cells)
    # sc.pp.normalize_total(adata, target_sum=total_counts)
    if log_transform:
        sc.pp.log1p(adata)
    if n_top_genes is not None:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        adata = adata[:, adata.var.highly_variable]
    return adata

def load_data(data_path , log_transform = True):
    adata = sc.read_h5ad(data_path)
    # if adata.X is not csr matrix, convert it to csr matrix
    if not isinstance(adata.X,scipy.sparse.csr.csr_matrix):
        adata.X = scipy.sparse.csr_matrix(adata.X)

    # adata.var.index is ensembl i
    adata = preprocess_method(adata,log_transform=log_transform)
    return adata

    
class SCDataset(Dataset):
    def __init__(self, data, mask_rate, gene_token = None,shuffle = True, max_len = [3000,1000]):
        '''
        scdataset for trainning
        - data : adata
        - mask_rate : mask rate
        - gene_token : gene token dictionary which maps gene id to token
        - shuffle : wheather shuffle the data
        '''
        super().__init__()
        self.data = data
        self.gene_token = gene_token
        self.mask_rate = mask_rate
        self.shuffle = shuffle
        self.shape = self.data.shape
        self.base_gene_pos = set(np.arange(self.shape[1]))
        self.max_len = max_len
        # update gene2token dictionary, if gene_token is None, create a new one, else update the old one and return
        if gene_token == None:
            self.gene_token = {gene : i+1 for i,gene in enumerate(self.data.var.index)}
        else:
            for g in self.data.var.index:
                if g not in gene_token:
                    self.gene_token[g] = len(self.gene_token)+1 # 0 is for padding

        self.gene_x = torch.tensor([self.gene_token[gene] for gene in self.data.var.index]).int()
        # self.count_x = torch.tensor(self.data.X.toarray())

    def __getitem__(self, index):
        count_x = self.data.X[index]

        #计算非零位置
        nonzero_pos = count_x.indices

        #shuffle and get max_len 
        # if self.shuffle:
        #     idx_x = torch.randperm(nonzero_pos.shape[0])
        #     nonzero_pos = nonzero_pos[idx_x]


        #计算零位置
        zero_pos = self.base_gene_pos - set(nonzero_pos)

        #计算非零个数
        len_i = len(nonzero_pos)
        #计算对应mask个数
        mask_num = int(len_i*self.mask_rate)

        #抽取encode和decode位置
        decode_pos = np.random.choice(len_i, mask_num,replace=False)
        encode_pos = np.setdiff1d(np.arange(len_i),decode_pos)

        #clip to max length
        if encode_pos.shape[0]>self.max_len[0]:
            encode_pos = encode_pos[:self.max_len[0]]

        encode_indice = count_x.indices[encode_pos]

        #抽取decoder的零位置
        decoder_zero = [zero_pos.pop() for _ in range(int(mask_num/2))]
        #合并decoder的零位置和decoder的非零位置
        decode_indice = np.concatenate([count_x.indices[decode_pos],decoder_zero])
        decode_val = np.concatenate([count_x.data[decode_pos],np.zeros(int(mask_num/2))])

        #shuffle decode_indice
        idx_x = torch.randperm(decode_indice.shape[0])[:self.max_len[1]]
        decode_indice = decode_indice[idx_x]
        decode_val = decode_val[idx_x]
        # return torch.tensor([0]),torch.tensor([0]),torch.tensor([0]),torch.tensor([0])
        return torch.tensor(count_x.data[encode_pos]), self.gene_x[encode_indice], torch.tensor(decode_val), self.gene_x[decode_indice]

    def __len__(self):
        return self.data.shape[0]

def collate_fn(batch):
    count_e, gene_e, count_d, gene_d = zip(*batch)
    count_e = pad_sequence(count_e, batch_first=True, padding_value=0)
    gene_e = pad_sequence(gene_e, batch_first=True, padding_value=0)
    count_d = pad_sequence(count_d, batch_first=True, padding_value=0)
    gene_d = pad_sequence(gene_d, batch_first=True, padding_value=0)

    return count_e,gene_e,count_d,gene_d

def collate_fn_tune(batch):
    count_x, gene_x, celltype_y = zip(*batch)
    count_x = pad_sequence(count_x, batch_first=True, padding_value=0)
    gene_x = pad_sequence(gene_x, batch_first=True, padding_value=0)
    celltype_y = torch.stack(celltype_y)
    return count_x, gene_x,celltype_y

def get_dataloader(data_path,gene2tok_dict = None, batch_size = 32,mask_rate=0.3,shuffle = True,log_transform = True,num_workers=0,label =False,log_file = None,max_len = [3000,1000]):
    '''
    get dataset
    - data_path : list of adata path(multiple adata path), or str(a single adata path)
    - gene2tok_dict : gene2token dictionary
    - mask_rate : mask rate
    - shuffle : wheather shuffle the data
    - process : preprocess method
    '''
    if type(data_path) == str:
        data_path = [data_path]

    dataset = []
    for path in data_path:
        if label:
            dataset_p = SCDataset_tune(
                data = load_data(path,log_transform),
                gene_token = gene2tok_dict,
                shuffle = shuffle
            )
            collate = collate_fn_tune
        else:
            dataset_p = SCDataset(
                data = load_data(path,log_transform),
                mask_rate = mask_rate,
                gene_token = gene2tok_dict,
                shuffle = shuffle,
                max_len = max_len,
            )
            collate = collate_fn
        if log_file is not None:
            log_file('add data {}'.format(path))
        gene2tok_dict = dataset_p.gene_token
        dataset.append(dataset_p)
        del dataset_p

    # save gene2tok_dict
    if len(dataset) == 1:
        return DataLoader(dataset[0], batch_size=batch_size, shuffle=shuffle, collate_fn=collate, drop_last=True,num_workers=num_workers),gene2tok_dict
    else:
        return DataLoader(ConcatDataset(dataset), batch_size=batch_size, shuffle=shuffle, collate_fn=collate, drop_last=True,num_workers=num_workers),gene2tok_dict

class SCDataset_tune(Dataset):
    def __init__(self, data, gene_token = None,shuffle = True,cell_type_dict = None, max_len = [3000,1000],clip = None, long = False):
        '''
        scdataset for trainning
        - data : adata
        - mask_rate : mask rate
        - gene_token : gene token dictionary which maps gene id to token
        - shuffle : wheather shuffle the data
        '''
        super().__init__()
        self.data = data
        self.gene_token = gene_token
        self.shape = self.data.shape
        self.shuffle = shuffle
        self.max_len = max_len
        self.base_gene_pos = set(np.arange(self.shape[1]))
        self.long = long
        self.clip = clip
        if cell_type_dict is None:
            RuntimeError('cell_type_dict is None')
        else:
            self.cell_type_dict = cell_type_dict
        
        # update gene2token dictionary, if gene_token is None, create a new one, else update the old one and return
        if gene_token == None:
            self.gene_token = {gene : i+1 for i,gene in enumerate(self.data.var.index)}
        else:
            for g in self.data.var.index:
                if g not in gene_token:
                    self.gene_token[g] = len(self.gene_token)+1 # 0 is for padding

        self.gene_x = torch.tensor([self.gene_token[gene] for gene in self.data.var.index])
        # self.count_x = torch.tensor(self.data.X.toarray())
        
        assert('celltype' in self.data.obs.columns)
        # create celltype token dictionary
        self.celltype_y = torch.tensor([self.cell_type_dict[celltype] for celltype in self.data.obs['celltype']])
        self.dx = self.data.X

    def __getitem__(self, index):
        count_x = self.dx[index]
        non_zero_pos = count_x.indices
        zero_pos = self.base_gene_pos - set(non_zero_pos)
        celltype_y = self.celltype_y[index]
        count = count_x.data
        if self.clip is not None:
            count[count>self.clip] = self.clip
        if self.long:
            return torch.tensor(count).long().float(), self.gene_x[non_zero_pos],celltype_y
        else:
            return torch.tensor(count), self.gene_x[non_zero_pos],celltype_y

        # count_x = self.dx[index]
        
        # if self.clip is not None:
        #     count_x.data[count_x.data>self.clip] = self.clip
        # if self.long:
        #     return  torch.tensor(count_x.toarray()).long().float().squeeze(0), self.gene_x, self.celltype_y[index]
        # else:
        #     return  torch.tensor(count_x.toarray()).squeeze(0), self.gene_x, self.celltype_y[index]
        

        
        # gene_x = self.gene_x.int()
        # celltype_y = self.celltype_y[index]
        
        # #shuffle count_x and gene_x
        # if self.shuffle:
        #     idx_x = torch.randperm(count_x.shape[0])
        #     count_x = count_x[idx_x]
        #     gene_x = gene_x[idx_x]

        # #select non zero position : nonzero position is 1, zero position is 0
        # nonzero_pos_bool = count_x != 0

        # return count_x, gene_x, nonzero_pos_bool, celltype_y

    def __len__(self):
        return self.data.shape[0]

def get_tune_dataloader_xtrimo(data_path,gene2tok_dict = None, batch_size = 32,shuffle = True,log_transform = True,num_workers=0,balanced = False,clip = None,long = False,seed = 0):
    '''
    get dataset
    - data_path : list of adata path(multiple adata path), or str(a single adata path)
    - gene2tok_dict : gene2token dictionary
    - mask_rate : mask rate
    - shuffle : wheather shuffle the data
    - process : preprocess method
    '''

    #assert one data

    dataset = []
    assert len(data_path) == 3
    adata_train = load_data(data_path[0],log_transform)
    adata_val = load_data(data_path[1],log_transform)
    adata_test = load_data(data_path[2],log_transform)
    
    if 'celltype' in adata_train.obs.columns:
        
        col = 'celltype'
    elif 'cell_type' in adata_train.obs.columns:
        col = 'cell_type'
    else:
        RuntimeError('celltype not found')

    cell_type_dict = {celltype : i for i,celltype in enumerate(adata_train.obs[col].unique())}
    num_class = len(adata_train.obs[col].unique())

    print('train size:',adata_train.shape[0])
    print('val size:',adata_val.shape[0])
    print('test size:',adata_test.shape[0])

    train_dataset = SCDataset_tune(
        data = adata_train,
        gene_token = gene2tok_dict,
        shuffle = shuffle,
        cell_type_dict = cell_type_dict,
        clip = clip,
        long = long
    )

    gene2tok_dict = train_dataset.gene_token
    val_dataset = SCDataset_tune(
        data = adata_val,
        gene_token = gene2tok_dict,
        shuffle = shuffle,
        cell_type_dict = cell_type_dict,
        clip = clip,
        long = long
    )

    test_dataset = SCDataset_tune(
        data = adata_test,
        gene_token = gene2tok_dict,
        shuffle = shuffle,
        cell_type_dict = cell_type_dict,
        clip = clip,
        long = long
    )
    
    collate = collate_fn_tune
    class_sample_count = np.array([len(np.where(train_dataset.celltype_y==t)[0]) for t in np.unique(train_dataset.celltype_y)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in train_dataset.celltype_y])
    train_sampler = WeightedRandomSampler(samples_weight, 2 * len(samples_weight))

    # save gene2tok_dict
    if balanced:
        return DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate, drop_last=True,num_workers=num_workers),\
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False,num_workers=num_workers),\
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False,num_workers=num_workers),gene2tok_dict,num_class,cell_type_dict
    else:
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True,num_workers=num_workers),\
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False,num_workers=num_workers),\
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False,num_workers=num_workers),gene2tok_dict,num_class,cell_type_dict

def get_tune_dataloader(data_path,gene2tok_dict = None, batch_size = 32,shuffle = True,log_transform = True,num_workers=0,balanced = False,clip = None,long = False,seed = 0):
    '''
    get dataset
    - data_path : list of adata path(multiple adata path), or str(a single adata path)
    - gene2tok_dict : gene2token dictionary
    - mask_rate : mask rate
    - shuffle : wheather shuffle the data
    - process : preprocess method
    '''

    #assert one data

    dataset = []
    for path in data_path:
        adata = load_data(path,log_transform)
        try:
            cell_type_dict = {celltype : i for i,celltype in enumerate(adata.obs['celltype'].unique())}
            col = 'celltype'
        except:
            cell_type_dict = {celltype : i for i,celltype in enumerate(adata.obs['cell_type'].unique())}
            col = 'cell_type'

        num_class = len(adata.obs[col].unique())
        #set seed
        torch.manual_seed(seed)
        shuffle_index = torch.randperm(adata.shape[0])
        train_index = shuffle_index[:int(adata.shape[0]*0.8)]
        val_index = shuffle_index[int(adata.shape[0]*0.8):int(adata.shape[0]*0.9)]
        test_index = shuffle_index[int(adata.shape[0]*0.9):]

        print('train size:',train_index.shape[0])
        print('val size:',val_index.shape[0])
        print('test size:',test_index.shape[0])

        adata_train = adata[train_index.numpy()]
        adata_val = adata[val_index.numpy()]
        adata_test = adata[test_index.numpy()]

        train_dataset = SCDataset_tune(
            data = adata_train,
            gene_token = gene2tok_dict,
            shuffle = shuffle,
            cell_type_dict = cell_type_dict,
            clip = clip,
            long = long
        )

        gene2tok_dict = train_dataset.gene_token
        val_dataset = SCDataset_tune(
            data = adata_val,
            gene_token = gene2tok_dict,
            shuffle = shuffle,
            cell_type_dict = cell_type_dict,
            clip = clip,
            long = long
        )

        test_dataset = SCDataset_tune(
            data = adata_test,
            gene_token = gene2tok_dict,
            shuffle = shuffle,
            cell_type_dict = cell_type_dict,
            clip = clip,
            long = long
        )
        
        collate = collate_fn_tune
        class_sample_count = np.array([len(np.where(train_dataset.celltype_y==t)[0]) for t in np.unique(train_dataset.celltype_y)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in train_dataset.celltype_y])
        train_sampler = WeightedRandomSampler(samples_weight, 2 * len(samples_weight))

    # save gene2tok_dict
    if balanced:
        return DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate, drop_last=True,num_workers=num_workers),\
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=True,num_workers=num_workers),\
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False,num_workers=num_workers),gene2tok_dict,num_class,cell_type_dict
    else:
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True,num_workers=num_workers),\
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=True,num_workers=num_workers),\
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False,num_workers=num_workers),gene2tok_dict,num_class,cell_type_dict

def get_tune_dataloader_scgpt(data_path,gene2tok_dict = None, batch_size = 32,shuffle = True,log_transform = True,num_workers=0,balanced = False,clip = None,long = False,seed = 0):
    '''
    get dataset
    - data_path : list of adata path(multiple adata path), or str(a single adata path)
    - gene2tok_dict : gene2token dictionary
    - mask_rate : mask rate
    - shuffle : wheather shuffle the data
    - process : preprocess method
    '''

    #assert one data

    dataset = []
    assert len(data_path) == 2
    adata_train = load_data(data_path[0],log_transform)
    adata_test = load_data(data_path[1],log_transform)
    
    if 'celltype' in adata_train.obs.columns:
        col = 'celltype'
    elif 'cell_type' in adata_train.obs.columns:
        col = 'cell_type'
        adata_train.obs['celltype'] = adata_train.obs['cell_type']
        adata_test.obs['celltype'] = adata_test.obs['cell_type']
    else:
        RuntimeError('celltype not found')

    cell_type_dict = {celltype : i for i,celltype in enumerate(adata_train.obs[col].unique())}
    for new_celltype in adata_test.obs[col].unique():
        if new_celltype not in cell_type_dict:
            cell_type_dict[new_celltype] = len(cell_type_dict) 
    print(cell_type_dict)
    # cell_type_dict['unknown'] = len(cell_type_dict)
    
    num_class = len(cell_type_dict.keys())

    #train valid split 9:1
    torch.manual_seed(seed)
    shuffle_index = torch.randperm(adata_train.shape[0])
    train_index = shuffle_index[:int(adata_train.shape[0]*0.9)]
    val_index = shuffle_index[int(adata_train.shape[0]*0.9):]

    adata_val = adata_train[val_index.numpy()]
    adata_train = adata_train[train_index.numpy()]

    print('train size:',adata_train.shape[0])
    print('val size:',adata_val.shape[0])
    print('test size:',adata_test.shape[0])

    train_dataset = SCDataset_tune(
        data = adata_train,
        gene_token = gene2tok_dict,
        shuffle = shuffle,
        cell_type_dict = cell_type_dict,
        clip = clip,
        long = long
    )

    gene2tok_dict = train_dataset.gene_token

    val_dataset = SCDataset_tune(
        data = adata_val,
        gene_token = gene2tok_dict,
        shuffle = shuffle,
        cell_type_dict = cell_type_dict,
        clip = clip,
        long = long
    )

    test_dataset = SCDataset_tune(
        data = adata_test,
        gene_token = gene2tok_dict,
        shuffle = shuffle,
        cell_type_dict = cell_type_dict,
        clip = clip,
        long = long
    )
    
    collate = collate_fn_tune
    class_sample_count = np.array([len(np.where(train_dataset.celltype_y==t)[0]) for t in np.unique(train_dataset.celltype_y)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in train_dataset.celltype_y])
    train_sampler = WeightedRandomSampler(samples_weight, 2 * len(samples_weight))

    # save gene2tok_dict
    if balanced:
        return DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate, drop_last=True,num_workers=num_workers),\
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=True,num_workers=num_workers),\
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False,num_workers=num_workers),gene2tok_dict,num_class,cell_type_dict
    else:
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True,num_workers=num_workers),\
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=True,num_workers=num_workers),\
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False,num_workers=num_workers),gene2tok_dict,num_class,cell_type_dict


if __name__ == '__main__':
    data_path =['/cluster/home/hanwen/scmodel/scdata/test_data_set/zheng68k_p_filtered.h5ad']
    with open('/cluster/home/hanwen/scmodel/scdata/cellXgene/Gid2tok.json','r') as f:
        gene2id_dict = json.load(f)

    trainloader,valloader,testloader,gene2id_dict,num_class,cell_type_dict = get_tune_dataloader(
            data_path=data_path,
            gene2tok_dict=gene2id_dict,
            batch_size=16,
            shuffle = True,
            num_workers=1,
            log_transform = True,
        )
    for i, (count_x, gene_x, celltype_y) in enumerate(trainloader):
        print(count_x.shape,gene_x.shape,celltype_y.shape)
        break
        
    