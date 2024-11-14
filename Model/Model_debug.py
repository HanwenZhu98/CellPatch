import torch 
import torch.nn as nn

from local_attention import LocalAttention
from performer_pytorch.performer_pytorch import *

class PathFormer_pretrain(nn.Module):
    def __init__(
            self,
            embedding_dim,
            max_gene_num,
            ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_gene_num = max_gene_num

        self.Embedding_Module = Embedding_block(self.embedding_dim,self.max_gene_num)

        self.Encoder_Module = None

        self.Decoder_Module = None

    def get_input_embedding(self, count_x, gene_x):
        # count_x: (batch_size, count_dim)
        # gene_x: (batch_size, gene_dim)
        # return: (batch_size, input_dim)
        pass

    def get_pathway_embedding(self, count_x, gene_x):
        pass
    

    def forward(self,count_x,gene_x,gene_y):

        count_x,gene_x,gene_y = self.get_input_embedding(count_x),\
                                self.get_input_embedding(gene_x,gene = True),\
                                self.get_input_embedding(gene_y,gene = True)
        


class Embedding_block(nn.Module):
    def __init__(self,
                 embedding_dim,
                 max_gene_num
                 ):
        
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_gene_num = max_gene_num
        self.count2vector = nn.Conv1d(1, self.embedding_dim, 1)
        self.gene2vector = nn.Embedding(self.max_gene_num, self.embedding_dim)
    
    def forward(self, x, gene = False):
        if gene:
            return self.gene2vector(x)
        else:
            return self.count2vector(x.unsqueeze(1)).permute(0,2,1)
        
