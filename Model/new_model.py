import torch
import torch.nn as nn
import math

class GGFormer_lp(nn.Module):
    def __init__(
            self,
            max_gene_num,
            embedding_dim,
            max_trainning_length,
            n_pathway,
            n_head,
            mlp_dim,
            encoder_cross_attn_depth ,
            encoder_self_attn_depth,
            encoder_projection_dim,
            decoder_extract_block_depth,
            decoder_selfattn_block_depth,
            decoder_projection_dim,
            n_cell_type,
            dropout,
            project_out,
            clip = None
        ):
        super(GGFormer_lp, self).__init__()
        self.max_gene_num = max_gene_num
        self.embedding_dim = embedding_dim
        self.max_trainning_length = max_trainning_length
        self.n_pathway = n_pathway
        self.n_head = n_head
        self.mlp_dim = mlp_dim if mlp_dim is not None else 2 * self.embedding_dim
        self.encoder_cross_attn_depth = encoder_cross_attn_depth
        self.encoder_self_attn_depth = encoder_self_attn_depth
        self.encoder_projection_dim = encoder_projection_dim
        self.decoder_extract_block_depth = decoder_extract_block_depth
        self.decoder_selfattn_block_depth = decoder_selfattn_block_depth
        self.decoder_projection_dim = decoder_projection_dim
        self.clip_count = clip+1
        # self.count_dropout = nn.Dropout(0.)
        self.count2vector = nn.Linear(1, self.embedding_dim)
        self.count2vector_norm = nn.LayerNorm(self.embedding_dim)
        # self.count2vector = nn.Embedding(self.clip_count, self.embedding_dim)
        self.gene2vector = nn.Embedding(self.max_gene_num, self.embedding_dim)
        self.pathway_token = nn.Parameter(torch.randn(1,self.n_pathway,self.embedding_dim))
        self.n_cell_type = n_cell_type

        #encoder
        self.encoder = GGencoder(
            self.embedding_dim,
            self.n_head,
            self.mlp_dim,
            self.encoder_self_attn_depth,
            self.encoder_projection_dim
        )

        self.ct_projector = nn.Linear(self.embedding_dim, 1)

        self.decoder = GGdecoder_lp(
            self.embedding_dim,
            self.n_pathway,
            self.encoder_projection_dim,
            self.n_cell_type,
            dropout
        )
    
    def get_mid_embeding(self, count_x, gene_x,output_attentions = False):
        # count_x = self.count_dropout(count_x)
        val_x = self.count2vector(count_x.unsqueeze(2))
        # val_x = self.count2vector(count_x)
        # val_x = self.count2vector_norm(val_x)
        id_x = self.gene2vector(gene_x) 
        # id_x *= math.sqrt(self.embedding_dim)
        val_x = val_x + id_x
        val_x = self.encoder(self.pathway_token, val_x)
        return val_x
    
    def train_forward(self, count_e, gene_e, gene_d=None):
        mid_embedding = self.get_mid_embeding(count_e, gene_e)
        contrast_out = self.ct_projector(mid_embedding)
        out = self.decoder(mid_embedding)
        return contrast_out, out

    def forward(self, count_e, gene_e, gene_d=None):
        mid_embedding = self.get_mid_embeding(count_e, gene_e)
        out = self.decoder(mid_embedding)
        return out


class GGencoder(nn.Module):
    def __init__(self, embedding_dim,n_head, mlp_dim, depth, projection_dim):
        super(GGencoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.mlp_dim = mlp_dim
        self.depth = depth
        self.projection_dim = projection_dim

        #cross extraction attention
        self.cross_extract_blocks_attn = nn.MultiheadAttention(
            self.embedding_dim,
            self.n_head,
            dropout = 0.1,
            batch_first=True
        )
        #add&norm
        self.cross_extract_blocks_attn_norm = nn.LayerNorm(self.embedding_dim)
        #feed forward
        self.cross_extract_blocks_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.embedding_dim)
        )
        #add&norm
        self.cross_extract_blocks_mlp_norm = nn.LayerNorm(self.embedding_dim)

        #self attention
        self.self_attention_blocks = nn.ModuleList([
            nn.MultiheadAttention(
                self.embedding_dim,
                self.n_head,
                dropout = 0.1,
                batch_first=True
            ) for _ in range(self.depth)
        ])
        #add&norm
        self.self_attention_blocks_norm = nn.ModuleList([
            nn.LayerNorm(self.embedding_dim) for _ in range(self.depth)
        ])
        #feed forward
        self.self_mlp_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embedding_dim, self.mlp_dim),
                nn.ReLU(),
                nn.Linear(self.mlp_dim, self.embedding_dim)
            ) for _ in range(self.depth)
        ])
        #add&norm
        self.self_mlp_blocks_norm = nn.ModuleList([
            nn.LayerNorm(self.embedding_dim) for _ in range(self.depth)
        ])

        if self.projection_dim is not None:
            self.projection = nn.Linear(self.embedding_dim, self.projection_dim)


    def forward(self, pathway_emb,cell_emb):
        pathway_emb = pathway_emb.repeat(cell_emb.shape[0],1,1)
        attn_out, attn_weights = self.cross_extract_blocks_attn(
            pathway_emb,
            cell_emb,
            cell_emb
        )
        attn_out = self.cross_extract_blocks_attn_norm(attn_out + pathway_emb)
        mlp_out = self.cross_extract_blocks_mlp(attn_out)
        mlp_out = self.cross_extract_blocks_mlp_norm(mlp_out + attn_out)

        for i in range(self.depth):
            attn_out, attn_weights = self.self_attention_blocks[i](
                mlp_out,
                mlp_out,
                mlp_out
            )
            attn_out = self.self_attention_blocks_norm[i](attn_out + mlp_out)
            mlp_out = self.self_mlp_blocks[i](attn_out)
            mlp_out = self.self_mlp_blocks_norm[i](mlp_out + attn_out)
        if self.projection_dim is not None:
            mlp_out = self.projection(mlp_out)
        return mlp_out

class GGdecoder_lp(nn.Module):
    def __init__(self, embedding_dim, n_pathway,encoder_projection_dim,n_class,dropout):
        super(GGdecoder_lp, self).__init__()

        self.embedding_dim = embedding_dim
        self.projection_dim = encoder_projection_dim
        self.n_class = n_class
        self.n_pathway = n_pathway

        if self.projection_dim is not None:
            self.projection = nn.Linear(self.projection_dim, self.embedding_dim)
        
        self.lp_dropout = nn.Dropout(dropout)
        self.lp_fc1 = nn.Linear(self.n_pathway * self.embedding_dim, 512)
        self.lp_act1 = nn.ReLU()
        self.lp_dropout1 = nn.Dropout(dropout)
        self.lp_fc2 = nn.Linear(512, 100)
        self.lp_act2 = nn.ReLU()
        self.lp_dropout2 = nn.Dropout(dropout)
        self.lp_fc3 = nn.Linear(100, self.n_class)

    def forward(self, pathway_emb):
        if self.projection_dim is not None:
            pathway_emb = self.projection(pathway_emb)
        pathway_emb = pathway_emb.view(pathway_emb.shape[0],-1)

        mlp_out = self.lp_fc1(self.lp_dropout(pathway_emb))
        mlp_out = self.lp_act1(mlp_out)
        mlp_out = self.lp_dropout1(mlp_out)
        mlp_out = self.lp_fc2(mlp_out)
        mlp_out = self.lp_act2(mlp_out)
        mlp_out = self.lp_dropout2(mlp_out)
        mlp_out = self.lp_fc3(mlp_out)
        return mlp_out