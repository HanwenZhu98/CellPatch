import torch
import torch.nn as nn




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
            decoder_selfattn_block_depth = 1,
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

        # decoder module
        self.decoder_extract_block_depth = decoder_extract_block_depth
        self.decoder_selfattn_block_depth = decoder_selfattn_block_depth
        self.decoder_projection_dim = decoder_projection_dim

        if self.encoder_projection_dim is not None:
            self.decoder_reverse_projection = nn.Linear(self.encoder_projection_dim,self.embedding_dim)
        
        # decoder extract block
        self.decoder_extract_blocks_attn = nn.MultiheadAttention(
            self.embedding_dim,
            self.n_head,
            dropout = 0.1,
            batch_first=True
        )
        self.decoder_extract_blocks_attn_norm = nn.LayerNorm(self.embedding_dim)
        self.decoder_extract_blocks_mlp = nn.Sequential(
            nn.Linear(self.embedding_dim,self.mlp_dim),
            nn.GELU(),
            nn.Linear(self.mlp_dim,self.embedding_dim)
        )
        self.decoder_extract_blocks_mlp_norm = nn.LayerNorm(self.embedding_dim)

        # decoder self attention module
        self.decoder_selfattn_blocks_attn = nn.ModuleList([
            nn.MultiheadAttention(
                self.embedding_dim,
                self.n_head,
                dropout = 0.1,
                batch_first=True
            ) for _ in range(self.decoder_selfattn_block_depth)
        ])
        self.decoder_selfattn_blocks_attn_norm = nn.ModuleList([
            nn.LayerNorm(self.embedding_dim) for _ in range(self.decoder_selfattn_block_depth)
        ])
        self.decoder_selfattn_blocks_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embedding_dim,self.mlp_dim),
                nn.GELU(),
                nn.Linear(self.mlp_dim,self.embedding_dim)
            ) for _ in range(self.decoder_selfattn_block_depth)
        ])
        self.decoder_selfattn_blocks_mlp_norm = nn.ModuleList([
            nn.LayerNorm(self.embedding_dim) for _ in range(self.decoder_selfattn_block_depth)
        ])
        if self.decoder_projection_dim is not None:
            self.dropout = nn.Dropout(p=0.1)
            self.decoder_projection = nn.Linear(self.embedding_dim,self.decoder_projection_dim)

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

    def get_gene_prediction(self, gene_x,pathway_emb,output_attentions = False):
        '''
        - gene_x : [batch_size, max_gene_num]
        - pathway_emb : [batch_size, n_pathway, embedding_dim]
        - output_attentions : bool type, whether need attention score output.
        '''
        gene_e = self.gene2vector(gene_x)
        pathway_emb = self.decoder_reverse_projection(pathway_emb)

        attn_out,attn_score = self.decoder_extract_blocks_attn(query = gene_e,key = pathway_emb,value = pathway_emb)
        #这一步是否因该加上gene_e 做residual
        gene_e = self.decoder_extract_blocks_attn_norm(attn_out + gene_e)
        gene_e = self.decoder_extract_blocks_mlp_norm(self.decoder_extract_blocks_mlp(gene_e) + gene_e)
        attn_out_total = {}

        for i in range(self.decoder_selfattn_block_depth):
            attn_out,attn_score = self.decoder_selfattn_blocks_attn[i](query = gene_e,key = gene_e,value = gene_e)
            gene_e = self.decoder_selfattn_blocks_attn_norm[i](attn_out + gene_e)
            gene_e = self.decoder_selfattn_blocks_mlp_norm[i](self.decoder_selfattn_blocks_mlp[i](gene_e) + gene_e)
            if output_attentions:
                attn_out_total['encoder_cross_extract_blocks_'+str(i)] = attn_score

        if self.decoder_projection_dim is not None:
            gene_e = self.dropout(gene_e)
            gene_e = self.decoder_projection(gene_e)
        
        return gene_e if not output_attentions else (gene_e,attn_out_total)

    def forward(self, count_e, gene_e, gene_d):
        '''
        - count_e : [batch_size, max_gene_num, embedding_dim]
        - gene_e  : [batch_size, max_gene_num, embedding_dim]
        - gene_d  : [batch_size, max_gene_num]
        '''
        pathway_emb = self.get_mid_embeding(count_e, gene_e)
        gene_d = self.get_gene_prediction(gene_d,pathway_emb)
        return gene_d


class PathFormer_lp(nn.Module):
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
            decoder_selfattn_block_depth = 1,
            decoder_projection_dim = 1,
            n_cell_type = None,
            dropout = 0.,
            project_out = True
            ):
        super().__init__()

        self.max_gene_num = max_gene_num
        self.embedding_dim = embedding_dim
        self.max_trainning_length = max_trainning_length
        self.n_pathway =n_pathway
        self.n_head = n_head
        self.n_cell_type = n_cell_type
        self.mlp_dim = embedding_dim * 2 if mlp_dim is None else mlp_dim
        self.dropout = dropout
        # embedding module
        self.count2vector = nn.Linear(1, self.embedding_dim)
        self.count2vector_norm = nn.LayerNorm(self.embedding_dim)
        self.gene2vector = nn.Embedding(self.max_gene_num, self.embedding_dim)
        
        self.emb_dropout = nn.Dropout(self.dropout)

        # encoder module
        self.encoder_cross_attn_depth = encoder_cross_attn_depth
        self.encoder_self_attn_depth = encoder_self_attn_depth
        self.encoder_projection_dim = encoder_projection_dim
        self.project_out = project_out

        # decoder module
        self.decoder_extract_block_depth = decoder_extract_block_depth
        self.decoder_selfattn_block_depth = decoder_selfattn_block_depth
        self.decoder_projection_dim = decoder_projection_dim

        
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

        # linear probing module
        if self.encoder_projection_dim is not None:
            self.lp_linear_probe_0 = nn.Linear(self.encoder_projection_dim,self.embedding_dim)
            self.lp_norm = nn.LayerNorm(self.embedding_dim)
        
        if self.encoder_projection_dim is not None:
            self.decoder_reverse_projection = nn.Linear(self.encoder_projection_dim,self.embedding_dim)

        self._init_weights()

        self.lp_fc1 = nn.Linear(self.n_pathway * self.embedding_dim, 512)
        self.lp_act1 = nn.ReLU()
        self.lp_dropout1 = nn.Dropout(dropout)
        self.lp_fc2 = nn.Linear(512, 100)
        self.lp_act2 = nn.ReLU()
        self.lp_dropout2 = nn.Dropout(dropout)
        self.lp_fc3 = nn.Linear(100, self.n_cell_type)

        # self._init_weights()

    def _init_weights(self):
        # Xavier或He初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                nn.init.constant_(m.in_proj_bias, 0)
        print('init weight')

    def get_mid_embeding(self, count_x, gene_x,output_attentions = False, project_out = True):
        '''
        - count_x : [batch_size, max_gene_num]
        - gene_x  : [batch_size, max_gene_num]
        - output_attentions : bool type, whether need attention score output.
        '''
        count_e = self.count2vector(count_x.unsqueeze(2))
        gene_e = self.gene2vector(gene_x)
        attn_out_total = {}

        count_e = count_e + gene_e
        count_e = self.emb_dropout(count_e)

        pathway_emb = self.pathway_token.repeat(count_e.shape[0],1,1)
        attn_out,attn_score = self.cross_extract_blocks_attn(query = pathway_emb,key = count_e,value = count_e)
        
        # pathway_emb = self.cross_extract_blocks_attn_norm(attn_out + self.pathway_token)
        #这里可以试试不残差
        # pathway_emb = self.cross_extract_blocks_attn_norm(attn_out)
        pathway_emb = self.cross_extract_blocks_attn_norm(attn_out + pathway_emb)

        pathway_emb = self.cross_extract_blocks_mlp_norm(self.cross_extract_blocks_mlp(pathway_emb) + pathway_emb)

        if output_attentions:
            attn_out_total['encoder_cross_extract_blocks'] = attn_score
        for i in range(self.encoder_self_attn_depth):
            attn_out,attn_score = self.self_extract_blocks_attn[i](query = pathway_emb,key = pathway_emb,value = pathway_emb)
            pathway_emb = self.self_extract_blocks_attn_norm[i](attn_out + pathway_emb)
            pathway_emb = self.self_extract_blocks_mlp_norm[i](self.self_extract_blocks_mlp[i](pathway_emb) + pathway_emb)
            if output_attentions:
                attn_out_total['encoder_self_extract_blocks_'+str(i)] = attn_score
        if (self.encoder_projection_dim is not None) and project_out:
            pathway_emb = self.encoder_projection(pathway_emb)
        return pathway_emb if not output_attentions else (pathway_emb,attn_out_total)
    
    def get_logits(self,mid_emb,project_out = True):
        '''
        - mid_emb : [batch_size, n_pathway, embedding_dim]
        '''
        if project_out:
            logits = self.lp_linear_probe_0(mid_emb)
            logits = self.lp_norm(logits)
        else:
            logits = mid_emb

        logits = logits.view(logits.size(0), -1)

        logits = self.lp_fc1(logits)
        logits = self.lp_act1(logits)
        logits = self.lp_dropout1(logits)

        logits = self.lp_fc2(logits)
        logits = self.lp_act2(logits)
        logits = self.lp_dropout2(logits)

        logits = self.lp_fc3(logits)
        return logits

    def forward(self, count_e, gene_e, return_mid = False):
        '''
        - count_e : [batch_size, max_gene_num, embedding_dim]
        - gene_e  : [batch_size, max_gene_num, embedding_dim]
        - gene_d  : [batch_size, max_gene_num]
        '''
        pathway_emb = self.get_mid_embeding(count_e, gene_e,project_out = self.project_out)
        logits = self.get_logits(pathway_emb,project_out = self.project_out)
        # gene_d = self.get_gene_prediction(gene_e,pathway_emb)
        return logits.squeeze(1) if not return_mid else (logits.squeeze(1),pathway_emb)
        
if __name__ == '__main__':
    test_x = torch.randn(10, 1000)
    test_y = torch.randint(0, 20000, (10, 1000))

    test_z = torch.randint(0, 20000, (10, 1000))
    test_model = PathFormer(20000, 64)
    test_out = test_model(test_x, test_y, test_z)
    print(test_model)
    