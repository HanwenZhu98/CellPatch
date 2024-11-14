import torch 
import torch.nn as nn

from local_attention import LocalAttention
from performer_pytorch.performer_pytorch import *

class PathFormer_annotator(nn.Module):
    def __init__(
            self,
            max_gene_num,
            embedding_dim,
            num_class,
            max_trainning_length = 3000,
            n_pathway = 128,
            n_head = 4,
            mlp_dim = None,
            encoder_cross_attn_depth = 1,
            encoder_self_attn_depth = 1,
            encoder_projection_dim = None,

            ):
        super().__init__()
        self.max_gene_num = max_gene_num
        self.embedding_dim = embedding_dim
        self.max_trainning_length = max_trainning_length
        self.n_pathway =n_pathway
        self.n_head = n_head
        self.num_class = num_class

        self.Embedding_block = Embedding_block(self.embedding_dim,self.max_gene_num)

        self.PathwayEncoder = Pathway_encoder(
            embedding_dim=self.embedding_dim,
            n_pathway=self.n_pathway,
            cross_attn_depth=encoder_cross_attn_depth,
            self_attn_depth=encoder_self_attn_depth,
            mlp_dim=mlp_dim,
            projection_dim=encoder_projection_dim,
            n_head=self.n_head
        )

        self.Downstream = Annotation_decoder(
            num_class = self.num_class,
            encoder_projection_dim = encoder_projection_dim,
            embedding_dim = self.embedding_dim,
            mlp_dim = mlp_dim,
            n_head = self.n_head,
        )

    def forward(self, count_x, gene_x):
        count_e,gene_e = self.Embedding_block(count_x),self.Embedding_block(gene_x,gene=True)
        pathway_emb = self.PathwayEncoder(count_e,gene_e)
        pred = self.Downstream(pathway_emb)
        return pred
    
    def get_embeding(self, count_x, gene_x,output_attentions = False):
        count_e,gene_e = self.Embedding_block(count_x),self.Embedding_block(gene_x,gene=True)
        if output_attentions:
            pathway_emb,attn_out = self.PathwayEncoder(count_e,gene_e,output_attentions = output_attentions)
            return pathway_emb,attn_out
        else:
            pathway_emb = self.PathwayEncoder(count_e,gene_e)
            return pathway_emb

    def get_pred(self,pathway_emb,output_attentions = False):
        if output_attentions:
            pred,attn_out = self.Downstream(pathway_emb,output_attentions = output_attentions)
            return pred,attn_out
        else:
            pred = self.Downstream(pathway_emb)
            return pred
    
class PathFormer_pretrain(nn.Module):
    '''
    - encoder projection dim : final pathway embedding dimention . if None, no projection layer in encoder
    '''
    def __init__(
            self,
            max_gene_num,
            embedding_dim,
            max_trainning_length = 3000,
            n_pathway = 128,
            n_head = 4,
            mlp_dim = None,
            encoder_cross_attn_depth = 1,
            encoder_self_attn_depth = 1,
            encoder_projection_dim = None,
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

        self.Embedding_block = Embedding_block(self.embedding_dim,self.max_gene_num)

        self.PathwayEncoder = Pathway_encoder(
            embedding_dim=self.embedding_dim,
            n_pathway=self.n_pathway,
            cross_attn_depth=encoder_cross_attn_depth,
            self_attn_depth=encoder_self_attn_depth,
            mlp_dim=mlp_dim,
            projection_dim=encoder_projection_dim,
            n_head=self.n_head
        )

        self.PathwayDecoder = Pathway_decoder(
            encoder_projection_dim = encoder_projection_dim,
            embedding_dim = self.embedding_dim,
            decoder_extract_block_depth = decoder_extract_block_depth,
            decoder_selfattn_block_depth = decoder_selfattn_block_depth,
            mlp_dim = mlp_dim,
            n_head = self.n_head,
            predictor_dim=decoder_projection_dim,
        )
    
    def get_mid_embeding(self, count_x, gene_x,output_attentions = False):
        count_e,gene_e = self.Embedding_block(count_x),self.Embedding_block(gene_x,gene=True)
        if output_attentions:
            pathway_emb,attn_out = self.PathwayEncoder(count_e,gene_e,output_attentions = output_attentions)
            return pathway_emb,attn_out
        else:
            pathway_emb = self.PathwayEncoder(count_e,gene_e)
            return pathway_emb

    def get_gene_prediction(self, gene_x,pathway_emb,output_attentions = False):
        '''
        get gene prediction : (batch_size, seq_len) -> (batch_size, seq_len, gene_num)
        '''
        gene_e = self.Embedding_block(gene_x,gene=True)
        return self.PathwayDecoder(pathway_emb,gene_e) if not output_attentions else self.PathwayDecoder(pathway_emb,gene_e,output_attentions = output_attentions)
    
    def forward(self, count_e, gene_e, gene_d):

        count_e,gene_e = self.Embedding_block(count_e),self.Embedding_block(gene_e,gene=True)

        pathway_emb = self.PathwayEncoder(count_e,gene_e) # batch_size, n_pathway, projection_dim

        gene_d_emb = self.Embedding_block(gene_d,gene=True)
        pred_d = self.PathwayDecoder(pathway_emb,gene_d_emb,output_attentions = False)

        return pred_d
    
    # def forward(self, count_x, gene_x, nonzero_pos_bool, mask_pos_bool):
    #     #get masked count_x and gene_x , nonzero and not masked
    #     encode_mask = (~nonzero_pos_bool)|(mask_pos_bool)
    #     count_e,gene_e = self.get_masked_tensor(count_x,encode_mask),self.get_masked_tensor(gene_x,encode_mask)
    #     #get embedding : (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
    #     count_e,gene_e = self.Embedding_block(count_e),self.Embedding_block(gene_e,gene=True)
    #     pathway_emb = self.PathwayEncoder(count_e,gene_e) # batch_size, n_pathway, projection_dim
    #     #get masked count_x and gene_x , masked
    #     gene_d_emb = self.Embedding_block(gene_d,gene=True)
    #     pred_d = self.PathwayDecoder(pathway_emb,gene_d_emb,output_attentions = False)
    #     return count_d,pred_d,gene_d

    def get_masked_tensor(self,x,mask,max_length = None):
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
        elif max_length is None and self.max_trainning_length is not None:
            max_length = self.max_trainning_length
            x_masked = x_masked[:,:max_length]
        elif max_length is None and self.max_trainning_length is None:
            print('max_length is None and self.max_trainning_length is None, use the max length of the batch')
        return x_masked


class Pathway_encoder(nn.Module):
    def __init__(
            self,
            embedding_dim,
            n_pathway,
            cross_attn_depth = 1,
            self_attn_depth = 2,
            mlp_dim = None,
            projection_dim = None,
            n_head = 4,
        ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_pathway = n_pathway
        self.mlp_dim = default(mlp_dim,embedding_dim * 2)
        self.cross_attn_depth = cross_attn_depth
        self.self_attn_depth = self_attn_depth
        self.projection_dim = projection_dim

        if self.projection_dim is None:
            self.projection_dim = self.embedding_dim
            
        self.n_head = n_head

        self.pathway_embedding = nn.Embedding(self.n_pathway, self.embedding_dim)
        
        self.cross_extract_blocks = nn.ModuleList()
        for i in range(self.cross_attn_depth):
            #only last layer not use bidirectional
            bidirectional = False if i == self.cross_attn_depth-1 else True
            self.cross_extract_blocks.append(CrossUpdateBlock(
                embedding_dim = self.embedding_dim,
                n_head = self.n_head,
                bidirectional = bidirectional,
                mlp_dim = self.mlp_dim,
            ))

        self.self_attn_blocks = nn.ModuleList()
        for i in range(self.self_attn_depth):
            self.self_attn_blocks.append(SelfAttentionBLock(
                embedding_dim = self.embedding_dim,
                n_head = self.n_head,
                mlp_dim = self.mlp_dim,
                fast = False
            ))

        if self.projection_dim != None:
            self.projector = nn.Conv1d(self.embedding_dim,self.projection_dim,1)
        else:
            self.projector = None

    def forward(self, count_emb, gene_emb,output_attentions=False):
        pathway_emb = self.pathway_embedding.weight.unsqueeze(0).repeat(count_emb.shape[0],1,1)

        count_emb = count_emb + gene_emb
        if output_attentions:
            attn_out_total = {}

        for i,layer in enumerate(self.cross_extract_blocks):
            pathway_emb,count_emb,attn_out_0,attn_out_1 = layer(pathway_emb,count_emb,output_attentions = output_attentions)

            

            layer_name = 'encoder_cross_extract_blocks_'+str(i)
            if output_attentions:
                attn_out_total[layer_name+'_0'] = [attn_out_0]
                attn_out_total[layer_name+'_1'] = [attn_out_1]

        for i,layer in enumerate(self.self_attn_blocks):
            pathway_emb,attn_out = layer(pathway_emb,output_attentions = output_attentions)
            layer_name = 'encoder_self_attn_blocks_'+str(i)
            if output_attentions:
                attn_out_total[layer_name] = [attn_out]

        # print(pathway_emb.shape)  
        # print(pathway_emb[:,:10,:])  

        if self.projector is not None:
            pathway_emb = self.projector(pathway_emb.transpose(1,2)).transpose(1,2)

        if output_attentions:
            return pathway_emb,attn_out_total
        else:
            return pathway_emb


class Pathway_decoder(nn.Module):
    def __init__(
            self,
            encoder_projection_dim,
            embedding_dim,
            decoder_extract_block_depth = 2,
            decoder_selfattn_block_depth = 2,
            mlp_dim = None,
            n_head = 4,
            predictor_dim = 1,
            ) -> None:
        super().__init__()
        self.encoder_projection_dim = encoder_projection_dim
        self.embedding_dim = embedding_dim
        self.mlp_dim = default(mlp_dim,embedding_dim * 2)
        self.n_head = n_head
        self.predictor_dim = predictor_dim

        if self.encoder_projection_dim is not None:
            self.projection_layer = nn.Conv1d(self.encoder_projection_dim,self.embedding_dim,1)

        self.decode_extraction_blocks = nn.ModuleList()
        for i in range(decoder_extract_block_depth):
            if i != decoder_extract_block_depth-1:
                bidirectional = True
            else:
                bidirectional = False
            self.decode_extraction_blocks.append(CrossUpdateBlock(
                embedding_dim = self.embedding_dim,
                n_head = self.n_head,
                bidirectional = bidirectional,
                mlp_dim = self.mlp_dim,
            ))

        self.self_attn_blocks = nn.ModuleList()
        for i in range(decoder_selfattn_block_depth):
            self.self_attn_blocks.append(SelfAttentionBLock(
                embedding_dim = self.embedding_dim,
                n_head = self.n_head,
                mlp_dim = self.mlp_dim,
                fast = True
            ))

        self.to_out = nn.ModuleList()
        self.to_out.append(nn.Dropout(0.1))
        self.to_out.append(nn.Linear(self.embedding_dim,self.predictor_dim))
        self.to_out.append(nn.ReLU())
    
    def forward(self, pathway_embedding,gene_embedding,output_attentions = False):
    
        attn_out_total = {}

        if self.encoder_projection_dim is not None:
            pathway_embedding = self.projection_layer(pathway_embedding.transpose(1,2)).transpose(1,2)

        for i,layer in enumerate(self.decode_extraction_blocks):
            gene_embedding, pathway_embedding, attn_0, attn_1 = layer(gene_embedding,pathway_embedding, output_attentions = output_attentions)
            attn_out_total['decoder_extract_blocks_'+str(i)+'_0'] = [attn_0]
            attn_out_total['decoder_extract_blocks_'+str(i)+'_1'] = [attn_1]

        for i,layer in enumerate(self.self_attn_blocks):
            gene_embedding, attn = layer(gene_embedding, output_attentions = output_attentions)
            attn_out_total['decoder_self_attn_blocks_'+str(i)] = [attn]

        for layer in self.to_out:
            gene_embedding = layer(gene_embedding)

        return gene_embedding if not output_attentions else (gene_embedding,attn_out_total)
    


class Annotation_decoder(nn.Module):
    def __init__(
            self,
            num_class,
            encoder_projection_dim,
            embedding_dim,
            mlp_dim = None,
            n_head = 4,
            ) -> None:
        super().__init__()
        self.encoder_projection_dim = encoder_projection_dim
        self.embedding_dim = embedding_dim
        self.mlp_dim = default(mlp_dim,embedding_dim * 2)
        self.n_head = n_head
        self.num_class = num_class

        self.class_token = nn.Embedding(1,self.embedding_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim = self.embedding_dim,
            num_heads = self.n_head,
            batch_first=True
        )
        self.norm0 = nn.LayerNorm(self.embedding_dim)
        self.mlp0 = MLPBlock(self.embedding_dim,self.mlp_dim)
        self.norm1 = nn.LayerNorm(self.embedding_dim)
        self.dropout = nn.Dropout(0.2)
        self.predictor = nn.Linear(self.embedding_dim,self.num_class)
    
    def forward(self, pathway_embedding,output_attentions = False):
        bs = pathway_embedding.shape[0]
        class_token = self.class_token.weight.unsqueeze(0).repeat(bs,1,1)
        attn_out,attn_weights = self.cross_attn(class_token,pathway_embedding,pathway_embedding)
        class_token = self.norm0(class_token + attn_out)
        mlp_out = self.mlp0(class_token)
        class_token = self.norm1(class_token + mlp_out)
        class_token = self.dropout(class_token)
        class_token = self.predictor(class_token)
        if output_attentions:
            return class_token,attn_weights
        else:
            return class_token


class SelfAttentionBLock(nn.Module):
    def __init__(
            self,
            embedding_dim,
            n_head,
            mlp_dim = None,
            fast = False,
        ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.mlp_dim = default(mlp_dim,embedding_dim * 2)
        self.fast = fast
        if fast:
            self.self_attn = nSelfAttention(
                dim = self.embedding_dim,
                heads = self.n_head,
            )
        else:
            self.self_attn = nn.MultiheadAttention(
                embed_dim = self.embedding_dim,
                num_heads = self.n_head,
                batch_first=True
            )

        self.norm0 = nn.LayerNorm(self.embedding_dim)
        self.mlp0 = MLPBlock(self.embedding_dim,self.mlp_dim)
        self.norm1 = nn.LayerNorm(self.embedding_dim)

    def forward(self, query, query_pe = None, output_attentions = False):
        if query_pe is not None:
            query = query+query_pe
        
        if self.fast:
            attn_out,attn_weights = self.self_attn(query,output_attentions = output_attentions)
        else:
            attn_out,attn_weights = self.self_attn(query,query,query,need_weights = output_attentions)
        
        #add&norm
        query = query + attn_out
        query = self.norm0(query)
        #mlp
        mlp_out = self.mlp0(query)
        #add&norm
        query = query + mlp_out
        query = self.norm1(query)

        if output_attentions:
            return query,attn_weights
        else:
            return query,None


class CrossUpdateBlock(nn.Module):
    def __init__(
            self,
            embedding_dim,
            n_head,
            bidirectional = True,
            mlp_dim = None,
        ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.bidirectional = bidirectional
        self.mlp_dim = default(mlp_dim,embedding_dim * 2)

        self.cross_attn = nCrossAttention(
            dim = self.embedding_dim,
            heads = self.n_head,
        )
        self.norm0 = nn.LayerNorm(self.embedding_dim)
        self.mlp0 = MLPBlock(self.embedding_dim,self.mlp_dim)
        self.norm1 = nn.LayerNorm(self.embedding_dim)
        
        if self.bidirectional:
            self.cross_attn_bi = nCrossAttention(
                dim = self.embedding_dim,
                heads = self.n_head,
            )

        self.norm2 = nn.LayerNorm(self.embedding_dim)
        self.mlp1 = MLPBlock(self.embedding_dim,self.mlp_dim)
        self.norm3 = nn.LayerNorm(self.embedding_dim)

    def forward(self, query, context, output_attentions = False):
        # encoder input : pathway , count
        # if output_attentions = False attn_weights_i = None
        attn_out,attn_weights_0 = self.cross_attn(query,context = context,output_attentions = output_attentions)
        #add&norm
        query = self.norm0(query + attn_out)
        #mlp
        #add&norm
        mlp_out = self.mlp0(query)
        query = self.norm1(query + mlp_out)

        #bidirectional
        if self.bidirectional:
            attn_out,attn_weights_1 = self.cross_attn_bi(context,context = query,output_attentions = output_attentions)

            #add&norm
            context = context + attn_out
            context = self.norm2(context)
            #mlp
            mlp_out = self.mlp1(context)
            #add&norm
            context = context + mlp_out
            context = self.norm3(context)
        else:
            attn_weights_1 = None
        if output_attentions:
            return query,context,attn_weights_0,attn_weights_1
        else:
            return query,context,None,None


class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act= nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class nFastAttention(FastAttention):
    def forward(self, q, k, v, output_attentions = False):
        device = q.device
        
        if self.no_projection:
            q = q.softmax(dim = -1)
            k = torch.exp(k) if self.causal else k.softmax(dim = -2)

        elif self.generalized_attention:
            create_kernel = partial(generalized_kernel, kernel_fn = self.kernel_fn, projection_matrix = self.projection_matrix, device = device)
            q, k = map(create_kernel, (q, k))

        else:
            create_kernel = partial(softmax_kernel, projection_matrix = self.projection_matrix, device = device)
            q = create_kernel(q, is_query = True)
            k = create_kernel(k, is_query = False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        out = attn_fn(q, k, v)
        if output_attentions:
            # 创建一个与v的维度匹配的单位矩阵
            v_diag = torch.eye(v.shape[-2]).to(device)
            v_diag = v_diag.unsqueeze(0).unsqueeze(0).repeat(v.shape[0],v.shape[1],1,1)

            # 初始化注意力权重张量
            attn_weights = torch.zeros(1, q.shape[0], q.shape[2], k.shape[2]).to(device).to(torch.float32)
            # 遍历每个头部维度来计算注意力权重
            for head_dim in range(q.shape[1]):
                # 应用注意力函数并计算绝对值
                head_attn_weights = attn_fn(q[:,head_dim].to(torch.float32), k[:,head_dim].to(torch.float32), v_diag[:,head_dim].to(torch.float32))
                # 累加到总的注意力权重中
                attn_weights += head_attn_weights
            attn_weights /= q.shape[1]

            return out, attn_weights
        else:
            return out

        # if output_attentions:
        #     v_diag = torch.eye(v.shape[-2]).to(device)
        #     v_diag = v_diag.unsqueeze(0).unsqueeze(0).repeat(v.shape[0],v.shape[1],1,1)
        #     # attn_weights = torch.zeros(1, 1, len(inds), len(inds)).to(device).to(torch.float16)
        #     # attn_weights = torch.zeros(1, q.shape[1], len(inds), len(inds)).to(device).to(torch.float16)
        #     attn_weights = torch.zeros(1, q.shape[0], q.shape[2], k.shape[2]).to(device).to(torch.float16)

        #     for head_dim in range(q.shape[1]):
        #         # attn_weights[0, head_dim] = torch.abs(attn_fn(q[:,head_dim].to(torch.float16), k[:,head_dim].to(torch.float16), v_diag[:,head_dim].to(torch.float16)))[0, inds][:, inds]
        #         attn_weights += torch.abs(attn_fn(q[:,head_dim].to(torch.float32), k[:,head_dim].to(torch.float32), v_diag[:,head_dim].to(torch.float32)))
        #         # attn_weights += norm_tensor(torch.abs(attn_fn(q[:,head_dim].to(torch.float16), k[:,head_dim].to(torch.float16), v_diag[:,head_dim].to(torch.float16))), dim=-1)
        #     attn_weights /= q.shape[1] # average attention weights over heads
        #     return out, attn_weights
        # else:
        #     return out 


class nAttention(nn.Module):
    #modified from performer_pytorch
    def __init__(
        self,
        dim,
        causal = False,
        heads = 8,
        dim_head = 64,
        local_heads = 0,
        local_window_size = 256,
        nb_features = None,
        feature_redraw_interval = 1000,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        dropout = 0.,
        no_projection = False,
        qkv_bias = False,
        attn_out_bias = True
    ):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        inner_dim = dim_head * heads
        self.fast_attention = nFastAttention(dim_head, nb_features, causal = causal, generalized_attention = generalized_attention, kernel_fn = kernel_fn, no_projection = no_projection)

        self.heads = heads
        self.global_heads = heads - local_heads
        self.local_attn = LocalAttention(window_size = local_window_size, causal = causal, autopad = True, dropout = dropout, look_forward = int(not causal), rel_pos_emb_config = (dim_head, local_heads)) if local_heads > 0 else None

        self.to_q = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_k = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_v = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias = attn_out_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb = None, context = None, mask = None, context_mask = None, output_attentions = False,**kwargs):
        b, n, _, h, gh = *x.shape, self.heads, self.global_heads

        cross_attend = exists(context)

        context = default(context, x)
        context_mask = default(context_mask, mask) if not cross_attend else context_mask

        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        attn_outs = []

        if not empty(q):
            if exists(context_mask):
                global_mask = context_mask[:, None, :, None]
                v.masked_fill_(~global_mask, 0.)

            if exists(pos_emb) and not cross_attend:
                q, k = apply_rotary_pos_emb(q, k, pos_emb)
            if output_attentions:
                out,attn_weights = self.fast_attention(q, k, v, output_attentions = output_attentions)
            else:
                out = self.fast_attention(q, k, v)
            attn_outs.append(out)

        if not empty(lq):
            assert not cross_attend, 'local attention is not compatible with cross attention'
            out = self.local_attn(lq, lk, lv, input_mask = mask)
            attn_outs.append(out)

        out = torch.cat(attn_outs, dim = 1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        if output_attentions:
            return self.dropout(out),attn_weights
        else:
            return self.dropout(out),None
        

class nSelfAttention(nAttention):
    def forward(self, *args, context = None, **kwargs):
        assert not exists(context), 'self attention should not receive context'
        return super().forward(*args, **kwargs)


class nCrossAttention(nAttention):
    def forward(self, *args, context = None, **kwargs):
        assert exists(context), 'cross attention should receive context'
        return super().forward(*args, context = context, **kwargs)


class Embedding_block(nn.Module):
    def __init__(self,embedding_dim,max_gene_num):
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