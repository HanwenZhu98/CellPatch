import torch 
import torch.nn as nn

from local_attention import LocalAttention
from performer_pytorch.performer_pytorch import *

# needed module
class SelfAttentionBLock(nn.Module):
    def __init__(
            self,
            embedding_dim,
            n_head,
            mlp_dim = None,
            fast = False,
        ):
        super().__init__()
        self.embedding_dim = embedding_dim  #embedding dim
        self.n_head = n_head            #head num
        self.mlp_dim = default(mlp_dim,embedding_dim * 2) #mlp dim if None, mlp_dim = embedding_dim * 2
        self.fast = fast            #if fast, use performer_pytorch, else use nn.MultiheadAttention
        if fast:
            self.self_attn = nSelfAttention(
                dim = self.embedding_dim,
                heads = self.n_head,
            )
        else:
            self.self_attn = nn.MultiheadAttention(
                embed_dim = self.embedding_dim,
                num_heads = self.n_head,
            )

        self.norm0 = nn.LayerNorm(self.embedding_dim)
        self.mlp0 = MLPBlock(self.embedding_dim,self.mlp_dim)
        self.norm1 = nn.LayerNorm(self.embedding_dim)

    def forward(self, query, query_pe = None, output_attentions = False):
        '''
        self attention + add&norm + mlp + add&norm
        '''
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


class PathFormer_allin1(nn.Module):
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
        
        self.encoder_cross_attn_depth = encoder_cross_attn_depth
        self.encoder_self_attn_depth = encoder_self_attn_depth
        self.encoder_projection_dim = encoder_projection_dim

        self.count_embedding_layer = nn.Conv1d(1,embedding_dim,1)
        self.gene_embedding_layer = nn.Embedding(max_gene_num,embedding_dim)

        self.pathway_token = nn.Embedding(n_pathway,embedding_dim)

        #encoder
        self.crossattn_0 = CrossAttention(embedding_dim,heads=n_head)
        self.cross_norm_0 = nn.LayerNorm(embedding_dim)
        self.cross_mlp_0 = MLPBlock(embedding_dim,default(mlp_dim,embedding_dim * 2))
        self.cross_norm_1 = nn.LayerNorm(embedding_dim)

        self.crossattn_1 = CrossAttention(embedding_dim,heads=n_head)
        self.cross_norm_2 = nn.LayerNorm(embedding_dim)
        self.cross_mlp_1 = MLPBlock(embedding_dim,default(mlp_dim,embedding_dim * 2))
        self.cross_norm_3 = nn.LayerNorm(embedding_dim)

        self.crossattn_2 = CrossAttention(embedding_dim,heads=n_head)
        self.cross_norm_4 = nn.LayerNorm(embedding_dim)
        self.cross_mlp_2 = MLPBlock(embedding_dim,default(mlp_dim,embedding_dim * 2))
        self.cross_norm_5 = nn.LayerNorm(embedding_dim)

        self.Encoder_selfattn_blocks = nn.ModuleList()
        for i in range(encoder_self_attn_depth):
            self.Encoder_selfattn_blocks.append(SelfAttentionBLock(embedding_dim,n_head,default(mlp_dim,embedding_dim * 2),fast = False))

        if encoder_projection_dim != None:
            self.encoder_projector = nn.Conv1d(embedding_dim,encoder_projection_dim,1)

        #decoder
        if encoder_projection_dim != None:
            self.decoder_projector = nn.Conv1d(encoder_projection_dim,embedding_dim,1)

        self.crossattn_3 = CrossAttention(embedding_dim,heads=n_head)
        self.cross_norm_6 = nn.LayerNorm(embedding_dim)
        self.cross_mlp_3 = MLPBlock(embedding_dim,default(mlp_dim,embedding_dim * 2))
        self.cross_norm_7 = nn.LayerNorm(embedding_dim)

        self.crossattn_4 = CrossAttention(embedding_dim,heads=n_head)
        self.cross_norm_8 = nn.LayerNorm(embedding_dim)
        self.cross_mlp_4 = MLPBlock(embedding_dim,default(mlp_dim,embedding_dim * 2))
        self.cross_norm_9 = nn.LayerNorm(embedding_dim)

        self.crossattn_5 = CrossAttention(embedding_dim,heads=n_head)
        self.cross_norm_10 = nn.LayerNorm(embedding_dim)
        self.cross_mlp_5 = MLPBlock(embedding_dim,default(mlp_dim,embedding_dim * 2))
        self.cross_norm_11 = nn.LayerNorm(embedding_dim)

        self.Decoder_selfattn_blocks = nn.ModuleList()
        for i in range(decoder_selfattn_block_depth):
            self.Decoder_selfattn_blocks.append(SelfAttentionBLock(embedding_dim,n_head,default(mlp_dim,embedding_dim * 2),fast = True))

        self.decoder_projector = nn.ModuleList()
        self.decoder_projector.append(nn.Dropout(0.1))
        self.decoder_projector.append(nn.Linear(embedding_dim,decoder_projection_dim))
        self.decoder_projector.append(nn.ReLU())

    def forward(self,count_in,gene_in,gene_out):
        '''
        count_in : b * len
        gene_in : b * len


        '''
        count_emb = self.count_embedding_layer(count_in.unsqueeze(1)).transpose(1,2)
        gene_emb = self.gene_embedding_layer(gene_in)

        pathway_emb = self.pathway_token.weight.unsqueeze(0).repeat(count_emb.shape[0],1,1)
        count_emb = count_emb + gene_emb

        # pathway encoder
        attn_out,attn_weights = self.crossattn_0(pathway_emb,context = count_emb)
        pathway_emb = self.cross_norm_0(pathway_emb + attn_out)
        mlp_out = self.cross_mlp_0(pathway_emb)
        pathway_emb = self.cross_norm_1(pathway_emb + mlp_out)

        attn_out,attn_weights = self.crossattn_1(count_emb,context = pathway_emb)
        count_emb = self.cross_norm_2(count_emb + attn_out)
        mlp_out = self.cross_mlp_1(count_emb)
        count_emb = self.cross_norm_3(count_emb + mlp_out)

        attn_out,attn_weights = self.crossattn_2(pathway_emb,context = count_emb)
        pathway_emb = self.cross_norm_4(pathway_emb + attn_out)
        mlp_out = self.cross_mlp_2(pathway_emb)
        pathway_emb = self.cross_norm_5(pathway_emb + mlp_out)

        for self_attn_block in self.Encoder_selfattn_blocks:
            pathway_emb,_ = self_attn_block(pathway_emb)
        
        if self.encoder_projection_dim != None:
            pathway_emb = self.encoder_projector(pathway_emb.transpose(1,2)).transpose(1,2)

        #decoder
        if self.encoder_projection_dim != None:
            pathway_emb = self.decoder_projector(pathway_emb.transpose(1,2)).transpose(1,2)

        attn_out,attn_weights = self.crossattn_3(gene_emb,context = pathway_emb)
        gene_emb = self.cross_norm_6(gene_emb + attn_out)
        mlp_out = self.cross_mlp_3(gene_emb)
        gene_emb = self.cross_norm_7(gene_emb + mlp_out)

        attn_out,attn_weights = self.crossattn_4(pathway_emb,context = gene_emb)
        pathway_emb = self.cross_norm_8(pathway_emb + attn_out)
        mlp_out = self.cross_mlp_4(pathway_emb)
        pathway_emb = self.cross_norm_9(pathway_emb + mlp_out)

        attn_out,attn_weights = self.crossattn_5(gene_emb,context = pathway_emb)
        gene_emb = self.cross_norm_10(gene_emb + attn_out)
        mlp_out = self.cross_mlp_5(gene_emb)
        gene_emb = self.cross_norm_11(gene_emb + mlp_out)

        for self_attn_block in self.Decoder_selfattn_blocks:
            gene_emb,_ = self_attn_block(gene_emb)
        
        for projector in self.decoder_projector:
            gene_emb = projector(gene_emb)


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
            v_diag = torch.eye(v.shape[-2]).to(device)
            v_diag = v_diag.unsqueeze(0).unsqueeze(0).repeat(v.shape[0],v.shape[1],1,1)
            # attn_weights = torch.zeros(1, 1, len(inds), len(inds)).to(device).to(torch.float16)
            # attn_weights = torch.zeros(1, q.shape[1], len(inds), len(inds)).to(device).to(torch.float16)
            attn_weights = torch.zeros(1, q.shape[0], q.shape[2], k.shape[2]).to(device).to(torch.float16)

            for head_dim in range(q.shape[1]):
                # attn_weights[0, head_dim] = torch.abs(attn_fn(q[:,head_dim].to(torch.float16), k[:,head_dim].to(torch.float16), v_diag[:,head_dim].to(torch.float16)))[0, inds][:, inds]
                attn_weights += torch.abs(attn_fn(q[:,head_dim].to(torch.float32), k[:,head_dim].to(torch.float32), v_diag[:,head_dim].to(torch.float32)))
                # attn_weights += norm_tensor(torch.abs(attn_fn(q[:,head_dim].to(torch.float16), k[:,head_dim].to(torch.float16), v_diag[:,head_dim].to(torch.float16))), dim=-1)
            attn_weights /= q.shape[1]
            return out, attn_weights
        else:
            return out 


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


# model = PathFormer_allin1(30,8,30,5)
# print(model)

count_in = torch.randn(8,123,1)
gene_in = torch.randint(1,100,(8,123,1))

print(count_in.shape,gene_in.shape)