import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, causal=False, q_mask=False):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.causal = causal
        self.q_mask = q_mask
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, q, k, v, key_padding_mask=None, query_padding_mask=None):
        def proj_and_split(x, proj_layer):
            x = proj_layer(x)
            bsz, sqlen, _ = x.shape
            head_dim = self.d_model // self.n_heads
            x = x.reshape(bsz, sqlen, self.n_heads, head_dim).permute(0, 2, 1, 3)
            return x
        
        Q = proj_and_split(q, self.q_proj)
        K = proj_and_split(k, self.k_proj)
        V = proj_and_split(v, self.v_proj)

        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)

        if key_padding_mask is not None:
            key_padding_mask = ~key_padding_mask[:, None, None, :].to(attn_weights.device)
            attn_weights = attn_weights.masked_fill(key_padding_mask, float('-inf'))
        
        if self.causal:
            causal_mask = torch.triu(torch.ones(q.shape[-2], k.shape[-2]), diagonal=1).to(q.device) > 0
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.q_mask and query_padding_mask is not None:
            query_padding_mask = ~query_padding_mask[:, None, :, None].to(attn_weights.device)
            attn_weights = attn_weights.masked_fill(query_padding_mask, 0.0)
        
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.permute(0, 2, 1, 3).reshape(q.shape[0], q.shape[1], -1)

        return out + q
            