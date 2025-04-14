import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):

    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, casual_mask=False):
        # x: (Batch_size, seq_len, dim)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interimum_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        #(batch_size, seq_len, dim) -> (batch_size, seq_len, h, dim * 3) -> 3 tensor of shape (batch_size, seq_len, dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        #(batch_size, seq_len, dim) -> (batch_size, seq_len, h, dim/h) -> (batch_size, h, seq_len, dim/h)
        q = q.view(interimum_shape).transpose(1, 2)
        k = k.view(interimum_shape).transpose(1, 2)
        v = v.view(interimum_shape).transpose(1, 2)

        #(Batch_size, h, seq_len, seq_len)
        weight = q @ k.transpose(-1, -2)

        if casual_mask:
            # mask where the upper traingle is made up of one
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim = -1)

        #(Batch_size, h, seq_len) @ (Batch_size, h, seq_len, dim/h) -> (Batch_size,)
        output = weight @ v

        #(Batch_size, h, seq_len, dim/h) -> (batch_size, seq_len, h, dim/h)
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        #(Batch_size, seq_len, dim)
        return output 


class CrossAttention(nn.Module):

    def __init__(self,n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias = True ):
        super().__init__()

        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)

        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        # x: (latent): (batch_size, seq_len_q, dim_q)
        # y: (context): (batch_size, seq_len_kv, dim_kv) -> (batch_size, 77, 768)

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape)
        interium_shape = (batch_size, -1, self.n_heads, self.d_head)

        #multiply query by Wq

        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interium_shape).transpose(1, 2)
        k = k.view(interium_shape).transpose(1, 2)
        v = v.view(interium_shape).transpose(1, 2)


        weight = q @ k.transpose(1,2)

        weigth /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        output = weight @ v

        output = output.transpose(1, 2).contiguous()

        output = output.view(input_shape)
        
        output = self.out_proj(output)

        return output




