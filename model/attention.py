import numpy as np
import torch
import torch.nn as nn
import copy
import math
from rope import RotaryEmbeddings

def clone(module,N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
#used to create copies of ecoders 

class Embeddings(nn.Module):
    def __init__(self,vocab,dim_model):
        super(Embeddings, self).__init__()
        self.look_up_tab = nn.Embedding(vocab,dim_model)
        self.dim_model = dim_model

    def forward(self,x):
        return self.look_up_tab(x)*np.sqrt(self.dim_model)
    

def attention(query,key,value,dropout=None,mask = None):
    d_k = query.size(-1)
    attn = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        attn= attn.masked_fill(mask == 0, -1e9)
    prob_attn = attn.softmax(dim = -1)
    if dropout is not None:
        prob_attn = dropout(prob_attn)
    return torch.matmul(prob_attn, value), prob_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self,h,dim_model,dropout =0.1):
        super(MultiHeadedAttention, self).__init__()
        assert dim_model % h == 0
        self.dk = dim_model//h
        self.h = h
        self.linears = clone(nn.Linear(dim_model,dim_model),4)#makes 4 clones
        self.attn = None
        self.dropout =nn.Dropout(p = dropout)
        self.rope = RotaryEmbeddings(self.dk)
        self.key_cache = None
        self.value_cache = None

    def forward(self,q,k,v,mask =None,use_cache = False):
        if mask is not None:
            mask = mask.unsqueeze(1)
        n_batches = q.size(0)
        query, key, value = [
            lin(x).view(n_batches, -1, self.h, self.dk).transpose(1, 2)
            for lin, x in zip(self.linears, (q, k, v))
        ]

        seq_len = query.shape[2]
        cache_len = 0 if self.key_cache is None else self.key_cache.shape[2]
        cos, sin = self.rope(seq_len,offset = cache_len)
        query, key = self.rope.apply_rotary_pos_emb(query,key,cos,sin)
        if use_cache:

            if self.key_cache is None:
                self.key_cache = key
                self.value_cache = value
            else:
                self.key_cache = torch.cat(
                    [self.key_cache, key],
                    dim=2
                )

                self.value_cache = torch.cat(
                    [self.value_cache, value],
                    dim=2
                )

            key = self.key_cache
            value = self.value_cache
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x= ( x.transpose(1, 2)
            .contiguous()
            .view(n_batches, -1, self.h * self.dk))

        return self.linears[-1](x)
    
    def reset_cache(self):
        self.key_cache = None
        self.value_cache = None
    

def causal_mask(seq_length):
    mask = torch.ones(1,seq_length,seq_length)
    mask= torch. tril(mask)
    mask = mask.bool()
    return mask

def test_output_shape():
    batch = 2
    seq_len = 5
    dim_model = 512
    heads = 8

    mha = MultiHeadedAttention(h=heads, dim_model=dim_model)
    x = torch.randn(batch, seq_len, dim_model)

    out = mha(x, x, x)

    assert out.shape == (batch, seq_len, dim_model)


#Test-1

batch = 2
seq_len = 5
dim_model = 512
heads = 8
mh = MultiHeadedAttention(h=heads, dim_model=dim_model,dropout=0.0)
x = torch.randn(batch, seq_len, dim_model)
mask = causal_mask(seq_len)
out = mh(x,x,x,mask=mask)
print(out.shape)

mh.attn

x = torch.randn(batch, seq_len, dim_model, requires_grad=True)

out = mh(x, x, x)

loss = out.mean()

loss.backward()

assert x.grad is not None
assert mh.linears[0].weight.grad is not None

batch = 2
seq_len = 5
d_model = 512
heads = 8

x = torch.randn(batch, seq_len, d_model, requires_grad=True)

mha = MultiHeadedAttention(heads, d_model)

out = mha(x, x, x)

loss = out.mean()
loss.backward()

print(x.grad is not None)

#test-2(positions)
rope = RotaryEmbeddings(64)
cos,sin = rope(6)
x = torch.randn(1, 1, 1, 64)
#pos0
cos0 = cos[:, :, 0:1, :]
sin0 = sin[:, :, 0:1, :]
x0, _ = rope.apply_rotary_pos_emb(x,x.clone(),cos0,sin0)
#pos5
cos5 = cos[:, :, 5:6, :]
sin5 = sin[:, :, 5:6, :]
x5, _ = rope.apply_rotary_pos_emb(x,x.clone(),cos5,sin5)

assert not torch.allclose(x0, x5)
print(torch.norm(x0 - x5))

#dot-product test
rope = RotaryEmbeddings(64)

q = torch.randn(1, 1, 1, 64)
k = torch.randn(1, 1, 1, 64)

cos, sin = rope(26)

q2, _ = rope.apply_rotary_pos_emb(q,q.clone(), cos[:, :, 2:3, :],sin[:, :, 2:3, :])
k7, _ = rope.apply_rotary_pos_emb(k,k.clone(),cos[:, :, 7:8, :],sin[:, :, 7:8, :])

dot1 = (q2 * k7).sum()

q20, _ = rope.apply_rotary_pos_emb(q,q.clone(),cos[:, :, 20:21, :],sin[:, :, 20:21, :])
k25, _ = rope.apply_rotary_pos_emb(k,k.clone(),cos[:, :, 25:26, :],sin[:, :, 25:26, :])

dot2 = (q20 * k25).sum()

print(dot1)
print(dot2)

assert torch.allclose(dot1, dot2, atol=1e-5)


mha = MultiHeadedAttention(
    heads,
    d_model,
    dropout=0.0
)
#auto regression test without KV Cache
import time

mha.eval()

tokens = torch.randn(1, 50, d_model)
start = time.perf_counter()

with torch.no_grad():
    for i in range(1, 51):
        mask = causal_mask(i)
        mha(
            tokens[:, :i],
            tokens[:, :i],
            tokens[:, :i],
            mask=mask,
            use_cache=False,
        )

no_cache_time = time.perf_counter() - start
mha.reset_cache()

start = time.perf_counter()

with torch.no_grad():
    for i in range(50):
        mha(
            tokens[:, i:i+1],
            tokens[:, i:i+1],
            tokens[:, i:i+1],
            use_cache=True,
        )

cache_time = time.perf_counter() - start

print(f"No cache: {no_cache_time:.6f} s")
print(f"Cache:    {cache_time:.6f} s")
print(f"Speedup:  {no_cache_time/cache_time:.2f}x")