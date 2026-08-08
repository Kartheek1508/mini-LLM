import torch
import torch.nn as nn

class RotaryEmbeddings(nn.Module):
    
    def __init__(self,dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0/(10000**(torch.arange(0,dim,2).float()/dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self,seq_len, offset = 0):
        pos = torch.arange(offset,offset+seq_len,device=self.inv_freq.device)
        freq = torch.outer(pos,self.inv_freq)
        cos = torch.cos(freq)
        sin = torch.sin(freq)
        cos = torch.repeat_interleave(cos, 2, dim=-1)
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        return cos,sin

    def rotate_half(self,x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x = torch.stack((-x2, x1), dim=-1)

        return x.flatten(-2)

    def apply_rotary_pos_emb(self,q, k, cos, sin):
        q = (q * cos) + (self.rotate_half(q) * sin)
        k = (k * cos) + (self.rotate_half(k) * sin)
    
        return q, k