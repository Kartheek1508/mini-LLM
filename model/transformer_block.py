from attention import MultiHeadAttention
from rms import RMSNorm
from SwigLUFFN import SwiGLUFFN
import torch.nn as nn
import torch

class TransformerBLock(nn.Module):
    def __init__(self,attention,rms1,rms2,swiglu):
        super(TransformerBLock, self).__init__()
        self.attention = attention
        self.norm1 = rms1
        self.norm2 = rms2
        self.swiglu = swiglu

    def forward(self,x):
        x_norm = self.norm1(x)
        attn = self.attention(x_norm,x_norm,x_norm)
        res = attn + x

        res_norm = self.norm2(res)
        out = self.swiglu(res_norm)
        res_2 = res+out
        return res_2
    
