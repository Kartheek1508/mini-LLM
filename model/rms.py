import torch
import time
x =torch.randn(10000)
ITER=10000
import torch.nn as nn

class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-8):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return x * self.weight
    
rms = RMSNorm(10000)
    
start = time.perf_counter()
for _ in range(ITER):
    rms(x)
rms_time = time.perf_counter() - start
    
layer_norm = nn.LayerNorm(10000)
start = time.perf_counter()
for _ in range(ITER):
    layer_norm(x)
layer_time = time.perf_counter() - start

print("LN  :", layer_time)
print(layer_time-rms_time)
print(layer_time>rms_time)