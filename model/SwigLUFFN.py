import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.up_gate_proj = nn.Linear(hidden_dim,2 * intermediate_dim,bias=False)
        self.down_proj = nn.Linear(intermediate_dim,hidden_dim,bias=False)
    def forward(self, x):
        #x shape: (batch_size, sequence_length, hidden_dim)
        combined = self.up_gate_proj(x)
        gate, value = torch.chunk(combined, 2, dim=-1) #splitting in exactly half 
        gate = F.silu(gate)
        x = gate * value
        output = self.down_proj(x)
        return output
        
hidden_dim = 512
intermediate_dim = 2048
model = SwiGLUFFN(hidden_dim, intermediate_dim)
x = torch.randn(2, 16, hidden_dim)
output = model(x)
print("Input Shape :", x.shape)
print("Output Shape:", output.shape)
total_params = sum(p.numel() for p in model.parameters())
print("Total Parameters:", total_params)
#silu(x)=x∗σ(x) where σ(x) is 1/(1+e**(-x))