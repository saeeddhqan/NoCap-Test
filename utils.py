import math
import torch
nn = torch.nn
F = nn.functional
from torch import Tensor

class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )

def norm1(x):
    return F.rms_norm(x, (x.size(-1),))

def norm2(x):
    rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt() + 1e-8
    return x / rms

if hasattr(F, 'rms_norm'):
    norm = norm1
else:
    norm = norm2
