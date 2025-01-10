import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, device: torch.device|None=None) -> torch.Tensor:
        x.to(device)
        # todo
        return x
