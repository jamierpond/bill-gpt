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

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(*t.shape[:-1], self.heads, -1).transpose(1, 2), qkv)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v).transpose(1, 2).reshape(*x.shape)
        return self.to_out(out)
