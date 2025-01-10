import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SingleSelfHeadAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scaling = 1 / math.sqrt(embed_dim)

    def forward(self, x, device=None):
        x.to(device)
        self.q_proj.to(device)
        self.k_proj.to(device)
        self.v_proj.to(device)

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        attention_weights = torch.bmm(q, k.transpose(1, 2)) * self.scaling
        attention_weights = F.softmax(attention_weights, dim=-1)
        output = torch.bmm(attention_weights, v)
        return self.out_proj(output)
