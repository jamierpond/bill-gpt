import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        # Single projection matrices for Q,K,V that we'll split into heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.scaling = 1 / math.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Project and reshape to [batch, heads, seq, head_dim]
        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Compute attention scores
        attention_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling

        if mask is not None:
            assert isinstance(mask, torch.Tensor), "mask must be a torch.Tensor"
            mask = mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, channel]
            # cat num heads in dim 1
            mask = mask.expand(batch_size, self.num_heads, seq_len, seq_len)
            assert (
                mask.shape == attention_weights.shape
            ), f"mask shape {mask.shape} must match attention_weights shape {attention_weights.shape}"
            attention_weights = attention_weights.masked_fill(mask, float("-inf"))

        attention_weights = F.softmax(attention_weights, dim=-1)

        # Apply attention to values
        output = torch.matmul(attention_weights, v)

        # Reshape back to [batch, seq, embed_dim]
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )

        return self.out_proj(output)
