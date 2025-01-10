import torch.nn as nn
from attention import Attention

class Transformer(nn.Module):
    def __init__(self, vocab_size, dim=512, depth=6, heads=8):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads),
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
                nn.LayerNorm(dim)
            ]))

        self.to_logits = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.token_emb(x) + self.pos_emb[:, :x.shape[1]]

        for attn, ln1, ff1, gelu, ff2, ln2 in self.layers:
            x = ln1(x + attn(x))
            x = ln2(x + ff2(gelu(ff1(x))))

        return self.to_logits(x)
