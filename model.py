import torch
import torch.nn.functional as F
import torch.nn as nn
from attention import SingleSelfHeadAttention

class Transformer(nn.Module):
    def __init__(self, vocab_size, dim=128, depth=128, heads=8):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                SingleSelfHeadAttention(dim),
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
                nn.LayerNorm(dim)
            ]))

        self.to_logits = nn.Linear(dim, vocab_size)

    def forward(self, x):

        token_emb = self.token_emb(x)
        pos_emb = self.pos_emb[:, :x.shape[1]]
        x = token_emb + pos_emb


        for sub_list in self.layers:
            assert isinstance(sub_list, nn.ModuleList)
            attn, ln1, ff1, gelu, ff2, ln2 = sub_list
            x = ln1(x + attn(x))
            x = ln2(x + ff2(gelu(ff1(x))))

        return self.to_logits(x)


if __name__ == "__main__":
    model = Transformer(vocab_size=256)
    x = torch.randint(0, 256, (1, 1024))
    out = model(x)
    print(out.shape)
