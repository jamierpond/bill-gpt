import torch
import torch.nn as nn
from attention import MultiHeadAttention

class Transformer(nn.Module):
    def __init__(self, vocab_size, dim=128, depth=16, heads=8):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)

        # sinusoidal positional embeddings
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MultiHeadAttention(dim, heads),
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
                nn.LayerNorm(dim)
            ]))

        self.to_logits = nn.Linear(dim, vocab_size)

    @staticmethod
    def get_causal_mask(seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.to(device)

    def forward(self, x, causal=False):
        batch_size, seq_len = x.size()
        token_emb = self.token_emb(x)

        # Ensure positional embeddings match the input sequence length
        seq_len = x.size(1)
        pos_emb = self.pos_emb[:, :seq_len, :].expand(batch_size, -1, -1)

        # Add token and positional embeddings
        assert token_emb.shape == pos_emb.shape
        x = token_emb + pos_emb

        mask = self.get_causal_mask(seq_len, x.device) if causal else None
        for sub_list in self.layers:
            assert isinstance(sub_list, nn.ModuleList)
            attn, ln1, ff1, gelu, ff2, ln2 = sub_list
            x = ln1(x + attn(x, mask=mask))
            x = ln2(x + ff2(gelu(ff1(x))))

        return self.to_logits(x)

if __name__ == "__main__":
    model = Transformer(vocab_size=256)
    x = torch.randint(0, 256, (1, 1024))
    out = model(x)
    print(out.shape)
