import torch
from torch.optim import AdamW
from data import TextDataset, DataLoader
from model import Transformer

def train():
    model = Transformer(vocab_size=256)  # Adjust vocab size
    opt = AdamW(model.parameters())
    dataset = TextDataset("path/to/shakespeare.txt")
    loader = DataLoader(dataset, batch_size=32)

    for epoch in range(100):
        print(f"Epoch {epoch}")
        for x, y in loader:
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(
                model(x).transpose(1, 2), y
            )
            loss.backward()
            opt.step()

if __name__ == "__main__":
    train()
