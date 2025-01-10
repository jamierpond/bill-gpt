import tqdm
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from data import TextDataset, BILL_PATH
from model import Transformer


def train():
    model = Transformer(vocab_size=256)  # Adjust vocab size
    opt = AdamW(model.parameters())
    dataset = TextDataset(BILL_PATH)
    MAX_TRAIN_SIZE = 2
    loader = DataLoader(dataset, batch_size=32)

    for epoch in range(1):
        print(f"Epoch {epoch}")
        step = 0
        for x, y in tqdm.tqdm(loader):
            if step > MAX_TRAIN_SIZE:
                print("Done training")
                break
            step += 1
            opt.zero_grad()
            out = model(x)
            loss = torch.nn.functional.cross_entropy(
                out.view(-1, 256), y.view(-1)
            )
            loss.backward()
            opt.step()
            print(loss.item())

    # save model
    torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    train()
