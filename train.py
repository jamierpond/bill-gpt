import tqdm
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from data import TextDataset, BILL_PATH
from model import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    model = Transformer(vocab_size=256)  # Adjust vocab size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    model.to(device)
    opt = AdamW(model.parameters(), lr=5e-3)
    dataset = TextDataset(BILL_PATH)
    loader = DataLoader(dataset, batch_size=64)

    lowest_loss = float("inf")
    for epoch in range(10):
        print(f"Epoch {epoch}")
        step = 0
        for x, y in tqdm.tqdm(loader):
            x, y = x.to(device), y.to(device)
            step += 1
            opt.zero_grad()
            out = model(x)
            out = out.permute(0, 2, 1)

            loss = torch.nn.functional.cross_entropy(
                out, y
            )
            loss.backward()
            opt.step()
            loss = loss.item()
            lowest_loss = min(lowest_loss, loss)
            if step % 50 == 0:
                tqdm.tqdm.write(f"Loss: {loss}, Lowest: {lowest_loss}")

        # save model
        tqdm.tqdm.write("Saving model...")
        torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    train()
