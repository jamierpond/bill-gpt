FAUSTUS = """
  FAUSTUS. Settle thy studies, Faustus, and begin
    To sound the depth of that thou wilt profess.
    Having commenced, be a divine in show,
    Yet level at the end of every art,
    And live and die in Aristotle's works.
"""
import tqdm
import torch.nn.functional as F
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from data import TextDataset, BILL_PATH
from model import Transformer
from infer import generate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def train():
    dataset = TextDataset(BILL_PATH)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    vocab_size = dataset.tokenizer.vocab_size
    model = Transformer(vocab_size=vocab_size)

    # load checkpoint
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    opt = AdamW(model.parameters(), lr=1e-3)

    lowest_loss = float("inf")
    for epoch in range(2):
        print(f"Epoch {epoch}")
        step = 0
        assert len(loader) > 0, "No data in loader"
        for x, y in tqdm.tqdm(loader):
            x, y = x.to(device), y.to(device)
            step += 1
            opt.zero_grad()
            out = model(x)
            out = out.permute(0, 2, 1)

            logits = model(x)

            # Reshape logits and target for loss calculation
            # Use contiguous() to ensure proper memory layout
            logits = logits.view(-1, vocab_size).contiguous()
            targets = y.view(-1).contiguous()

            loss = F.cross_entropy(logits, targets)
            loss.backward()
            opt.step()
            lowest_loss = min(lowest_loss, loss.item())
            if lowest_loss == loss.item():
                tqdm.tqdm.write(f"New lowest loss: {lowest_loss}, saving model...")
                torch.save(model.state_dict(), f'best-model.pth')

            if step % 150 == 0:
                faustus_tokens = dataset.tokenizer.tokenize(FAUSTUS)
                faustus_tokens = torch.Tensor(faustus_tokens).int().unsqueeze(0).to(device)
                generate(model, faustus_tokens)

            if step % 20 == 0:
                tqdm.tqdm.write(f"Loss: {loss.item()}, Lowest: {lowest_loss}")

        # save model
        tqdm.tqdm.write("Saving model...")
        torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    train()
