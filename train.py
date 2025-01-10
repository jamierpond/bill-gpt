
import tqdm
import torch.nn.functional as F
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from data import TextDataset, BILL_PATH
import torch.nn as nn
from model import Transformer
from data import DumbTokenizer
from infer import generate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def analyze_predictions(model, loader):
    model.eval()
    with torch.no_grad():
        x, y = next(iter(loader))
        x = x.to(device)
        logits = model(x)

        # Get top predictions for a few positions
        probs = F.softmax(logits[0, :5], dim=-1)  # First 5 positions of first batch item
        top_preds = torch.topk(probs, k=3, dim=-1)

        tokenizer = DumbTokenizer()
        for pos, (values, indices) in enumerate(zip(top_preds.values, top_preds.indices)):
            actual = tokenizer.decode([y[0, pos].item()])
            predictions = [tokenizer.decode([idx.item()]) for idx in indices]
            print(f"\nPosition {pos}")
            print(f"Actual: '{actual}'")
            print("Top predictions:")
            for pred, prob in zip(predictions, values):
                print(f"'{pred}': {prob:.3f}")


def train():
    model = Transformer(vocab_size=256)
    # load checkpoint
    model.load_state_dict(torch.load("best-model.pth", weights_only=True, map_location=device))

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")
    model.to(device)
    opt = AdamW(model.parameters(), lr=1e-3)
    dataset = TextDataset(BILL_PATH)
    loader = DataLoader(dataset, batch_size=128)

    lowest_loss = float("inf")
    for epoch in range(2000):
        print(f"Epoch {epoch}")
        step = 0
        example = None
        for x, y in tqdm.tqdm(loader):
            x, y = x.to(device), y.to(device)
            example = x
            step += 1
            opt.zero_grad()
            out = model(x)
            out = out.permute(0, 2, 1)

            logits = model(x)

            # Reshape logits and target for loss calculation
            # Use contiguous() to ensure proper memory layout
            logits = logits.view(-1, 256).contiguous()
            targets = y.view(-1).contiguous()

            loss = F.cross_entropy(logits, targets)
            loss.backward()
            opt.step()
            lowest_loss = min(lowest_loss, loss.item())
            if lowest_loss == loss.item():
                tqdm.tqdm.write(f"New lowest loss: {lowest_loss}, saving model...")
                torch.save(model.state_dict(), f'best-model.pth')

            if step % 150 == 0:
                generate(model, x[0].unsqueeze(0))

            if step % 20 == 0:
                tqdm.tqdm.write(f"Loss: {loss.item()}, Lowest: {lowest_loss}")

        # analyze_predictions()
        tqdm.tqdm.write("======== Generated text =================")
        assert example is not None
        generate(model, example[0].unsqueeze(0))
        tqdm.tqdm.write("=========================================")

        # save model
        tqdm.tqdm.write("Saving model...")
        torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    train()
