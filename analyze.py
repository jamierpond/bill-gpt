import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader


def analyze_predictions():
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
            actual = tokenizer.detokenize([y[0, pos].item()])
            predictions = [tokenizer.detokenize([idx.item()]) for idx in indices]
            print(f"\nPosition {pos}")
            print(f"Actual: '{actual}'")
            print("Top predictions:")
            for pred, prob in zip(predictions, values):
                print(f"'{pred}': {prob:.3f}")
