import torch
from model import Transformer

def generate(model, start_text, max_len=100, temp=0.8):
    model.eval()
    x = torch.tensor([ord(c) for c in start_text], dtype=torch.long)[None, ...]

    with torch.no_grad():
        for _ in range(max_len):
            logits = model(x)
            next_logits = logits[0, -1] / temp
            probs = torch.softmax(next_logits, dim=-1)
            next_char = torch.multinomial(probs, 1)
            x = torch.cat([x, next_char[None]], dim=1)

            if next_char.item() == ord('\n'):
                break

    return ''.join(chr(i) for i in x[0].tolist())

if __name__ == "__main__":
    model = Transformer(vocab_size=256)
    model.load_state_dict(torch.load("model.pt"))
    print(generate(model, "To be, or not to be: "))
