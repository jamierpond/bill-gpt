import torch
from data import DumbTokenizer
from model import Transformer

def generate(model, start_text, max_len=256, temp=0.8):
    model.eval()
    tokenizer = DumbTokenizer()
    x = tokenizer.tokenize(start_text)
    x = torch.Tensor(x).to(torch.int64).unsqueeze(1)

    logits = model(x)
    y = torch.multinomial(logits[:, -1, :].squeeze().softmax(-1), 1)


    y = y.cpu().int().tolist()
    # flatten nested lists
    y = [item for sublist in y for item in sublist]
    breakpoint()

    x = tokenizer.detokenize(y)

    return x

if __name__ == "__main__":
    model = Transformer(vocab_size=256)
    model.load_state_dict(torch.load("model.pth", weights_only=True))
    print(generate(model, "To be, or not to be: "))
