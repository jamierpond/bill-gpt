import torch
from torch.utils.data import DataLoader
from data import BILL_PATH, DumbTokenizer, TextDataset
from model import Transformer

def generate(model, initial_context, max_len=256, temp=0.8):
    tokenizer = DumbTokenizer()
    for _ in range(max_len):
        logits = model(initial_context)
        next_token = torch.multinomial(torch.softmax(logits[:, -1, :] / temp, dim=-1), 1)
        next_token = (torch.Tensor(next_token).int())
        initial_context = torch.cat([initial_context, next_token], dim=-1)
        token = next_token.item()
        char = tokenizer.decode([int(token)])
        print(char, end="")
    print()


if __name__ == "__main__":
    ds = TextDataset(BILL_PATH)
    input_tokens, y = ds[5000]
    model = Transformer(vocab_size=256)
    model.load_state_dict(torch.load("best-model.pth", weights_only=True))

    model.to("cuda")
    input_tokens = input_tokens.to("cuda")

    initial_context = input_tokens.unsqueeze(0)
    out = generate(model, initial_context)

