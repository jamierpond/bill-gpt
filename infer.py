import torch
from data import BILL_PATH, DumbTokenizer, TextDataset
from model import Transformer
import math


def advance_sine_phase(current_phase, freq, num_steps):
    new = current_phase + (2 * math.pi * freq / num_steps)
    while new > 2 * math.pi:
        new -= 2 * math.pi
    return new

def generate(model, initial_context, max_len=256, temp=0.8):
    tokenizer = DumbTokenizer()

    # cpm = cycles per 1000 tokens
    temp_cpm = 50
    temp_bias = 0.7
    temp_amplitude = 0.30
    phase = 0

    for i in range(max_len):
        if initial_context.size(1) > 1024:
            # pop the first token
            initial_context = initial_context[:, 1:]
        logits = model(initial_context, causal=True)
        phase = advance_sine_phase(phase, 1, temp_cpm)
        temp = temp_bias + abs(math.sin(phase)) * temp_amplitude
        #¢ print(temp)
        next_token = torch.multinomial(torch.softmax(logits[:, -1, :] / temp, dim=-1), 1)
        next_token = (torch.Tensor(next_token).int())
        initial_context = torch.cat([initial_context, next_token], dim=-1)
        token = next_token.item()
        char = tokenizer.decode([int(token)])
        print(char, end="")
    print()


from train import FAUSTUS

if __name__ == "__main__":
    ds = TextDataset(BILL_PATH)
    input_tokens = ds.tokenizer.tokenize(FAUSTUS)
    input_tokens = torch.Tensor(input_tokens).int().unsqueeze(0)
    model = Transformer(vocab_size=ds.tokenizer.vocab_size)
    model.load_state_dict(torch.load("best-model.pth", weights_only=True))

    model.to("cuda")
    input_tokens = input_tokens.to("cuda")

    initial_context = input_tokens
    out = generate(model, initial_context, max_len=int(1e18))

