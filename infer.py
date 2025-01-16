from .data import BILL_PATH, DumbTokenizer, TextDataset
from .model import Transformer
from .train import FAUSTUS
import torch
import math


device = "cuda" if torch.cuda.is_available() else "cpu"


def advance_sine_phase(current_phase, freq, num_steps):
    new = current_phase + (2 * math.pi * freq / num_steps)
    while new > 2 * math.pi:
        new -= 2 * math.pi
    return new


def print_token_str(token_str):
    print(token_str, end="")


def generate(model, initial_context, max_len=256, temp=0.8, next_token_callback=print_token_str):
    tokenizer = DumbTokenizer()

    cycle_size_samples = 50
    temp_bias = 0.7
    temp_amplitude = 0.30
    phase = 0

    for _ in range(max_len):
        if initial_context.size(1) > 1024:
            # pop the first token
            initial_context = initial_context[:, 1:]
        logits = model(initial_context, causal=True)
        phase = advance_sine_phase(phase, 1, cycle_size_samples)
        temp = temp_bias + abs(math.sin(phase)) * temp_amplitude
        next_token = torch.multinomial(torch.softmax(logits[:, -1, :] / temp, dim=-1), 1)
        next_token = (torch.Tensor(next_token).int())
        initial_context = torch.cat([initial_context, next_token], dim=-1)
        token = next_token.item()
        char = tokenizer.decode([int(token)])
        char = next_token_callback(char)


if __name__ == "__main__":
    ds = TextDataset(BILL_PATH)
    input_tokens = ds.tokenizer.tokenize(FAUSTUS)
    input_tokens = torch.Tensor(input_tokens).int().unsqueeze(0)
    model = Transformer(vocab_size=ds.tokenizer.vocab_size)
    model.load_state_dict(torch.load("best-model.pth", weights_only=True, map_location=device))

    model.to(device)
    input_tokens = input_tokens.to(device)

    initial_context = input_tokens
    out = generate(model, initial_context, max_len=int(1e18))

