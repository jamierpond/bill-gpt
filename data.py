from pathlib import Path
from torch.utils.data import Dataset
import torch

import tiktoken

THIS_DIR = Path(__file__).parent
BILL_PATH = THIS_DIR / Path("data/bill.txt")
# VOCAB_SIZE = 256 tiktoken.get_encoding("cl100k_base").max_token_value


class DumbTokenizer:
    def __init__(self):
        self.enc = tiktoken.get_encoding("cl100k_base")

    def forward(self, x: list[str]):
        return [self.tokenize(i) for i in x]

    def __call__(self, x: list[str]):
        return self.forward(x)

    def tokenize(self, x: str):
        return self.enc.encode(x)

    @property
    def vocab_size(self):
        return self.enc.max_token_value

    def old_tokenize(self, x: str):
        characters = list(x)
        to_ascii_int = lambda x: ord(x)
        token_index = [to_ascii_int(i) for i in characters]
        return token_index

    def decode(self, x: list[int]):
        return self.enc.decode(x)

    def old_decode(self, x: list[int]):
        return "".join([chr(i) for i in x])


class TextDataset(Dataset):
    def __init__(self, path, ctx_len=256):
        self.text = Path(path).read_text()
        self.ctx_len = ctx_len
        self.tokenizer = DumbTokenizer()
        self.tokens = self.tokenizer([self.text])[0]

    def __len__(self):
        l = len(self.tokens) - self.ctx_len
        # margin
        return l - 2

    def __getitem__(self, i):
        x = self.tokens[i : i + self.ctx_len]
        y = self.tokens[i + 1 : i + self.ctx_len + 1]
        return torch.Tensor(x).long(), torch.Tensor(y).long()


def test_basic_tokenizer():
    tokenizer = DumbTokenizer()
    tokens = tokenizer(["hello world", "goodbye world"])
    print(tokens)
    detokenized = tokenizer.decode(tokens[0])
    assert detokenized == "hello world"
    detokenized = tokenizer.decode(tokens[1])
    assert detokenized == "goodbye world"
    print("Success!")


def test_text_dataset():
    dataset = TextDataset(BILL_PATH)
    size = len(dataset)
    print(size)
    x, y = dataset[size // 3]
    assert x.dtype == torch.int64, f"Expected int64, got {x.dtype}"
    assert y.dtype == torch.int64, f"Expected int64, got {y.dtype}"
    assert x.shape == y.shape, f"Expected same shape, got {x.shape} and {y.shape}"
    tokenizer = DumbTokenizer()
    x = tokenizer.decode(x.int().tolist())
    y = tokenizer.decode(y.int().tolist())

    print(x)
    print(y)
    print("Success!")


if __name__ == "__main__":
    test_text_dataset()
    test_basic_tokenizer()
