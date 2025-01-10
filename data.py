
from pathlib import Path
from torch.utils.data import Dataset
import torch

BILL_PATH=Path('data/bill.txt')

class DumbTokenizer():
    def forward(self, x: list[str]):
        return [self.tokenize(i) for i in x]

    def __call__(self, x: list[str]):
        return self.forward(x)

    def tokenize(self, x: str):
        characters = list(x)
        to_ascii_int = lambda x: ord(x)
        token_index = [to_ascii_int(i) for i in characters]
        return token_index

    def detokenize(self, x: list[int]):
        return ''.join([chr(i) for i in x])


class TextDataset(Dataset):
    def __init__(self, path, ctx_len=256):
        self.text = Path(path).read_text()
        self.ctx_len = ctx_len
        self.tokenizer = DumbTokenizer()

    def __len__(self):
        # only unique sequences
        return len(self.text) // self.ctx_len

    def __getitem__(self, i):
        context_index = i * self.ctx_len
        other_index = (i + 1) * self.ctx_len
        def get_text(i):
            text = self.text[i:i+self.ctx_len]
            tokens = self.tokenizer([text])
            return torch.Tensor(tokens[0]).long()
        return get_text(context_index), get_text(other_index)



def test_basic_tokenizer():
    tokenizer = DumbTokenizer()
    tokens = tokenizer(['hello world', 'goodbye world'])
    detokenized = tokenizer.detokenize(tokens[0])
    assert detokenized == 'hello world'
    detokenized = tokenizer.detokenize(tokens[1])
    assert detokenized == 'goodbye world'
    print('Success!')


def test_text_dataset():
    dataset = TextDataset(BILL_PATH)
    size = len(dataset)
    print(size)
    x, y = dataset[size // 2]
    assert x.dtype == torch.int64, f'Expected int64, got {x.dtype}'
    assert y.dtype == torch.int64, f'Expected int64, got {y.dtype}'
    assert len(x) == 256, f'Expected 1, got {len(x)}, shape: {x.shape}'
    assert len(y) == 256, f'Expected 1, got {len(y)}, shape: {y.shape}'
    tokenizer = DumbTokenizer()
    x = tokenizer.detokenize(x.int().tolist())
    y = tokenizer.detokenize(y.int().tolist())

    print(x)
    print(y)
    print('Success!')

if __name__ == '__main__':
    test_text_dataset()
    test_basic_tokenizer()

