from torch.utils.data import Dataset, DataLoader
from pathlib import Path

BILL_PATH=Path('data/bill.txt')

class TextDataset(Dataset):
    def __init__(self, path, ctx_len=256):
        self.text = Path(path).read_text()
        self.ctx_len = ctx_len
        # Add tokenization logic here

    def __len__(self): return len(self.text) - self.ctx_len
    def __getitem__(self, i):
        return self.text[i:i+self.ctx_len], self.text[i+self.ctx_len]



if __name__ == '__main__':
    ds = TextDataset(BILL_PATH)
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    for x, y in dl:
        print(x, y)
        break

