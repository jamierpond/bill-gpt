from data import BILL_PATH, VOCAB_SIZE, DumbTokenizer
import torch

if __name__ == "__main__":
    tokenizer = DumbTokenizer()
    bill = BILL_PATH.read_text()
    tokens = tokenizer([bill])
    tokens = torch.Tensor(tokens[0]).long()
    torch.save(tokens, BILL_PATH.with_suffix('.pth'))

    loaded = torch.load(BILL_PATH.with_suffix('.pth'), map_location='cpu', weights_only=True)
    assert torch.allclose(tokens, loaded)
    decoded = tokenizer.decode(loaded.tolist())
    print(decoded)
    print('done')



