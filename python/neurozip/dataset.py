import torch
from torch.utils.data import Dataset

class ByteDataset(Dataset):
    """
    Converts a long byte array into overlapping sequences of fixed length.

    Example:
        data = bytearray(...)
        ds = ByteDataset(data, seq_len=256)
    """
    def __init__(self, data: bytes, seq_len: int):
        self.data = torch.tensor(list(data), dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y
