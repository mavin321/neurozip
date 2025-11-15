import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import ByteDataset
from .model import TinyLSTM
from .tokenize import file_to_bytes

def evaluate(checkpoint_path: str, data_path: str, seq_len: int = 256):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    hidden_size = ckpt["hidden_size"]

    model = TinyLSTM(hidden_size=hidden_size)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    data = file_to_bytes(data_path)
    dataset = ByteDataset(data, seq_len)
    loader = DataLoader(dataset, batch_size=1)

    total_logloss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for x, y in loader:
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction="sum")
            total_logloss += loss.item()
            total_tokens += y.numel()

    bpb = total_logloss / total_tokens / torch.log(torch.tensor(2.0))
    print(f"Bits per byte: {bpb:.4f}")
