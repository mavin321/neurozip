import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import ByteDataset
from .model import TinyLSTM
from .nz_tokenize import file_to_bytes

import argparse
import time


def train_model(
    data_path: str,
    seq_len: int,
    batch_size: int,
    hidden_size: int,
    epochs: int,
    lr: float,
    output: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[+] Loading data from {data_path}")
    data = file_to_bytes(data_path)
    dataset = ByteDataset(data, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TinyLSTM(hidden_size=hidden_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print("[+] Starting training")
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, 256), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} — loss: {avg:.4f}")

    dur = time.time() - start
    print(f"[+] Training finished in {dur:.1f}s")

    print(f"[+] Saving checkpoint to {output}")
    torch.save({
        "hidden_size": hidden_size,
        "model_state": model.state_dict(),
    }, output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--hidden-size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--output", required=True)

    args = ap.parse_args()

    train_model(
        data_path=args.data,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        lr=args.lr,
        output=args.output,
    )


if __name__ == "__main__":
    main()
