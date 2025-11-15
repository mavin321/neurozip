import torch
import torch.nn as nn

class TinyLSTM(nn.Module):
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=256,     # byte one-hot
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, 256)

    def forward(self, x, h=None):
        # x: (batch, seq_len) bytes 0..255
        # Convert to one-hot
        onehot = torch.nn.functional.one_hot(x, num_classes=256).float()

        out, h = self.lstm(onehot, h)
        logits = self.fc(out)
        return logits, h
