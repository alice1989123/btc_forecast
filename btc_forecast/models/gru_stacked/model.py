# btc_forecast/models/gru_stacked.py
import torch
import torch.nn as nn

class GRUStacked(nn.Module):
    def __init__(self, input_width, label_width, num_features, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=int(num_features),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(int(hidden_size), int(label_width))
        self.label_width = int(label_width)
        self.horizon_bias = nn.Parameter(torch.zeros(self.label_width))

    def forward(self, x):
        out, _ = self.gru(x)       # (B, T, H)
        out = out[:, -1, :]        # (B, H)
        out = self.fc(out)         # (B, out_steps)
        out = out + self.horizon_bias
        return out.unsqueeze(-1)   # (B, out_steps, 1)
