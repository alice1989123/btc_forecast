import torch
import torch.nn as nn


class PatchTST(nn.Module):
    """
    Minimal Transformer encoder that behaves like your GRUStacked:
      input:  (B, T, num_features)
      output: (B, label_width, 1)

    It’s not the full PatchTST paper implementation, but it’s a clean transformer
    backbone that works with your existing returns pipeline.
    """

    def __init__(
        self,
        input_width: int,
        label_width: int,
        num_features: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        dim_feedforward: int = 256,
    ):
        super().__init__()
        self.input_width = int(input_width)
        self.label_width = int(label_width)
        self.num_features = int(num_features)

        self.in_proj = nn.Linear(self.num_features, int(d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, self.input_width, int(d_model)))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(nhead),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))

        self.head = nn.Linear(int(d_model), self.label_width)
        self.horizon_bias = nn.Parameter(torch.zeros(self.label_width))

        # tiny init that usually helps stability
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        B, T, F = x.shape
        if T != self.input_width:
            # you *can* relax this, but keeping strict prevents silent bugs
            raise ValueError(f"Expected T={self.input_width}, got T={T}")

        h = self.in_proj(x) + self.pos_emb  # (B, T, d_model)
        h = self.encoder(h)                 # (B, T, d_model)
        h_last = h[:, -1, :]                # (B, d_model)

        out = self.head(h_last) + self.horizon_bias  # (B, label_width)
        return out.unsqueeze(-1)                      # (B, label_width, 1)
