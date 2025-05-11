# btc_forecast/models_torch/lstm.py

import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_width, label_width, hidden_size, num_layers, num_inputs, num_outputs):
        super().__init__()
        self.label_width = label_width
        self.num_outputs = num_outputs

        self.lstm = nn.LSTM(
        input_size=num_inputs,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=0.3,  # 💧 Dropout between layers (if num_layers > 1)
        batch_first=True
         )

        self.linear = nn.Linear(hidden_size, label_width * num_outputs)

    def forward(self, x):
        # x: (batch_size, input_width, num_inputs)
        out, _ = self.lstm(x)  # out: (batch_size, input_width, hidden_size)
        out = out[:, -1, :]    # take the last timestep
        out = self.linear(out) # (batch_size, label_width * num_outputs)
        out = out.view(-1, self.label_width, self.num_outputs)
        return out
