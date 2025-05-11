import torch.nn as nn

class GRUStacked(nn.Module):
    def __init__(self, input_width, label_width, num_inputs, hidden_size=64, num_layers=2):
        super(GRUStacked, self).__init__()
        self.gru = nn.GRU(
            input_size=num_inputs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, label_width * num_inputs)
        self.label_width = label_width
        self.num_features = num_inputs

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # take the output from the last timestep
        out = self.fc(out)
        return out.view(-1, self.label_width, self.num_inputs)