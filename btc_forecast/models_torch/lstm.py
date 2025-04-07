import torch.nn as nn

class LSTMForecastModel(nn.Module):
    def __init__(self, input_width, label_width, hidden_size, num_layers, num_inputs, num_outputs):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_inputs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_outputs * label_width)
        self.label_width = label_width
        self.num_outputs = num_outputs

    def forward(self, x):
        output, _ = self.lstm(x)
        x = self.fc(output[:, -1])
        x = x.view(-1, self.label_width, self.num_outputs)
        return x
