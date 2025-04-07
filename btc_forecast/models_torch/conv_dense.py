## MODEL

import torch
import torch.nn as nn

class ConvDenseTorch(nn.Module):
    def __init__(self, input_width, label_width, num_inputs, num_outputs):
        super().__init__()
        self.label_width = label_width
        self.num_outputs = num_outputs

        self.conv1 = nn.Conv1d(in_channels=num_inputs, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        conv_output_size = (input_width - 3 + 1) * 32
        self.fc = nn.Linear(conv_output_size, label_width * num_outputs)

    def forward(self, x):
        # x shape: [batch_size, input_width, num_inputs]
        x = x.permute(0, 2, 1)  # → [batch_size, num_inputs, input_width] for Conv1d
        x = self.conv1(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = x.view(-1, self.label_width, self.num_outputs)  # reshape to match target
        return x


