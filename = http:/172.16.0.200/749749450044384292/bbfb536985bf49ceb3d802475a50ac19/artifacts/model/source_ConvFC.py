    class ConvFC(nn.Module):

        def __init__(self, input_width, label_width, num_features, conv_channels=16, kernel_size=3):
            super(ConvFC, self).__init__()

            self.input_width = input_width
            self.label_width = label_width
            self.num_features = num_features
            self.conv_channels = conv_channels
            self.kernel_size = kernel_size

            conv_output_width = input_width - kernel_size + 1
            input_dim = conv_output_width * conv_channels
            output_dim = label_width * num_features

            self.conv = nn.Conv1d(
                in_channels=num_features,
                out_channels=conv_channels,
                kernel_size=kernel_size
            )

            self.fc = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim)
            )

        def forward(self, x):
            # x: (batch_size, input_width, num_features)
            x = x.permute(0, 2, 1)        # → (batch_size, num_features, input_width)
            x = self.conv(x)              # → (batch_size, conv_channels, output_width)
            x = x.flatten(start_dim=1)    # → (batch_size, conv_channels * output_width)
            x = self.fc(x)                # → (batch_size, label_width * num_features)
            return x.view(-1, self.label_width, self.num_features)
