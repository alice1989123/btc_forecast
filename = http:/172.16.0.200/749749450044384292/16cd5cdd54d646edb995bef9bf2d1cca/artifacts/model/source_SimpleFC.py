    class SimpleFC(nn.Module):
        def __init__(self, input_width, label_width, num_features):
            super(SimpleFC, self).__init__()
            self.input_width = input_width
            self.label_width = label_width
            self.num_features = num_features
            input_dim = input_width * num_features
            output_dim = label_width * num_features

            self.fc = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim)
            )
        def forward(self, x):
            x = self.fc(x)                             # → (batch_size, label_width * num_features)
            return x.view(-1, self.label_width, self.num_features) 
