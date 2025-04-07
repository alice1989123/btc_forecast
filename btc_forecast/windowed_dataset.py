import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class WindowedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, input_width: int, label_width: int, shift: int, variables_used: list, label_columns=None):
        """
        df: normalized dataframe (no NaNs)
        input_width: # of past timesteps (X)
        label_width: # of timesteps to predict (Y)
        shift: how far ahead to start labels
        variables_used: columns used in X
        label_columns: optional list of target columns (default = same as input)
        """
        self.df = df
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.variables_used = variables_used
        self.label_columns = label_columns if label_columns else variables_used

        self.total_window_size = input_width + shift + label_width - 1

        # convert to numpy arrays
        self.x_data = df[variables_used].values.astype(np.float32)
        self.y_data = df[self.label_columns].values.astype(np.float32)

    def __len__(self):
        return len(self.df) - self.total_window_size

    def __getitem__(self, idx):
        # X = past input_width steps
        x_start = idx
        x_end = idx + self.input_width
        x = self.x_data[x_start:x_end]

        # Y = future label_width steps, starting after shift
        y_start = x_end + self.shift - 1
        y_end = y_start + self.label_width
        y = self.y_data[y_start:y_end]

        return torch.tensor(x), torch.tensor(y)
