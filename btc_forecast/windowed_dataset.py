import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class WindowedDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        input_width: int,
        label_width: int,
        shift: int,
        variables_used: list[str],
        label_columns: list[str] | None = None,
    ):
        self.df = df
        self.input_width = int(input_width)
        self.label_width = int(label_width)
        self.shift = int(shift)
        self.variables_used = variables_used
        self.label_columns = label_columns if label_columns is not None else variables_used

        # ✅ correct: X + gap + Y (no -1)
        self.total_window_size = self.input_width + self.shift + self.label_width

        self.x_data = df[self.variables_used].to_numpy(dtype=np.float32)
        self.y_data = df[self.label_columns].to_numpy(dtype=np.float32)

    def __len__(self):
        n = len(self.df)
        # ✅ correct: number of full windows
        return max(0, n - self.total_window_size + 1)

    def __getitem__(self, idx):
        x_start = idx
        x_end = idx + self.input_width
        x = self.x_data[x_start:x_end]

        # ✅ labels start AFTER the input window (+ optional shift)
        y_start = x_end + self.shift
        y_end = y_start + self.label_width
        y = self.y_data[y_start:y_end]

        return torch.from_numpy(x), torch.from_numpy(y)
