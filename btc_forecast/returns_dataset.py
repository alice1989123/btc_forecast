import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def compute_log_returns(close: pd.Series) -> pd.Series:
    close = close.astype("float64")
    return np.log(close).diff().dropna()

def make_return_windows(r: np.ndarray, input_width: int, label_width: int):
    # r is 1D numpy array of log-returns
    X, y = [], []
    n = len(r)
    for i in range(input_width, n - label_width + 1):
        X.append(r[i - input_width:i])
        y.append(r[i:i + label_width])
    return np.stack(X), np.stack(y)

class ReturnsWindowDataset(Dataset):
    def __init__(self, returns: np.ndarray, input_width: int, label_width: int):
        X, y = make_return_windows(returns, input_width, label_width)
        # shape -> (N, input_width, 1) and (N, label_width, 1)
        self.X = torch.tensor(X[:, :, None], dtype=torch.float32)
        self.y = torch.tensor(y[:, :, None], dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
