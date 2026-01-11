# btc_forecast/training/eval_price.py
from __future__ import annotations

from typing import Dict, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from btc_forecast.windowed_dataset import WindowedDataset

def reconstruct_prices_from_returns(last_close: float, future_returns: np.ndarray) -> np.ndarray:
    log_p0 = np.log(float(last_close))
    log_path = log_p0 + np.cumsum(future_returns)
    return np.exp(log_path)

def eval_model_price_from_returns(
    *,
    model,
    device: torch.device,
    close_raw: pd.Series,
    test_df_std: pd.DataFrame,
    returns_mean: float,
    returns_std: float,
    input_width: int,
    label_width: int,
) -> Dict[str, Any]:
    ds = WindowedDataset(test_df_std, int(input_width), int(label_width), 0, ["ret"])
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    if len(ds) == 0:
        raise RuntimeError("Test dataset is empty after windowing.")

    preds_prices = []
    targets_prices = []

    model.eval()
    with torch.no_grad():
        for i, (xb, yb) in enumerate(loader):
            xb = xb.to(device)
            yb = yb.to(device)

            yhat = model(xb)

            yhat_r = (yhat[0, :, 0].detach().cpu().numpy() * returns_std + returns_mean)
            ytrue_r = (yb[0, :, 0].detach().cpu().numpy() * returns_std + returns_mean)

            if i + int(input_width) >= len(test_df_std.index):
                break

            y_start_time = test_df_std.index[i + int(input_width)]
            if y_start_time in close_raw.index:
                last_close = float(close_raw.loc[y_start_time])
            else:
                pos = close_raw.index.searchsorted(y_start_time, side="right") - 1
                pos = max(0, min(pos, len(close_raw) - 1))
                last_close = float(close_raw.iloc[pos])

            preds_prices.append(reconstruct_prices_from_returns(last_close, yhat_r))
            targets_prices.append(reconstruct_prices_from_returns(last_close, ytrue_r))

    preds_prices = np.asarray(preds_prices)
    targets_prices = np.asarray(targets_prices)

    mae = float(np.mean(np.abs(targets_prices - preds_prices)))
    rmse = float(np.sqrt(np.mean((targets_prices - preds_prices) ** 2)))
    mae_steps = np.mean(np.abs(targets_prices - preds_prices), axis=0)
    rmse_steps = np.sqrt(np.mean((targets_prices - preds_prices) ** 2, axis=0))

    return {"mae_steps": mae_steps, "rmse_steps": rmse_steps, "mae": mae, "rmse": rmse}
