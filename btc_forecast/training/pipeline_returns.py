# btc_forecast/training/pipeline_returns.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from btc_forecast.data_loader import load_or_download
from btc_forecast.data_processing import train_test
from btc_forecast.windowed_dataset import WindowedDataset


EpochCallback = Callable[[int, float, float, Dict[str, Any]], None]


# ---------------------------
# Data utils
# ---------------------------
def _coerce_pred_shape(pred: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    """
    Make pred match yb shape in common cases:
      yb:   [B, L, C]
      pred: [B, L] or [B, L, C] or [B, L, 1]
    """
    if pred.shape == yb.shape:
        return pred

    # y: [B,L,1] and pred: [B,L] -> add channel dim
    if (
        yb.ndim == 3
        and yb.shape[-1] == 1
        and pred.ndim == 2
        and pred.shape[:2] == yb.shape[:2]
    ):
        return pred.unsqueeze(-1)

    # pred: [B,L,1] and y: [B,L] -> squeeze channel dim
    if (
        pred.ndim == 3
        and pred.shape[-1] == 1
        and yb.ndim == 2
        and pred.shape[:2] == yb.shape[:2]
    ):
        return pred.squeeze(-1)

    # both 3D, same B,L but channel mismatch and y expects 1 channel
    if yb.ndim == 3 and pred.ndim == 3 and pred.shape[:2] == yb.shape[:2]:
        if pred.shape[-1] != yb.shape[-1] and yb.shape[-1] == 1:
            return pred[..., :1]

    raise RuntimeError(
        f"Prediction shape {tuple(pred.shape)} not compatible with target {tuple(yb.shape)}"
    )


def compute_log_returns(close: pd.Series) -> pd.Series:
    close = close.astype("float64")
    return np.log(close).diff().dropna()


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    for c in ("open_time", "timestamp", "date", "datetime"):
        if c in df.columns:
            df = df.copy()
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=False)
            df = df.dropna(subset=[c]).set_index(c)
            return df

    try:
        idx = pd.to_datetime(df.index, errors="coerce", utc=False)
        if isinstance(idx, pd.DatetimeIndex) and idx.notna().any():
            df = df.copy()
            df.index = idx
            df = df.loc[df.index.notna()]
            return df
    except Exception:
        pass

    return df


def standardize_with_train_stats(
    train_s: pd.Series, s: pd.Series
) -> Tuple[pd.Series, float, float]:
    mean = float(train_s.mean())
    std = float(train_s.std(ddof=1)) + 1e-8
    return (s - mean) / std, mean, std


def huber_zero_baseline(loader: DataLoader, beta: float, device: torch.device) -> float:
    """
    Baseline: predict 0 returns (in standardized space) -> SmoothL1Loss mean per element.
    """
    loss_fn = nn.SmoothL1Loss(beta=float(beta), reduction="sum")
    s, n = 0.0, 0
    with torch.no_grad():
        for _, yb in loader:
            yb = yb.to(device)
            s += loss_fn(torch.zeros_like(yb), yb).item()
            n += yb.numel()
    return s / max(1, n)


# ---------------------------
# Config objects
# ---------------------------
@dataclass(frozen=True)
class ReturnsTrainSpec:
    coin: str
    interval: str
    input_width: int
    label_width: int
    batch_size: int
    lr: float
    max_epochs: int
    patience: int
    grad_clip: float
    weight_decay: float
    huber_beta: float
    target_clip: Optional[float] = 6.0
    num_workers: int = 0              # 👈 safe default
    pin_memory: bool = True           # 👈 useful with cuda
    persistent_workers: bool = False  # 👈 set True if num_workers>0


@dataclass
class ReturnsArtifacts:
    returns_mean: float
    returns_std: float
    train_df_std: pd.DataFrame
    val_df_std: pd.DataFrame
    test_df_std: pd.DataFrame
    returns_df_raw: pd.DataFrame
    close_raw: pd.Series


@dataclass
class TrainResult:
    best_state_dict: Dict[str, torch.Tensor]
    train_losses: List[float]
    val_losses: List[float]
    artifacts: ReturnsArtifacts
    baseline_train: float
    baseline_val: float


# ---------------------------
# Dataset build
# ---------------------------
def build_returns_datasets(
    coin: str,
    interval: str,
    input_width: int,
    label_width: int,
    target_clip: Optional[float],
) -> Tuple[ReturnsArtifacts, WindowedDataset, WindowedDataset]:
    df = ensure_datetime_index(load_or_download(coin, interval))
    if "close" not in df.columns:
        raise RuntimeError(f"{coin} df missing 'close' column. cols={list(df.columns)}")

    close = df["close"].astype(float)
    r = compute_log_returns(close)
    returns_df = pd.DataFrame({"ret": r}, index=r.index)

    train_df, val_df, test_df = train_test(returns_df)

    _, r_mean, r_std = standardize_with_train_stats(train_df["ret"], train_df["ret"])

    train_std = (train_df["ret"] - r_mean) / r_std
    val_std = (val_df["ret"] - r_mean) / r_std
    test_std = (test_df["ret"] - r_mean) / r_std

    if target_clip is not None and float(target_clip) > 0:
        tc = float(target_clip)
        train_std = train_std.clip(-tc, tc)
        val_std = val_std.clip(-tc, tc)
        test_std = test_std.clip(-tc, tc)

    train_df_std = pd.DataFrame({"ret": train_std.astype(np.float32)}, index=train_std.index)
    val_df_std = pd.DataFrame({"ret": val_std.astype(np.float32)}, index=val_std.index)
    test_df_std = pd.DataFrame({"ret": test_std.astype(np.float32)}, index=test_std.index)

    variables_used = ["ret"]
    train_ds = WindowedDataset(train_df_std, int(input_width), int(label_width), 0, variables_used)
    val_ds = WindowedDataset(val_df_std, int(input_width), int(label_width), 0, variables_used)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            "Not enough data after returns+split+windowing: "
            f"train_len={len(train_df_std)} val_len={len(val_df_std)} "
            f"needs >= input_width+label_width ({input_width}+{label_width})."
        )

    artifacts = ReturnsArtifacts(
        returns_mean=float(r_mean),
        returns_std=float(r_std),
        train_df_std=train_df_std,
        val_df_std=val_df_std,
        test_df_std=test_df_std,
        returns_df_raw=returns_df,
        close_raw=close,
    )
    return artifacts, train_ds, val_ds


# ---------------------------
# Train loop
# ---------------------------
def train_returns_model(
    *,
    model: nn.Module,
    spec: ReturnsTrainSpec,
    device: torch.device,
    debug_once: bool = False,
    on_epoch_end: Optional[EpochCallback] = None,
) -> TrainResult:
    # ---- datasets + loaders
    artifacts, train_ds, val_ds = build_returns_datasets(
        coin=spec.coin,
        interval=spec.interval,
        input_width=int(spec.input_width),
        label_width=int(spec.label_width),
        target_clip=spec.target_clip,
    )

    use_cuda = (device.type == "cuda")
    num_workers = int(spec.num_workers)
    pin_memory = bool(spec.pin_memory and use_cuda)
    persistent_workers = bool(spec.persistent_workers and num_workers > 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(spec.batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(spec.batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    # ---- baseline (predict 0)
    baseline_train = huber_zero_baseline(train_loader, beta=float(spec.huber_beta), device=device)
    baseline_val = huber_zero_baseline(val_loader, beta=float(spec.huber_beta), device=device)

    # ---- loss + optimizer
    loss_fn = nn.SmoothL1Loss(beta=float(spec.huber_beta))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(spec.lr),
        weight_decay=float(spec.weight_decay),
    )

    model = model.to(device)

    best_val = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    patience_left = int(spec.patience)

    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(1, int(spec.max_epochs) + 1):
        t0 = time.time()

        # ---- TRAIN
        model.train()
        running = 0.0
        n_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            pred = _coerce_pred_shape(pred, yb)

            loss = loss_fn(pred, yb)
            loss.backward()

            if spec.grad_clip is not None and float(spec.grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(spec.grad_clip))

            optimizer.step()

            running += float(loss.item())
            n_batches += 1

            if debug_once:
                print(
                    f"[debug] xb={tuple(xb.shape)} yb={tuple(yb.shape)} "
                    f"pred={tuple(pred.shape)} loss={loss.item():.6f}"
                )
                debug_once = False

        train_loss = running / max(n_batches, 1)
        train_losses.append(float(train_loss))

        # ---- VAL
        model.eval()
        v_running = 0.0
        v_batches = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                pred = _coerce_pred_shape(pred, yb)
                v_loss = loss_fn(pred, yb)
                v_running += float(v_loss.item())
                v_batches += 1

        val_loss = v_running / max(v_batches, 1)
        val_losses.append(float(val_loss))

        extra = {
            "epoch_time_sec": float(time.time() - t0),
            "patience_left": int(patience_left),
            "baseline_train": float(baseline_train),
            "baseline_val": float(baseline_val),
        }

        if on_epoch_end is not None:
            on_epoch_end(int(epoch), float(train_loss), float(val_loss), extra)

        # ---- Early stopping
        if val_loss < best_val:
            best_val = float(val_loss)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(spec.patience)
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return TrainResult(
        best_state_dict=best_state,
        train_losses=train_losses,
        val_losses=val_losses,
        artifacts=artifacts,
        baseline_train=float(baseline_train),
        baseline_val=float(baseline_val),
    )
