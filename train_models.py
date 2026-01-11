# train_gru_returns_mlflow.py
# Full working training script (keeps label_width=12) to predict standardized LOG-RETURNS.
#
# Run:
#   export TRACKING_URI=http://172.16.0.200
#   export INTERVAL=1h
#   # optional:
#   # export COINS=BTCUSDT,ETHUSDT
#   # export DEBUG_COIN=BTCUSDT
#   python3 train_gru_returns_mlflow.py
#
# Notes:
# - Uses WindowedDataset from your project (expects it returns (xb, yb))
# - Standardizes returns using TRAIN stats only (mean/std) and clips targets (optional)
# - Logs canonical metadata at predict/metadata.json (stable path)
# - Registers model under name: gru-<coin>-<interval> (lowercase)

from __future__ import annotations

from itertools import product
import os
import copy
import json
import inspect
import traceback
from typing import Dict, Any, Tuple, Optional, List

import dotenv
import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchinfo import summary

from btc_forecast.data_loader import load_or_download
from btc_forecast.data_processing import train_test
from btc_forecast.windowed_dataset import WindowedDataset

from config.config import coins as DEFAULT_COINS


# ---------------------------
# Env / MLflow
# ---------------------------
dotenv.load_dotenv()

SEED = int(os.getenv("SEED", "42"))
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRACKING_URI = os.getenv("TRACKING_URI") or os.getenv("MLFLOW_TRACKING_URI")
INTERVAL = os.getenv("INTERVAL")
MODEL_NAME = "GRU"
interval_str = str(INTERVAL)

if not TRACKING_URI:
    raise ValueError("TRACKING_URI env var not set (or MLFLOW_TRACKING_URI)")
if not INTERVAL:
    raise ValueError("INTERVAL env var not set")

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(f"{MODEL_NAME}_ALL_COINS_RETURNS_V2_{interval_str}")

DEBUG_COIN = (os.getenv("DEBUG_COIN") or "BTCUSDT").strip()
COINS_ENV = (os.getenv("COINS") or "").strip()
COINS: List[str] = [c.strip() for c in COINS_ENV.split(",") if c.strip()] if COINS_ENV else list(DEFAULT_COINS)


# ---------------------------
# Helpers
# ---------------------------
def compute_log_returns(close: pd.Series) -> pd.Series:
    close = close.astype("float64")
    return np.log(close).diff().dropna()


def standardize_with_train_stats(train_s: pd.Series, s: pd.Series) -> Tuple[pd.Series, float, float]:
    mean = float(train_s.mean())
    std = float(train_s.std(ddof=1)) + 1e-8
    return (s - mean) / std, mean, std


def reconstruct_prices_from_returns(last_close: float, future_returns: np.ndarray) -> np.ndarray:
    log_p0 = np.log(float(last_close))
    log_path = log_p0 + np.cumsum(future_returns)
    return np.exp(log_path)


def _interval_to_pandas_freq(interval: str) -> str:
    interval = (interval or "").strip().lower()
    unit = interval[-1]
    n = int(interval[:-1])
    if unit not in ("m", "h", "d"):
        raise ValueError(f"Unsupported interval={interval!r} (use like '1h','4h','15m','1d')")
    return f"{n}{unit}"


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


def huber_zero_baseline(loader: DataLoader, beta: float) -> float:
    """
    Apples-to-apples baseline for SmoothL1Loss:
    predictor yhat = 0 => SmoothL1(yhat=0, y)
    computed per-element.
    """
    loss_fn = nn.SmoothL1Loss(beta=float(beta), reduction="sum")
    s, n = 0.0, 0
    for _, yb in loader:
        s += loss_fn(torch.zeros_like(yb), yb).item()
        n += yb.numel()
    return s / max(1, n)


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def get_registered_version_for_run(registered_model_name: str, run_id: str) -> Optional[int]:
    """
    Best-effort: find a registered model version created for this run_id.
    May return None if registry is slow/eventually consistent.
    """
    try:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{registered_model_name}'")
        matched = []
        for mv in versions:
            mv_run_id = getattr(mv, "run_id", None) or (mv.get("run_id") if isinstance(mv, dict) else None)
            mv_ver = getattr(mv, "version", None) or (mv.get("version") if isinstance(mv, dict) else None)
            if mv_run_id == run_id and mv_ver is not None:
                matched.append(int(mv_ver))
        return max(matched) if matched else None
    except Exception:
        return None


def log_predict_metadata_json(
    *,
    artifact_file: str,
    model_name: str,
    coin: str,
    interval: str,
    reg_name: str,
    reg_version: Optional[int],
    variables_used: List[str],
    input_width: int,
    label_width: int,
    num_features: int,
    returns_mean: float,
    returns_std: float,
    val_losses: List[float],
    train_losses: List[float],
    eval_out: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
):
    # Best epoch by val loss
    best_idx = int(np.argmin(val_losses)) if val_losses else -1
    best_epoch = int(best_idx + 1) if best_idx >= 0 else None
    val_best = float(val_losses[best_idx]) if best_idx >= 0 else None
    train_at_best = float(train_losses[best_idx]) if (best_idx >= 0 and best_idx < len(train_losses)) else None

    mae = _safe_float(eval_out.get("mae"))
    rmse = _safe_float(eval_out.get("rmse"))

    mae_steps = eval_out.get("mae_steps")
    rmse_steps = eval_out.get("rmse_steps")
    mae_per_step = [float(x) for x in mae_steps] if mae_steps is not None else []
    rmse_per_step = [float(x) for x in rmse_steps] if rmse_steps is not None else []

    run = mlflow.active_run()
    run_id = run.info.run_id if run else None

    payload = {
        # identity
        "model_name": model_name,
        "coin": coin,
        "interval": interval,
        "registry_name": reg_name,
        "version": reg_version,
        "run_id": run_id,

        # window config
        "input_width": int(input_width),
        "label_width": int(label_width),
        "input_shape": [int(input_width), int(num_features)],
        "variables_used": list(variables_used),

        # training summary
        "best_epoch": best_epoch,
        "val_loss": val_best,
        "train_loss_at_best": train_at_best,

        # scalers
        "returns_mean": float(returns_mean),
        "returns_std": float(returns_std),

        # evaluation in PRICE space
        "mae": mae,
        "rmse": rmse,
        "mae_per_step": mae_per_step,
        "rmse_per_step": rmse_per_step,
    }

    if extra:
        payload.update(extra)

    mlflow.log_dict(payload, artifact_file=artifact_file)

    # convenient tags/metrics
    if val_best is not None:
        mlflow.log_metric("val_loss_best", float(val_best))
        mlflow.set_tag("val_loss_best", str(val_best))
    if mae is not None:
        mlflow.set_tag("price_mae", str(mae))
    if best_epoch is not None:
        mlflow.set_tag("best_epoch", str(best_epoch))
    if reg_version is not None:
        mlflow.set_tag("registered_version", str(reg_version))


# ---------------------------
# MLflow architecture logging
# ---------------------------
def log_model_architecture(model, input_width, num_features, label_width, extra_config=None):
    cfg = {
        "model_class": model.__class__.__name__,
        "input_width": int(input_width),
        "label_width": int(label_width),
        "num_features": int(num_features),
    }
    if extra_config:
        cfg.update(extra_config)
    mlflow.log_text(json.dumps(cfg, indent=2), artifact_file="model/config.json")

    arch_str = str(model)
    try:
        s = summary(
            model,
            input_size=(1, int(input_width), int(num_features)),
            col_names=("input_size", "output_size", "num_params"),
            depth=4,
            verbose=0,
        )
        arch_str = str(s)
    except Exception:
        pass
    mlflow.log_text(arch_str + "\n", artifact_file="model/architecture.txt")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mlflow.log_text(
        json.dumps({"total_params": int(total_params), "trainable_params": int(trainable_params)}, indent=2),
        artifact_file="model/parameter_counts.json",
    )

    state_shapes = {k: list(v.shape) for k, v in model.state_dict().items()}
    mlflow.log_text(json.dumps(state_shapes, indent=2), artifact_file="model/state_dict_shapes.json")

    try:
        src = inspect.getsource(model.__class__)
        mlflow.log_text(src, artifact_file=f"model/source_{model.__class__.__name__}.py")
    except Exception:
        pass


# ---------------------------
# Model
# ---------------------------
class GRUStacked(nn.Module):
    def __init__(self, input_width, label_width, num_features, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=int(num_features),
            hidden_size=int(hidden_size),
            num_layers=int(num_layers),
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(int(hidden_size), int(label_width))
        self.label_width = int(label_width)
        self.horizon_bias = nn.Parameter(torch.zeros(self.label_width))

    def forward(self, x):
        out, _ = self.gru(x)             # (B, T, H)
        out = out[:, -1, :]              # (B, H)
        out = self.fc(out)               # (B, out_steps)
        out = out + self.horizon_bias    # (B, out_steps)
        return out.unsqueeze(-1)         # (B, out_steps, 1)


# ---------------------------
# Grid (returns-only) - KEEPS label_width=12
# ---------------------------
variable_sets = [["ret"]]

param_grid = {
    "input_width": [250],
    "label_width": [12],
    "batch_size": [128],
    "learning_rate": [0.003],
    "num_epochs": [80],
    "coin": COINS,
    "hidden_size": [128],
    "num_layers": [2],
    "dropout": [0.1],
    "patience": [10],
    "grad_clip": [1.0],
    "weight_decay": [0.0],
    "huber_beta": [0.2],
    "target_clip": [6.0],  # clip standardized returns to [-6, 6]
}

all_combinations = list(
    product(
        variable_sets,
        param_grid["input_width"],
        param_grid["label_width"],
        param_grid["batch_size"],
        param_grid["learning_rate"],
        param_grid["num_epochs"],
        param_grid["coin"],
        param_grid["hidden_size"],
        param_grid["num_layers"],
        param_grid["dropout"],
        param_grid["patience"],
        param_grid["grad_clip"],
        param_grid["weight_decay"],
        param_grid["huber_beta"],
        param_grid["target_clip"],
    )
)


# ---------------------------
# Training (returns)
# ---------------------------
def train_model_returns(
    coin: str,
    model: nn.Module,
    *,
    interval: str,
    input_width: int,
    label_width: int,
    batch_size: int,
    lr: float,
    max_epochs: int,
    patience: int,
    grad_clip: float,
    weight_decay: float,
    huber_beta: float,
    target_clip: Optional[float],
    debug_once: bool = False,
):
    df = load_or_download(coin, interval)
    df = ensure_datetime_index(df)

    if "close" not in df.columns:
        raise RuntimeError(f"{coin} df missing 'close' column. cols={list(df.columns)}")

    close = df["close"].astype(float)

    r = compute_log_returns(close)
    returns_df = pd.DataFrame({"ret": r}, index=r.index)

    # split in return space
    train_df, val_df, test_df = train_test(returns_df)

    # standardize using TRAIN stats only
    _, r_mean, r_std = standardize_with_train_stats(train_df["ret"], train_df["ret"])
    train_std = (train_df["ret"] - r_mean) / r_std
    val_std = (val_df["ret"] - r_mean) / r_std
    test_std = (test_df["ret"] - r_mean) / r_std

    # optional clipping (fat tails)
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
            f"Not enough data after returns+split+windowing: "
            f"train_len={len(train_df_std)} val_len={len(val_df_std)} "
            f"needs >= input_width+label_width."
        )

    train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, drop_last=False)

    if debug_once:
        xb, yb = next(iter(train_loader))
        print("xb shape", xb.shape, "yb shape", yb.shape)
        print("xb stats", xb.mean().item(), xb.std().item(), xb.min().item(), xb.max().item())
        print("yb stats", yb.mean().item(), yb.std().item(), yb.min().item(), yb.max().item())
        print("example: last_x vs first_y (first 5)")
        print(torch.stack([xb[:5, -1, 0], yb[:5, 0, 0]], dim=1))

    optimizer = optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-5)
    loss_fn = nn.SmoothL1Loss(beta=float(huber_beta))

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_weights = None
    early_stop_counter = 0

    baseline_train = huber_zero_baseline(train_loader, beta=float(huber_beta))
    baseline_val = huber_zero_baseline(val_loader, beta=float(huber_beta))
    print("baseline huber train/val (per-element):", baseline_train, baseline_val)
    if mlflow.active_run():
        mlflow.log_metric("baseline_train_huber_zero", float(baseline_train))
        mlflow.log_metric("baseline_val_huber_zero", float(baseline_val))

    for epoch in range(int(max_epochs)):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
            optimizer.step()

            running_loss += float(loss.item())

        train_loss = running_loss / max(1, len(train_loader))
        train_losses.append(train_loss)

        model.eval()
        v = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                v += float(loss_fn(preds, yb).item())
        val_loss = v / max(1, len(val_loader))
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        if mlflow.active_run():
            mlflow.log_metric("train_loss", float(train_loss), step=epoch + 1)
            mlflow.log_metric("val_loss", float(val_loss), step=epoch + 1)
            mlflow.log_metric("lr", float(current_lr), step=epoch + 1)

        print(f"📉 Epoch {epoch+1}: Train={train_loss:.6f} | Val={val_loss:.6f} | lr={current_lr:.6g}")

        if np.isnan(train_loss) or np.isnan(val_loss):
            raise RuntimeError("NaN detected in loss — aborting run.")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= int(patience):
                print("🛑 Early stopping")
                break

    artifacts = {
        "scaler": {"returns_mean": float(r_mean), "returns_std": float(r_std)},
        "train_df_std": train_df_std,
        "val_df_std": val_df_std,
        "test_df_std": test_df_std,
        "returns_df_raw": returns_df,
        "close_raw": close,
    }
    return best_weights, train_losses, val_losses, artifacts


# ---------------------------
# Evaluation (price reconstruction)
# ---------------------------
def eval_model_price_from_returns(
    *,
    model: nn.Module,
    close_raw: pd.Series,
    test_df_std: pd.DataFrame,
    returns_mean: float,
    returns_std: float,
    interval: str,
    input_width: int,
    label_width: int,
) -> Dict[str, Any]:
    ds = WindowedDataset(test_df_std, int(input_width), int(label_width), 0, ["ret"])
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    if len(ds) == 0:
        raise RuntimeError("Test dataset is empty after windowing.")

    preds_prices = []
    targets_prices = []

    close_idx = close_raw.index
    is_dt = isinstance(close_idx, pd.DatetimeIndex)

    model.eval()
    with torch.no_grad():
        for i, (xb, yb) in enumerate(loader):
            xb = xb.to(device)
            yb = yb.to(device)

            yhat = model(xb)

            # de-standardize returns (no extra eps; scaler already has eps)
            yhat_r = (yhat[0, :, 0].detach().cpu().numpy() * returns_std + returns_mean)
            ytrue_r = (yb[0, :, 0].detach().cpu().numpy() * returns_std + returns_mean)

            if i + int(input_width) >= len(test_df_std.index):
                break
            y_start_time = test_df_std.index[i + int(input_width)]

            if is_dt and isinstance(y_start_time, (pd.Timestamp, np.datetime64)):
                if y_start_time in close_raw.index:
                    last_close = float(close_raw.loc[y_start_time])
                else:
                    pos = close_raw.index.searchsorted(y_start_time, side="right") - 1
                    pos = max(0, min(pos, len(close_raw) - 1))
                    last_close = float(close_raw.iloc[pos])
            else:
                pos = min(len(close_raw) - 1, (len(close_raw) - len(test_df_std)) + (i + int(input_width)))
                last_close = float(close_raw.iloc[pos])

            pred_path = reconstruct_prices_from_returns(last_close, yhat_r)
            true_path = reconstruct_prices_from_returns(last_close, ytrue_r)

            preds_prices.append(pred_path)
            targets_prices.append(true_path)

    preds_prices = np.asarray(preds_prices)
    targets_prices = np.asarray(targets_prices)

    mae = float(np.mean(np.abs(targets_prices - preds_prices)))
    rmse = float(np.sqrt(np.mean((targets_prices - preds_prices) ** 2)))
    mae_steps = np.mean(np.abs(targets_prices - preds_prices), axis=0)
    rmse_steps = np.sqrt(np.mean((targets_prices - preds_prices) ** 2, axis=0))

    if isinstance(close_raw.index, pd.DatetimeIndex):
        pd_freq = _interval_to_pandas_freq(interval)
        last_hist_end_time = close_raw.index[-1]
        _ = pd.date_range(
            start=last_hist_end_time + pd.Timedelta(pd_freq),
            periods=int(label_width),
            freq=pd_freq,
        )

    return {"mae_steps": mae_steps, "rmse_steps": rmse_steps, "mae": mae, "rmse": rmse}


# ---------------------------
# Run grid
# ---------------------------
for (
    variables_used,
    input_width,
    label_width,
    batch_size,
    learning_rate,
    num_epochs,
    coin,
    hidden_size,
    num_layers,
    dropout,
    patience,
    grad_clip,
    weight_decay,
    huber_beta,
    target_clip,
) in all_combinations:

    variables_used = ["ret"]
    num_features = 1

    model_config = {
        "input_width": int(input_width),
        "label_width": int(label_width),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "num_features": int(num_features),
    }
    model = GRUStacked(**model_config)

    run_name = (
        f"{model.__class__.__name__}"
        f"|RET"
        f"|in{input_width}-out{label_width}"
        f"|bs{batch_size}|lr{learning_rate}"
        f"|hs{hidden_size}|L{num_layers}|do{dropout}"
        f"|gc{grad_clip}|wd{weight_decay}"
        f"|hb{huber_beta}|clip{target_clip}"
        f"|{coin}"
        f"|{interval_str}"
    )

    with mlflow.start_run(run_name=run_name):
        try:
            mlflow.log_params(
                {
                    "model_class": str(model.__class__.__name__),
                    "target": "log_return",
                    "num_features": int(num_features),
                    "input_width": int(input_width),
                    "label_width": int(label_width),
                    "batch_size": int(batch_size),
                    "learning_rate": float(learning_rate),
                    "num_epochs": int(num_epochs),
                    "patience": int(patience),
                    "coin": str(coin),
                    "interval": interval_str,
                    "variables_used": json.dumps(list(variables_used)),
                    "hidden_size": int(hidden_size),
                    "num_layers": int(num_layers),
                    "dropout": float(dropout),
                    "grad_clip": float(grad_clip),
                    "weight_decay": float(weight_decay),
                    "loss": f"SmoothL1(beta={float(huber_beta)})",
                    "target_clip": float(target_clip),
                    "seed": int(SEED),
                }
            )

            mlflow.set_tags(
                {
                    "Model": str(model.__class__.__name__),
                    "Coin": str(coin),
                    "Interval": str(interval_str),
                    "status": "running",
                }
            )

            log_model_architecture(
                model,
                input_width=int(input_width),
                num_features=int(num_features),
                label_width=int(label_width),
                extra_config={
                    "hidden_size": int(hidden_size),
                    "num_layers": int(num_layers),
                    "dropout": float(dropout),
                },
            )

            model.to(device)

            best_weights, train_l, val_l, artifacts = train_model_returns(
                coin=str(coin),
                model=model,
                interval=interval_str,
                input_width=int(input_width),
                label_width=int(label_width),
                batch_size=int(batch_size),
                lr=float(learning_rate),
                max_epochs=int(num_epochs),
                patience=int(patience),
                grad_clip=float(grad_clip),
                weight_decay=float(weight_decay),
                huber_beta=float(huber_beta),
                target_clip=float(target_clip),
                debug_once=(str(coin) == DEBUG_COIN),
            )

            if best_weights is None:
                best_weights = copy.deepcopy(model.state_dict())
            model.load_state_dict(best_weights)
            model.eval()

            scaler = artifacts["scaler"]
            mlflow.log_param("returns_mean", float(scaler["returns_mean"]))
            mlflow.log_param("returns_std", float(scaler["returns_std"]))
            mlflow.log_dict(scaler, artifact_file="scaler/returns_scaler.json")

            eval_out = eval_model_price_from_returns(
                model=model,
                close_raw=artifacts["close_raw"],
                test_df_std=artifacts["test_df_std"],
                returns_mean=float(scaler["returns_mean"]),
                returns_std=float(scaler["returns_std"]),
                interval=interval_str,
                input_width=int(input_width),
                label_width=int(label_width),
            )

            mae_steps = eval_out["mae_steps"]
            rmse_steps = eval_out["rmse_steps"]

            mlflow.log_metric("train_loss_last", float(train_l[-1]))
            mlflow.log_metric("val_loss_last", float(val_l[-1]))
            mlflow.log_metric("price_mae", float(eval_out["mae"]))
            mlflow.log_metric("price_rmse", float(eval_out["rmse"]))

            mlflow.log_dict({"mae_per_step": [float(x) for x in mae_steps]}, artifact_file="metrics/price_mae_per_step.json")
            mlflow.log_dict({"rmse_per_step": [float(x) for x in rmse_steps]}, artifact_file="metrics/price_rmse_per_step.json")

            for step, v in enumerate(mae_steps, start=1):
                mlflow.log_metric(f"price_mae_step_{step:02d}", float(v))
            for step, v in enumerate(rmse_steps, start=1):
                mlflow.log_metric(f"price_rmse_step_{step:02d}", float(v))

            # ---- Register model ----
            model_to_log = copy.deepcopy(model).to("cpu").eval()
            input_example = np.zeros((1, int(input_width), int(num_features)), dtype=np.float32)

            with torch.no_grad():
                sample_out = model_to_log(torch.from_numpy(input_example))
            signature = infer_signature(input_example, sample_out.cpu().numpy())

            pip_reqs = [
                "mlflow",
                "torch==2.5.1",
                "numpy>=1.24,<3",
                "pandas>=2.0,<3",
                "scikit-learn>=1.3,<2",
            ]

            reg_name = f"{MODEL_NAME}-{coin}-{interval_str}".lower()

            mlflow.pytorch.log_model(
                model_to_log,
                artifact_path="model",
                input_example=input_example,
                signature=signature,
                pip_requirements=pip_reqs,
                registered_model_name=reg_name,
            )

            run_id = mlflow.active_run().info.run_id
            reg_version = get_registered_version_for_run(reg_name, run_id)

            # ---- Canonical metadata JSON (predict/API reads this) ----
            log_predict_metadata_json(
                artifact_file="predict/metadata.json",
                model_name=MODEL_NAME,
                coin=str(coin),
                interval=str(interval_str),
                reg_name=reg_name,
                reg_version=reg_version,
                variables_used=list(variables_used),
                input_width=int(input_width),
                label_width=int(label_width),
                num_features=int(num_features),
                returns_mean=float(scaler["returns_mean"]),
                returns_std=float(scaler["returns_std"]),
                val_losses=[float(x) for x in val_l],
                train_losses=[float(x) for x in train_l],
                eval_out=eval_out,
                extra={
                    "windows_normalization_length": 30,  # adjust to your pipeline if needed
                    "target": "log_return",
                    "loss": f"SmoothL1(beta={float(huber_beta)})",
                    "target_clip": float(target_clip),
                },
            )

            mlflow.set_tag("status", "success")

        except KeyboardInterrupt:
            try:
                mlflow.set_tag("status", "killed")
            except Exception:
                pass
            raise

        except Exception as e:
            try:
                mlflow.set_tag("status", "failed")
                mlflow.set_tag("error_type", type(e).__name__)
                mlflow.set_tag("error_msg", str(e)[:200])
            except Exception:
                pass

            tb_str = traceback.format_exc()
            try:
                with open("error_trace.txt", "w") as f:
                    f.write(tb_str)
                mlflow.log_artifact("error_trace.txt", artifact_path="errors")
                mlflow.log_dict(
                    {"error": str(e), "traceback": tb_str.splitlines()},
                    artifact_file="errors/error.json",
                )
            except Exception:
                pass
            continue
