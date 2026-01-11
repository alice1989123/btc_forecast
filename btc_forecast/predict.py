# btc_forecast/predict.py
# Return-forecasting version (predict log-returns -> reconstruct prices)
# Assumes your MLflow model outputs shape (B, label_width) OR (B, label_width, 1)

import os
import re
import logging
import warnings
import datetime
from typing import Optional, Dict, Any

import dotenv
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch

from btc_forecast.binance_data import get_binance_data
from btc_forecast.data_processing import data_parser  # keep your existing parser

dotenv.load_dotenv()
log = logging.getLogger(__name__)
warnings.simplefilter(action="ignore", category=FutureWarning)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI") or os.getenv("TRACKING_URI")
if not TRACKING_URI:
    raise ValueError("TRACKING_URI (or MLFLOW_TRACKING_URI) environment variable not set.")
mlflow.set_tracking_uri(TRACKING_URI)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# helpers
# ----------------------------
def _interval_to_pandas_freq(interval: str) -> str:
    interval = (interval or "").strip().lower()
    m = re.fullmatch(r"(\d+)\s*([mhd])", interval)
    if not m:
        raise ValueError(f"Unsupported interval format: {interval!r} (expected like '1h','4h','15m','1d')")
    return f"{int(m.group(1))}{m.group(2)}"


def _apply_s3_params(
    endpoint: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    region: Optional[str] = "us-east-1",
    verify: bool = True,
) -> None:
    if endpoint:
        endpoint = endpoint.strip()
        if not endpoint.startswith("http"):
            endpoint = ("https://" if verify else "http://") + endpoint
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = endpoint

    if access_key:
        os.environ["AWS_ACCESS_KEY_ID"] = access_key
    if secret_key:
        os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key

    if region and not (os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")):
        os.environ["AWS_REGION"] = region

    os.environ.setdefault("AWS_S3_ADDRESSING_STYLE", "path")
    if not verify:
        os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _describe_array(name: str, arr: np.ndarray, max_items: int = 5) -> None:
    if arr is None:
        log.debug("%s: <None>", name)
        return
    a = np.asarray(arr)
    if a.size == 0:
        log.debug("%s: empty", name)
        return
    flat = a.reshape(-1)
    head = [float(x) for x in flat[:max_items]]
    log.debug(
        "%s: shape=%s dtype=%s min=%s max=%s mean=%s head=%s",
        name,
        a.shape,
        a.dtype,
        float(np.nanmin(flat)),
        float(np.nanmax(flat)),
        float(np.nanmean(flat)),
        head,
    )


def compute_log_returns(close: pd.Series) -> pd.Series:
    close = close.astype("float64")
    return np.log(close).diff().dropna()


def reconstruct_prices(last_close: float, pred_returns: np.ndarray) -> np.ndarray:
    # pred_returns: shape (label_width,)
    log_p0 = np.log(float(last_close))
    log_path = log_p0 + np.cumsum(pred_returns)
    return np.exp(log_path)


def predict_returns_to_prices(
    df: pd.DataFrame,
    input_width: int,
    label_width: int,
    interval: str,
    model,
    device,
    *,
    r_mean: float = 0.0,
    r_std: float = 1.0,
) -> pd.Series:
    close = df["close"].astype(float)
    r = compute_log_returns(close)

    if len(r) < input_width:
        raise RuntimeError(f"Not enough returns: need {input_width}, got {len(r)}")

    # input window of returns (last input_width)
    x = r.iloc[-input_width:].to_numpy(dtype=np.float32)
    x_std = (x - float(r_mean)) / (float(r_std) + 1e-8)

    # (B, T, F) = (1, input_width, 1)
    x_tensor = torch.tensor(x_std[None, :, None], dtype=torch.float32).to(device)
    _describe_array("x_returns_std", x_std)

    model.eval()
    with torch.no_grad():
        yhat = model(x_tensor).detach().cpu().numpy()

    # normalize output shape to (label_width,)
    if yhat.ndim == 3:
        # (B, label_width, 1) or (B, label_width, F)
        yhat = yhat[0, :, 0]
    elif yhat.ndim == 2:
        # (B, label_width)
        yhat = yhat[0, :]
    else:
        raise ValueError(f"Unexpected model output shape: {yhat.shape}")

    if yhat.shape[0] != label_width:
        raise ValueError(f"Model returned {yhat.shape[0]} steps, expected label_width={label_width}")

    _describe_array("yhat_returns_std", yhat)

    # de-standardize returns
    pred_r = yhat * (float(r_std) + 1e-8) + float(r_mean)

    # future index
    pd_freq = _interval_to_pandas_freq(interval)
    start_pred = close.index[-1] + pd.Timedelta(pd_freq)
    idx_future = pd.date_range(start=start_pred, periods=label_width, freq=pd_freq)

    # reconstruct future prices
    last_close = float(close.iloc[-1])
    future_prices = reconstruct_prices(last_close, pred_r)

    # history for chart + future
    hist = close.tail(input_width)
    fut = pd.Series(future_prices, index=idx_future, name="close")
    combined = pd.concat([hist, fut])

    # debug boundary
    if len(combined) >= input_width + 1:
        hist_last_t = combined.index[input_width - 1]
        pred_first_t = combined.index[input_width]
        hist_last_p = float(combined.iloc[input_width - 1])
        pred_first_p = float(combined.iloc[input_width])
        log.info(
            "BOUNDARY: hist_last=%s %0.2f | pred_first=%s %0.2f | delta=%0.2f",
            hist_last_t, hist_last_p, pred_first_t, pred_first_p, (pred_first_p - hist_last_p)
        )

    return combined


# ----------------------------
# main predict()
# ----------------------------
def predict(
    config: Dict[str, Any],
    coin: str,
    model_name: str,
    version: int = 1,
    *,
    tracking_uri: Optional[str] = None,
    s3_endpoint: Optional[str] = None,
    s3_access_key: Optional[str] = None,
    s3_secret_key: Optional[str] = None,
    s3_region: Optional[str] = "us-east-1",
    s3_verify: bool = True,
) -> pd.Series:
    interval = config.get("interval")
    if not interval:
        raise ValueError("config['interval'] is required (prevents mixing intervals).")

    label_width = int(config.get("label_width", 12))
    input_width = int(config.get("input_width", 100))

    # These must come from training (train returns mean/std) for stability.
    # If missing, defaults are okay but training should really set them.
    r_mean = float(config.get("returns_mean", 0.0))
    r_std = float(config.get("returns_std", 1.0))

    log.info("📈 Predicting %s with %s (v%s) | interval=%s | target=log_return", coin, model_name, version, interval)
    log.debug("config: input_width=%s label_width=%s returns_mean=%s returns_std=%s device=%s",
              input_width, label_width, r_mean, r_std, device)

    _apply_s3_params(s3_endpoint, s3_access_key, s3_secret_key, s3_region, s3_verify)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # fetch a decent history window (tune as you like)
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(hours=2000)
    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)

    log.debug("binance fetch: coin=%s interval=%s start=%s end=%s", coin, interval, start_time, end_time)
    raw_data = get_binance_data(coin, start_ts, end_ts, interval)
    if not raw_data:
        raise RuntimeError(f"Binance returned 0 rows for {coin} interval={interval}.")

    df = data_parser(raw_data)
    if df.empty:
        raise RuntimeError(f"No data returned for {coin} in the requested window.")
    if "close" not in df.columns:
        raise RuntimeError(f"df missing 'close' column. cols={list(df.columns)}")

    # time sanity
    now_utc = datetime.datetime.utcnow()
    df_last = df.index[-1].to_pydatetime()
    delta_h = (df_last - now_utc).total_seconds() / 3600
    log.warning("TIME sanity: now_utc=%s df_last=%s delta_hours=%0.2f", now_utc, df_last, delta_h)

    last_close = float(df["close"].astype(float).iloc[-1])
    log.info("RAW last_close=%0.2f at %s", last_close, df.index[-1])

    # load model from registry
    registry_name = f"{model_name}-{coin}-{interval}".lower()
    model_uri = f"models:/{registry_name}/{version}"
    log.info("Loading model | uri=%s device=%s", model_uri, device)

    model = mlflow.pytorch.load_model(model_uri).to(device)
    model.eval()
    log.debug("✅ Loaded model from '%s'", model_uri)

    # predict + reconstruct prices
    combined = predict_returns_to_prices(
        df=df,
        input_width=input_width,
        label_width=label_width,
        interval=interval,
        model=model,
        device=device,
        r_mean=r_mean,
        r_std=r_std,
    )

    return combined
