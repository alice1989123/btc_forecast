# btc_forecast/predict.py
import os
import re
import logging
import warnings
import datetime
from typing import Optional, Dict, Any

import dotenv
import mlflow
import mlflow.pytorch
import pandas as pd
import torch

from utils.rolling import resolve_rolling_window
from btc_forecast.binance_data import get_binance_data
from btc_forecast.data_processing import normalize, data_parser, data_for_prediction_parser

# ------------------------------------------------------------------------------
# Env + logging
# ------------------------------------------------------------------------------
dotenv.load_dotenv()  # or dotenv.load_dotenv(".keys.env") in your caller

log = logging.getLogger(__name__)
warnings.simplefilter(action="ignore", category=FutureWarning)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI") or os.getenv("TRACKING_URI")
if not TRACKING_URI:
    raise ValueError("TRACKING_URI (or MLFLOW_TRACKING_URI) environment variable not set.")
mlflow.set_tracking_uri(TRACKING_URI)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _interval_to_pandas_freq(interval: str) -> str:
    """
    Accept "1h", "4h", "15m", "1d" etc. Return pandas offset alias string.
    """
    interval = (interval or "").strip().lower()
    m = re.fullmatch(r"(\d+)\s*([mhd])", interval)
    if not m:
        raise ValueError(
            f"Unsupported interval format: {interval!r} (expected like '1h', '4h', '15m', '1d')"
        )
    n = int(m.group(1))
    unit = m.group(2)
    return f"{n}{unit}"


def _apply_s3_params(
    endpoint: Optional[str] = None,
    access_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    region: Optional[str] = "us-east-1",
    verify: bool = True,  # set False for self-signed MinIO
) -> None:
    """
    Set just enough env vars for MLflow/boto3 to use S3-compatible storage.
    """
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

    # path-style works best with MinIO
    os.environ.setdefault("AWS_S3_ADDRESSING_STYLE", "path")

    if not verify:
        os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"

    # masked debug log
    if log.isEnabledFor(logging.DEBUG):
        mask = lambda s: (s[:3] + "..." + s[-4:]) if s else None
        log.debug(
            "S3 params applied endpoint=%s access=%s region=%s verify=%s",
            os.getenv("MLFLOW_S3_ENDPOINT_URL"),
            mask(os.getenv("AWS_ACCESS_KEY_ID")),
            os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"),
            verify,
        )


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
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
    """
    Returns a Series with:
      - last input_width historical closes (raw close)
      - followed by label_width predicted closes (denormalized)
    """

    # --- interval MUST exist to avoid mixing registry/models & timestamps ---
    interval = config.get("interval")
    if not interval:
        raise ValueError("config['interval'] is required (prevents mixing intervals).")

    # --- make sure MLflow knows where to fetch artifacts from ---
    log.info("📊 Predicting %s with %s (v%s) | interval=%s", coin, model_name, version, interval)
    _apply_s3_params(s3_endpoint, s3_access_key, s3_secret_key, s3_region, s3_verify)

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # --- fetch enough raw history ---
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(hours=1000)
    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)

    raw_data = get_binance_data(coin, start_ts, end_ts,interval)
    if not raw_data:
        raise RuntimeError(f"Binance returned 0 rows for {coin} interval={interval}.")

    first = raw_data[0]
    if not isinstance(first, (list, tuple)):
        raise RuntimeError(f"Unexpected kline row type: {type(first)}")

    if len(first) < 6:
        raise RuntimeError(f"Kline row too short: len={len(first)} row={first}")

    # Most common is 12; we accept >= 11 just in case libs differ slightly
    if len(first) < 11:
        raise RuntimeError(f"Unexpected kline schema length={len(first)} row={first}")
    log.debug("Kline schema ok | len=%s first=%s", len(first), first[:6])
    
    df = data_parser(raw_data)  # expects DateTimeIndex
    if df.empty:
        raise RuntimeError(f"No data returned for {coin} in the requested window.")

    # --- normalize (your normalize() is causal; uses rolling window & shift(label_width)) ---
    label_width = int(config.get("label_width", 12))
    input_width = int(config.get("input_width", 100))
    win = int(config.get("windows_normalization_length", 30))

    variables_used = config.get("variables_used") or ["close"]
    input_shape = config.get("input_shape") or (input_width, len(variables_used))

    df_norm = normalize(df, label_width=label_width, window=win)

    # last input window, normalized
    recent_data = df_norm.tail(input_width)[variables_used]
    if len(recent_data) < input_width:
        raise RuntimeError(f"Not enough rows to build input window: need {input_width}, got {len(recent_data)}")

    input_arr = data_for_prediction_parser(recent_data, input_shape=input_shape)
    input_tensor = torch.tensor(input_arr, dtype=torch.float32).to(device)

    # --- load the RIGHT model (must include interval suffix) ---
    registry_name = f"{model_name}-{coin}-{interval}".lower()
    model_uri = f"models:/{registry_name}/{version}"
    log.info("Loading model | uri=%s device=%s", model_uri, device)

    model = mlflow.pytorch.load_model(model_uri).to(device)
    model.eval()
    log.debug("✅ Loaded model from '%s'", model_uri)

    # --- predict ---
    with torch.no_grad():
        preds = model(input_tensor).detach().cpu().numpy()

    # shape: (batch, label_width, num_features) or (label_width, num_features)
    if preds.ndim == 3 and preds.shape[0] == 1:
        preds = preds[0]
    elif preds.ndim == 3:
        raise ValueError(f"Unexpected prediction shape (batch>1): {preds.shape}")
    elif preds.ndim == 1:
        preds = preds.reshape(-1, 1)

    # --- timestamps for predicted steps (based on interval) ---
    freq = _interval_to_pandas_freq(interval)
    dti_new = pd.date_range(
        start=df.index[-1] + pd.Timedelta(freq),
        periods=label_width,
        freq=freq,
    )

    pred_df = pd.DataFrame(preds, columns=variables_used, index=dti_new)

    # --- denormalize using same logic you used before ---
    # IMPORTANT: your existing logic denormalizes from the normalized space back to raw space
    denorm_df = pred_df.copy()

    roll_win = resolve_rolling_window(df.index, win)

    for var in variables_used:
        mean = (
            df[var]
            .shift(label_width)
            .rolling(window=roll_win)
            .mean()
            .tail(label_width)
        )
        std = (
            df[var]
            .shift(label_width)
            .rolling(window=roll_win)
            .std()
            .tail(label_width)
        )

        mean.index = dti_new
        std.index = dti_new

        denorm_df[var] = pred_df[var] * std + mean

    # return series: last input historical closes + predicted closes
    combined = pd.concat([df["close"].tail(input_width), denorm_df["close"]])
    return combined
