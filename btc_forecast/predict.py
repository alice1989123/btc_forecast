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
import numpy as np
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

    label_width = int(config.get("label_width", 12))
    input_width = int(config.get("input_width", 100))
    win = int(config.get("windows_normalization_length", 30))
    variables_used = config.get("variables_used") or ["close"]
    input_shape = config.get("input_shape") or (input_width, len(variables_used))

    # --- make sure MLflow knows where to fetch artifacts from ---
    log.info("📊 Predicting %s with %s (v%s) | interval=%s", coin, model_name, version, interval)
    log.debug(
        "config: input_width=%s label_width=%s win=%s vars=%s input_shape=%s device=%s",
        input_width, label_width, win, variables_used, input_shape, device
    )

    _apply_s3_params(s3_endpoint, s3_access_key, s3_secret_key, s3_region, s3_verify)

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # --- fetch enough raw history ---
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(hours=1000)
    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)

    log.debug("binance fetch: coin=%s interval=%s start=%s end=%s", coin, interval, start_time, end_time)

    raw_data = get_binance_data(coin, start_ts, end_ts, interval)
    if not raw_data:
        raise RuntimeError(f"Binance returned 0 rows for {coin} interval={interval}.")

    first = raw_data[0]
    if not isinstance(first, (list, tuple)):
        raise RuntimeError(f"Unexpected kline row type: {type(first)}")

    if len(first) < 11:
        raise RuntimeError(f"Unexpected kline schema length={len(first)} row={first}")
    log.debug("Kline schema ok | len=%s first=%s", len(first), first[:6])

    df = data_parser(raw_data)  # expects DateTimeIndex
    if df.empty:
        raise RuntimeError(f"No data returned for {coin} in the requested window.")

    # ---- df sanity ----
    log.debug("df cols=%s rows=%s index_first=%s index_last=%s tz=%s freq_guess=%s",
              list(df.columns), len(df), df.index[0], df.index[-1], getattr(df.index, "tz", None),
              getattr(pd.infer_freq(df.index[-min(len(df), 200):]), "upper", lambda: pd.infer_freq(df.index[-min(len(df), 200):]))())

    if "close" not in df.columns:
        raise RuntimeError(f"df missing 'close' column. cols={list(df.columns)}")

    # --------------------------------------------------------------------------
    # DEBUG: last_close and rolling stats in RAW space
    # --------------------------------------------------------------------------
    close_raw = df["close"].astype(float)
    last_close = float(close_raw.iloc[-1])
    log.info("RAW last_close=%0.2f at %s", last_close, df.index[-1])

    # match your rolling definition helper (it converts window=30 into Timedelta for time index)
    roll_win = resolve_rolling_window(df.index, win)

    # Rolling mean/std *in the SAME STYLE you denormalize with*:
    # NOTE: you denormalize using shift(label_width) then rolling then tail(label_width)
    mu_series = close_raw.shift(label_width).rolling(window=roll_win).mean()
    std_series = close_raw.shift(label_width).rolling(window=roll_win).std()  # ddof=1 default

    mu_last = _safe_float(mu_series.iloc[-1])
    std_last = _safe_float(std_series.iloc[-1])

    # implied z-score of the most recent close using those stats (just a diagnostic)
    last_z = (last_close - mu_last) / std_last if np.isfinite(std_last) and std_last != 0 else float("nan")

    log.info(
        "DENORM-STATS(last idx) mu_last=%0.2f std_last=%0.2f implied_last_z=%0.4f roll_win=%s shift=%s",
        mu_last, std_last, last_z, roll_win, label_width
    )

    # --------------------------------------------------------------------------
    # normalize (your normalize() is causal; uses rolling window & shift(label_width)) ---
    # --------------------------------------------------------------------------
    df_norm = normalize(df, label_width=label_width, window=win)

    # DEBUG: check normalization produced required columns and no NaN explosion
    missing = [c for c in variables_used if c not in df_norm.columns]
    if missing:
        raise RuntimeError(f"normalize(df) did not produce required columns {missing}. cols={list(df_norm.columns)}")

    # last input window, normalized
    recent_data = df_norm.tail(input_width)[variables_used]
    if len(recent_data) < input_width:
        raise RuntimeError(f"Not enough rows to build input window: need {input_width}, got {len(recent_data)}")

    # input window diagnostics
    log.info("INPUT window: first_time=%s last_time=%s (len=%s)",
             recent_data.index[0], recent_data.index[-1], len(recent_data))
    for v in variables_used:
        s = recent_data[v].astype(float)
        log.info("INPUT norm[%s]: last=%0.5f min=%0.5f max=%0.5f mean=%0.5f",
                 v, float(s.iloc[-1]), float(s.min()), float(s.max()), float(s.mean()))

    input_arr = data_for_prediction_parser(recent_data, input_shape=input_shape)
    input_tensor = torch.tensor(input_arr, dtype=torch.float32).to(device)
    _describe_array("input_arr", input_arr)

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

    _describe_array("preds_norm_raw", preds)

    # --- timestamps for predicted steps (based on interval) ---
    freq = _interval_to_pandas_freq(interval)
    start_pred = df.index[-1] + pd.Timedelta(freq)
    dti_new = pd.date_range(start=start_pred, periods=label_width, freq=freq)
    log.info("PRED timeline: hist_last=%s -> pred_start=%s freq=%s steps=%s",
             df.index[-1], dti_new[0], freq, label_width)

    pred_df = pd.DataFrame(preds, columns=variables_used, index=dti_new)

    # --- denormalize using same logic you used before ---
    denorm_df = pred_df.copy()

    # This is your denorm method: mean/std computed from RAW df
    # for each predicted step, use tail(label_width) of shifted rolling stats aligned to dti_new
    for var in variables_used:
        base = df[var].astype(float)

        mean = (
            base
            .shift(label_width)
            .rolling(window=roll_win)
            .mean()
            .tail(label_width)
        )
        std = (
            base
            .shift(label_width)
            .rolling(window=roll_win)
            .std()  # ddof=1 by default
            .tail(label_width)
        )

        # align to predicted timestamps
        mean = mean.copy()
        std = std.copy()
        mean.index = dti_new
        std.index = dti_new

        # --- DEBUG: canary for NaN/0 std ---
        n_nan_mean = int(mean.isna().sum())
        n_nan_std = int(std.isna().sum())
        min_std = float(np.nanmin(std.values)) if len(std) else float("nan")
        log.info("DENORM series[%s]: nan_mean=%s nan_std=%s min_std=%s",
                 var, n_nan_mean, n_nan_std, min_std)

        # log step1 stats
        step1_mu = _safe_float(mean.iloc[0])
        step1_std = _safe_float(std.iloc[0])
        step1_pred_norm = _safe_float(pred_df[var].iloc[0])

        step1_price = step1_pred_norm * step1_std + step1_mu if np.isfinite(step1_std) else float("nan")
        log.info(
            "STEP1[%s]: pred_norm=%0.5f mu=%0.2f std=%0.2f -> price=%0.2f (delta_vs_last_close=%0.2f)",
            var, step1_pred_norm, step1_mu, step1_std, step1_price, (step1_price - last_close)
        )

        denorm_df[var] = pred_df[var] * std + mean

    # return series: last input historical closes + predicted closes
    hist_close = df["close"].astype(float).tail(input_width)
    combined = pd.concat([hist_close, denorm_df["close"].astype(float)])

    # boundary check (where cliff appears on your chart)
    if len(combined) >= input_width + 1:
        hist_last_t = combined.index[input_width - 1]
        pred_first_t = combined.index[input_width]
        hist_last_p = float(combined.iloc[input_width - 1])
        pred_first_p = float(combined.iloc[input_width])
        log.info("BOUNDARY: hist_last=%s %0.2f | pred_first=%s %0.2f | delta=%0.2f",
                 hist_last_t, hist_last_p, pred_first_t, pred_first_p, (pred_first_p - hist_last_p))

    return combined
