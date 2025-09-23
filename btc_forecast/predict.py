import torch
import pandas as pd
import datetime
import os
import mlflow.pytorch
from datetime import timedelta as td
from utils.rolling import resolve_rolling_window
from btc_forecast.binance_data import get_binance_data
from btc_forecast.data_processing import normalize, data_parser, data_for_prediction_parser
from config import config
import logger
import warnings
import dotenv
from mlflow import MlflowClient
import os, mlflow
import logging

dotenv.load_dotenv()  # take environment variables from .env.

TRACKING_URI = os.getenv("TRACKING_URI")
if TRACKING_URI:
    mlflow.set_tracking_uri(TRACKING_URI)
else:
    raise ValueError("TRACKING_URI environment variable not set.")



client = MlflowClient()
warnings.simplefilter(action='ignore', category=FutureWarning)

# Setup logging
log = logging.getLogger(__name__)

def _apply_s3_params(
    endpoint: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    region: str | None = "us-east-1",
    verify: bool = True,          # set False for self-signed MinIO
):
    """Set just enough env vars for MLflow/boto3 to use S3-compatible storage."""
    if endpoint:
        if not endpoint.startswith("http"):
            endpoint = "https://" + endpoint if verify else "http://" + endpoint
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
        # lets MLflow/boto3 skip TLS verification for the S3 endpoint
        os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
    # optional: masked debug
    if log.isEnabledFor(logging.DEBUG):
        mask = lambda s: s[:3] + "..." + s[-4:] if s else None
        log.debug(
            "S3 params applied endpoint=%s access=%s region=%s verify=%s",
            os.getenv("MLFLOW_S3_ENDPOINT_URL"), mask(os.getenv("AWS_ACCESS_KEY_ID")),
            os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"), verify
        )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(config :dict , coin: str, model_name: str, version: int = 1, *,
    tracking_uri: str | None = None,
    s3_endpoint: str | None = None,
    s3_access_key: str | None = None,
    s3_secret_key: str | None = None,
    s3_region: str | None = "us-east-1",
    s3_verify: bool = True,):

    # 1) Wire S3 creds for MLflow artifacts (MinIO or AWS S3)
    logging.info(f"📊 Predicting {coin} with {model_name} (v{version})...")
    _apply_s3_params(s3_endpoint, s3_access_key, s3_secret_key, s3_region, s3_verify)
    # 2) Tracking server (HTTP URL of MLflow)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    # ──⏳ Time window
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(hours=1000)
    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)

    # ──📥 Get & normalize data
    raw_data = get_binance_data(coin, start_ts, end_ts)
    df = data_parser(raw_data)
    df_norm = normalize(df, label_width=config.get("label_width"), window=config.get("windows_normalization_length"))

    # Prepare last input window
    recent_data = df_norm.tail(config["input_width"])[config["variables_used"]]
    input_tensor = data_for_prediction_parser(
        recent_data, input_shape=config["input_shape"]
    )
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32).to(device)

    # ──🔍 Load model from MLflow Registry
    model_uri = f"models:/{model_name}-{coin}/{version}".lower()
    model = mlflow.pytorch.load_model(model_uri).to(device)
    model.eval()
    logging.debug(f"✅ Loaded model from '{model_uri}'") 
    # ──📈 Make predictions
    with torch.no_grad():
        preds = model(input_tensor).cpu().numpy()

    # Handle shape
    if preds.ndim == 3 and preds.shape[0] == 1:
        preds = preds[0]  # squeeze batch
    elif preds.ndim == 3:
        raise ValueError(f"❌ Unexpected shape: batch size > 1: {preds.shape}")
    elif preds.ndim == 1:
        preds = preds.reshape(-1, 1)

 
    # ──📅 Create timestamps for predicted steps
    dti_new = pd.date_range(
        start=df.index[-1] + pd.Timedelta(hours=1),
        periods=config["label_width"],
        freq="h"
    )
    pred_df = pd.DataFrame(preds, columns=config["variables_used"], index=dti_new)

    # ──📈 Denormalize
    denorm_df = pred_df.copy()
    win_raw = config.get("windows_normalization_length", 30)
    roll_win = resolve_rolling_window(df.index, win_raw)
    
    for var in config["variables_used"]:

        mean = (
            df[var]
            .shift(config["label_width"])
            .rolling(window=roll_win)
            .mean()
            .tail(config["label_width"])
        )
        std = (
            df[var]
            .shift(config["label_width"])
            .rolling(window=roll_win)
            .std()
            .tail(config["label_width"])
        )
        mean.index, std.index = dti_new, dti_new
        denorm_df[var] = pred_df[var] * std + mean

    # ──💾 Save result
    combined = pd.concat([df["close"].tail(config["input_width"]), denorm_df["close"]])

    return combined
