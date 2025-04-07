import torch
import pandas as pd
import datetime
import glob
import os
from datetime import timedelta as td

from btc_forecast.binance_data import get_binance_data
from btc_forecast.data_processing import normalize, data_parser, data_for_prediction_parser
from btc_forecast.models_torch.conv_dense import ConvDenseTorch
from config import config
import logger

# Setup logging
logging = logger.configure_logging(config.LOG_DIR, config.LOG_FILE_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(coin: str):
    logging.info(f"📊 Predicting {coin}...")

    # ──⏳ Time window
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(hours=1000)

    # Convert to timestamp
    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)

    # ──📥 Get & normalize data
    raw_data = get_binance_data(coin, start_ts, end_ts)
    df = data_parser(raw_data)
    df_norm = normalize(df, label_width=config.label_width, window=30)

    # Use the last input_width rows as model input
    recent_data = df_norm.tail(config.input_width)[config.variables_used]
    input_tensor = data_for_prediction_parser(recent_data, input_shape=config.input_shape)
    input_tensor = torch.tensor(input_tensor, dtype=torch.float32).to(device)

    # ──🔍 Find model
    model_path = f"models/ConvDenseTorch_{coin}_best.pt"
    if not os.path.exists(model_path):
        logging.error(f"❌ Model not found for {coin}")
        return None

    # ──📦 Load model
    model = ConvDenseTorch(
        input_width=config.input_width,
        label_width=config.label_width,
        num_inputs=len(config.variables_used),
        num_outputs=len(config.variables_used)
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    

    # ──📅 Create timestamps for predicted steps
    with torch.no_grad():
        preds = model(input_tensor).cpu().numpy()
        print("Preds shape BEFORE fix:", preds.shape)

    if preds.ndim == 3 and preds.shape[0] == 1:
        preds = preds[0]  # squeeze batch
    elif preds.ndim == 3:
        raise ValueError(f"❌ Unexpected shape: batch size > 1: {preds.shape}")
    elif preds.ndim == 1:
        preds = preds.reshape(-1, 1)

    # Transpose if necessary
    expected_shape = (config.label_width, len(config.variables_used))

    if preds.shape != expected_shape:
        
        raise ValueError(f"❌ Final shape mismatch: got {preds.shape}, expected {expected_shape}")
    dti_new = pd.date_range(end=df.index[-1], periods=config.label_width + 1, freq="h")[1:]
    
    pred_df = pd.DataFrame(preds, columns=["close"], index=dti_new)

   


    # ──📈 Denormalize "close" (example for 1 variable; adapt if more)
    denorm_df = pred_df.copy()
    for var in config.variables_used:
        mean = df[var].shift(config.label_width).rolling(window=30).mean().tail(config.label_width)
        std = df[var].shift(config.label_width).rolling(window=30).std().tail(config.label_width)

        mean.index = dti_new
        std.index = dti_new

        denorm_df[var] = pred_df[var] * std + mean

    # ──💾 Save result
    out_path = f"predictions/{coin}.csv"
    os.makedirs("predictions", exist_ok=True)

    # Combine with previous close for plotting
    combined = pd.concat([df["close"].tail(config.input_width), denorm_df["close"]])
    combined.to_csv(out_path)

    logging.info(f"✅ Saved prediction for {coin} to {out_path}")
    return combined

coins = config.coins
        
      