import os
import datetime
from datetime import timedelta as td
import pandas as pd
from btc_forecast.binance_data import get_binance_data

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data")
os.makedirs(DATA_DIR, exist_ok=True)

from btc_forecast.data_processing import data_parser  # make sure to import this!



start_time = "1 Jun 2010"
now = datetime.datetime.now() - td(hours=24)
end_time = now.strftime("%Y-%m-%d %H:%M:%S")

def get_data_path(coin: str, interval: str) -> str:
    return os.path.join(DATA_DIR, f"{coin}_{interval}.csv")

def download_and_save(coin , interval: str) -> pd.DataFrame:
    print(f"📥 Downloading {coin}...")
    raw_data = get_binance_data(coin, start_time, end_time , interval)
    df = data_parser(raw_data)  # 🔁 convert list to DataFrame
    df.to_csv(get_data_path(coin , interval), index=False)
    print(f"✅ Saved to {get_data_path(coin , interval= interval)}")
    return df

def load_or_download(coin: str, interval: str) -> pd.DataFrame:
    path = get_data_path(coin , interval)
    if os.path.exists(path):
        print(f"📁 Loading {coin} from {path}")
        return pd.read_csv(path)
    else:
        return download_and_save(coin , interval)