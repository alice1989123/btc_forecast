from btc_forecast import predict
import datetime
from typing import List, Dict
import model_info
import boto3
from datetime import datetime, UTC
import numpy as np
from typing import List, Dict
import pandas as pd
from decimal import Decimal
import argparse
import time
import psycopg2
import uuid
import json
import logging
import os
import dotenv

dotenv.load_dotenv(".keys.env")

DB_HOST = os.getenv("DBHOST")
DB_USER = os.getenv("DBUSER")
DB_PASSWORD = os.getenv("DBPASSWORD")
DB_NAME = os.getenv("DBNAME")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "mx-central-1")


logger = logging.getLogger(__name__)

# --- Logging helpers ---
_STR_TO_LEVEL = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}

def _coerce_level(level_str: str | int | None, default: int = logging.INFO) -> int:
    if level_str is None:
        return default
    if isinstance(level_str, int):
        return level_str
    return _STR_TO_LEVEL.get(level_str.upper(), default)

def setup_logging(cli_level: str | None, verbose: int, quiet: int) -> None:
    level = _coerce_level(os.getenv("LOG_LEVEL"), logging.INFO)

    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    if quiet > 0:
        level = logging.WARNING

    if cli_level:
        level = _coerce_level(cli_level, level)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

    # ✅ ensure everything is aligned to the chosen level
    root = logging.getLogger()
    root.setLevel(level)                                     # <—
    for h in root.handlers:
        h.setLevel(level)                                    # <—
    logging.getLogger(__name__).setLevel(level)              # <—

    # quiet noisy libs (optional)
    logging.getLogger("botocore").setLevel(max(level, logging.WARNING))
    logging.getLogger("boto3").setLevel(max(level, logging.WARNING))
    logging.getLogger("urllib3").setLevel(max(level, logging.WARNING))

    logging.getLogger(__name__).debug("Logging initialized (effective=%s)",
                                      logging.getLevelName(root.getEffectiveLevel()))




def generate_prediction(coin: str, model_name: str, version: int, config: dict) -> List[Dict[str, str]]:
    data = []
    prediction = predict.predict(config, coin, model_name=model_name, version=version)
    for i in range(len(prediction)):
        data.append({"date": prediction.index[i], "price": prediction.values[i]})
    return data


def convert_types(obj):
    if isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_types(v) for v in obj]
    elif isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, (np.floating, np.integer)):   # ✅ optional but recommended
        return obj.item()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    else:
        return obj



def save_prediction_to_dynamodb(predictions: List[Dict[str, str]], metadata, coin: str):

    cleaned_predictions = convert_types(predictions)
    cleaned_metadata = convert_types(metadata)
    
    # TTL: auto-expire in 12 hours (43200 seconds)
    ttl = int(time.time()) + 12 * 3600  # current epoch time + 12 hours

    dynamodb = boto3.resource('dynamodb', region_name=AWS_DEFAULT_REGION)
    table = dynamodb.Table('crypto_predictions_')

    table.put_item(Item={
        'coin': coin,
        'timestamp': datetime.now(UTC).isoformat(),
        'predictions': cleaned_predictions,
        'metadata': cleaned_metadata,
        'ttl': ttl  
    })
def save_prediction_to_postgres(predictions, metadata, coin):
    interval = metadata.get("interval")
    if not interval:
        raise ValueError("metadata.interval is required (do not default; prevents mixing intervals).")

    conn = psycopg2.connect(
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST
    )
    cursor = conn.cursor()

    pred_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()

    model_name = metadata.get("model_name")
    input_width = int(metadata.get("input_width", 0))

    cursor.execute("""
        INSERT INTO prediction_metadata (id, coin, model_name, interval, input_width, created_at, metadata_json)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (pred_id, coin, model_name, interval, input_width, now, json.dumps(metadata)))

    label_width = int(metadata.get("label_width", 12))
    for i, p in enumerate(predictions):
        date_str = p["date"]
        price_val = float(p["price"])
        is_predicted = (i >= len(predictions) - label_width)

        cursor.execute("""
            INSERT INTO predicted_prices (id, prediction_time, price, is_historical)
            VALUES (%s, %s, %s, %s)
        """, (pred_id, date_str, price_val, not is_predicted))

    conn.commit()
    cursor.close()
    conn.close()




def get_new_predictions(model_name: str, version: int = 1 , coin: str = "BTCUSDT", interval: str = ""):
    try:
        
        if interval == "":
            raise ValueError("Interval not specified in config or environment variable.")
        logger.info("Generating predictions | coin=%s model=%s interval=%s version=%s",
                    coin, model_name, interval, version)

        config = model_info.build_predict_config(model_name=model_name, coin=coin, interval=interval, version=version)
        logger.debug("Using config: %s", config)

        predictions = generate_prediction(coin, model_name=model_name, version=version, config=config)

        save_prediction_to_postgres(predictions, config, coin)
        save_prediction_to_dynamodb(predictions, config, coin)

    except Exception:
        logger.exception("Error generating predictions for %s", coin) 
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictions and upload to DynamoDB/Postgres.")
    parser.add_argument("--model_name", type=str, default="GRU", help="Registered model name (no version suffix).")
    parser.add_argument("--version", type=int, default=1, help="Model version to load from MLflow registry.")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Coin ID to generate predictions for.")
    parser.add_argument("--interval", type=str, default="", help="Interval (e.g., '4h'). Overrides config if set.")


    # --- Logging flags ---
    parser.add_argument("--log-level", choices=[k.lower() for k in _STR_TO_LEVEL.keys()],
                        help="Set log level (overrides -v/--quiet and LOG_LEVEL env).")
    parser.add_argument("-v", "--verbose", action="count", default=0,
                        help="-v for INFO, -vv for DEBUG (ignored if --log-level used).")
    parser.add_argument("-q", "--quiet", action="count", default=0,
                        help="Reduce output (WARNING+) (ignored if --log-level used).")
    args = parser.parse_args()

    setup_logging(args.log_level, args.verbose, args.quiet)
    get_new_predictions(model_name=args.model_name, version=args.version, coin=args.symbol, interval=args.interval)