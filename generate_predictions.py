from btc_forecast import predict
import datetime
from typing import List, Dict
import config.config as config
import metadata
import boto3
from datetime import datetime
from typing import List, Dict
import pandas as pd
from decimal import Decimal
import argparse
import time
import psycopg2
import uuid
import json

import os
import dotenv
dotenv.load_dotenv(".keys.env")

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")



def generate_prediction(coin : str , model_name :str) -> List[Dict[str, str]]:
    data = []
    prediction = predict.predict(coin , model_name=model_name )
    
    for i in range(0, len(prediction)):
        data.append({'date': prediction.index[i], 'price': prediction.values[i]})
    return data


def convert_types(obj):
    if isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_types(v) for v in obj]
    elif isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, pd.Timestamp):
        
        
        return obj.isoformat()
    else:
        return obj



def save_prediction_to_dynamodb(predictions: List[Dict[str, str]], metadata, coin: str):
    cleaned_predictions = convert_types(predictions)
    cleaned_metadata = convert_types(metadata)
    
    # TTL: auto-expire in 12 hours (43200 seconds)
    ttl = int(time.time()) + 12 * 3600  # current epoch time + 12 hours

    dynamodb = boto3.resource('dynamodb', region_name='eu-central-1')
    table = dynamodb.Table('crypto_predictions_')

    table.put_item(Item={
        'coin': coin,
        'timestamp': datetime.utcnow().isoformat(),
        'predictions': cleaned_predictions,
        'metadata': cleaned_metadata,
        'ttl': ttl  
    })
def save_prediction_to_postgres(predictions, metadata, coin):
    conn = psycopg2.connect(
        database='crypto_predictions',
        user='alice',
        password='pollo3.1416bill',
        host=DB_HOST
    )
    cursor = conn.cursor()

    pred_id = str(uuid.uuid4())
    now = datetime.utcnow()

    model_name = metadata.get('model_name')
    input_width = int(metadata.get('config_input_width', 0))
    label_width = int(metadata.get('config_label_width', 12))

    cursor.execute("""
        INSERT INTO prediction_metadata (id, coin, model_name, input_width, created_at, metadata_json)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (pred_id, coin, model_name, input_width, now, json.dumps(metadata)))

    # Save each prediction with is_predicted flag
    for i, p in enumerate(predictions):
        date_str = p['date']
        price_val = float(p['price'])
        is_predicted = (i >= len(predictions) - label_width)
        cursor.execute("""
            INSERT INTO predicted_prices (id, prediction_time, price, is_historical)
            VALUES (%s, %s, %s, %s)
        """, (pred_id, date_str, price_val, not is_predicted))  # `is_historical = not is_predicted`

    conn.commit()
    cursor.close()
    conn.close()


def get_new_predictions(model_name: str = "ConvDenseTorch"):
    # List of coins from config
    coins = config.coins
    
    # Iterate through each coin
    for coin in coins:
        # Generate predictions for the coin
        predictions = generate_prediction(coin , model_name=model_name)
        
        # Read metadata for the coin
        metadata_ = metadata.read_metadata(coin , model_name=model_name)
        metadata_["model_name"] = model_name
        
        # Save the predictions to the database
        #save_prediction_to_db(predictions, metadata_, coin)
        save_prediction_to_postgres(predictions , metadata_ , coin)
        save_prediction_to_dynamodb(predictions , metadata_ , coin)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictions and upload to DynamoDB.")
    parser.add_argument("--model_name", type=str, default="ConvDenseTorch", help="Model to use for prediction.")
    args = parser.parse_args()

    get_new_predictions(model_name=args.model_name)