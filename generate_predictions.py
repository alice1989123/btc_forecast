from btc_forecast import predict
import datetime
from typing import List, Dict
import os 
import config.config as config
import metadata
import boto3
from datetime import datetime
from typing import List, Dict
import pandas as pd
from decimal import Decimal
import argparse
import sys







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
    dynamodb = boto3.resource('dynamodb', region_name='eu-central-1')  # choose your region
    table = dynamodb.Table('crypto_predictions_')  # Create this table first (via AWS console or code)

    formatted_predictions = [
        {'date': prediction['date'], 'price': prediction['price']}
        for prediction in predictions
    ]

    #print( "Formatted predictions:", formatted_predictions)
    #print("Formatted metadata:", cleaned_metadata)
    

    # Save or update item
    table.put_item(Item={
    'coin': coin,
    'timestamp': datetime.utcnow().isoformat(),
    'predictions': cleaned_predictions,
    'metadata': cleaned_metadata
    })

def get_new_predictions(model_name: str = "ConvDenseTorch"):
    # List of coins from config
    coins = config.coins
    
    # Iterate through each coin
    for coin in coins:
        # Generate predictions for the coin
        predictions = generate_prediction(coin , model_name=model_name)
        
        # Read metadata for the coin
        metadata_ = metadata.read_metadata(coin)
        
        # Save the predictions to the database
        #save_prediction_to_db(predictions, metadata_, coin)
        save_prediction_to_dynamodb(predictions , metadata_ , coin)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run predictions and upload to DynamoDB.")
    parser.add_argument("--model_name", type=str, default="ConvDenseTorch", help="Model to use for prediction.")
    args = parser.parse_args()

    get_new_predictions(model_name=args.model_name)