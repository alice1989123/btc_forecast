import predict
import datetime
from typing import List, Dict
import os 
import config.config as config
import metadata
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import boto3
from datetime import datetime
from typing import List, Dict
import pandas as pd
from decimal import Decimal




DB_KEY =  os.getenv("DB_KEY") 





def generate_prediction(coin : str) -> List[Dict[str, str]]:
    data = []
    prediction = predict.predict(coin )
    print(prediction)
    for i in range(0, len(prediction)):
        data.append({'date': prediction.index[i], 'price': prediction.values[i]})
    return data

def save_prediction_to_db(predictions: List[Dict[str, str]], metadata , coin : str):
    # Configure your MongoDB connection
    connection_string = DB_KEY
    print(connection_string)
    client = MongoClient("mongodb+srv://Alice2:pollo3.1416bill@cluster0.eshcn.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0", server_api=ServerApi('1'))

    # Select the database and collection
    db = client['forecast_crypto_currency']
    general_collection = db[coin]

    # Convert 'date' to a Python datetime object before saving
    formatted_predictions = [
        {'date': prediction['date'], 'price': prediction['price']}
        for prediction in predictions
    ]

    # Save the predictions to the collection under a single key
    general_collection.update_one(
        {'_id': 'prediction_data'},
        {'$set': {'predictions': formatted_predictions , "timestamp": datetime.utcnow() , 'metadata': metadata }},
        upsert=True
    )

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
    table = dynamodb.Table('crypto_predictions')  # Create this table first (via AWS console or code)

    formatted_predictions = [
        {'date': prediction['date'], 'price': prediction['price']}
        for prediction in predictions
    ]

    # Save or update item
    table.put_item(Item={
    'coin': coin,
    'timestamp': datetime.utcnow().isoformat(),
    'predictions': cleaned_predictions,
    'metadata': cleaned_metadata
    })

def get_new_predictions():
    # List of coins from config
    coins = config.coins
    
    # Iterate through each coin
    for coin in coins:
        # Generate predictions for the coin
        predictions = generate_prediction(coin)
        
        # Read metadata for the coin
        metadata_ = metadata.read_metadata(coin)
        
        # Save the predictions to the database
        #save_prediction_to_db(predictions, metadata_, coin)
        save_prediction_to_dynamodb(predictions , metadata_ , coin)
if __name__ == "__main__":
    get_new_predictions()

