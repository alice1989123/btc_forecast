import predict
import datetime
from typing import List, Dict
from pymongo import MongoClient
import time
import os 
import config.config as config
import metadata
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
    client = MongoClient(connection_string)

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
        {'$set': {'predictions': formatted_predictions , "timestamp": datetime.datetime.utcnow() , 'metadata': metadata }},
        upsert=True
    )

def main():
    while True:
        coins = config.coins
        for coin in coins:
            predictions = generate_prediction(coin)
            metadata_ = metadata.read_metadata(coin)
            save_prediction_to_db(predictions ,metadata_ , coin)
        time.sleep(3600)  # Sleep for 1 hour (3600 seconds)

if __name__ == "__main__":
    main()
