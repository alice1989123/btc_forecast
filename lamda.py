import json
from pymongo import MongoClient
from typing import List, Dict


def fetch_predictions_from_db() -> List[Dict[str, str]]:
    
    # Configure your MongoDB connection
    client = MongoClient(DB_KEY)
   
     
            """
    # Select the database and collection
    db = client['forecast_crypto_currency']
    general_collection = db['btc']

    # Fetch the data from the collection, sorting by the 'timestamp' field in descending order
    prediction_data = general_collection.find().sort('timestamp', -1).limit(1)

    if prediction_data:
        latest_prediction = list(prediction_data)[0]
        return latest_prediction['predictions']
    else:
      return []
    """
def lambda_handler(event, context):
    try:
        predictions = fetch_predictions_from_db()
    catch Exception as e :
        print(e)
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json'
        },
        'body':  "test" #json.dumps(predictions)
    }