import json
from pymongo import MongoClient
from typing import List, Dict
import traceback
import os
import datetime
class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super(JSONEncoder, self).default(obj)

    
def lambda_handler(event, context):
    predictions = "none"
    try: 
        DB_KEY =  os.environ.get('DB_KEY')

            # Select the database and collection
        def get_prediction(): 
                client = MongoClient(DB_KEY)

                db = client['forecast_crypto_currency']
                general_collection = db['btc']
        
                # Fetch the data from the collection, sorting by the 'timestamp' field in descending order
                prediction_data = general_collection.find().sort('timestamp', -1).limit(1)
        
                if prediction_data:
                    latest_prediction = list(prediction_data)[0]
                    return latest_prediction['predictions']
                else:
                    return []

    except Exception as e :
        print(e)
        predictions = traceback.format_exc()

    predictions = get_prediction()
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',  # Update this value to the appropriate domain in production
            'Access-Control-Allow-Methods': 'GET, POST',
            'Access-Control-Allow-Headers': 'Content-Type'
        },
        'body':  JSONEncoder().encode({'predictions': predictions})
    }