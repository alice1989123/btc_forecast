from flask import Flask, request, jsonify
from pymongo import MongoClient
from typing import List, Dict
import datetime
import random
from dotenv import load_dotenv
import os 

from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app , origins=["*"])
DB_KEY =  os.getenv("DB_KEY") 


def fetch_predictions_from_db() -> List[Dict[str, str]]:
    # Configure your MongoDB connection
    client = MongoClient(DB_KEY)

    # Select the database and collection
    db = client['forecast_crypto_currency']
    general_collection = db['btc']

    # Fetch the data from the collection, sorting by the 'timestamp' field in descending order
    prediction_data = general_collection.find().sort('timestamp', -1).limit(1)

    if prediction_data:
        print(prediction_data)
        return list(prediction_data)[0]["data"]
        #return prediction_data['predictions']
    else:
        return []


@app.route('/predict', methods=['GET'])

def generate_prediction() -> List[Dict[str, str]]:
    predictions = fetch_predictions_from_db()
    return jsonify(predictions)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3001)