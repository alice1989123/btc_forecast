from flask import Flask, request, jsonify
import glob
import tensorflow as tf
import pandas as pd
import datetime
import random
from typing import List, Dict
import predict

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

from typing import List, Dict

@app.route('/predict', methods=['GET'])
def generate_prediction() -> List[Dict[str, str]]:
    data=[]
    prediction = predict.predict()
    for i in range(0, len(prediction)):
        data.append({'date': prediction.index[i], 'price': prediction.values[i]})
    return (data)
    
def generate_dummy_data() -> List[Dict[str, str]]:
    data = []
    start_date = datetime.date(2022, 1, 1)
    end_date = datetime.date(2022, 3, 1)
    time_diff = abs((end_date - start_date).days)

    for i in range(time_diff):
        current_date = start_date + datetime.timedelta(days=i)
        date_string = current_date.isoformat()
        price = random.randint(1, 1000)
        data.append({'date': date_string, 'price': price})

    return data



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3001)