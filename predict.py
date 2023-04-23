
import pandas as pd
import pandas as pd
import datetime
from  window_generator  import WindowGenerator
import pandas as pd
import tensorflow as tf
from binance_data import get_binance_data
from data_processing import train_test   , compile_and_fit , normalize , data_parser  , data_for_prediction_parser
import pandas as pd
import models
import pickle
from datetime import timedelta as td
import glob
import tensorflow as tf
import config.config as config

import pandas as pd
import matplotlib.pyplot as plt

def predict():
    # Get the data from Binance and Parse for preditions

    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(hours=1000)

    # Convert the start and end time to milliseconds

    start_timestamp = int(start_time.timestamp() * 1000)
    end_timestamp = int(end_time.timestamp() * 1000)
    data_pred = get_binance_data('BTCUSDT',start_timestamp, end_timestamp)
    df_pred = data_parser(data_pred)
    df_pred_norm = normalize(df_pred,label_width=config.label_width,window=30)
    prediction_data = df_pred_norm.tail(config.input_width)
    data_for_prediction = data_for_prediction_parser(prediction_data , input_shape=config.input_shape)

    #loads the models

    model_files = glob.glob("models/*.h5")
    loaded_models={}
    for model_file in model_files:
        loaded_models[model_file.split(".h5")[0]] = tf.keras.models.load_model(model_file) 

    #predicts 

    first_key = next(iter(loaded_models))
    model = loaded_models[first_key]
    prediction = model.predict(data_for_prediction)[0]


    # create a new DatetimeIndex for the next 24 hours

    td = pd.Timedelta(hours=config.label_width)
    dti_new = df_pred["close"].tail(config.label_width).index + td
    normalized_prediction = pd.DataFrame(prediction, columns=["close"] , index=dti_new ) 
    #fits total data
    combined_df = pd.concat([df_pred_norm[config.variables_used].tail(48), normalized_prediction])

    # Denormalize the prediction

    mean = df_pred["close"].shift( config.label_width).rolling(30).mean().tail(config.label_width )
    mean.index = dti_new
    std = df_pred["close"].shift(config.label_width).rolling(30).std().tail(config.label_width )
    std.index =dti_new
    predictions_no_ma  = normalized_prediction["close"]  *std + mean
    
    # Concatenate the two DataFrames
    df1 = df_pred["close"].tail(config.input_width)
    df2 = predictions_no_ma .squeeze()  # Convert the DataFrame into a pandas Series
    combined_df_ = pd.concat([df1, df2])
    combined_df_.columns = ["close"]
    #print(combined_df_.index )
    #print(combined_df_.values) 



    return combined_df_
