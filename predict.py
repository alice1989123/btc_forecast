
import datetime
import pandas as pd
import tensorflow as tf
from btc_forecast.binance_data import get_binance_data , get_stored_klines
from btc_forecast.data_processing import      normalize , data_parser  , data_for_prediction_parser
import pandas as pd
from datetime import timedelta as td
import glob
import tensorflow as tf
import config.config as config
import logger
import pandas as pd

# Configure the logger
logging = logger.configure_logging(config.LOG_DIR, config.LOG_FILE_NAME)

def predict(coin:str):
    # Get the data from Binance and Parse for preditions

    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(hours=1000)

    # Convert the start and end time to milliseconds

    start_timestamp = int(start_time.timestamp() * 1000)
    end_timestamp = int(end_time.timestamp() * 1000)
    #TODO: Use hitorical get_stored_klines(coin, start_time, end_time)
    data_pred = get_binance_data(coin,start_timestamp, end_timestamp)
    df_pred = data_parser(data_pred)
    df_pred_norm = normalize(df_pred,label_width=config.label_width,window=30)
    prediction_data = df_pred_norm.tail(config.input_width)
    logging.info("data parsed for prediction" , prediction_data)

    data_for_prediction = data_for_prediction_parser(prediction_data , input_shape=config.input_shape)
    print("data_for_prediction" , data_for_prediction)
    #loads the models
    model_files = glob.glob("models/*.h5")
    loaded_models={}
    logging.info("loading local model for " + coin + "...")
    try:
        matching_models = [
        model for model in model_files
        if coin in model.split("/")[-1].split(".")[0].split("_")[1]
            ]
        if len(matching_models) == 0:
            raise Exception("No matching models found")
        elif len(matching_models) > 1:
            raise Exception("More than one matching model found")
    except Exception as e:
    # logging.info(matching_models)
        logging.error(e)

    for model_file in matching_models:
        logging.info("loading model: " + model_file)
        loaded_models[model_file.split(".h5")[0]] = tf.keras.models.load_model(model_file) 
        #predicts 
        first_key = next(iter(loaded_models))
        model = loaded_models[first_key]
        logging.info("predicting..." + coin)
        prediction = model.predict(data_for_prediction)[0]
        # create a new DatetimeIndex for the next 24 hours
        #TODO: change this to config.variables_used ustead of close
        logging.info("creating new DatetimeIndex for the next 24 hours")
        td = pd.Timedelta(hours=config.label_width)
        dti_new = df_pred["close"].tail(config.label_width).index + td
        normalized_prediction = pd.DataFrame(prediction, columns=["close"] , index=dti_new ) 
        #fits total data
        #combined_df = pd.concat([df_pred_norm[config.variables_used].tail(48), normalized_prediction])
        logging.info("denormalizing the prediction")
        # Denormalize the prediction

        mean = df_pred["close"].shift( config.label_width).rolling(30).mean().tail(config.label_width )
        mean.index = dti_new
        std = df_pred["close"].shift(config.label_width).rolling(30).std().tail(config.label_width )
        std.index =dti_new
        predictions_no_ma  = normalized_prediction["close"]  *std + mean
        
        # Concatenate the two DataFrames
        logging.info("concatenating the two DataFrames past and future")
        df1 = df_pred["close"].tail(config.input_width)
        df2 = predictions_no_ma .squeeze()  # Convert the DataFrame into a pandas Series
        combined_df_ = pd.concat([df1, df2])
        combined_df_.columns = ["close"]
        combined_df_.to_csv("predictions/" + coin + ".csv") 
        return combined_df_
