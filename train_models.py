
import pandas as pd
import pandas as pd
import datetime
from  window_generator  import WindowGenerator
import pandas as pd
import tensorflow as tf
from binance_data import get_binance_data
from data_processing import train_test   ,  normalize , data_parser  , data_for_prediction_parser
import pandas as pd
import models
import pickle
from datetime import timedelta as td
from models import ConvDense
import json


config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
from config import config
import numpy as np

start_time = "1 Jun 2010"
# Get current date and time
now = datetime.datetime.now()-td(hours=24)
# Format date and time as a string
end_time = now.strftime("%Y-%m-%d %H:%M:%S")

coins = config.coins

patience=5

MAX_EPOCHS = 100

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')


 

def train_model (coin:str):

        data = get_binance_data (coin,  start_time, end_time )
        training_df = data_parser(data)
        training_df_norm = normalize(training_df, label_width=config.label_width, window=30)
        train_df, val_df, test_df = train_test(training_df_norm[config.variables_used])
        wide_window = WindowGenerator(
        input_width=config.input_width, label_width=config.label_width, shift=0,
        label_columns=config.variables_used, train_df=train_df, val_df=val_df, test_df=test_df)

        
        model= ConvDense(input_shape=config.input_shape, label_width=config.label_width)
        history = model.fit(wide_window.train, epochs=MAX_EPOCHS,
                      validation_data=wide_window.val,
                      callbacks=[early_stopping])
        metadata = { "symbol":coin, "model_name" : model.name , "history": json.dumps(history.history) , "config_label_width" : config.label_width , "config_input_width" : config.input_width , "config_input_shape" : config.input_shape , "config_variables_used" : config.variables_used }
        
        json_data = json.dumps(metadata)

        with open(f"models/{model.name}_{coin}.txt", 'wb') as f:
                f.write(json_data.encode('utf-8'))
        #save models
        model.save(f"models/{model.name}_{coin}.h5" ,overwrite=True)


for coin in coins:
        train_model(coin)