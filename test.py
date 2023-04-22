
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
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
  except RuntimeError as e:
    print(e)
# Loading the list from the file
with open('btc_data', 'rb') as f:
    loaded_list = pickle.load(f)

# Printing the loaded list
#print(loaded_list)

data = loaded_list
### Configuration of the prediction
#variables_used = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume','num_trades','taker_base_vol','taker_quote_vol' ]
#variables_used = df.columns
#label_width = 12
#input_width=label_width*4
variables_used = ['close']
import numpy as np
from scipy.optimize import minimize , Bounds


# Define the hyperparameters to optimize
hyperparameters = ["label_width" , "input_width"  ]
# Define the objective function to optimize
def objective(hyperparameters , args  ):
    data = args
    label_width = int(hyperparameters[0])
    input_width = int(hyperparameters[1])
    input_shape = (input_width, 1)

    print(f"label_width: {label_width}")
    print(f"input_width: {input_width}")
    print(f"input_shape: {input_shape}")
    model = models.create_Dense(input_shape=input_shape, label_width=label_width)
    print(data)
    training_df = data_parser(data)
    training_df_norm = normalize(training_df, label_width=label_width, window=30)
    train_df, val_df, test_df = train_test(training_df_norm[variables_used])
    wide_window = WindowGenerator(
        input_width=input_width, label_width=label_width, shift=0,
        label_columns=["close"], train_df=train_df, val_df=val_df, test_df=test_df)

    history = compile_and_fit(model, wide_window)

    print(f"Model output shape: {model.output_shape}")

    return model.evaluate(wide_window.test, verbose=0)[0] / label_width


    
# Define the integer bounds for the hyperparameters

# Define the integer bounds for the hyperparameters
lower_bounds = np.array([12, 1])[0:2]
upper_bounds = np.array([48, 12])[0:2]
bounds = Bounds(lower_bounds.astype(int), upper_bounds.astype(int))
extra_args = (data,)
initial_guess = np.array([12, 48]).astype(int)

# Perform the gradient search over the hyperparameters
result = minimize(objective, x0=initial_guess,
  bounds=bounds, method='L-BFGS-B' , args=extra_args)

# Get the optimized hyperparameters
optimized_hyperparameters = result.x