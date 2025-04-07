
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def train_test(df):
    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    return train_df, val_df, test_df





#Normalize data
def normalize(df, label_width ,window=30):
  # normalize data
  df_normalized = df.copy()
  for value in df.columns:
    rolling_mean = df_normalized[value].shift(label_width).rolling(window=window).mean()
    rolling_std = df_normalized[value].shift(label_width).rolling(window=window).std()
    df_normalized[value] = (df_normalized[value] - rolling_mean) / rolling_std
  return df_normalized.dropna()


def  data_parser(data):
    df_pred = pd.DataFrame(data , columns = ['open_time','open', 'high', 'low', 'close', 'volume','close_time', 'quote_asset_volume','num_trades','taker_base_vol','taker_quote_vol', 'ignore'] )
    # Convert Unix time to datetime format
    df_pred.drop("ignore", axis=1, inplace=True)
    df_pred['open_time'] = pd.to_datetime(df_pred['open_time'], unit='ms')

    # Set the datetime column as the index
    df_pred.set_index('open_time', inplace=True)

    # Convert the rest of the columns to float
    df_pred = df_pred.astype(float)
    return df_pred
    # Print the first few rows of the dataframe
def data_for_prediction_parser(df, input_shape):
    # Ensure it's a 2D array first
    prediction_data = df.values.astype(np.float32)

    if prediction_data.ndim == 1:
        # If shape is (2000,), make it (2000, 1)
        prediction_data = prediction_data.reshape(-1, 1)

    if prediction_data.shape != input_shape:
        raise ValueError(f"❌ Input shape mismatch: got {prediction_data.shape}, expected {input_shape}")

    # Add batch dimension: (1, input_width, num_features)
    return prediction_data.reshape(1, *input_shape)



# reshape the 'new_data' to match the shape of the training data
    return df.values.reshape((-1,) + input_shape)