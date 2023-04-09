import pandas as pd
import numpy as np
from datetime import timedelta as td
from data_processing import *
from binance_data import *


def predict(model , data_for_prediction , df_pred , label_width , imput_width , input_shape):
    td = pd.Timedelta(hours=6)
    prediction = model.predict(data_for_prediction)[0]

    # create a new DatetimeIndex for the next 24 hours
    dti_new = df_pred["close"].tail(label_width).index + td
    normalized_prediction = pd.DataFrame(prediction, columns=["close"] , index=dti_new ) 

    mean = df_pred["close"].shift(imput_width -label_width).rolling(30).mean().tail(label_width )
    mean.index = dti_new
    std = df_pred["close"].shift(imput_width -label_width).rolling(30).std().tail(label_width )
    std.index =dti_new
    predictions_no_ma  = normalized_prediction["close"]  *std + mean
    #print( predictions_no_ma)

    # create a new DataFrame with the same columns as 'df' but with NaN values for the 'open_time' column and the new DatetimeIndex
    prediction__ = pd.DataFrame({'close': predictions_no_ma.values }, index=dti_new, columns=df_pred.columns)

    return prediction__ 
