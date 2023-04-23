import tensorflow as tf
from transformer import create_transformer
def create_Dense(input_shape, label_width, units=64, dropout_rate=0.3):
    Dense = tf.keras.Sequential([
        tf.keras.layers.Dense(units, input_shape=input_shape),
        #tf.keras.layers.Dropout(dropout_rate),
        #tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(label_width)    ], name='Dense')
    return Dense


def create_LSTM(input_shape, label_width):
    LSTM =  tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=input_shape ),

        tf.keras.layers.Dense(label_width)
    ] , name='LSTM')

    return LSTM

def create_LSTM2(input_shape, label_width):
    LSTM2 =  tf.keras.Sequential([
        tf.keras.layers.LSTM(128, input_shape=input_shape ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(label_width)
    ], name='LSTM2')

    return LSTM2

def create_Conv1D(input_shape, label_width):
    
    Conv1D = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=256, kernel_size=3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(label_width)
    ] , name='Conv1D')
    return Conv1D  


