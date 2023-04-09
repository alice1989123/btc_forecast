import tensorflow as tf
def scaled_dot_product_attention(query, key, value):
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    attention_weights = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attention_weights, value)
    return output, attention_weights

def positional_encoding(position, d_model):
    angle_rads = tf.range(position, dtype=tf.float32)[:, tf.newaxis] / tf.cast(2 * (tf.range(d_model // 2, dtype=tf.float32)), tf.float32) * 3.141592653589793
    angle_rads = tf.math.sin(angle_rads)
    sin_encoding = tf.math.sin(angle_rads)
    cos_encoding = tf.math.cos(angle_rads)
    pos_encoding = tf.stack([sin_encoding, cos_encoding], axis=-1)
    pos_encoding = tf.reshape(pos_encoding, (1, position, d_model))
    return pos_encoding

class SingleHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(SingleHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def call(self, v, k, q):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        scaled_attention, _ = scaled_dot_product_attention(q, k, v)
        output = self.dense(scaled_attention)
        return output
def create_transformer_block(input_layer, d_model, dff):
    attention = SingleHeadSelfAttention(d_model)
    x = attention(input_layer, input_layer, input_layer)
    x = tf.keras.layers.Dense(d_model)(x)
    out1 = tf.keras.layers.Add()([input_layer, x])
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1)
    
    ffn_output = tf.keras.layers.Dense(dff, activation='relu')(out1)
    ffn_output = tf.keras.layers.Dense(d_model)(ffn_output)
    out2 = tf.keras.layers.Add()([out1, ffn_output])
    out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out2)

    return out2


def create_transformer(input_shape, d_model, dff):
    inputs = tf.keras.Input(shape=input_shape)

    x = inputs
    x *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    x += positional_encoding(tf.shape(x)[1], d_model)

    x = create_transformer_block(x, d_model, dff)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(30, activation='relu')(x)
    x = tf.keras.layers.Dense(10, activation='relu')(x)
    x = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model
