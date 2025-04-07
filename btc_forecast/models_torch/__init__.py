from .conv_dense import ConvDenseTorch
from .lstm import LSTMForecastModel  # You'll create this

MODEL_REGISTRY = {
    "conv_dense": ConvDenseTorch,
    "lstm": LSTMForecastModel
}