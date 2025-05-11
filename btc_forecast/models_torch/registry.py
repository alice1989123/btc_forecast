from btc_forecast.models_torch.conv_dense import ConvDenseTorch
from btc_forecast.models_torch.lstm import LSTMModel
from btc_forecast.models_torch.gru_stacked import GRUStacked  # 👈 NEW

MODEL_REGISTRY = {
    "ConvDenseTorch": ConvDenseTorch,
    "LSTMModel": LSTMModel,
    "GRUStacked": GRUStacked, 
}

def get_model(model_name, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"❌ Model '{model_name}' not found in registry.")
    return MODEL_REGISTRY[model_name](**kwargs)