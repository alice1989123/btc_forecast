# config/model_configs.py

def get_model_config(model_name):
    if model_name == "conv_dense":
        return {
            "input_width": 200,
            "label_width": 12,
            "num_inputs": 1,
            "num_outputs": 1
        }
    elif model_name == "lstm":
        return {
            "input_width": 200,
            "label_width": 12,
            "hidden_size": 64,
            "num_layers": 2,
            "num_inputs": 1,
            "num_outputs": 1
        }
    else:
        raise ValueError(f"Model config for '{model_name}' not found.")
