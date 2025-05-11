variables_used = ['close' , 'open', 'high', 'low', 'volume',] # 'quote_asset_volume','num_trades','taker_base_vol','taker_quote_vol' ]
input_width= 200 #label_width*4
label_width = 12

input_shape = (input_width, len(variables_used))

def get_model_config(model_name):
    name = model_name.lower()
    if name == "convdensetorch":
        return {
            "input_width": 200,
            "label_width": 12,
            "num_inputs": len(variables_used),
            "num_outputs": len(variables_used)
        }
    elif name == "lstmmodel":
        return {
            "input_width": 200,
            "label_width": 12,
            "hidden_size": 128,        # ⬆️ Increased for more expressive power
            "num_layers": 3,           # ⬆️ More layers for capturing long-term dependencies
            "num_inputs": len(variables_used),
            "num_outputs": len(variables_used),
                         
        }
    elif name == "grustacked":
        return {
            "input_width": 200,
            "label_width": 12,
            "hidden_size": 64,
            "num_layers": 2,
            "num_features": len(variables_used)  # ✅ match constructor exactly
        }


    else:
        raise ValueError(f"Model config for '{model_name}' not found.")