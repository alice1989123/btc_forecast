import json
import config.config as config
import glob

coins = config.coins
# Read and load the JSON file
def read_metadata(coin: str, model_name: str):
    # Build path based on coin and model
    pattern = f"./models/*{model_name}*{coin}*.json"
    matches = glob.glob(pattern)
    print(pattern)
    if not matches:
        raise FileNotFoundError(f"No metadata file found for coin {coin} and model {model_name}")

    with open(matches[0], "r") as f:
        data = json.load(f)
    return data

