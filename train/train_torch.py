import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime, timedelta as td
from btc_forecast.data_loader import load_or_download
from  window_generator  import WindowGenerator
from btc_forecast.windowed_dataset import WindowedDataset
from btc_forecast.models_torch.conv_dense import ConvDenseTorch
import torch
import json
from btc_forecast.models_torch.registry import get_model
from btc_forecast.binance_data import get_binance_data
from btc_forecast.data_processing import train_test, normalize, data_parser
from config import config
from config.models_config import get_model_config 
import config.models_config as models_config
from btc_forecast.models_torch.registry import get_model
import argparse

#import json

#from dataset import WindowedDataset  # you'll build this
#from model import ConvDenseTorch     # you'll port this
coins = config.coins
#print(coins)
def train(coin, model_name="ConvDenseTorch"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_config = get_model_config(model_name)
    # Update config values like label_width if needed

    model = get_model(model_name , **model_config).to(device)
    training_df = load_or_download(coin)
    print("training_df size", training_df.size)

    training_df_norm = normalize(training_df, label_width=models_config.label_width, window=30)
    train_df, val_df, test_df = train_test(training_df_norm[models_config.variables_used])

    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 5
    best_model_path = f"models/{model_name}_{coin}_best.pt"
    best_state_dict = None

    train_ds = WindowedDataset(
        df=train_df,
        input_width=models_config.input_width,
        label_width=models_config.label_width,
        shift=0,
        variables_used=models_config.variables_used
    )
    val_ds = WindowedDataset(
        df=val_df,
        input_width=models_config.input_width,
        label_width=models_config.label_width,
        shift=0,
        variables_used=models_config.variables_used
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    x_batch, y_batch = next(iter(train_loader))
    print("X shape:", x_batch.shape)
    print("Y shape:", y_batch.shape)



    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    for epoch in range(100):
        model.train()
        running_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_loss += loss_fn(preds, yb).item()
            val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
            early_stop_counter = 0
            print(f"\u2705 New best val loss at epoch {epoch+1}")
        else:
            early_stop_counter += 1
            print(f"\u23F3 Patience {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print("\U0001F6D1 Early stopping triggered.")
                break

        print(f"\U0001F4C9 Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

    if best_state_dict:
        torch.save(best_state_dict, best_model_path)
        print(f"\U0001F4BE Saved best model to {best_model_path}")

        metadata = {
            "symbol": coin,
            "model_name": model_name,
            "val_loss": best_val_loss,
            "config_label_width": models_config.label_width,
            "config_input_width": models_config.input_width,
            "config_variables_used": models_config.variables_used
        }
        with open(best_model_path.replace(".pt", ".json"), "w") as f:
            json.dump(metadata, f)

        print(f"\u2705 Finished training for {coin}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains the models for all the coins.")
    parser.add_argument("--model_name", type=str, default="ConvDenseTorch", help="Model to use for prediction.")
    args = parser.parse_args()
    for coin in config.coins:

        try:
            train(coin, model_name=args.model_name)
        except Exception as e:
            print(f"\u274C Failed training {coin}: {e}")
