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

from btc_forecast.binance_data import get_binance_data
from btc_forecast.data_processing import train_test, normalize, data_parser
from config import config
#import json

#from dataset import WindowedDataset  # you'll build this
#from model import ConvDenseTorch     # you'll port this
coins = config.coins
#print(coins)

def train(coin):
    training_df = load_or_download(coin)
    #print(training_df)
    print( "training_df size" , training_df.size)
    training_df_norm =normalize(training_df, label_width=config.label_width, window=30)
    #print(training_df_norm)
    train_df, val_df, test_df = train_test(training_df_norm[config.variables_used])

    #print(train_df , val_df , test_df)

    best_val_loss = float('inf')
    early_stop_counter = 0
    patience = 5  # ← adjust based on your dataset
    best_model_path = f"models/ConvDenseTorch_{coin}_best.pt"
    best_state_dict = None

    train_ds = WindowedDataset(
        df=train_df,
        input_width=config.input_width,
        label_width=config.label_width,
        shift=0,  # or use 1 for forecasting 1 step ahead
        variables_used=config.variables_used
    )

    val_ds = WindowedDataset(
        df=val_df,
        input_width=config.input_width,
        label_width=config.label_width,
        shift=0,
        variables_used=config.variables_used
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    x_batch, y_batch = next(iter(train_loader))





    print("X shape:", x_batch.shape)  # (batch_size, input_width, num_features)
    print("Y shape:", y_batch.shape)

    model = ConvDenseTorch(
        input_width=config.input_width,
        label_width=config.label_width,
        num_inputs=len(config.variables_used),
        num_outputs=len(config.variables_used)  # or 1 if predicting "close" only
    ).to("cuda" )#if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    MAX_EPOCHS = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    for epoch in range(MAX_EPOCHS):
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
            val_loss = val_loss / len(val_loader)

        # 🧠 Only save to memory, not disk
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = model.state_dict()
            early_stop_counter = 0
            print(f"✅ New best val loss at epoch {epoch+1}")
        else:
            early_stop_counter += 1
            print(f"⏳ Patience {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print("🛑 Early stopping triggered.")
                break

        print(f"📉 Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f}")

    # 💾 Save model ONCE after early stopping
    if best_state_dict:
        torch.save(best_state_dict, best_model_path)
        print(f"💾 Saved best model to {best_model_path}")
        metadata = {
                "symbol": coin,
                "model_name": "ConvDenseTorch",
                "val_loss": best_val_loss,
                "config_label_width": config.label_width,
                "config_input_width": config.input_width,
                "config_variables_used": config.variables_used
            }
        with open(best_model_path.replace(".pt", ".json"), "w") as f:
                json.dump(metadata, f)

        print(f"✅ Finished training for {coin}")

if __name__ == "__main__":
    for coin in config.coins:
        try:
            train(coin)
        except Exception as e:
            print(f"❌ Failed training {coin}: {e}")
