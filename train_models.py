from itertools import combinations, product
# --- MLflow setup ---
import os, mlflow, mlflow.pytorch
from mlflow.models import infer_signature
import copy
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from btc_forecast.data_loader import load_or_download
from torchinfo import summary
import numpy as np
from btc_forecast.data_processing import train_test, normalize
from btc_forecast.windowed_dataset import WindowedDataset
import dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error
import traceback

from config.config import coins

dotenv.load_dotenv()  # take environment variables from .env.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



TRACKING_URI = os.getenv("TRACKING_URI")


mlflow.set_tracking_uri(TRACKING_URI)

# 3) Select experiment
mlflow.set_experiment("GRU_ALL_COINS_V1")

import json, inspect

def log_model_architecture(model, input_width, num_features, label_width, extra_config=None):
    """
    Logs architecture details as MLflow artifacts.
    Call this *inside* the `with mlflow.start_run(...)` block, after the model is built.
    """
    # 1) Config needed to recreate the model
    cfg = {
        "model_class": model.__class__.__name__,
        "input_width": int(input_width),
        "label_width": int(label_width),
        "num_features": int(num_features),
    }
    if extra_config:
        cfg.update(extra_config)
    mlflow.log_text(json.dumps(cfg, indent=2), artifact_file="model/config.json")

    # 2) Readable architecture (fallback to str(model))
    arch_str = str(model)
    try:
        # Prefer torchinfo if available (pip install torchinfo)
        s = summary(
            model,
            input_size=(1, input_width, num_features),  # (batch, seq_len, features)
            col_names=("input_size", "output_size", "num_params"),
            depth=4,
            verbose=0,
        )
        arch_str = str(s)
    except Exception:
        pass
    mlflow.log_text(arch_str + "\n", artifact_file="model/architecture.txt")

    # 3) Param counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mlflow.log_text(
        json.dumps(
            {"total_params": int(total_params), "trainable_params": int(trainable_params)},
            indent=2,
        ),
        artifact_file="model/parameter_counts.json",
    )

    # 4) State dict shapes (names + shapes only)
    state_shapes = {k: list(v.shape) for k, v in model.state_dict().items()}
    mlflow.log_text(json.dumps(state_shapes, indent=2), artifact_file="model/state_dict_shapes.json")

    # 5) Try to capture the class source (works best if defined in a .py module)
    try:
        src = inspect.getsource(model.__class__)
        mlflow.log_text(src, artifact_file=f"model/source_{model.__class__.__name__}.py")
    except Exception:
        # If class is defined in a notebook cell, source retrieval may fail—this is ok.
        pass


models =[]

base_features = ['close']
variable_sets = [list(c) for i in range(1, len(base_features)+1) for c in combinations(base_features, i)]


# Hyperparameter grid
param_grid = {
    "input_width": [ 100 ],
    "label_width": [12],
    "batch_size": [32],
    "learning_rate": [0.001, ],
    "num_epochs": [50],  
    "windows_normalization_length": [30 ],
    "coin": coins,
}

all_combinations = list(product(
    variable_sets,
    param_grid["input_width"],
    param_grid["label_width"],
    param_grid["batch_size"],
    param_grid["learning_rate"],
    param_grid["num_epochs"],
    param_grid["windows_normalization_length"],
    param_grid["coin"],    
))

for (variables_used, input_width, label_width, batch_size,
     learning_rate, num_epochs, windows_normalization_length,coin) in all_combinations:
    
    if "close" not in variables_used:
        continue

    # ✅ Reset models per-combo (avoid growing/duplicating across runs)
    models = []

  
    # Load the model

    close_idx = variables_used.index("close")
    
    ## SImple Linear 
    model_config = {
    "input_width": input_width,
    "label_width": label_width,
    "num_features": len(variables_used),
    "conv_channels": 16,        # NEW
    "kernel_size": 3            # NEW
        }


    model_config = {
        "input_width": input_width,
        "label_width": label_width,
        #"hidden_size": 64,
        #"num_layers": 2,
        "num_features": len(variables_used),
        }
  
    #GRU
    model_config = {
        "input_width": input_width,
        "label_width": label_width,
        "hidden_size": 64,
        "num_layers": 2,
        "num_features": len(variables_used),
        }
    class GRUStacked(nn.Module):
        def __init__(self, input_width, label_width, num_features, hidden_size=64, num_layers=2):
            super(GRUStacked, self).__init__()
            self.gru = nn.GRU(
                input_size=num_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_size, label_width * num_features)
            self.label_width = label_width
            self.num_features = num_features

        def forward(self, x):
            out, _ = self.gru(x)
            out = out[:, -1, :]  # take the output from the last timestep
            out = self.fc(out)
            return out.view(-1, self.label_width, self.num_features)
    models.append(GRUStacked(**model_config))

   

    #TRAIN THE MODEL 

    def train_model(
        coin,
        model,
        input_width=200,
        label_width=12,
        lr=1e-3,
        max_epochs=num_epochs,
        patience=5,
        
    ):

    

        df = load_or_download(coin)
        #if 'volume' in df.columns:
        #    df["volume"] = np.log1p(df["volume"])
        #df_norm = normalize(df, label_width=label_width, window=30)

        # 1) Normalize causally (ok to do before split)
        df_z = normalize(df[variables_used], label_width=label_width, window=windows_normalization_length)

        # 2) Split
        train_df, val_df, test_df = train_test(df_z)

        # 3) Fit MinMax *only* on train, transform val/test
        scaler = MinMaxScaler(feature_range=(-1, 1))
        train_df[train_df.columns] = scaler.fit_transform(train_df)
        val_df[val_df.columns]     = scaler.transform(val_df)
        test_df[test_df.columns]    = scaler.transform(test_df)

        train_ds = WindowedDataset(train_df, input_width, label_width, 0, variables_used)


        val_ds = WindowedDataset(val_df, input_width, label_width, 0, variables_used)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        train_losses, val_losses = [], []
        best_val_loss = float("inf")
        best_weights = None
        early_stop_counter = 0

        



        for epoch in range(max_epochs):
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
            train_losses.append(train_loss)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    preds = model(xb)
                    val_loss += loss_fn(preds, yb).item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            if mlflow.active_run():
                mlflow.log_metric("train_loss", float(train_loss), step=epoch+1)
                mlflow.log_metric("val_loss", float(val_loss), step=epoch+1)

            print(f"📉 Epoch {epoch+1}: Train={train_loss:.4f} | Val={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = model.state_dict()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print("🛑 Early stopping")
                    break

        return model, best_weights, train_losses, val_losses , scaler
  

  

    for model in models:
        run_name = (
            f"{model.__class__.__name__}"
            f"|in{input_width}-out{label_width}"
            f"|bs{batch_size}|lr{learning_rate}"
            f"|win{windows_normalization_length}"
            f"|feat{len(variables_used)}"
            f"|{coin}"
        )

        with mlflow.start_run(run_name=run_name):
            try: 
                mlflow.log_params({
                    "model_class": model.__class__.__name__,
                    "num_features": len(variables_used),
                    "input_width": input_width,
                    "label_width": label_width,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "windows_normalization_length": windows_normalization_length,
                    "coin": coin,                     
                })
                # Per-feature boolean params
                for f in base_features:
                    mlflow.log_param(f"feature_{f}", str(f in variables_used))
                mlflow.set_tags({
                    "project": "crypto-forecast",
                })
                 # Log metadata as tags
                mlflow.set_tags({
                    "Model": model.__class__.__name__,
                    "Coin": coin,
                    "status": "success",   # or failed if exception
                })
                # Log the architecture early (pre-training is fine)
                extra_cfg = {}
                if hasattr(model, "gru"):              # GRUStacked
                    extra_cfg.update({"hidden_size": 64, "num_layers": 2})
                if hasattr(model, "conv"):             # ConvFC
                    extra_cfg.update({"conv_channels": 16, "kernel_size": 3})
                if hasattr(model, "transformer_encoder"):
                    extra_cfg.update({"hidden_size": 64, "num_layers": 2, "nhead": 4})

                log_model_architecture(
                    model,
                    input_width=input_width,
                    num_features=len(variables_used),
                    label_width=label_width,
                    extra_config=extra_cfg,
                )

                model.to(device)
                model, weights, train_l, val_l, scaler = train_model(
                    coin=coin,
                    model=model,
                    input_width=input_width,
                    label_width=label_width,
                    lr=learning_rate,
                    max_epochs=num_epochs,
                    patience=5
                )

                # --- NEW: ensure we evaluate the BEST checkpoint, not the last ---
                if weights is None:
                    weights = copy.deepcopy(model.state_dict())
                model.load_state_dict(weights)
                model.eval()

                # ---- Eval (unchanged logic, now runs with best weights) ----
                df = load_or_download(coin)
                _, _, validate_raw = train_test(df[variables_used])

                test_df_zscore = normalize(validate_raw, label_width, window=windows_normalization_length)
                test_df_scaled = scaler.transform(test_df_zscore)
                test_df_scaled = pd.DataFrame(test_df_scaled, columns=test_df_zscore.columns, index=test_df_zscore.index)

                test_ds = WindowedDataset(test_df_scaled, input_width, label_width, 0, variables_used)
                test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

                all_preds, all_targets = [], []

            
                with torch.no_grad():
                    for xb, yb in test_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        pred = model(xb)
                        all_preds.append(pred.cpu().numpy())
                        all_targets.append(yb.cpu().numpy())

                all_preds = np.concatenate(all_preds, axis=0)
                all_targets = np.concatenate(all_targets, axis=0)

                preds_minmax = all_preds[:, :, close_idx]
                targets_minmax = all_targets[:, :, close_idx]

                close_min = scaler.data_min_[close_idx]
                close_max = scaler.data_max_[close_idx]
                preds_zscore = 0.5 * (preds_minmax + 1) * (close_max - close_min) + close_min
                targets_zscore = 0.5 * (targets_minmax + 1) * (close_max - close_min) + close_min

                denorm_preds = np.zeros_like(preds_zscore)
                denorm_targets = np.zeros_like(targets_zscore)
                for i in range(preds_zscore.shape[0]):
                    for j in range(label_width):
                        idx = i + j + input_width
                        mean = validate_raw["close"].shift(label_width).rolling(window=windows_normalization_length).mean().iloc[idx]
                        std  = validate_raw["close"].shift(label_width).rolling(window=windows_normalization_length).std().iloc[idx]
                        denorm_preds[i, j]   = preds_zscore[i, j] * std + mean
                        denorm_targets[i, j] = targets_zscore[i, j] * std + mean

                
                mae  = mean_absolute_error(denorm_targets.flatten(), denorm_preds.flatten())
                rmse = np.sqrt(mean_squared_error(denorm_targets.flatten(), denorm_preds.flatten()))

                mlflow.log_metric("final_mae", float(mae))
                mlflow.log_metric("final_rmse", float(rmse))
                mlflow.log_metric("final_mae_per_step", float(mae / label_width))
                mlflow.log_metric("final_rmse_per_step", float(rmse / label_width))
                mlflow.log_metric("train_loss_last", float(train_l[-1]))
                mlflow.log_metric("val_loss_last", float(val_l[-1]))
                
                # Extra metadata as tags
                mlflow.set_tag("Coin", coin)


                # --- Log model (already using best weights from above) ---
                model_to_log = copy.deepcopy(model).to("cpu").eval()
                input_example = np.zeros((1, input_width, len(variables_used)), dtype=np.float32)
                with torch.no_grad():
                    sample_out = model_to_log(torch.from_numpy(input_example))
                signature = infer_signature(input_example, sample_out.cpu().numpy())
                pip_reqs = ["mlflow", "torch==2.5.1", "numpy>=1.24,<3", "pandas>=2.0,<3", "scikit-learn>=1.3,<2"]
                reg_name = f"gru-{coin.lower()}" 

                mlflow.pytorch.log_model(
                    model_to_log,
                    artifact_path="model",
                    input_example=input_example,
                    signature=signature,
                    pip_requirements=pip_reqs,
                    registered_model_name=reg_name,
                )
                
            except Exception as e:
                # --- Quick filterable tags ---
                mlflow.set_tag("status", "failed")
                mlflow.set_tag("error_type", type(e).__name__)
                mlflow.set_tag("error_msg", str(e)[:200])  # truncate long messages

                # --- Detailed artifacts ---
                tb_str = traceback.format_exc()

                # Save full traceback as plain text
                with open("error_trace.txt", "w") as f:
                    f.write(tb_str)
                mlflow.log_artifact("error_trace.txt", artifact_path="errors")

                # Save structured JSON (error + traceback lines)
                mlflow.log_dict(
                    {"error": str(e), "traceback": tb_str.splitlines()},
                    artifact_file="errors/error.json",
                )

                continue
    