#!/usr/bin/env python3
# train_model_returns.py
#
# Train returns-based models (GRU now; PatchTST later) and register to MLflow.
# ✅ MLflow Registry is the source of truth:
#    - Logs predict_config.json + train_config.json INSIDE the registered model artifact
#    - Ships your repo code with the model to avoid "Can't get attribute GRUStacked" unpickle errors
#
# Run:
#   export MLFLOW_TRACKING_URI=http://172.16.0.200
#   export INTERVAL=1h
#   export MODEL_KEY=GRU
#   export COINS=BTCUSDT
#   python3 train_model_returns.py
#
from __future__ import annotations
from btc_forecast.training.eval_price import eval_model_price_from_returns
from btc_forecast.training.mlflow_utils import (
    log_model_architecture,
    log_predict_metadata_json,
    get_registered_version_for_run,
)

import os
import json


import copy
import traceback
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import dotenv
import numpy as np
import torch

import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from config.config import coins as DEFAULT_COINS

from btc_forecast.training.registry import build_model_bundle
from btc_forecast.training.pipeline_returns import ReturnsTrainSpec, train_returns_model

def cb(epoch, tr, va, extra):
    print(f"Epoch {epoch} tr={tr:.4f} va={va:.4f} t={extra['epoch_time_sec']:.1f}s")

# ---------------------------
# Env / seeding
# ---------------------------
dotenv.load_dotenv()

SEED = int(os.getenv("SEED", "42"))
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRACKING_URI = os.getenv("TRACKING_URI") or os.getenv("MLFLOW_TRACKING_URI")
INTERVAL = (os.getenv("INTERVAL") or "").strip()
MODEL_KEY = (os.getenv("MODEL_KEY") or "GRU").strip()  # "GRU" or later "PatchTST"

if not TRACKING_URI:
    raise ValueError("TRACKING_URI env var not set (or MLFLOW_TRACKING_URI)")
if not INTERVAL:
    raise ValueError("INTERVAL env var not set")

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(f"{MODEL_KEY}_ALL_COINS_RETURNS_{INTERVAL}")

DEBUG_COIN = (os.getenv("DEBUG_COIN") or "BTCUSDT").strip()
COINS_ENV = (os.getenv("COINS") or "").strip()
COINS: List[str] = [c.strip() for c in COINS_ENV.split(",") if c.strip()] if COINS_ENV else list(DEFAULT_COINS)

# For pickled torch models: ensure the code that defines classes is available.
# We ship the *repo root* (folder containing btc_forecast/) with the model.
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent  # if train_model_returns.py sits in repo root
if not (REPO_ROOT / "btc_forecast").exists():
    # fallback: go one up if script is in scripts/ or similar
    if (REPO_ROOT.parent / "btc_forecast").exists():
        REPO_ROOT = REPO_ROOT.parent

# ---------------------------
# Helpers
# ---------------------------
def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _write_json(tmpdir: Path, filename: str, payload: Dict[str, Any]) -> str:
    p = tmpdir / filename
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(p)


def run_one(*, coin: str) -> None:
    coin = str(coin)
    interval = str(INTERVAL)

    # Build model/config bundle (currently from local configs; prediction will read config from Registry)
    bundle = build_model_bundle(
        model_key=MODEL_KEY,
        coin=coin,
        interval=interval,
        overrides={
            # override examples:
            # "data": {"input_width": 300},
            # "train": {"learning_rate": 0.001},
        },
    )

    cfg = bundle.config
    model = bundle.model.to(device)
    registry_name = bundle.registry_name            # e.g. "gru-btcusdt-1h"
    model_family = bundle.model_family              # e.g. "GRU" (used by predictor as model_name)

    data_cfg: Dict[str, Any] = cfg["data"]
    train_cfg: Dict[str, Any] = cfg["train"]

    input_width = int(data_cfg["input_width"])
    label_width = int(data_cfg["label_width"])
    num_features = int(data_cfg.get("num_features", 1))
    variables_used = list(data_cfg.get("variables_used", ["ret"]))

    spec = ReturnsTrainSpec(
        coin=coin,
        interval=interval,
        input_width=input_width,
        label_width=label_width,
        batch_size=int(train_cfg["batch_size"]),
        lr=float(train_cfg["learning_rate"]),
        max_epochs=int(train_cfg["num_epochs"]),
        patience=int(train_cfg["patience"]),
        grad_clip=float(train_cfg["grad_clip"]),
        weight_decay=float(train_cfg["weight_decay"]),
        huber_beta=float(train_cfg["huber_beta"]),
        target_clip=float(train_cfg["target_clip"]) if train_cfg.get("target_clip") is not None else None,
    )

    run_name = f"{MODEL_KEY}|{coin}|{interval}|in{input_width}-out{label_width}"

    with mlflow.start_run(run_name=run_name):
        try:
            # -----------------------
            # Params + tags
            # -----------------------
            mlflow.log_params(
                {
                    "model_key": MODEL_KEY,
                    "model_family": model_family,
                    "registry_name": registry_name,
                    "coin": coin,
                    "interval": interval,
                    "target": "log_return",
                    "input_width": input_width,
                    "label_width": label_width,
                    "num_features": num_features,
                    "variables_used": json.dumps(variables_used),
                    **{f"train.{k}": v for k, v in train_cfg.items()},
                    **{f"data.{k}": v for k, v in data_cfg.items()},
                    "seed": SEED,
                }
            )
            mlflow.set_tags({"status": "running", "Coin": coin, "Interval": interval, "ModelKey": MODEL_KEY})

            # Architecture artifact
            log_model_architecture(
                model,
                input_width=input_width,
                num_features=num_features,
                label_width=label_width,
                extra_config={"model_key": MODEL_KEY, "model_family": model_family},
            )

            # -----------------------
            # Train
            # -----------------------
            train_res = train_returns_model(
                model=model,
                spec=spec,              # ✅ use the spec you built from cfg
                device=device,          # ✅ reuse the device you computed
                debug_once=True,
                on_epoch_end=cb,
            )

            mlflow.log_metric("baseline_train_huber_zero", float(train_res.baseline_train))
            mlflow.log_metric("baseline_val_huber_zero", float(train_res.baseline_val))

            # Restore best
            model.load_state_dict(train_res.best_state_dict)
            model.eval()

            train_l = [float(x) for x in train_res.train_losses]
            val_l = [float(x) for x in train_res.val_losses]
            if train_l:
                mlflow.log_metric("train_loss_last", train_l[-1])
            if val_l:
                mlflow.log_metric("val_loss_last", val_l[-1])

            # Scaler artifacts
            returns_mean = float(train_res.artifacts.returns_mean)
            returns_std = float(train_res.artifacts.returns_std)
            mlflow.log_dict(
                {"returns_mean": returns_mean, "returns_std": returns_std},
                artifact_file="scaler/returns_scaler.json",
            )

            # -----------------------
            # Eval (price space)
            # -----------------------
            eval_out = eval_model_price_from_returns(
                model=model,
                device=device,
                close_raw=train_res.artifacts.close_raw,
                test_df_std=train_res.artifacts.test_df_std,
                returns_mean=returns_mean,
                returns_std=returns_std,
                input_width=input_width,
                label_width=label_width,
            )

            mae = _safe_float(eval_out.get("mae"))
            rmse = _safe_float(eval_out.get("rmse"))
            if mae is not None:
                mlflow.log_metric("price_mae", mae)
            if rmse is not None:
                mlflow.log_metric("price_rmse", rmse)

            # -----------------------
            # Build registry-truth config files (stored inside model artifact)
            # -----------------------
            tmpdir = Path(tempfile.mkdtemp(prefix="mlflow_cfg_"))

            predict_config: Dict[str, Any] = {
                "model_name": model_family,       # predictor expects family here (e.g., "GRU")
                "model_key": MODEL_KEY,           # useful for selecting architecture
                "coin": coin,
                "interval": interval,
                "registry_name": registry_name,
                "input_width": input_width,
                "label_width": label_width,
                "num_features": num_features,
                "variables_used": variables_used,
                "returns_mean": returns_mean,
                "returns_std": returns_std,
                "windows_normalization_length": int(data_cfg.get("windows_normalization_length", 30)),
                "target": "log_return",
            }

            predict_cfg_path = _write_json(tmpdir, "predict_config.json", predict_config)
            train_cfg_path = _write_json(tmpdir, "train_config.json", cfg)

            # -----------------------
            # Register model
            # -----------------------
            model_to_log = copy.deepcopy(model).to("cpu").eval()
            input_example = np.zeros((1, input_width, num_features), dtype=np.float32)
            with torch.no_grad():
                sample_out = model_to_log(torch.from_numpy(input_example))
            signature = infer_signature(input_example, sample_out.cpu().numpy())

            mlflow.pytorch.log_model(
                model_to_log,
                artifact_path="model",
                input_example=input_example,
                signature=signature,
                pip_requirements=None,
                # pip_requirements=[
                #     "mlflow",
                #     "torch==2.5.1",
                #     "numpy>=1.24,<3",
                #     "pandas>=2.0,<3",
                #     "scikit-learn>=1.3,<2",
                # ],
                registered_model_name=registry_name,
                # ✅ critical: ship code for torch pickle class resolution
                #code_paths=[str(REPO_ROOT)],
                code_paths=None ,
                # ✅ critical: ship configs inside the model artifact
                extra_files=[predict_cfg_path, train_cfg_path]
            )

            run_id = mlflow.active_run().info.run_id
            reg_version = get_registered_version_for_run(registry_name, run_id)

            # Tag model version with config file names (nice discoverability)
            try:
                client = MlflowClient()
                client.set_model_version_tag(registry_name, str(reg_version), "predict_config_file", "predict_config.json")
                client.set_model_version_tag(registry_name, str(reg_version), "train_config_file", "train_config.json")
                client.set_model_version_tag(registry_name, str(reg_version), "model_key", MODEL_KEY)
                client.set_model_version_tag(registry_name, str(reg_version), "model_family", model_family)
            except Exception:
                # Don't fail training if registry tags fail
                pass

            # Canonical metadata JSON (also stored in run artifacts)
            log_predict_metadata_json(
                artifact_file="predict/metadata.json",
                model_name=model_family,
                coin=coin,
                interval=interval,
                reg_name=registry_name,
                reg_version=reg_version,
                variables_used=variables_used,
                input_width=input_width,
                label_width=label_width,
                num_features=num_features,
                returns_mean=returns_mean,
                returns_std=returns_std,
                val_losses=val_l,
                train_losses=train_l,
                eval_out=eval_out,
            )

            mlflow.set_tag("status", "success")
            mlflow.set_tag("registered_version", str(reg_version))

        except KeyboardInterrupt:
            try:
                mlflow.set_tag("status", "killed")
            except Exception:
                pass
            raise

        except Exception as e:
            try:
                mlflow.set_tag("status", "failed")
                mlflow.set_tag("error_type", type(e).__name__)
                mlflow.set_tag("error_msg", str(e)[:200])
            except Exception:
                pass

            tb = traceback.format_exc()
            try:
                with open("error_trace.txt", "w", encoding="utf-8") as f:
                    f.write(tb)
                mlflow.log_artifact("error_trace.txt", artifact_path="errors")
                mlflow.log_dict({"error": str(e), "traceback": tb.splitlines()}, artifact_file="errors/error.json")
            except Exception:
                pass
            return


def main() -> None:
    print("MLFLOW_TRACKING_URI:", TRACKING_URI)
    print("INTERVAL:", INTERVAL)
    print("MODEL_KEY:", MODEL_KEY)
    print("COINS:", COINS)
    print("DEVICE:", device)
    print("REPO_ROOT shipped with model:", str(REPO_ROOT))

    for coin in COINS:
        run_one(coin=coin)


if __name__ == "__main__":
    main()
