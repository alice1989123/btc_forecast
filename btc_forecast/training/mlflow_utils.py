

from __future__ import annotations


import json
import inspect
from typing import Dict, Any, Optional, List

import mlflow

import numpy as np


from torchinfo import summary



def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None
    

def log_predict_metadata_json(
    *,
    artifact_file: str,
    model_name: str,
    coin: str,
    interval: str,
    reg_name: str,
    reg_version: Optional[int],
    variables_used: List[str],
    input_width: int,
    label_width: int,
    num_features: int,
    returns_mean: float,
    returns_std: float,
    val_losses: List[float],
    train_losses: List[float],
    eval_out: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
):
    # Best epoch by val loss
    best_idx = int(np.argmin(val_losses)) if val_losses else -1
    best_epoch = int(best_idx + 1) if best_idx >= 0 else None
    val_best = float(val_losses[best_idx]) if best_idx >= 0 else None
    train_at_best = float(train_losses[best_idx]) if (best_idx >= 0 and best_idx < len(train_losses)) else None

    mae = _safe_float(eval_out.get("mae"))
    rmse = _safe_float(eval_out.get("rmse"))

    mae_steps = eval_out.get("mae_steps")
    rmse_steps = eval_out.get("rmse_steps")
    mae_per_step = [float(x) for x in mae_steps] if mae_steps is not None else []
    rmse_per_step = [float(x) for x in rmse_steps] if rmse_steps is not None else []

    run = mlflow.active_run()
    run_id = run.info.run_id if run else None

    payload = {
        # identity
        "model_name": model_name,
        "coin": coin,
        "interval": interval,
        "registry_name": reg_name,
        "version": reg_version,
        "run_id": run_id,

        # window config
        "input_width": int(input_width),
        "label_width": int(label_width),
        "input_shape": [int(input_width), int(num_features)],
        "variables_used": list(variables_used),

        # training summary
        "best_epoch": best_epoch,
        "val_loss": val_best,
        "train_loss_at_best": train_at_best,

        # scalers
        "returns_mean": float(returns_mean),
        "returns_std": float(returns_std),

        # evaluation in PRICE space
        "mae": mae,
        "rmse": rmse,
        "mae_per_step": mae_per_step,
        "rmse_per_step": rmse_per_step,
    }

    if extra:
        payload.update(extra)

    mlflow.log_dict(payload, artifact_file=artifact_file)

    # convenient tags/metrics
    if val_best is not None:
        mlflow.log_metric("val_loss_best", float(val_best))
        mlflow.set_tag("val_loss_best", str(val_best))
    if mae is not None:
        mlflow.set_tag("price_mae", str(mae))
    if best_epoch is not None:
        mlflow.set_tag("best_epoch", str(best_epoch))
    if reg_version is not None:
        mlflow.set_tag("registered_version", str(reg_version))


# ---------------------------
# MLflow architecture logging
# ---------------------------
def log_model_architecture(model, input_width, num_features, label_width, extra_config=None):
    cfg = {
        "model_class": model.__class__.__name__,
        "input_width": int(input_width),
        "label_width": int(label_width),
        "num_features": int(num_features),
    }
    if extra_config:
        cfg.update(extra_config)
    mlflow.log_text(json.dumps(cfg, indent=2), artifact_file="model/config.json")

    arch_str = str(model)
    try:
        s = summary(
            model,
            input_size=(1, int(input_width), int(num_features)),
            col_names=("input_size", "output_size", "num_params"),
            depth=4,
            verbose=0,
        )
        arch_str = str(s)
    except Exception:
        pass
    mlflow.log_text(arch_str + "\n", artifact_file="model/architecture.txt")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mlflow.log_text(
        json.dumps({"total_params": int(total_params), "trainable_params": int(trainable_params)}, indent=2),
        artifact_file="model/parameter_counts.json",
    )

    state_shapes = {k: list(v.shape) for k, v in model.state_dict().items()}
    mlflow.log_text(json.dumps(state_shapes, indent=2), artifact_file="model/state_dict_shapes.json")

    try:
        src = inspect.getsource(model.__class__)
        mlflow.log_text(src, artifact_file=f"model/source_{model.__class__.__name__}.py")
    except Exception:
        pass

def get_registered_version_for_run(registered_model_name: str, run_id: str) -> Optional[int]:
    """
    Best-effort: find a registered model version created for this run_id.
    May return None if registry is slow/eventually consistent.
    """
    try:
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{registered_model_name}'")
        matched = []
        for mv in versions:
            mv_run_id = getattr(mv, "run_id", None) or (mv.get("run_id") if isinstance(mv, dict) else None)
            mv_ver = getattr(mv, "version", None) or (mv.get("version") if isinstance(mv, dict) else None)
            if mv_run_id == run_id and mv_ver is not None:
                matched.append(int(mv_ver))
        return max(matched) if matched else None
    except Exception:
        return None