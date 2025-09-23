# model_info.py
import os, re, yaml
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts
from typing import Optional, Dict, Any, Tuple, List
import dotenv

def get_model_info(
    model_name: str,
    *,
    version: Optional[int] = None,
    alias: Optional[str] = None,
    with_histories: bool = False
) -> Dict[str, Any]:
    """
    Return MLflow run info for a registered model version/alias.

    Example:
        get_model_info("gru", alias="prod")
        get_model_info("gru", version=2, with_histories=True)

    Returns:
        {
          "run_id": "...",
          "params": { ... },
          "metrics": { "val_loss": 0.123, "mae": 0.045, ... },
          "tags": { ... },
          "artifact_uri": "mlflow-artifacts:/.../artifacts",
          "experiment_id": "123",
          # only if with_histories=True
          "metric_history": { "val_loss": [(step, ts_ms, value), ...], ... }
        }
    """
    if (version is None) == (alias is None):
        raise ValueError("Provide exactly one of: version or alias")

    dotenv.load_dotenv(override=True)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or os.getenv("TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("Set MLFLOW_TRACKING_URI (or TRACKING_URI)")
    mlflow.set_tracking_uri(tracking_uri)

    c = MlflowClient()

    # Resolve model version and build the registry URI
    if alias is not None:
        mv = c.get_model_version_by_alias(model_name, alias)
        model_uri = f"models:/{model_name}@{alias}"
    else:
        mv = c.get_model_version(model_name, str(version))
        model_uri = f"models:/{model_name}/{version}"

    # Try direct run_id first (may be empty for copied versions)
    run_id = getattr(mv, "run_id", None) or ""

    if not run_id:
        # Fallback: read MLmodel from the registry to extract run_id
        # download_artifacts returns a local path to the file/folder
        mlmodel_local = download_artifacts(f"{model_uri}/MLmodel")
        with open(mlmodel_local, "r", encoding="utf-8") as f:
            raw = f.read()
        data = yaml.safe_load(raw) or {}
        run_id = data.get("run_id") or _extract_hex_32(raw)

    if not run_id:
        raise RuntimeError("Could not resolve run_id for this model version.")

    run = c.get_run(run_id)

    info: Dict[str, Any] = {
        "run_id": run_id,
        "params": dict(run.data.params),             # str -> str
        "metrics": dict(run.data.metrics),           # str -> float (latest values)
        "tags": dict(run.data.tags),
        "artifact_uri": run.info.artifact_uri,
        "experiment_id": run.info.experiment_id,
    }

    if with_histories:
        # For each metric key, pull full history (step, timestamp_ms, value)
        mh: Dict[str, List[Tuple[int, int, float]]] = {}
        # client.list_metrics(run_id) exists in newer MLflow; to be compatible,
        # derive keys from latest metrics dict:
        for key in info["metrics"].keys():
            records = c.get_metric_history(run_id, key)
            # Each record has .step, .timestamp, .value
            mh[key] = [(m.step, m.timestamp, m.value) for m in records]
        info["metric_history"] = mh

    return info


def _extract_hex_32(text: str) -> Optional[str]:
    m = re.search(r"\b[a-f0-9]{32}\b", text or "")
    return m.group(0) if m else None
