# model_params.py
import os, re, yaml
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts
from typing import Optional, Dict
import dotenv

def get_params(model_name: str, *, version: Optional[int] = None, alias: Optional[str] = None) -> Dict[str, str]:
    """
    Return the MLflow run params for a registered model.
    Works whether the ModelVersion is linked to a run or is a registry copy.
    
    Usage:
        get_params("gru", alias="prod")
        get_params("gru", version=2)
    """
    if (version is None) == (alias is None):
        raise ValueError("Provide exactly one of: version or alias")

    # Load env + set tracking
    dotenv.load_dotenv(override=True)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or os.getenv("TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("Set MLFLOW_TRACKING_URI (or TRACKING_URI)")
    mlflow.set_tracking_uri(tracking_uri)

    c = MlflowClient()

    # Resolve model URI and try to grab run_id directly
    if alias is not None:
        mv = c.get_model_version_by_alias(model_name, alias)
        model_uri = f"models:/{model_name}@{alias}"
    else:
        mv = c.get_model_version(model_name, str(version))
        model_uri = f"models:/{model_name}/{version}"

    run_id = getattr(mv, "run_id", None) or ""
    if not run_id:
        # Fallback: read the registry MLmodel and extract run_id
        # (download_artifacts works with 'models:/...' URIs)
        mlmodel_local = download_artifacts(f"{model_uri}/MLmodel")
        with open(mlmodel_local, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        run_id = data.get("run_id") or _extract_hex_32(f.read())

    if not run_id:
        raise RuntimeError("Could not resolve run_id for this model version.")

    # Fetch params from the tracking server
    run = c.get_run(run_id)
    return dict(run.data.params)

def _extract_hex_32(text: str) -> Optional[str]:
    m = re.search(r"\b[a-f0-9]{32}\b", text or "")
    return m.group(0) if m else None
