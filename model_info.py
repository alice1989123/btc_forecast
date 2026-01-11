# model_info.py
import os, re, yaml, json, logging
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts
from typing import Optional, Dict, Any, Tuple, List
import dotenv

logger = logging.getLogger(__name__)

def build_predict_config(model_name: str, coin: str, interval: str, *, version: int = 1) -> dict:
    if not interval:
        raise ValueError("interval is required")

    registry_name = f"{model_name}-{coin}-{interval}".lower()
    logger.info("Building predict config | model=%s coin=%s interval=%s version=%s registry_name=%s",
                model_name, coin, interval, version, registry_name)

    info = get_model_info(model_name=registry_name, version=version)
    params = info.get("params", {}) or {}
    metrics = info.get("metrics", {}) or {}

    input_width = int(params["input_width"])
    label_width = int(params["label_width"])
    win = int(params.get("windows_normalization_length", 30))

    variables_used = params.get("variables_used")
    if isinstance(variables_used, str):
        try:
            variables_used = json.loads(variables_used)
        except Exception:
            variables_used = ["close"]
    if not variables_used:
        variables_used = ["close"]

    # ✅ FIX: use your training metric keys
    mae_per_step = [
        metrics.get(f"price_mae_step_{i:02d}") for i in range(1, label_width + 1)
    ]
    mae_per_step = [float(x) for x in mae_per_step if x is not None]

    rmse_per_step = [
        metrics.get(f"price_rmse_step_{i:02d}") for i in range(1, label_width + 1)
    ]
    rmse_per_step = [float(x) for x in rmse_per_step if x is not None]

    cfg = {
        "interval": interval,
        "model_name": model_name,  # base family name (e.g. "GRU")
        "registry_name": registry_name,
        "version": int(version),

        "input_width": input_width,
        "label_width": label_width,
        "windows_normalization_length": win,
        "variables_used": variables_used,
        "input_shape": (input_width, len(variables_used)),

        # ✅ FIX: correct overall metrics
        "val_loss": metrics.get("val_loss_best") or metrics.get("val_loss"),
        "mae": metrics.get("price_mae"),
        "rmse": metrics.get("price_rmse"),

        # ✅ FIX: per-step arrays
        "mae_per_step": mae_per_step,
        "rmse_per_step": rmse_per_step,

        # (very useful for returns models if you logged them as params)
        "returns_mean": float(params["returns_mean"]) if "returns_mean" in params else None,
        "returns_std": float(params["returns_std"]) if "returns_std" in params else None,
    }

    return cfg


def get_model_info(
    model_name: str,
    *,
    version: Optional[int] = None,
    alias: Optional[str] = None,
    with_histories: bool = False
) -> Dict[str, Any]:
    if (version is None) == (alias is None):
        raise ValueError("Provide exactly one of: version or alias")

    dotenv.load_dotenv(override=True)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or os.getenv("TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError("Set MLFLOW_TRACKING_URI (or TRACKING_URI)")
    mlflow.set_tracking_uri(tracking_uri)

    logger.debug("MLflow tracking_uri=%s", tracking_uri)
    c = MlflowClient()

    if alias is not None:
        logger.info("Resolving model by alias | name=%s alias=%s", model_name, alias)
        mv = c.get_model_version_by_alias(model_name, alias)
        model_uri = f"models:/{model_name}@{alias}"
    else:
        logger.info("Resolving model by version | name=%s version=%s", model_name, version)
        mv = c.get_model_version(model_name, str(version))
        model_uri = f"models:/{model_name}/{version}"

    run_id = getattr(mv, "run_id", None) or ""
    logger.debug("Registry resolved | model_uri=%s run_id=%s", model_uri, run_id or "(empty)")

    if not run_id:
        logger.warning("ModelVersion.run_id empty. Falling back to reading MLmodel from registry | %s", model_uri)
        mlmodel_local = download_artifacts(f"{model_uri}/MLmodel")
        with open(mlmodel_local, "r", encoding="utf-8") as f:
            raw = f.read()
        data = yaml.safe_load(raw) or {}
        run_id = data.get("run_id") or _extract_hex_32(raw)
        logger.debug("Extracted run_id from MLmodel | run_id=%s", run_id)

    if not run_id:
        raise RuntimeError("Could not resolve run_id for this model version.")

    run = c.get_run(run_id)

    info: Dict[str, Any] = {
        "run_id": run_id,
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "tags": dict(run.data.tags),
        "artifact_uri": run.info.artifact_uri,
        "experiment_id": run.info.experiment_id,
    }

    if with_histories:
        mh: Dict[str, List[Tuple[int, int, float]]] = {}
        for key in info["metrics"].keys():
            records = c.get_metric_history(run_id, key)
            mh[key] = [(m.step, m.timestamp, m.value) for m in records]
        info["metric_history"] = mh
        logger.debug("Loaded metric histories | keys=%s", sorted(list(mh.keys())))

    return info


def _extract_hex_32(text: str) -> Optional[str]:
    m = re.search(r"\b[a-f0-9]{32}\b", text or "")
    return m.group(0) if m else None
