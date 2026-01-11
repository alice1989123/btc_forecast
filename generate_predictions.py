#!/usr/bin/env python3
"""
save_predictions_to_pg.py

Generate predictions (via btc_forecast.predict) and persist to Postgres.

"Kubernetes-safe" + "MLflow Registry is source of truth":
- Can run in a different pod than training.
- Loads predict config from the *registered model artifact* (preferred):
    models:/<registry_name>/<version>/predict_config.json
  (this is what we logged via mlflow.pytorch.log_model(extra_files=[...]) )
- Fallbacks to run artifacts (older style):
    runs:/<run_id>/predict/metadata.json

Requires:
  - Env vars in .keys.env:
      DBHOST, DBUSER, DBPASSWORD, DBNAME
  - MLflow env:
      MLFLOW_TRACKING_URI (or TRACKING_URI)
  - Network access + credentials to MLflow artifact store (S3/MinIO)

Run:
  python3 save_predictions_to_pg.py --interval 1h --symbol BTCUSDT --model_name GRU --version 0 -vv
"""

from __future__ import annotations

from datetime import datetime, UTC
import argparse
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

import dotenv
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

import mlflow
from mlflow.tracking import MlflowClient

from btc_forecast import predict
from btc_forecast import get_latest_mlflow_version
from config.config import coins

# ---------------------------
# Env
# ---------------------------
dotenv.load_dotenv(".keys.env")

DB_HOST = os.getenv("DBHOST")
DB_USER = os.getenv("DBUSER")
DB_PASSWORD = os.getenv("DBPASSWORD")
DB_NAME = os.getenv("DBNAME")

logger = logging.getLogger(__name__)

_STR_TO_LEVEL = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


# ---------------------------
# Logging helpers
# ---------------------------
def _coerce_level(level_str: Optional[str | int], default: int = logging.INFO) -> int:
    if level_str is None:
        return default
    if isinstance(level_str, int):
        return level_str
    return _STR_TO_LEVEL.get(str(level_str).upper(), default)


def setup_logging(cli_level: Optional[str], verbose: int, quiet: int) -> None:
    level = _coerce_level(os.getenv("LOG_LEVEL"), logging.INFO)

    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    if quiet > 0:
        level = logging.WARNING

    if cli_level:
        level = _coerce_level(cli_level, level)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )

    root = logging.getLogger()
    root.setLevel(level)
    for h in root.handlers:
        h.setLevel(level)

    logging.getLogger("urllib3").setLevel(max(level, logging.WARNING))
    logging.getLogger("psycopg2").setLevel(max(level, logging.WARNING))
    logging.getLogger("mlflow").setLevel(max(level, logging.WARNING))


# ---------------------------
# Core helpers
# ---------------------------
def _env_sanity() -> None:
    missing = [k for k in ("DBHOST", "DBUSER", "DBPASSWORD", "DBNAME") if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing env vars: {missing}")

    tracking = os.getenv("MLFLOW_TRACKING_URI") or os.getenv("TRACKING_URI")
    if not tracking:
        raise RuntimeError("Missing MLFLOW_TRACKING_URI (or TRACKING_URI) env var")


def _as_timestamptz(x: Any) -> datetime:
    """Convert common timestamp types to timezone-aware datetime (UTC)."""
    if isinstance(x, datetime):
        return x if x.tzinfo else x.replace(tzinfo=UTC)
    if isinstance(x, pd.Timestamp):
        if x.tzinfo is None:
            x = x.tz_localize("UTC")
        return x.to_pydatetime()
    dt = pd.to_datetime(x, utc=True, errors="raise")
    return dt.to_pydatetime()


def _load_dict(uri: str) -> Dict[str, Any]:
    return mlflow.artifacts.load_dict(uri)


def _try_load_dict(uri: str) -> Optional[Dict[str, Any]]:
    try:
        return _load_dict(uri)
    except Exception:
        return None


def load_predict_config_from_registry(*, registry_name: str, version: int) -> Dict[str, Any]:
    """
    Preferred source of truth:
      models:/<registry_name>/<version>/predict_config.json

    Fallback:
      runs:/<run_id>/predict/metadata.json
    """
    client = MlflowClient()

    # Resolve run_id for traceability + fallback
    mv = client.get_model_version(name=registry_name, version=str(version))
    run_id = mv.run_id

    # 1) Preferred: config packaged with the registered model
    #    (what we created via mlflow.pytorch.log_model(extra_files=[predict_config.json,...]))
    model_cfg_uri = f"models:/{registry_name}/{version}/predict_config.json"
    cfg = _try_load_dict(model_cfg_uri)
    if cfg:
        cfg["registry_name"] = registry_name
        cfg["version"] = int(version)
        cfg["mlflow_run_id"] = run_id
        cfg["config_source"] = "models:/.../predict_config.json"
        return cfg

    # 2) Fallback: run artifact metadata.json (older approach)
    meta_uri = f"runs:/{run_id}/predict/metadata.json"
    meta = _try_load_dict(meta_uri)
    if not meta:
        raise RuntimeError(
            "Could not load predict config from MLflow.\n"
            f"Tried:\n  - {model_cfg_uri}\n  - {meta_uri}\n\n"
            "Fix in training: log predict_config.json as extra_files in the registered model "
            "or ensure predict/metadata.json exists in the run."
        )

    meta["registry_name"] = registry_name
    meta["version"] = int(version)
    meta["mlflow_run_id"] = run_id
    meta["config_source"] = "runs:/.../predict/metadata.json"
    return meta


def generate_prediction_series(
    *,
    coin: str,
    model_name: str,
    version: int,
    config: Dict[str, Any],
) -> pd.Series:
    # IMPORTANT: your predict() likely uses registry_name+version to load the model.
    # We pass model_name/version for backward compatibility, but config is the truth.
    s = predict.predict(config, coin, model_name=model_name, version=version)

    if not isinstance(s, pd.Series):
        raise TypeError(f"predict.predict() must return pd.Series, got {type(s).__name__}")
    if len(s) == 0:
        raise RuntimeError("predict.predict() returned empty Series")

    idx0, idxN = s.index[0], s.index[-1]
    logger.info("predict_return_type=%s len=%s", type(s).__name__, len(s))
    logger.info(
        "index_first=%s tz=%s | index_last=%s tz=%s",
        idx0, getattr(idx0, "tzinfo", None),
        idxN, getattr(idxN, "tzinfo", None),
    )
    return s


def save_prediction_to_postgres(*, pred_series: pd.Series, metadata: Dict[str, Any], coin: str) -> str:
    interval = metadata.get("interval")
    if not interval:
        raise ValueError("metadata.interval is required (prevents mixing intervals).")

    model_name = metadata.get("model_name")
    if not model_name:
        raise ValueError("metadata.model_name is required.")

    model_version = metadata.get("version")  # int
    input_width = int(metadata.get("input_width", 0) or 0)
    label_width = int(metadata.get("label_width", 12) or 12)

    if label_width <= 0:
        raise ValueError(f"label_width must be > 0, got {label_width}")

    n = len(pred_series)
    if n < label_width:
        raise RuntimeError(f"Series too short: len={n} < label_width={label_width}")

    split_index = n - label_width  # forecast starts here

    prediction_id = uuid.uuid4()
    created_at = datetime.now(UTC)

    points: List[Tuple[uuid.UUID, int, datetime, float, bool]] = []
    for i, (ts, val) in enumerate(pred_series.items()):
        points.append(
            (
                prediction_id,
                int(i),
                _as_timestamptz(ts),
                float(val),
                bool(i < split_index),
            )
        )

    logger.info(
        "DB write | total=%s input_width=%s label_width=%s split_index=%s hist=%s forecast=%s",
        n,
        input_width,
        label_width,
        split_index,
        split_index,
        label_width,
    )

    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
    )

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO prediction_runs
                      (prediction_id, coin, interval, model_name, model_version,
                       input_width, label_width, split_index, created_at, metadata_json)
                    VALUES
                      (%s, %s, %s, %s, %s,
                       %s, %s, %s, %s, %s::jsonb)
                    """,
                    (
                        str(prediction_id),
                        coin,
                        interval,
                        model_name,
                        int(model_version) if model_version is not None else None,
                        input_width,
                        label_width,
                        split_index,
                        created_at,
                        json.dumps(metadata),
                    ),
                )

                execute_values(
                    cur,
                    """
                    INSERT INTO prediction_points
                      (prediction_id, point_index, point_time, value, is_historical)
                    VALUES %s
                    """,
                    [(str(pid), idx, t, v, hist) for (pid, idx, t, v, hist) in points],
                    page_size=2000,
                )

        logger.info("✅ Saved prediction_id=%s coin=%s interval=%s", prediction_id, coin, interval)
        return str(prediction_id)

    finally:
        conn.close()


def get_new_predictions(*, model_name: str, version: int, coin: str, interval: str) -> str:
    _env_sanity()

    registry_name = f"{model_name}-{coin}-{interval}".lower()
    logger.info(
        "Generating | coin=%s model=%s version=%s interval=%s registry=%s",
        coin,
        model_name,
        version,
        interval,
        registry_name,
    )

    # ✅ Load config from MLflow Registry (source of truth)
    config = load_predict_config_from_registry(registry_name=registry_name, version=int(version))

    # Defensive reconciliation: ensure identity matches invocation
    config["coin"] = coin
    config["interval"] = config.get("interval") or interval
    config["model_name"] = config.get("model_name") or model_name
    config["registry_name"] = registry_name
    config["version"] = int(version)

    # Sanity required keys
    for k in ("interval", "model_name", "input_width", "label_width"):
        if k not in config or config[k] is None:
            raise RuntimeError(f"MLflow config missing {k!r}. got keys={list(config.keys())}")

    # returns-based models need these (warn only: maybe transformer forecast won’t)
    if "returns_mean" not in config or "returns_std" not in config:
        logger.warning(
            "Config missing returns_mean/returns_std (OK for price-models; risky for returns-models). keys=%s",
            list(config.keys()),
        )

    logger.info("Config source: %s | run_id=%s", config.get("config_source"), config.get("mlflow_run_id"))
    logger.debug("Using config: %s", config)

    s = generate_prediction_series(
        coin=coin,
        model_name=model_name,  # keep for compatibility
        version=int(version),
        config=config,
    )

    prediction_id = save_prediction_to_postgres(pred_series=s, metadata=config, coin=coin)

    label_width = int(config.get("label_width", 12))
    split_index = len(s) - label_width
    logger.info(
        "Boundary | split_index=%s | hist_last=%s | pred_first=%s",
        split_index,
        s.index[split_index - 1] if split_index - 1 >= 0 else None,
        s.index[split_index] if split_index < len(s) else None,
    )

    logger.info("Done | prediction_id=%s | mlflow_run_id=%s", prediction_id, config.get("mlflow_run_id"))
    return prediction_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Run predictions and save to Postgres (MLflow Registry config).")
    parser.add_argument("--model_name", type=str, default="GRU", help="Model family name (e.g., GRU).")
    parser.add_argument("--version", type=int, default=0, help="Registry version. Use 0 for latest.")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Coin symbol (e.g., BTCUSDT).")
    parser.add_argument("--interval", type=str, required=True, help="Interval (e.g., '1h').")

    parser.add_argument(
        "--log-level",
        choices=[k.lower() for k in _STR_TO_LEVEL.keys()],
        help="Set log level (overrides -v/--quiet and LOG_LEVEL env).",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("-q", "--quiet", action="count", default=0)

    args = parser.parse_args()
    setup_logging(args.log_level, args.verbose, args.quiet)

    # MLflow tracking must be set in this pod too
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or os.getenv("TRACKING_URI")
    mlflow.set_tracking_uri(tracking_uri)

    if args.symbol == "ALL":
        for coin in coins:
            registry_name = f"{args.model_name}-{coin}-{args.interval}".lower()
            version = args.version
            if version == 0:
                version = get_latest_mlflow_version.latest_version(registry_name)
                logger.info("Resolved latest MLflow version=%s for registry_name=%s", version, registry_name)
            get_new_predictions(
                model_name=args.model_name,
                version=version,
                coin=coin,
                interval=args.interval,
            )
    else:

        registry_name = f"{args.model_name}-{args.symbol}-{args.interval}".lower()
        version = args.version
        if args.version == 0:
            version = get_latest_mlflow_version.latest_version(registry_name)
            logger.info("Resolved latest MLflow version=%s for registry_name=%s", args.version, registry_name)
        get_new_predictions(
            model_name=args.model_name,
            version=version,
            coin=args.symbol,
            interval=args.interval,
        )



if __name__ == "__main__":
    main()
