# mlflow_mv_source_loader.py
import os, re, json, yaml, tempfile, boto3, mlflow
from typing import Dict, List, Tuple, Optional
from botocore.config import Config
from mlflow.tracking import MlflowClient
from botocore.exceptions import ClientError
import dotenv
dotenv.load_dotenv()  # take environment variables from .env.
# ---------- S3 / MinIO client ----------
def _s3():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"),   # e.g. http://172.16.0.205:9000
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        config=Config(s3={"addressing_style": "path"}),
    )

def _parse_source_to_bucket_prefix(source: str) -> Tuple[str, str, str]:
    if source.startswith("s3://"):
        m = re.match(r"s3://([^/]+)/(.*)$", source)
        if not m:
            raise ValueError(f"Bad s3 URL: {source}")
        return m.group(1), m.group(2).rstrip("/"), "s3"

    if source.startswith("mlflow-artifacts:/"):
        raw = os.getenv("MLFLOW_ARTIFACTS_BUCKET", "mlflow")
        bucket = raw.strip().strip('\'"')           # <<< strip quotes & spaces
        prefix = source[len("mlflow-artifacts:/"):].lstrip("/")  # keep leading '7/'
        print(f"[loader] bucket={bucket!r} endpoint={os.getenv('MLFLOW_S3_ENDPOINT_URL')!r}")
        return bucket, prefix.rstrip("/"), "mlflow-artifacts"

    raw = os.getenv("MLFLOW_ARTIFACTS_BUCKET", "mlflow")
    bucket = raw.strip().strip('\'"')
    return bucket, source.strip("/"), "mlflow-artifacts"

# ---------- Simple S3 getters ----------
def _get_json_s3(bucket: str, key: str) -> dict:
    s3 = _s3()
    try:
        print(f"[loader] get_json_s3 s3://{bucket}/{key}")
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))
    except Exception:
        alt = "mlflow-artifacts" if bucket == "mlflow" else "mlflow"
        obj = s3.get_object(Bucket=alt, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))

def _get_text_s3(bucket: str, key: str) -> str:
    s3 = _s3()
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
    except Exception:
        alt = "mlflow-artifacts" if bucket == "mlflow" else "mlflow"
        obj = s3.get_object(Bucket=alt, Key=key)
    return obj["Body"].read().decode("utf-8")

# ---------- List / Download helpers ----------
def _list_keys(bucket: str, prefix: str) -> Tuple[str, List[str]]:
    s3 = _s3()

    def _do_list(bkt: str) -> List[str]:
        out: List[str] = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bkt, Prefix=prefix):
            for obj in page.get("Contents", []):
                out.append(obj["Key"])
        return out

    # try provided bucket
    try:
        keys = _do_list(bucket)
        if keys:
            return bucket, keys
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code == "NoSuchBucket":
            print(f"[loader] NoSuchBucket on {bucket!r} — creds likely don’t own/see this bucket")
        else:
            raise

    # try conventional alternate
    alt = "mlflow-artifacts" if bucket == "mlflow" else "mlflow"
    try:
        keys = _do_list(alt)
        if keys:
            return alt, keys
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") not in ("NoSuchBucket",):
            raise

    # last resort: probe visible buckets (diagnostic)
    try:
        visible = [b["Name"] for b in _s3().list_buckets().get("Buckets", [])]
        print(f"[loader] visible buckets for {os.getenv('AWS_ACCESS_KEY_ID')!r}: {visible}")
        for bkt in visible:
            try:
                keys = _do_list(bkt)
                if keys:
                    return bkt, keys
            except ClientError:
                continue
    except Exception:
        pass

    return bucket, []

def _download_prefix(bucket: str, prefix: str, local_dir: str) -> Tuple[str, List[str]]:
    """Download all objects under s3://bucket/prefix to local_dir, preserving structure."""
    eff_bucket, keys = _list_keys(bucket, prefix)
    if not keys:
        return eff_bucket, []

    s3 = _s3()
    base = prefix.rstrip("/")
    for key in keys:
        if key.endswith("/") or len(key) <= len(base):
            continue
        rel = key[len(base):].lstrip("/")
        dst = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        s3.download_file(eff_bucket, key, dst)
    return eff_bucket, keys

# ---------- Public APIs ----------
def read_json_from_model_version(model_name: str, version: int, relative_path: str = "model/config.json") -> dict:
    """
    Load a JSON artifact for a ModelVersion:
      1) Try models/<mv>/artifacts/<relative_path>
      2) If missing, read models/<mv>/artifacts/MLmodel, extract run_id,
         then fetch 7/<run_id>/artifacts/<relative_path>
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    c = MlflowClient()
    mv = c.get_model_version(name=model_name, version=version)
    bucket, models_prefix, _ = _parse_source_to_bucket_prefix(mv.source)

    # 1) registry path
    key_models_cfg = f"{models_prefix.rstrip('/')}/{relative_path.lstrip('/')}"
    try:
        return _get_json_s3(bucket, key_models_cfg)
    except Exception:
        pass

    # 2) run path (via MLmodel → run_id)
    key_mlmodel = f"{models_prefix.rstrip('/')}/MLmodel"
    mlmodel_text = _get_text_s3(bucket, key_mlmodel)
    data = yaml.safe_load(mlmodel_text) or {}
    run_id = data.get("run_id")
    if not run_id:
        m = re.search(r"\b[a-f0-9]{32}\b", mlmodel_text)
        if not m:
            raise RuntimeError(f"MLmodel at s3://{bucket}/{key_mlmodel} has no run_id; cannot locate {relative_path}")
        run_id = m.group(0)

    key_run_cfg = f"7/{run_id}/artifacts/{relative_path.lstrip('/')}"
    cfg = _get_json_s3(bucket, key_run_cfg)
    cfg.setdefault("model_name", model_name)
    cfg.setdefault("model_version", int(version))
    cfg.setdefault("run_id", run_id)
    return cfg


def download_artifacts_from_model_version(
    model_name: str,
    version: int,
    subdir: str = "model/",
    dest_dir: Optional[str] = None,
) -> Dict[str, object]:
    """
    Download ALL artifacts under `subdir` for a ModelVersion.
    Strategy:
      1) Download models/<mv>/artifacts/<subdir>
      2) If empty, read MLmodel to get run_id and download 7/<run_id>/artifacts/<subdir>
    Returns:
      {
        "local_dir": <path>,
        "bucket": <effective bucket>,
        "registry_prefix": <models-prefix>/<subdir>,
        "run_prefix": <run-prefix or None>,
        "run_id": <run_id or None>,
        "downloaded_keys": [ ... ]
      }
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    c = MlflowClient()
    mv = c.get_model_version(name=model_name, version=version)
    bucket, models_prefix, _ = _parse_source_to_bucket_prefix(mv.source)

    local_dir = dest_dir or tempfile.mkdtemp(prefix=f"{model_name}_v{version}_")
    subdir = subdir.lstrip("/")
    registry_path = f"{models_prefix.rstrip('/')}/{subdir}"

    # 1) registry copy
    eff_bucket, keys = _download_prefix(bucket, registry_path, local_dir)
    if keys:
        return {
            "local_dir": local_dir,
            "bucket": eff_bucket,
            "registry_prefix": registry_path,
            "run_prefix": None,
            "run_id": None,
            "downloaded_keys": keys,
        }

    # 2) run copy
    mlmodel_key = f"{models_prefix.rstrip('/')}/MLmodel"
    mlmodel_text = _get_text_s3(bucket, mlmodel_key)
    data = yaml.safe_load(mlmodel_text) or {}
    run_id = data.get("run_id")
    if not run_id:
        m = re.search(r"\b[a-f0-9]{32}\b", mlmodel_text)
        if not m:
            raise RuntimeError(f"No artifacts under s3://{bucket}/{registry_path} and MLmodel has no run_id")
        run_id = m.group(0)

    run_path = f"7/{run_id}/artifacts/{subdir}"
    eff_bucket, keys = _download_prefix(bucket, run_path, local_dir)
    if not keys:
        raise FileNotFoundError(
            f"No artifacts found under either:\n"
            f"  s3://{bucket}/{registry_path}\n"
            f"  s3://{bucket}/{run_path}\n"
        )

    return {
        "local_dir": local_dir,
        "bucket": eff_bucket,
        "registry_prefix": registry_path,
        "run_prefix": run_path,
        "run_id": run_id,
        "downloaded_keys": keys,
    }
