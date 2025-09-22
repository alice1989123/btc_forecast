# mlflow_mv_source_loader.py
import os, json, re, boto3, mlflow
from mlflow.tracking import MlflowClient

def _boto3_s3():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL"),  # ej: http://mlflow-minio.default.svc.cluster.local:9000
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        config=boto3.session.Config(signature_version="s3v4"),
    )

def _parse_source_to_bucket_prefix(source: str):
    """
    Devuelve (bucket, prefix, scheme) donde:
      - scheme ∈ {"s3", "mlflow-artifacts", "file"}
      - bucket/prefix solo aplican a s3/minio; para file se devuelve (None, path, "file")
    """
    if source.startswith("s3://"):
        m = re.match(r"s3://([^/]+)/(.*)$", source)
        if not m:
            raise ValueError(f"Bad s3 URL: {source}")
        return m.group(1), m.group(2).rstrip("/"), "s3"

    if source.startswith("mlflow-artifacts:/"):
        # Mapea al bucket MinIO (por defecto "mlflow"). Quita el "7/" si está presente.
        bucket = os.getenv("MLFLOW_ARTIFACTS_BUCKET", "mlflow")
        prefix = source[len("mlflow-artifacts:/"):].lstrip("/")
        if prefix.startswith("7/"):
            prefix = prefix[2:]
        return bucket, prefix.rstrip("/"), "mlflow-artifacts"

    if source.startswith("file:/"):
        # Devuelve ruta local como "prefix"
        return None, source[len("file:"):], "file"

    # Fallback: trata como prefix dentro del bucket por defecto
    return os.getenv("MLFLOW_ARTIFACTS_BUCKET", "mlflow"), source.strip("/"), "mlflow-artifacts"

def read_json_from_model_version(model_name: str, version: int, relative_path: str = "model/config.json") -> dict:
    """
    Lee un JSON dentro del artifact de un ModelVersion usando su 'source' sin necesitar run_id.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-tracking.default.svc.cluster.local:5000"))
    c = MlflowClient()
    mv = c.get_model_version(name=model_name, version=version)

    bucket, prefix, scheme = _parse_source_to_bucket_prefix(mv.source)

    if scheme == "file":
        full = os.path.join(prefix, relative_path)  # prefix es ruta local
        with open(full, "r") as f:
            return json.load(f)

    # s3 / mlflow-artifacts → usa boto3 contra MinIO/S3
    key = f"{prefix.rstrip('/')}/{relative_path.lstrip('/')}"
    s3 = _boto3_s3()
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read().decode("utf-8"))

def list_artifacts_in_model_version(model_name: str, version: int, subdir: str = "model/"):
    """Lista los archivos bajo un subdirectorio (útil para debug)."""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-tracking.default.svc.cluster.local:5000"))
    c = MlflowClient()
    mv = c.get_model_version(name=model_name, version=version)

    bucket, prefix, scheme = _parse_source_to_bucket_prefix(mv.source)
    if scheme == "file":
        path = os.path.join(prefix, subdir)
        return [os.path.join(dp, f) for dp, _, files in os.walk(path) for f in files]

    s3 = _boto3_s3()
    base = f"{prefix.rstrip('/')}/{subdir.lstrip('/')}"
    paginator = s3.get_paginator("list_objects_v2")
    items = []
    for page in paginator.paginate(Bucket=bucket, Prefix=base):
        for obj in page.get("Contents", []):
            items.append(obj["Key"])
    return items
