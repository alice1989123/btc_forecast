import os
import argparse
from mlflow.tracking import MlflowClient

def latest_version(model_name: str, stage: str | None = None) -> int:
    client = MlflowClient()
    if stage:
        # Returns latest version for that stage (Production/Staging/Archived/None)
        vs = client.get_latest_versions(model_name, stages=[stage])
        if not vs:
            raise SystemExit(f"No versions found for model={model_name!r} stage={stage!r}")
        return int(vs[0].version)

    # No stage: pick the max numeric version
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise SystemExit(f"No versions found for model={model_name!r}")
    return max(int(v.version) for v in versions)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Registered model name, e.g. gru-btcusdt-1h")
    ap.add_argument("--stage", default=None, help="Optional: Production | Staging | Archived | None")
    args = ap.parse_args()

    # Make sure these match your training env
    # export MLFLOW_TRACKING_URI=http://172.16.0.200
    v = latest_version(args.model, args.stage)
    print(v)
