#!/usr/bin/env python3
import os, sys, time, json, argparse, traceback
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts
import dotenv

def env_or_fallback():
    dotenv.load_dotenv(override=True)
    uri = os.getenv("MLFLOW_TRACKING_URI") or os.getenv("TRACKING_URI")
    if not uri:
        print("ERROR: Set MLFLOW_TRACKING_URI (or TRACKING_URI).", file=sys.stderr)
        sys.exit(1)
    mlflow.set_tracking_uri(uri)
    return uri

def wait_until_ready(client: MlflowClient, name: str, version: str, timeout=180, poll=2):
    t0 = time.time()
    while True:
        mv = client.get_model_version(name=name, version=version)
        if mv.status == "READY":
            return mv
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Timed out waiting for {name} v{version} -> READY (last status: {mv.status})")
        time.sleep(poll)

def _list_dir(c: MlflowClient, run_id: str, path: str):
    return c.list_artifacts(run_id, path or "")

def _find_mlmodel_parent(c: MlflowClient, run_id: str) -> str:
    """
    Returns the artifact-relative directory that contains 'MLmodel'.
    BFS through the run's artifacts so we don't assume 'model/'.
    """
    from collections import deque
    q = deque([""])
    while q:
        path = q.popleft()
        for item in _list_dir(c, run_id, path):
            if item.is_dir:
                q.append(item.path)
            else:
                # item.path is like 'model/MLmodel' or 'something/MLmodel'
                if item.path.endswith("MLmodel"):
                    # parent directory (strip trailing '/MLmodel')
                    parent = item.path.rsplit("/", 1)[0] if "/" in item.path else ""
                    return parent
    raise FileNotFoundError("Could not locate an 'MLmodel' file in this run's artifacts.")

def main():
    p = argparse.ArgumentParser(description="Register an MLflow ModelVersion from an existing run (robust path discovery).")
    p.add_argument("--run-id", required=True, help="Run ID that logged the model (contains an MLmodel file).")
    p.add_argument("--name", required=True, help="Registered model name (e.g., 'gru').")
    p.add_argument("--stage", default="", help="(Optional) Stage to set (e.g., 'Production'). Stages are deprecated; prefer aliases.")
    p.add_argument("--alias", default="", help="(Optional) Alias to set (e.g., 'prod').")
    p.add_argument("--desc", default="", help="(Optional) Model version description.")
    p.add_argument("--print-config", action="store_true", help="Print model/config.json if present.")
    args = p.parse_args()

    uri = env_or_fallback()
    c = MlflowClient()

    run = c.get_run(args.run_id)
    print(f"[info] Tracking URI: {uri}")
    print(f"[info] Run ID: {args.run_id}")
    print(f"[info] artifact_uri: {run.info.artifact_uri}")

    # Find where the MLmodel actually lives under the run
    subdir = _find_mlmodel_parent(c, args.run_id)  # e.g., "model" or "my_export" or ""
    model_src = f"runs:/{args.run_id}/{subdir}" if subdir else f"runs:/{args.run_id}"
    print(f"[step] Found MLmodel under: '{subdir or '(run root)'}'")
    print(f"[step] Registering: {model_src} → name='{args.name}'")

    mv = mlflow.register_model(model_src, args.name)
    print(f"[ok] Created ModelVersion: name='{mv.name}' version='{mv.version}' status='{mv.status}'")

    if args.desc:
        c.update_model_version(name=mv.name, version=mv.version, description=args.desc)

    print("[step] Waiting for ModelVersion to become READY …")
    mv = wait_until_ready(c, mv.name, mv.version)
    print(f"[ok] READY: name='{mv.name}' version='{mv.version}' source='{mv.source}' run_id='{mv.run_id}'")

    # Stages are deprecated; attempt only if user asked, and with a compatible signature
    if args.stage:
        print(f"[step] Setting stage: {args.stage} (stages are deprecated in newer MLflow)")
        try:
            # Old signature without archive_existing
            c.transition_model_version_stage(name=mv.name, version=mv.version, stage=args.stage)
            print(f"[ok] Stage set: {args.stage}")
        except TypeError:
            # Some older versions used archive_existing, some newer reject it
            try:
                c.transition_model_version_stage(name=mv.name, version=mv.version, stage=args.stage, archive_existing=False)
                print(f"[ok] Stage set (with archive_existing=False): {args.stage}")
            except Exception as e:
                print(f"[warn] Could not set stage ({e}); continuing with aliases only.")

    if args.alias:
        print(f"[step] Setting alias '{args.alias}' → version {mv.version}")
        c.set_registered_model_alias(name=mv.name, alias=args.alias, version=mv.version)
        print(f"[ok] Alias set: {args.alias} → v{mv.version}")

    print("[info] Run params:")
    if run.data.params:
        for k, v in run.data.params.items():
            print(f"  - {k}: {v}")
    else:
        print("  (no params logged in this run)")

    if args.print_config:
        try:
            cfg_rel = f"{subdir}/config.json" if subdir else "config.json"
            local_cfg = download_artifacts(f"runs:/{args.run_id}/{cfg_rel}")
            print("[info] model/config.json:", json.load(open(local_cfg)))
        except Exception as e:
            print(f"[warn] Could not read model/config.json: {e}")

    print(f"\n[done] Model registered:\n  name:    {mv.name}\n  version: {mv.version}\n  stage:   {args.stage or '(none)'}\n  alias:   {args.alias or '(none)'}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(2)
