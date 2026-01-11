# btc_forecast/training/registry.py
from __future__ import annotations

import json
import os
import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


# -------------------------
# utils
# -------------------------
def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _models_root() -> str:
    return os.path.join(_project_root(), "models")


def _resolve_model_dir(model_key: str) -> str:
    key = str(model_key).strip().lower()
    aliases = {
        "gru": "gru_stacked",
        "gru_stacked": "gru_stacked",
        "grustacked": "gru_stacked",

        "patchtst": "patchtst",
        "patch_tst": "patchtst",
        "transformer": "patchtst",
    }

    folder = aliases.get(key, key)
    folder_path = os.path.join(_models_root(), folder)

    if not os.path.isdir(folder_path):
        raise ValueError(
            f"Unknown model_key={model_key!r}. Expected folder at: {folder_path}\n"
            f"Available folders: {sorted([d for d in os.listdir(_models_root()) if os.path.isdir(os.path.join(_models_root(), d))])}"
        )
    return folder


def _get_model_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    m = cfg.get("model") or {}
    if isinstance(m, dict) and isinstance(m.get("params"), dict):
        return dict(m["params"])
    if isinstance(m, dict):
        return dict(m)
    return {}


def _filter_kwargs_for_callable(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return kwargs

    accepted = set(sig.parameters.keys())
    accepted.discard("self")

    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return kwargs

    return {k: v for k, v in kwargs.items() if k in accepted}


def _resolve_class_name(model_dir: str, cfg: Dict[str, Any]) -> str:
    class_name = cfg.get("class_name") or (cfg.get("model") or {}).get("class_name")
    if class_name:
        return str(class_name)

    defaults = {
        "gru_stacked": "GRUStacked",
        "patchtst": "PatchTST",   # ✅ match your real class
    }
    if model_dir in defaults:
        return defaults[model_dir]

    raise ValueError(
        f"Config missing class_name and no default known for model_dir={model_dir!r}. "
        f"Add class_name to config.json."
    )


# -------------------------
# API
# -------------------------
@dataclass
class ModelBundle:
    model: torch.nn.Module
    config: Dict[str, Any]
    registry_name: str
    model_family: str
    model_key: str
    model_dir: str


def build_model_bundle(
    *,
    model_key: str,
    coin: str,
    interval: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> ModelBundle:
    model_dir = _resolve_model_dir(model_key)

    cfg_path = os.path.join(_models_root(), model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing config.json for model {model_key!r} at: {cfg_path}")

    cfg = _load_json(cfg_path)
    cfg = _deep_update(cfg, overrides or {})

    model_family = str(cfg.get("model_family") or cfg.get("model_key") or cfg.get("model_name") or model_key).strip()

    data_cfg = cfg.get("data") or {}
    input_width = int(data_cfg.get("input_width", 250))
    label_width = int(data_cfg.get("label_width", 12))
    num_features = int(data_cfg.get("num_features", 1))

    model_params = _get_model_params(cfg)

    module_path = f"btc_forecast.models.{model_dir}.model"
    mod = importlib.import_module(module_path)

    class_name = _resolve_class_name(model_dir, cfg)
    if not hasattr(mod, class_name):
        raise AttributeError(
            f"Model class {class_name!r} not found in {module_path}. "
            f"Available: {[x for x in dir(mod) if x[0].isupper()]}"
        )

    ModelClass = getattr(mod, class_name)

    base_kwargs = {
        "input_width": input_width,
        "label_width": label_width,
        "num_features": num_features,
        **model_params,
    }

    safe_kwargs = _filter_kwargs_for_callable(ModelClass.__init__, base_kwargs)
    model = ModelClass(**safe_kwargs)

    registry_name = f"{model_family}-{coin}-{interval}".lower()

    return ModelBundle(
        model=model,
        config=cfg,
        registry_name=registry_name,
        model_family=model_family,
        model_key=model_key,
        model_dir=model_dir,
    )
