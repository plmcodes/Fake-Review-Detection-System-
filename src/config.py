"""Configuration loading and seeding.

Every script calls `load_config(...)` once and then `set_seed(cfg["seed"])`, so that
all randomness (Python, numpy, PyTorch on CPU and GPU) is reproducible.

Config layering (later layers win):
    1. configs/default.yaml                     shared defaults
    2. configs/models/<model>.yaml              model-specific settings
    3. configs/models/tuned/<model>_<data>.yaml  Optuna's best params (only with --tuned)
    4. --set key.sub=value                      command-line overrides
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

# Repository root (the folder that contains src/). All relative paths are resolved from here,
# so scripts behave the same whether you run them from the repo root or from notebooks/.
ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base` (nested dicts are merged, not replaced)."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def parse_override(text: str) -> dict:
    """Turn 'train.lr=0.0005' into {'train': {'lr': 0.0005}}.

    The value is parsed with YAML so numbers, booleans, lists and null work naturally.
    """
    if "=" not in text:
        raise ValueError(f"Override must look like key.sub=value, got: {text!r}")
    key, raw_value = text.split("=", 1)
    value = yaml.safe_load(raw_value)
    nested: dict = {}
    cursor = nested
    parts = key.strip().split(".")
    for part in parts[:-1]:
        cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value
    return nested


def read_yaml(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(
    model: str | None = None,
    overrides: Iterable[str] | None = None,
    extra_files: Iterable[Path | str] | None = None,
) -> dict:
    """Load default.yaml, then the model's YAML, then extra files, then CLI overrides."""
    cfg = read_yaml(CONFIG_DIR / "default.yaml")
    if model:
        model_file = CONFIG_DIR / "models" / f"{model}.yaml"
        if not model_file.exists():
            available = sorted(p.stem for p in (CONFIG_DIR / "models").glob("*.yaml"))
            raise FileNotFoundError(f"No config for model '{model}'. Available: {available}")
        cfg = deep_update(cfg, read_yaml(model_file))
    for extra in extra_files or []:
        cfg = deep_update(cfg, read_yaml(extra))
    for item in overrides or []:
        cfg = deep_update(cfg, parse_override(item))
    return cfg


def resolve_path(path: str | Path) -> Path:
    """Make a config path absolute (relative paths are relative to the repo root)."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def config_hash(cfg: dict, ignore: Iterable[str] = ("seed", "seeds", "paths")) -> str:
    """Short, stable fingerprint of a config.

    Runs with the same hash used the same settings, which makes `runs.csv` easy to group.
    The seed is excluded on purpose so the 3 repeated seeds of one config share a hash.
    """
    clean = {k: v for k, v in cfg.items() if k not in set(ignore)}
    blob = json.dumps(clean, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fix every source of randomness we use: Python, numpy and PyTorch (CPU + CUDA).

    `deterministic=True` also asks cuDNN to pick deterministic kernels. That can make GPU
    training slightly slower but means the same seed gives (near-)identical metrics.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:  # torch is optional for e.g. the classical baselines
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
