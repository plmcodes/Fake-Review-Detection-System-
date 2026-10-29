"""Small shared helpers: logging, device choice, early stopping, checkpoints and the run logger.

The run logger (`log_run`) is what makes the project honest: every training/evaluation
appends one row to results/runs.csv, and every table in the report is generated from that
file by a script - never typed by hand.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import resolve_path

# ----------------------------------------------------------------------------- logging

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "frd") -> logging.Logger:
    """Return a logger that prints to stdout once (no duplicated handlers in notebooks)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# ----------------------------------------------------------------------------- device

def get_device(preference: str = "auto"):
    """'auto' picks CUDA when available (Colab/Kaggle T4), otherwise CPU (laptop)."""
    import torch

    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda" or (preference == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError("train.device=cuda requested but no GPU is available")
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------------------- early stopping

class EarlyStopping:
    """Stop training when the validation metric has not improved for `patience` epochs.

    mode="min" for losses (lower is better), mode="max" for scores such as F1 or AUC.
    `improved` tells the caller when to save the best checkpoint.
    """

    def __init__(self, patience: int = 5, mode: str = "min", min_delta: float = 1e-6):
        assert mode in {"min", "max"}
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = math.inf if mode == "min" else -math.inf
        self.best_epoch = -1
        self.bad_epochs = 0
        self.improved = False

    def step(self, value: float, epoch: int) -> bool:
        """Record this epoch's value. Returns True when training should stop."""
        if self.mode == "min":
            better = value < self.best - self.min_delta
        else:
            better = value > self.best + self.min_delta
        self.improved = bool(better)
        if better:
            self.best, self.best_epoch, self.bad_epochs = value, epoch, 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


# ----------------------------------------------------------------------------- json / timing

def save_json(obj: Any, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    """Let json.dump handle numpy numbers/arrays and Paths."""
    if hasattr(o, "item"):
        try:
            return o.item()
        except (ValueError, AttributeError):
            pass
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


class Timer:
    """`with Timer() as t: ...` then read `t.seconds`."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start


# ----------------------------------------------------------------------------- run logger

# Fixed column order for results/runs.csv. Metrics a model doesn't produce are left blank.
RUN_COLUMNS = [
    "timestamp", "run_id", "model", "family", "needs_labels", "train_dataset", "eval_dataset",
    "split", "tag", "config_hash", "seed", "sample",
    "precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc", "threshold",
    "tn", "fp", "fn", "tp", "val_f1", "val_roc_auc", "val_pr_auc",
    "n_train", "n_eval", "train_time_s", "checkpoint", "notes",
]


def log_run(row: dict, csv_path: Path | str | None = None) -> Path:
    """Append one result row to results/runs.csv (creating the file with a header if needed).

    Unknown keys are rejected loudly - that prevents silently losing a metric because of a typo.
    """
    path = resolve_path(csv_path or "results/runs.csv")
    unknown = set(row) - set(RUN_COLUMNS)
    if unknown:
        raise KeyError(f"log_run got unknown columns {sorted(unknown)}; add them to RUN_COLUMNS")
    full = {col: row.get(col, "") for col in RUN_COLUMNS}
    if not full["timestamp"]:
        full["timestamp"] = datetime.now().isoformat(timespec="seconds")
    for key, value in full.items():  # round floats so the CSV stays readable
        if isinstance(value, float):
            full[key] = round(value, 6)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(full)
    return path


def make_run_id(model: str, dataset: str, tag: str, seed: int, sample) -> str:
    """Human-readable id that also names the checkpoint folder, e.g. ae_dense__salminen__seed42."""
    parts = [model + (f"-{tag}" if tag else ""), dataset + (f"-s{sample}" if sample else ""), f"seed{seed}"]
    return "__".join(parts)
