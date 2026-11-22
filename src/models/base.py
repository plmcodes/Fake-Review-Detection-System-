"""Common interface for every model + shared PyTorch training helpers.

Every model is wrapped in a *Runner* with the same four methods:

    fit(data)            train on data.idx('train') (and early-stop on 'val')
    score(data, idx)     return one number per review - HIGHER = MORE LIKELY FAKE
    save(folder)         write weights + fitted preprocessing (vectoriser, vocab, scaler)
    load(folder, cfg)    rebuild the runner from a checkpoint folder

Because of this, `src/train.py` and `src/evaluate.py` treat all 13 models the same way.

Thresholds: supervised models output a probability, so the threshold is 0.5. Anomaly
detectors output a reconstruction error with no natural cut-off, so `tune_threshold=True`
makes train.py pick the F1-maximising threshold on the VALIDATION split and freeze it for test.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from ..utils import EarlyStopping, get_device, get_logger, load_json, save_json

log = get_logger(__name__)


class Runner:
    family: str = ""              # classical | deep | transformer | autoencoder | hybrid
    needs_labels: bool = True     # does TRAINING need fake/genuine labels?
    tune_threshold: bool = False  # True for anomaly scores (choose threshold on val)
    default_threshold: float = 0.5

    def __init__(self, cfg: dict, seed: int):
        self.cfg = cfg
        self.mcfg = cfg["model"]
        self.seed = seed
        self.history: list[dict] = []

    # --- to implement -------------------------------------------------------------
    def fit(self, data) -> None:
        raise NotImplementedError

    def score(self, data, idx: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, folder: Path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, folder: Path, cfg: dict) -> "Runner":
        raise NotImplementedError

    # --- shared helpers -------------------------------------------------------------
    def save_history(self, folder: Path) -> None:
        save_json(self.history, Path(folder) / "history.json")

    def load_history(self, folder: Path) -> None:
        path = Path(folder) / "history.json"
        self.history = load_json(path) if path.exists() else []


# ============================================================================ torch helpers

def make_bce_loss(y_train: np.ndarray, train_cfg: dict):
    """Binary loss with the configured class-imbalance handling.

    class_weight: weight positives by n_neg / n_pos (no effect on the balanced Salminen data,
                  important on imbalanced Yelp).
    focal:        focal loss (Lin et al. 2017) - down-weights easy examples.
    We never oversample val/test - imbalance is handled only in the training loss.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    n_pos = max(int(y_train.sum()), 1)
    n_neg = max(int(len(y_train) - y_train.sum()), 1)
    mode = train_cfg.get("imbalance", "class_weight")
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32)

    if mode == "none":
        return nn.BCEWithLogitsLoss()
    if mode == "class_weight":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if mode == "focal":
        gamma = float(train_cfg.get("focal_gamma", 2.0))
        alpha = n_neg / (n_pos + n_neg)  # weight of the positive (fake) class

        def focal(logits, target):
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            p_t = torch.exp(-bce)
            a_t = alpha * target + (1 - alpha) * (1 - target)
            return (a_t * (1 - p_t) ** gamma * bce).mean()

        return focal
    raise ValueError(f"train.imbalance must be none|class_weight|focal, got {mode!r}")


def iterate_minibatches(n: int, batch_size: int, shuffle: bool, rng: np.random.Generator):
    order = rng.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        yield order[start : start + batch_size]


def fit_torch(model, train_step, val_loss_fn, n_train: int, train_cfg: dict, seed: int,
              history: list, label: str = "model", mode: str = "min"):
    """Generic epoch loop with early stopping on a validation value.

    train_step(batch_positions) -> float loss   (does forward, backward, optimizer step)
    val_loss_fn() -> float                       (validation value to early-stop on)
    The best weights (lowest val loss) are restored at the end.
    """
    import torch

    rng = np.random.default_rng(seed)
    stopper = EarlyStopping(patience=int(train_cfg["patience"]), mode=mode)
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(int(train_cfg["epochs"])):
        model.train()
        losses = [train_step(b) for b in iterate_minibatches(n_train, int(train_cfg["batch_size"]), True, rng)]
        model.eval()
        with torch.no_grad():
            val = float(val_loss_fn())
        stop = stopper.step(val, epoch)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val": val})
        if stopper.improved:
            best_state = copy.deepcopy(model.state_dict())
        log.info("%s epoch %02d  train_loss=%.5f  val=%.5f%s", label, epoch, np.mean(losses), val,
                 "  *best*" if stopper.improved else "")
        if stop:
            log.info("%s early stop at epoch %d (best epoch %d)", label, epoch, stopper.best_epoch)
            break
    model.load_state_dict(best_state)
    return model


def batched_predict(fn, n: int, batch_size: int = 512) -> np.ndarray:
    """Run fn(positions) -> np.ndarray over n items in batches (no grad) and concatenate."""
    import torch

    outs = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            outs.append(fn(np.arange(start, min(start + batch_size, n))))
    return np.concatenate(outs) if outs else np.zeros(0)


def device_from_cfg(cfg: dict):
    return get_device(cfg["train"].get("device", "auto"))
