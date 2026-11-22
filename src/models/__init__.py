"""Model registry: config `model.runner` -> Runner class, plus checkpoint loading."""
from __future__ import annotations

from pathlib import Path

from ..utils import load_json


def runner_class(runner: str):
    # Imports are inside the function so that e.g. the classical baselines don't import torch.
    if runner == "classical":
        from .baselines import ClassicalRunner
        return ClassicalRunner
    if runner == "sequence":
        from .seq_runner import SeqClassifierRunner
        return SeqClassifierRunner
    if runner == "transformer":
        from .transformer import TransformerRunner
        return TransformerRunner
    if runner == "dense_ae":
        from .ae_dense import DenseAERunner
        return DenseAERunner
    if runner == "lstm_ae":
        from .lstm_ae import LSTMAERunner
        return LSTMAERunner
    if runner == "hybrid":
        from .hybrid import HybridRunner
        return HybridRunner
    raise KeyError(f"Unknown runner type {runner!r}")


def build_runner(cfg: dict, seed: int):
    return runner_class(cfg["model"]["runner"])(cfg, seed)


def load_runner(ckpt_dir: Path | str):
    """Rebuild a trained model from checkpoints/<run_id>/ (meta.json holds its full config)."""
    ckpt_dir = Path(ckpt_dir)
    meta = load_json(ckpt_dir / "meta.json")
    cfg = meta["config"]
    runner = runner_class(cfg["model"]["runner"]).load(ckpt_dir, cfg)
    runner.load_history(ckpt_dir)
    return runner, meta
