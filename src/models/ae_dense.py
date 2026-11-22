"""Model A - dense autoencoder on SBERT embeddings (UNSUPERVISED anomaly detector).

Idea (Dong et al. 2018 use a similar autoencoder for opinion fraud): train the autoencoder
ONLY on genuine reviews. It learns to compress and rebuild "normal" review embeddings well.
A fake review looks different, so the decoder rebuilds it poorly ->
    fake score = reconstruction error = mean squared error between input and output.

No fake/genuine labels are used for TRAINING. Labels on the validation split are used only to
pick the decision threshold (F1-maximising), which is then frozen for the test split.

This file also holds `DenseAERunner`, which trains Model A and the Model C variants
(denoising AE and VAE in ae_variants.py) with the same code path.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from ..utils import get_logger, load_json, save_json
from .base import Runner, batched_predict, device_from_cfg, fit_torch

log = get_logger(__name__)


class AE(nn.Module):
    """The reference architecture from the project plan: 384 -> 256 -> 64 -> z -> 64 -> 256 -> 384."""

    def __init__(self, d: int = 384, z: int = 32, hidden=(256, 64), dropout: float = 0.2):
        super().__init__()
        h1, h2 = hidden
        self.enc = nn.Sequential(nn.Linear(d, h1), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(h1, h2), nn.ReLU(),
                                 nn.Linear(h2, z))
        self.dec = nn.Sequential(nn.Linear(z, h2), nn.ReLU(),
                                 nn.Linear(h2, h1), nn.ReLU(),
                                 nn.Linear(h1, d))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def forward(self, x: torch.Tensor):
        z = self.enc(x)
        return self.dec(z), z

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """Training loss = mean squared reconstruction error."""
        recon, _ = self(x)
        return ((recon - x) ** 2).mean()

    def recon_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-review anomaly score (one number per row)."""
        recon, _ = self(x)
        return ((recon - x) ** 2).mean(dim=1)


def build_dense_ae(mcfg: dict, d: int) -> nn.Module:
    """Pick Model A / C variant from config `model.variant`."""
    from .ae_variants import DenoisingAE, VAE

    kwargs = dict(d=d, z=int(mcfg.get("latent_dim", 32)), hidden=tuple(mcfg.get("hidden", [256, 64])),
                  dropout=float(mcfg.get("dropout", 0.2)))
    variant = mcfg.get("variant", "ae")
    if variant == "ae":
        return AE(**kwargs)
    if variant == "dae":
        return DenoisingAE(noise=mcfg.get("noise", "gaussian"), noise_std=float(mcfg.get("noise_std", 0.3)),
                           mask_prob=float(mcfg.get("mask_prob", 0.1)), **kwargs)
    if variant == "vae":
        return VAE(beta=float(mcfg.get("beta", 1.0)), **kwargs)
    raise ValueError(f"Unknown autoencoder variant {variant!r}")


class DenseAERunner(Runner):
    """Trains AE / DAE / VAE on genuine-only SBERT embeddings; scores by reconstruction error."""

    family = "autoencoder"
    needs_labels = False
    tune_threshold = True

    def __init__(self, cfg, seed):
        super().__init__(cfg, seed)
        self.device = device_from_cfg(cfg)
        self.scaler: StandardScaler | None = None
        self.net: nn.Module | None = None

    # SBERT vectors are unit-length, so each of the 384 dimensions has a tiny variance and the
    # MSE would be dominated by a few dims. Standardising (fit on genuine TRAIN rows only) puts
    # every dimension on the same scale. Switch off with model.standardize=false.
    def _features(self, data, idx) -> np.ndarray:
        X = data.sbert()[idx]
        return self.scaler.transform(X).astype(np.float32) if self.scaler is not None else X.astype(np.float32)

    def fit(self, data):
        tr_g, va_g = data.genuine_idx("train"), data.genuine_idx("val")   # GENUINE ONLY
        log.info("Training %s on %d genuine reviews (val: %d genuine)", self.mcfg["name"], len(tr_g), len(va_g))
        if self.mcfg.get("standardize", True):
            self.scaler = StandardScaler().fit(data.sbert()[tr_g])
        Xtr = torch.from_numpy(self._features(data, tr_g))
        Xva = torch.from_numpy(self._features(data, va_g))
        self.net = build_dense_ae(self.mcfg, Xtr.shape[1]).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=float(self.cfg["train"]["lr"]),
                               weight_decay=float(self.cfg["train"].get("weight_decay", 0.0)))

        def step(pos):
            opt.zero_grad()
            loss = self.net.loss(Xtr[pos].to(self.device))
            loss.backward()
            opt.step()
            return loss.item()

        def val_loss():
            # Early stopping on genuine-only validation reconstruction error (no labels needed).
            errs = batched_predict(lambda p: self.net.recon_error(Xva[p].to(self.device)).cpu().numpy(), len(Xva))
            return float(errs.mean())

        fit_torch(self.net, step, val_loss, len(Xtr), self.cfg["train"], self.seed, self.history, self.mcfg["name"])

    def _apply(self, data, idx, fn) -> np.ndarray:
        self.net.eval()
        X = torch.from_numpy(self._features(data, idx))
        return batched_predict(lambda p: fn(X[p].to(self.device)).cpu().numpy(), len(X))

    def score(self, data, idx):
        if self.mcfg.get("score", "recon") == "elbo":
            return self._apply(data, idx, self.net.neg_elbo)
        return self._apply(data, idx, self.net.recon_error)

    def latent(self, data, idx) -> np.ndarray:
        """Encoder output z (VAE: the mean mu) - used by the hybrid model and the UMAP plot."""
        return self._apply(data, idx, self.net.encode)

    # --- helpers for raw-feature input (hybrid model, app) --------------------------
    def transform_embeddings(self, emb: np.ndarray) -> torch.Tensor:
        X = self.scaler.transform(emb) if self.scaler is not None else emb
        return torch.from_numpy(np.asarray(X, dtype=np.float32))

    def save(self, folder: Path):
        torch.save(self.net.state_dict(), Path(folder) / "model.pt")
        joblib.dump(self.scaler, Path(folder) / "scaler.joblib")
        save_json({"input_dim": int(next(self.net.parameters()).shape[1])}, Path(folder) / "arch.json")

    @classmethod
    def load(cls, folder: Path, cfg: dict):
        obj = cls(cfg, cfg["seed"])
        obj.scaler = joblib.load(Path(folder) / "scaler.joblib")
        d = load_json(Path(folder) / "arch.json")["input_dim"]
        obj.net = build_dense_ae(obj.mcfg, d)
        obj.net.load_state_dict(torch.load(Path(folder) / "model.pt", map_location=obj.device))
        obj.net.to(obj.device).eval()
        return obj
