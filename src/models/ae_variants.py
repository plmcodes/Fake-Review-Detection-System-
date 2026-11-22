"""Model C - variants of Model A (same encoder/decoder sizes, different training objective).

DenoisingAE: corrupt the INPUT during training (Gaussian noise or random masking) but ask the
    network to rebuild the CLEAN input. It cannot simply copy, so it learns more robust
    structure of genuine reviews. At scoring time no noise is added.

VAE (Kingma & Welling 2014): the encoder outputs a distribution N(mu, sigma^2) instead of a point.
    z = mu + sigma * eps   (reparameterisation trick - lets gradients flow through the sampling)
    loss = reconstruction + beta * KL(q(z|x) || N(0, I))
    The KL term keeps the latent space smooth. Scoring uses z = mu (deterministic) and either
    the reconstruction error or the negative ELBO (config `model.score: recon | elbo`).
    Favuzzi's thesis (Politecnico di Milano) uses a VAE for deep-fake reviews [verify details].
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .ae_dense import AE


class DenoisingAE(AE):
    def __init__(self, noise: str = "gaussian", noise_std: float = 0.3, mask_prob: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.noise, self.noise_std, self.mask_prob = noise, noise_std, mask_prob

    def corrupt(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise == "gaussian":
            return x + self.noise_std * torch.randn_like(x)
        if self.noise == "mask":   # randomly zero input features (input dropout)
            return x * (torch.rand_like(x) > self.mask_prob).float()
        raise ValueError(f"Unknown noise type {self.noise!r}")

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        recon, _ = self(self.corrupt(x))          # noisy in ...
        return ((recon - x) ** 2).mean()          # ... clean target


class VAE(nn.Module):
    def __init__(self, d: int = 384, z: int = 32, hidden=(256, 64), dropout: float = 0.2, beta: float = 1.0):
        super().__init__()
        h1, h2 = hidden
        self.beta = beta
        self.trunk = nn.Sequential(nn.Linear(d, h1), nn.ReLU(), nn.Dropout(dropout),
                                   nn.Linear(h1, h2), nn.ReLU())
        self.mu = nn.Linear(h2, z)
        self.logvar = nn.Linear(h2, z)
        self.dec = nn.Sequential(nn.Linear(z, h2), nn.ReLU(),
                                 nn.Linear(h2, h1), nn.ReLU(),
                                 nn.Linear(h1, d))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.mu(self.trunk(x))

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        mu, logvar = self.mu(h), self.logvar(h).clamp(-10, 10)
        if self.training:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)   # reparameterisation trick
        else:
            z = mu
        return self.dec(z), mu, logvar

    @staticmethod
    def kl(mu, logvar) -> torch.Tensor:
        """KL(N(mu, sigma^2) || N(0, 1)) summed over latent dims -> one value per review."""
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        recon, mu, logvar = self(x)
        rec = ((recon - x) ** 2).sum(dim=1)
        return (rec + self.beta * self.kl(mu, logvar)).mean() / x.shape[1]   # per-dimension scale

    def recon_error(self, x: torch.Tensor) -> torch.Tensor:
        recon, _, _ = self(x)
        return ((recon - x) ** 2).mean(dim=1)

    def neg_elbo(self, x: torch.Tensor) -> torch.Tensor:
        recon, mu, logvar = self(x)
        return ((recon - x) ** 2).sum(dim=1) + self.kl(mu, logvar)
