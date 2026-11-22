"""Model B - LSTM sequence-to-sequence autoencoder over GloVe word sequences (UNSUPERVISED).

Basis: "Spam review detection using LSTM autoencoder: an unsupervised approach" [verify
authors/venue]. Like Model A it is trained on GENUINE reviews only, but it works on the word
sequence itself instead of a sentence embedding, so it can notice unusual word ORDER.

    words -> GloVe embeddings -> LSTM encoder -> last hidden state -> latent z (64-d)
    z -> initial state of an LSTM decoder -> predicts the review again, word by word

Teacher forcing: while decoding, the decoder is fed the TRUE previous word (not its own guess).
This makes training stable and lets us score a review in one pass.

Fake score = mean cross-entropy per real token (padding positions are masked out), i.e. how
"surprised" the genuine-review model is by this review.

T4/CPU budget: sequences are cut to `max_len` (default 64) and the output vocabulary to
`max_vocab` (default 10k) - the output layer is [batch x len x vocab], so these two numbers
control memory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from ..features import Vocab, glove_matrix
from ..utils import get_logger, load_json, save_json
from .base import Runner, batched_predict, device_from_cfg, fit_torch

log = get_logger(__name__)


class LSTMAutoencoder(nn.Module):
    def __init__(self, emb_matrix, hidden: int = 256, z: int = 64, dropout: float = 0.2, freeze_emb: bool = True):
        super().__init__()
        emb = torch.as_tensor(emb_matrix, dtype=torch.float32)
        vocab_size, dim = emb.shape
        self.emb = nn.Embedding.from_pretrained(emb, freeze=freeze_emb, padding_idx=0)
        self.encoder = nn.LSTM(dim, hidden, batch_first=True)
        self.to_z = nn.Linear(hidden, z)
        self.from_z = nn.Linear(z, 2 * hidden)       # latent -> decoder's initial (h0, c0)
        self.decoder = nn.LSTM(dim, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, vocab_size)

    def encode(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(self.emb(ids), lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.encoder(packed)
        return self.to_z(h[-1])                       # [B, z]

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor):
        z = self.encode(ids, lengths)
        h0, c0 = torch.tanh(self.from_z(z)).chunk(2, dim=1)
        # Teacher forcing: decoder input = <bos> + the review shifted right by one word.
        bos = torch.full_like(ids[:, :1], Vocab.BOS)
        dec_in = self.drop(self.emb(torch.cat([bos, ids[:, :-1]], dim=1)))
        dec_out, _ = self.decoder(dec_in, (h0.unsqueeze(0).contiguous(), c0.unsqueeze(0).contiguous()))
        return self.out(self.drop(dec_out)), z       # logits [B, L, V]

    def token_loss(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Mean cross-entropy over the real (non-padding) tokens of each review -> [B]."""
        logits, _ = self(ids, lengths)
        ce = F.cross_entropy(logits.transpose(1, 2), ids, reduction="none")   # [B, L]
        mask = (ids != Vocab.PAD).float()
        return (ce * mask).sum(1) / mask.sum(1).clamp(min=1)


class LSTMAERunner(Runner):
    family = "autoencoder"
    needs_labels = False
    tune_threshold = True

    def __init__(self, cfg, seed):
        super().__init__(cfg, seed)
        self.device = device_from_cfg(cfg)
        self.vocab: Vocab | None = None
        self.net: LSTMAutoencoder | None = None

    def _encode(self, data, idx):
        ids, lengths = self.vocab.encode(data.texts(idx, "text_seq"), int(self.mcfg["max_len"]))
        return torch.from_numpy(ids), torch.from_numpy(lengths)

    def _per_review(self, ids, lengths, pos):
        L = int(lengths[pos].max())
        return self.net.token_loss(ids[pos, :L].to(self.device), lengths[pos].to(self.device))

    def fit(self, data):
        tr_g, va_g = data.genuine_idx("train"), data.genuine_idx("val")   # GENUINE ONLY
        # Vocabulary from genuine training reviews only: the model never sees a fake review.
        self.vocab = Vocab.build(data.texts(tr_g, "text_seq"), int(self.mcfg["max_vocab"]),
                                 int(self.cfg["features"]["glove"]["min_freq"]))
        matrix, coverage = glove_matrix(self.vocab, self.cfg, seed=self.seed)
        log.info("LSTM-AE vocab %d, GloVe coverage %.1f%%, %d genuine train reviews",
                 len(self.vocab), 100 * coverage, len(tr_g))
        self.net = LSTMAutoencoder(matrix, hidden=int(self.mcfg["hidden"]), z=int(self.mcfg["latent_dim"]),
                                   dropout=float(self.mcfg["dropout"]),
                                   freeze_emb=bool(self.mcfg.get("freeze_emb", True))).to(self.device)
        Xtr, Ltr = self._encode(data, tr_g)
        Xva, Lva = self._encode(data, va_g)
        opt = torch.optim.Adam([p for p in self.net.parameters() if p.requires_grad],
                               lr=float(self.cfg["train"]["lr"]))

        def step(pos):
            opt.zero_grad()
            loss = self._per_review(Xtr, Ltr, pos).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
            opt.step()
            return loss.item()

        def val_loss():
            return float(batched_predict(lambda p: self._per_review(Xva, Lva, p).cpu().numpy(), len(Xva), 256).mean())

        fit_torch(self.net, step, val_loss, len(Xtr), self.cfg["train"], self.seed, self.history, "lstm_ae")

    def score(self, data, idx):
        self.net.eval()
        X, L = self._encode(data, idx)
        return batched_predict(lambda p: self._per_review(X, L, p).cpu().numpy(), len(X), 256)

    def latent(self, data, idx):
        self.net.eval()
        X, L = self._encode(data, idx)
        return batched_predict(lambda p: self.net.encode(X[p].to(self.device), L[p].to(self.device)).cpu().numpy(),
                               len(X), 256)

    def save(self, folder: Path):
        torch.save(self.net.state_dict(), Path(folder) / "model.pt")
        save_json({"itos": self.vocab.itos}, Path(folder) / "vocab.json")

    @classmethod
    def load(cls, folder: Path, cfg: dict):
        obj = cls(cfg, cfg["seed"])
        obj.vocab = Vocab(load_json(Path(folder) / "vocab.json")["itos"])
        dim = cfg["features"]["glove"]["dim"]
        obj.net = LSTMAutoencoder(np.zeros((len(obj.vocab), dim), np.float32), hidden=int(obj.mcfg["hidden"]),
                                  z=int(obj.mcfg["latent_dim"]), dropout=float(obj.mcfg["dropout"]))
        obj.net.load_state_dict(torch.load(Path(folder) / "model.pt", map_location=obj.device))
        obj.net.to(obj.device).eval()
        return obj
