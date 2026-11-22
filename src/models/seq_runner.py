"""Runner shared by the GloVe sequence classifiers (CNN and BiLSTM).

Steps: build vocabulary from TRAIN texts -> look up GloVe vectors -> pad sequences ->
train with early stopping on validation loss -> score = sigmoid(logit) = P(fake).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..features import Vocab, glove_matrix
from ..utils import get_logger, load_json, save_json
from .base import Runner, batched_predict, device_from_cfg, fit_torch, make_bce_loss
from .bilstm import BiLSTMClassifier
from .cnn import TextCNN

log = get_logger(__name__)


def _build_net(arch: str, emb_matrix, mcfg: dict):
    if arch == "cnn":
        return TextCNN(emb_matrix, n_filters=mcfg.get("n_filters", 100),
                       kernel_sizes=tuple(mcfg.get("kernel_sizes", [3, 4, 5])),
                       dropout=mcfg.get("dropout", 0.5), freeze_emb=mcfg.get("freeze_emb", False))
    if arch == "bilstm":
        return BiLSTMClassifier(emb_matrix, hidden=mcfg.get("hidden", 128),
                                num_layers=mcfg.get("num_layers", 1), dropout=mcfg.get("dropout", 0.3),
                                freeze_emb=mcfg.get("freeze_emb", False))
    raise ValueError(f"Unknown sequence architecture {arch!r}")


class SeqClassifierRunner(Runner):
    family = "deep"
    needs_labels = True

    def __init__(self, cfg, seed):
        super().__init__(cfg, seed)
        self.arch = self.mcfg["arch"]
        self.device = device_from_cfg(cfg)
        self.vocab: Vocab | None = None
        self.net = None

    def _encode(self, data, idx):
        g = self.cfg["features"]["glove"]
        ids, lengths = self.vocab.encode(data.texts(idx, "text_seq"), g["max_len"])
        return torch.from_numpy(ids), torch.from_numpy(lengths)

    def _logits(self, ids, lengths, pos):
        L = int(lengths[pos].max())                    # trim padding to the longest review in batch
        return self.net(ids[pos, :L].to(self.device), lengths[pos].to(self.device))

    def fit(self, data):
        g = self.cfg["features"]["glove"]
        tr, va = data.idx("train"), data.idx("val")
        self.vocab = Vocab.build(data.texts(tr, "text_seq"), g["max_vocab"], g["min_freq"])  # TRAIN only
        matrix, coverage = glove_matrix(self.vocab, self.cfg, seed=self.seed)
        log.info("Vocab %d words, GloVe coverage %.1f%%", len(self.vocab), 100 * coverage)
        self.net = _build_net(self.arch, matrix, self.mcfg).to(self.device)

        Xtr, Ltr = self._encode(data, tr)
        Xva, Lva = self._encode(data, va)
        ytr = torch.tensor(data.y[tr], dtype=torch.float32)
        yva = torch.tensor(data.y[va], dtype=torch.float32)
        loss_fn = make_bce_loss(data.y[tr], self.cfg["train"])
        opt = torch.optim.Adam(self.net.parameters(), lr=float(self.cfg["train"]["lr"]),
                               weight_decay=float(self.cfg["train"].get("weight_decay", 0.0)))

        def step(pos):
            opt.zero_grad()
            loss = loss_fn(self._logits(Xtr, Ltr, pos), ytr[pos].to(self.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
            opt.step()
            return loss.item()

        def val_loss():
            logits = batched_predict(lambda p: self._logits(Xva, Lva, p).cpu().numpy(), len(va))
            return float(loss_fn(torch.from_numpy(logits), yva))

        fit_torch(self.net, step, val_loss, len(tr), self.cfg["train"], self.seed, self.history, self.arch)

    def score(self, data, idx):
        self.net.eval()
        X, L = self._encode(data, idx)
        logits = batched_predict(lambda p: self._logits(X, L, p).cpu().numpy(), len(idx))
        return 1.0 / (1.0 + np.exp(-logits))

    def save(self, folder: Path):
        torch.save(self.net.state_dict(), Path(folder) / "model.pt")
        save_json({"itos": self.vocab.itos}, Path(folder) / "vocab.json")

    @classmethod
    def load(cls, folder: Path, cfg: dict):
        obj = cls(cfg, cfg["seed"])
        obj.vocab = Vocab(load_json(Path(folder) / "vocab.json")["itos"])
        dim = cfg["features"]["glove"]["dim"]
        obj.net = _build_net(obj.arch, np.zeros((len(obj.vocab), dim), np.float32), obj.mcfg)
        obj.net.load_state_dict(torch.load(Path(folder) / "model.pt", map_location=obj.device))
        obj.net.to(obj.device).eval()
        return obj
