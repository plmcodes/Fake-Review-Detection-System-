r"""Model D - HYBRID (the proposed contribution).

    Review text -> SBERT (384-d) -> AE encoder -> latent z (32-d) ----------------\
                                          \-> AE decoder -> reconstruction error --+--> classifier -> P(fake)
    Review text -> style features (13) -------------------------------------------/
    Yelp only   -> behavioural features (6) -----------------------------------/

Idea (close to Dong et al. 2018, who feed autoencoder features into a random forest):
the autoencoder trained on genuine reviews gives two complementary signals - WHERE the review
sits in "genuine-review space" (z) and HOW UNUSUAL it is (reconstruction error). A supervised
classifier combines them with cheap stylistic (and, on Yelp, behavioural) evidence.

Two classifiers:  MLP (primary) and XGBoost (comparison)          -> model.classifier
Two regimes:      frozen  - AE pretrained on genuine reviews, then frozen; classifier on top
                  joint   - start from the pretrained AE and fine-tune encoder + decoder +
                            classifier together:  loss = BCE + lambda * reconstruction
Feature groups (latent / recon_error / style / behaviour) are switchable for the ablation.

Avoiding an optimistic bias (frozen regime): the AE has SEEN the genuine training reviews,
so their reconstruction errors are unrealistically low compared with unseen genuine reviews
at test time. We therefore compute the errors of genuine TRAIN rows out-of-fold: K extra AEs,
each trained without one fold, score the fold they did not see (`model.oof_folds`).
"""
from __future__ import annotations

import copy
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from ..config import deep_update
from ..features import BEHAVIOUR_COLUMNS, STYLE_COLUMNS, encode_texts, style_features
from ..utils import get_logger, load_json, save_json
from .ae_dense import DenseAERunner
from .base import Runner, batched_predict, device_from_cfg, fit_torch, make_bce_loss

log = get_logger(__name__)


def mlp_head(in_dim: int, hidden=(64, 32), dropout: float = 0.3, batchnorm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = [nn.BatchNorm1d(in_dim)] if batchnorm else []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


class JointHybridNet(nn.Module):
    """AE + classifier trained together (joint regime). Reconstruction error stays differentiable."""

    def __init__(self, ae: nn.Module, n_extra: int, z_dim: int, use_latent: bool, use_err: bool,
                 hidden, dropout: float):
        super().__init__()
        self.ae, self.use_latent, self.use_err = ae, use_latent, use_err
        in_dim = z_dim * use_latent + int(use_err) + n_extra
        self.head = mlp_head(in_dim, hidden, dropout, batchnorm=True)

    def forward(self, x: torch.Tensor, extra: torch.Tensor):
        z = self.ae.enc(x)
        err = ((self.ae.dec(z) - x) ** 2).mean(dim=1)
        parts = ([z] if self.use_latent else []) + ([torch.log(err + 1e-8).unsqueeze(1)] if self.use_err else [])
        if extra.shape[1]:
            parts.append(extra)
        return self.head(torch.cat(parts, dim=1)).squeeze(1), err


class HybridRunner(Runner):
    family = "hybrid"
    needs_labels = True

    def __init__(self, cfg, seed):
        super().__init__(cfg, seed)
        self.device = device_from_cfg(cfg)
        g = self.mcfg["features"]
        self.use_latent, self.use_err, self.use_style = bool(g["latent"]), bool(g["recon_error"]), bool(g["style"])
        self.use_beh = g.get("behaviour", "auto")   # auto -> only when the dataset has it (Yelp)
        self.ae: DenseAERunner | None = None
        self.scaler: StandardScaler | None = None   # for the frozen-regime feature matrix
        self.extra_scaler: StandardScaler | None = None  # style/behaviour scaler (joint regime)
        self.clf = None
        self.notes = ""

    # ---------------------------------------------------------------- helpers
    def _ae_cfg(self) -> dict:
        """Config for the inner autoencoder = Model A settings (model.ae + model.ae_train)."""
        return deep_update(self.cfg, {
            "model": deep_update({"name": "hybrid-ae", "variant": "ae"}, self.mcfg["ae"]),
            "train": self.mcfg.get("ae_train", {}),
        })

    def _resolve_behaviour(self, data) -> None:
        if self.use_beh == "auto":
            self.use_beh = bool(data.has_behaviour())
        self.use_beh = bool(self.use_beh)

    def _extra_raw(self, style: np.ndarray | None, beh: np.ndarray | None) -> np.ndarray:
        parts = ([style] if self.use_style else []) + ([beh] if self.use_beh else [])
        n = len(style) if style is not None else len(beh)
        return np.hstack(parts).astype(np.float32) if parts else np.zeros((n, 0), np.float32)

    def _data_extra(self, data, idx) -> np.ndarray:
        style = data.style().to_numpy()[idx]
        beh = None
        if self.use_beh:
            b = data.behaviour()
            if b is None:
                raise ValueError(f"{data.name} has no behavioural features but this hybrid model uses them")
            beh = b.to_numpy()[idx]
        return self._extra_raw(style, beh)

    def _ae_parts(self, emb: np.ndarray, ae: DenseAERunner | None = None):
        ae = ae or self.ae
        ae.net.eval()
        X = ae.transform_embeddings(emb)
        z = batched_predict(lambda p: ae.net.encode(X[p].to(self.device)).cpu().numpy(), len(X))
        err = batched_predict(lambda p: ae.net.recon_error(X[p].to(self.device)).cpu().numpy(), len(X))
        return z, err

    def _assemble(self, z, err, extra) -> np.ndarray:
        parts = ([z] if self.use_latent else []) + ([np.log(err + 1e-8)[:, None]] if self.use_err else []) + [extra]
        return np.hstack(parts).astype(np.float32)

    def feature_names(self) -> list[str]:
        names = [f"z_{i}" for i in range(int(self.mcfg["ae"]["latent_dim"]))] if self.use_latent else []
        names += ["log_recon_error"] if self.use_err else []
        names += STYLE_COLUMNS if self.use_style else []
        names += BEHAVIOUR_COLUMNS if self.use_beh else []
        return names

    # ---------------------------------------------------------------- training
    def _train_ae(self, data, train_positions=None) -> DenseAERunner:
        ae = DenseAERunner(self._ae_cfg(), self.seed)
        if train_positions is not None:   # fold model: pretend only these genuine rows exist in train
            view = copy.copy(data)
            view.splits = dict(data.splits, train=train_positions)
            ae.fit(view)
        else:
            ae.fit(data)
        return ae

    def _oof_errors(self, data, tr_g: np.ndarray) -> np.ndarray:
        """Out-of-fold reconstruction errors for genuine training reviews (see module docstring)."""
        k = int(self.mcfg.get("oof_folds", 5))
        emb = data.sbert()
        out = np.zeros(len(tr_g), dtype=np.float32)
        for fold, (fit_part, held_out) in enumerate(KFold(k, shuffle=True, random_state=self.seed).split(tr_g)):
            log.info("Out-of-fold AE %d/%d", fold + 1, k)
            fold_ae = self._train_ae(data, train_positions=tr_g[fit_part])
            out[held_out] = self._ae_parts(emb[tr_g[held_out]], fold_ae)[1]
        return out

    def fit(self, data):
        self._resolve_behaviour(data)
        log.info("Hybrid: classifier=%s regime=%s features: latent=%s err=%s style=%s behaviour=%s",
                 self.mcfg["classifier"], self.mcfg["regime"], self.use_latent, self.use_err,
                 self.use_style, self.use_beh)
        self.ae = self._train_ae(data)                       # Model A, genuine-only pretraining
        if self.mcfg["regime"] == "joint":
            self._fit_joint(data)
        else:
            self._fit_frozen(data)

    def _frozen_matrix(self, data, idx, oof: dict | None = None) -> np.ndarray:
        z, err = self._ae_parts(data.sbert()[idx])
        if oof:
            pos = {p: i for i, p in enumerate(idx)}
            for p, e in oof.items():
                if p in pos:
                    err[pos[p]] = e
        return self._assemble(z, err, self._data_extra(data, idx))

    def _fit_frozen(self, data):
        tr, va = data.idx("train"), data.idx("val")
        oof = None
        if self.use_err and int(self.mcfg.get("oof_folds", 5)) > 1:
            tr_g = data.genuine_idx("train")
            oof = dict(zip(tr_g.tolist(), self._oof_errors(data, tr_g)))
            self.notes = f"genuine-train recon errors out-of-fold (k={self.mcfg.get('oof_folds', 5)})"
        Xtr, Xva = self._frozen_matrix(data, tr, oof), self._frozen_matrix(data, va)
        self.scaler = StandardScaler().fit(Xtr)               # fit on TRAIN only
        Xtr, Xva = self.scaler.transform(Xtr), self.scaler.transform(Xva)
        ytr, yva = data.y[tr], data.y[va]
        if self.mcfg["classifier"] == "xgb":
            self._fit_xgb(Xtr, ytr, Xva, yva)
        else:
            self._fit_mlp(Xtr, ytr, Xva, yva)

    def _fit_mlp(self, Xtr, ytr, Xva, yva):
        torch.manual_seed(self.seed)
        m = self.mcfg["mlp"]
        self.clf = mlp_head(Xtr.shape[1], tuple(m["hidden"]), float(m["dropout"])).to(self.device)
        loss_fn = make_bce_loss(ytr, self.cfg["train"])
        opt = torch.optim.Adam(self.clf.parameters(), lr=float(self.cfg["train"]["lr"]),
                               weight_decay=float(self.cfg["train"].get("weight_decay", 0.0)))
        Xt, yt = torch.from_numpy(Xtr.astype(np.float32)), torch.from_numpy(ytr.astype(np.float32))
        Xv, yv = torch.from_numpy(Xva.astype(np.float32)), torch.from_numpy(yva.astype(np.float32))

        def step(pos):
            opt.zero_grad()
            loss = loss_fn(self.clf(Xt[pos].to(self.device)).squeeze(1), yt[pos].to(self.device))
            loss.backward()
            opt.step()
            return loss.item()

        def val_loss():
            return float(loss_fn(self.clf(Xv.to(self.device)).squeeze(1).cpu(), yv))

        fit_torch(self.clf, step, val_loss, len(Xt), self.cfg["train"], self.seed, self.history, "hybrid-mlp")

    def _fit_xgb(self, Xtr, ytr, Xva, yva):
        from xgboost import XGBClassifier

        x = self.mcfg["xgb"]
        n_pos, n_neg = max(ytr.sum(), 1), max(len(ytr) - ytr.sum(), 1)
        spw = n_neg / n_pos if self.cfg["train"].get("imbalance", "class_weight") != "none" else 1.0
        self.clf = XGBClassifier(n_estimators=int(x["n_estimators"]), max_depth=int(x["max_depth"]),
                                 learning_rate=float(x["learning_rate"]), subsample=float(x["subsample"]),
                                 colsample_bytree=float(x["colsample_bytree"]), scale_pos_weight=spw,
                                 early_stopping_rounds=int(x["early_stopping_rounds"]), eval_metric="logloss",
                                 tree_method="hist", random_state=self.seed, n_jobs=-1)
        self.clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        self.history = [{"best_iteration": int(self.clf.best_iteration)}]

    def _fit_joint(self, data):
        """Fine-tune encoder + decoder + classifier together: loss = BCE + lambda * recon."""
        tr, va = data.idx("train"), data.idx("val")
        lam = float(self.mcfg.get("lambda_recon", 0.5))
        m = self.mcfg["mlp"]
        Xtr, Xva = self.ae.transform_embeddings(data.sbert()[tr]), self.ae.transform_embeddings(data.sbert()[va])
        Etr, Eva = self._data_extra(data, tr), self._data_extra(data, va)
        self.extra_scaler = StandardScaler().fit(Etr) if Etr.shape[1] else None
        if self.extra_scaler is not None:
            Etr, Eva = self.extra_scaler.transform(Etr), self.extra_scaler.transform(Eva)
        Etr, Eva = torch.from_numpy(Etr.astype(np.float32)), torch.from_numpy(Eva.astype(np.float32))
        ytr, yva = torch.from_numpy(data.y[tr].astype(np.float32)), torch.from_numpy(data.y[va].astype(np.float32))
        torch.manual_seed(self.seed)
        self.clf = JointHybridNet(self.ae.net, Etr.shape[1], int(self.mcfg["ae"]["latent_dim"]), self.use_latent,
                                  self.use_err, tuple(m["hidden"]), float(m["dropout"])).to(self.device)
        bce = make_bce_loss(data.y[tr], self.cfg["train"])
        opt = torch.optim.Adam(self.clf.parameters(), lr=float(self.mcfg.get("joint_lr", 3e-4)))

        def step(pos):
            opt.zero_grad()
            x, e, y = Xtr[pos].to(self.device), Etr[pos].to(self.device), ytr[pos].to(self.device)
            logits, err = self.clf(x, e)
            # Reconstruction term on GENUINE rows only, so the AE keeps "genuine = easy to rebuild".
            genuine = y == 0
            rec = err[genuine].mean() if genuine.any() else torch.zeros((), device=self.device)
            loss = bce(logits, y) + lam * rec
            loss.backward()
            opt.step()
            return loss.item()

        def val_loss():
            logits = batched_predict(lambda p: self.clf(Xva[p].to(self.device), Eva[p].to(self.device))[0].cpu().numpy(),
                                     len(Xva))
            return float(bce(torch.from_numpy(logits), yva))   # early stop on classification loss

        fit_torch(self.clf, step, val_loss, len(Xtr), self.cfg["train"], self.seed, self.history, "hybrid-joint")
        self.notes = f"joint fine-tuning, lambda_recon={lam}"

    # ---------------------------------------------------------------- scoring
    def _proba_from_parts(self, emb: np.ndarray, extra: np.ndarray) -> np.ndarray:
        if self.mcfg["regime"] == "joint":
            self.clf.eval()
            X = self.ae.transform_embeddings(emb)
            E = self.extra_scaler.transform(extra) if self.extra_scaler is not None else extra
            E = torch.from_numpy(np.asarray(E, dtype=np.float32))
            logits = batched_predict(lambda p: self.clf(X[p].to(self.device), E[p].to(self.device))[0].cpu().numpy(), len(X))
            return 1 / (1 + np.exp(-logits))
        z, err = self._ae_parts(emb)
        return self.predict_matrix(self.scaler.transform(self._assemble(z, err, extra)))

    def predict_matrix(self, X: np.ndarray) -> np.ndarray:
        """P(fake) from an already-scaled frozen-regime feature matrix (used by SHAP)."""
        if self.mcfg["classifier"] == "xgb":
            return self.clf.predict_proba(X)[:, 1]
        self.clf.eval()
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        logits = batched_predict(lambda p: self.clf(Xt[p].to(self.device)).squeeze(1).cpu().numpy(), len(Xt))
        return 1 / (1 + np.exp(-logits))

    def feature_matrix(self, data, idx) -> np.ndarray:
        """Scaled frozen-regime features (for SHAP)."""
        return self.scaler.transform(self._frozen_matrix(data, idx))

    def score(self, data, idx):
        return self._proba_from_parts(data.sbert()[idx], self._data_extra(data, idx))

    def score_texts(self, texts: list[str]) -> dict:
        """Score raw review texts (Streamlit app / LIME). Behavioural features are not available."""
        if self.use_beh:
            raise ValueError("This hybrid model needs behavioural features; raw text alone is not enough")
        from ..preprocess import light_clean

        clean = [light_clean(t, self.cfg["preprocess"]["emoji"]) for t in texts]
        emb = encode_texts(clean, self.cfg)
        extra = self._extra_raw(style_features(clean).to_numpy(), None)
        ae_err = self._ae_parts(emb)[1]
        return {"proba": self._proba_from_parts(emb, extra), "recon_error": ae_err}

    # ---------------------------------------------------------------- persistence
    def save(self, folder: Path):
        folder = Path(folder)
        (folder / "ae").mkdir(exist_ok=True)
        self.ae.save(folder / "ae")
        save_json({"use_beh": self.use_beh}, folder / "hybrid.json")
        joblib.dump({"scaler": self.scaler, "extra_scaler": self.extra_scaler}, folder / "scalers.joblib")
        if self.mcfg["classifier"] == "xgb":
            joblib.dump(self.clf, folder / "clf.joblib")
        else:
            torch.save(self.clf.state_dict(), folder / "clf.pt")

    @classmethod
    def load(cls, folder: Path, cfg: dict):
        folder = Path(folder)
        obj = cls(cfg, cfg["seed"])
        obj.use_beh = load_json(folder / "hybrid.json")["use_beh"]
        obj.ae = DenseAERunner.load(folder / "ae", obj._ae_cfg())
        sc = joblib.load(folder / "scalers.joblib")
        obj.scaler, obj.extra_scaler = sc["scaler"], sc["extra_scaler"]
        m = obj.mcfg["mlp"]
        if obj.mcfg["classifier"] == "xgb":
            obj.clf = joblib.load(folder / "clf.joblib")
        elif obj.mcfg["regime"] == "joint":
            n_extra = len(STYLE_COLUMNS) * obj.use_style + len(BEHAVIOUR_COLUMNS) * obj.use_beh
            obj.clf = JointHybridNet(obj.ae.net, n_extra, int(obj.mcfg["ae"]["latent_dim"]), obj.use_latent,
                                     obj.use_err, tuple(m["hidden"]), float(m["dropout"]))
            obj.clf.load_state_dict(torch.load(folder / "clf.pt", map_location=obj.device))
        else:
            n_in = len(obj.feature_names())
            obj.clf = mlp_head(n_in, tuple(m["hidden"]), float(m["dropout"]))
            obj.clf.load_state_dict(torch.load(folder / "clf.pt", map_location=obj.device))
        if obj.mcfg["classifier"] != "xgb":
            obj.clf.to(obj.device).eval()
        return obj
