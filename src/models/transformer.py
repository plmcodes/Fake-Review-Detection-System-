"""Fine-tuned DistilBERT (or RoBERTa) baseline on raw, lightly cleaned text.

Designed for a single Colab/Kaggle T4:
  * mixed precision (fp16 autocast + GradScaler) when a GPU is present,
  * max length 256, batch 32, 2-4 epochs, AdamW + linear warm-up/decay,
  * a resumable checkpoint (`resume.pt`) is written after EVERY epoch into the run's checkpoint
    folder - on Colab that folder lives on Google Drive, so a session timeout loses <= 1 epoch.

On a CPU laptop this model is very slow; configs/models/distilbert_cpu.yaml trains on a
stratified subset (`max_train`) so it can still be smoke-tested. That subset size is recorded
in runs.csv `notes`, so CPU-subset numbers are never confused with full Colab runs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from ..utils import get_logger
from .base import Runner, device_from_cfg, make_bce_loss

log = get_logger(__name__)


class TransformerRunner(Runner):
    family = "transformer"
    needs_labels = True

    def __init__(self, cfg, seed):
        super().__init__(cfg, seed)
        self.device = device_from_cfg(cfg)
        self.ckpt_dir: Path | None = None   # set by train.py so we can write per-epoch checkpoints
        self.tokenizer = None
        self.net = None
        self.notes = ""

    def _init_model(self, source: str):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(source)
        # num_labels=1 -> one logit; we apply our own BCE loss (with class weights/focal option).
        self.net = AutoModelForSequenceClassification.from_pretrained(source, num_labels=1).to(self.device)

    def _batches(self, texts, batch_size):
        """Tokenise per batch; sorting by length cuts padding and speeds up inference a lot."""
        order = np.argsort([len(t) for t in texts])
        for start in range(0, len(texts), batch_size):
            pos = order[start : start + batch_size]
            enc = self.tokenizer([texts[i] for i in pos], truncation=True, padding=True,
                                 max_length=int(self.mcfg["max_len"]), return_tensors="pt")
            yield pos, {k: v.to(self.device) for k, v in enc.items()}

    def _predict_logits(self, texts) -> np.ndarray:
        self.net.eval()
        out = np.zeros(len(texts), dtype=np.float32)
        use_amp = self.device.type == "cuda"
        with torch.no_grad():
            for pos, enc in self._batches(texts, int(self.mcfg.get("eval_batch_size", 64))):
                with torch.autocast(self.device.type, dtype=torch.float16, enabled=use_amp):
                    out[pos] = self.net(**enc).logits.squeeze(-1).float().cpu().numpy()
        return out

    def fit(self, data):
        from transformers import get_linear_schedule_with_warmup

        self._init_model(self.mcfg["pretrained"])
        tr, va = data.idx("train"), data.idx("val")
        max_train = self.mcfg.get("max_train")
        if max_train and max_train < len(tr):
            tr, _ = train_test_split(tr, train_size=int(max_train), stratify=data.y[tr], random_state=self.seed)
            self.notes = f"trained on stratified subset of {len(tr)} train reviews (CPU budget)"
            log.info(self.notes)
        texts, y = data.texts(tr), data.y[tr].astype(np.float32)
        val_texts, yva = data.texts(va), data.y[va].astype(np.float32)

        bs, epochs = int(self.mcfg["batch_size"]), int(self.mcfg["epochs"])
        loss_fn = make_bce_loss(data.y[tr], self.cfg["train"])
        opt = torch.optim.AdamW(self.net.parameters(), lr=float(self.mcfg["lr"]), weight_decay=0.01)
        total = epochs * int(np.ceil(len(tr) / bs))
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        start_epoch, best_val, best_state = 0, float("inf"), None
        resume = self.ckpt_dir / "resume.pt" if self.ckpt_dir else None
        if resume and resume.exists():   # continue after a Colab timeout
            st = torch.load(resume, map_location=self.device, weights_only=False)
            self.net.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
            sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
            start_epoch, best_val, best_state, self.history = st["epoch"] + 1, st["best_val"], st["best_state"], st["history"]
            log.info("Resumed from epoch %d", start_epoch)

        rng = np.random.default_rng(self.seed)
        for epoch in range(start_epoch, epochs):
            self.net.train()
            order, losses = rng.permutation(len(tr)), []
            for step, start in enumerate(range(0, len(tr), bs)):
                pos = order[start : start + bs]
                enc = self.tokenizer([texts[i] for i in pos], truncation=True, padding=True,
                                     max_length=int(self.mcfg["max_len"]), return_tensors="pt").to(self.device)
                target = torch.from_numpy(y[pos]).to(self.device)
                opt.zero_grad()
                with torch.autocast(self.device.type, dtype=torch.float16, enabled=use_amp):
                    logits = self.net(**enc).logits.squeeze(-1)
                loss = loss_fn(logits.float(), target)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                losses.append(loss.item())
                if step % 50 == 0:
                    log.info("epoch %d step %d/%d loss %.4f", epoch, step, int(np.ceil(len(tr) / bs)), loss.item())
            val_logits = torch.from_numpy(self._predict_logits(val_texts))
            val = float(loss_fn(val_logits, torch.from_numpy(yva)))
            self.history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val": val})
            log.info("epoch %d train_loss=%.4f val_loss=%.4f", epoch, np.mean(losses), val)
            if val < best_val:
                best_val, best_state = val, {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            if resume:
                torch.save({"model": self.net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "scaler": scaler.state_dict(), "epoch": epoch, "best_val": best_val,
                            "best_state": best_state, "history": self.history}, resume)
        if best_state is not None:
            self.net.load_state_dict(best_state)   # keep the epoch with the lowest val loss
        if resume and resume.exists():
            resume.unlink()  # training finished; the final model is saved by save()

    def score(self, data, idx):
        logits = self._predict_logits(data.texts(idx))
        return 1.0 / (1.0 + np.exp(-logits))

    def score_texts(self, texts):
        return 1.0 / (1.0 + np.exp(-self._predict_logits(list(texts))))

    def save(self, folder: Path):
        self.net.save_pretrained(Path(folder) / "hf")
        self.tokenizer.save_pretrained(Path(folder) / "hf")

    @classmethod
    def load(cls, folder: Path, cfg: dict):
        obj = cls(cfg, cfg["seed"])
        obj._init_model(str(Path(folder) / "hf"))
        return obj
