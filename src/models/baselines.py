"""Classical baselines on TF-IDF: Logistic Regression, Linear SVM, Multinomial Naive Bayes.

These follow the baseline choices of Shahariar et al. (2022, arXiv 2211.01675), who compare
deep models against NB/SVM-style classifiers on spam reviews.

The TF-IDF vectoriser is fit on the TRAIN split only (no vocabulary leakage from val/test).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from ..features import make_tfidf
from .base import Runner


def _build_classifier(kind: str, mcfg: dict, seed: int, balanced: bool):
    cw = "balanced" if balanced else None
    if kind == "logreg":
        return LogisticRegression(C=mcfg.get("C", 1.0), max_iter=2000, class_weight=cw, random_state=seed)
    if kind == "svm":
        return LinearSVC(C=mcfg.get("C", 1.0), class_weight=cw, random_state=seed)
    if kind == "nb":
        # NB has no class_weight; its class prior is learned from the training data.
        return MultinomialNB(alpha=mcfg.get("alpha", 1.0))
    raise ValueError(f"Unknown classical model {kind!r}")


class ClassicalRunner(Runner):
    family = "classical"
    needs_labels = True

    def __init__(self, cfg, seed):
        super().__init__(cfg, seed)
        self.kind = self.mcfg["kind"]
        # LinearSVC outputs a margin, not a probability: positive margin = fake.
        self.default_threshold = 0.0 if self.kind == "svm" else 0.5
        self.vectorizer = None
        self.clf = None

    def fit(self, data):
        tr = data.idx("train")
        self.vectorizer = make_tfidf(self.cfg)
        X = self.vectorizer.fit_transform(data.texts(tr, "text_tfidf"))   # fit on TRAIN only
        balanced = self.cfg["train"].get("imbalance", "class_weight") != "none"
        self.clf = _build_classifier(self.kind, self.mcfg, self.seed, balanced)
        self.clf.fit(X, data.y[tr])

    def score(self, data, idx):
        X = self.vectorizer.transform(data.texts(idx, "text_tfidf"))
        if self.kind == "svm":
            return self.clf.decision_function(X)
        return self.clf.predict_proba(X)[:, 1]

    def top_features(self, k: int = 20) -> dict:
        """Most fake-indicative / genuine-indicative n-grams (for the report and viva)."""
        names = np.array(self.vectorizer.get_feature_names_out())
        if self.kind == "nb":
            w = self.clf.feature_log_prob_[1] - self.clf.feature_log_prob_[0]
        else:
            w = self.clf.coef_.ravel()
        order = np.argsort(w)
        return {"fake": names[order[::-1][:k]].tolist(), "genuine": names[order[:k]].tolist()}

    def save(self, folder: Path):
        joblib.dump({"vectorizer": self.vectorizer, "clf": self.clf}, Path(folder) / "model.joblib")

    @classmethod
    def load(cls, folder: Path, cfg: dict):
        obj = cls(cfg, cfg["seed"])
        saved = joblib.load(Path(folder) / "model.joblib")
        obj.vectorizer, obj.clf = saved["vectorizer"], saved["clf"]
        return obj
