"""Feature representations, all cached under data/processed/ so they are computed once.

| Representation | Fit on             | Used by                          |
|----------------|--------------------|----------------------------------|
| TF-IDF         | train split only   | LR, SVM, NB                      |
| GloVe seqs     | vocab: train only  | CNN, BiLSTM, LSTM-AE             |
| SBERT          | nothing (frozen)   | Models A, C, D                   |
| Style          | nothing (per text) | Model D  (scaler fit on train)   |
| Behavioural    | no labels used     | Model D on Yelp only             |

LEAKAGE RULE: anything with learnable state (vectoriser, vocabulary, scaler) is fit inside a
model runner on the TRAIN positions only, then applied to val/test/cross-domain data.
SBERT and the style features have no learned state, so caching them for all rows is safe.
"""
from __future__ import annotations

import hashlib
import json
import string
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve_path
from .preprocess import genuine_only, prepare_dataset, tokenize
from .utils import get_logger

log = get_logger(__name__)


# ============================================================================ data bundle

@dataclass
class DataBundle:
    """Everything a model needs about one dataset: cleaned rows, fixed splits and features.

    `splits` maps 'train'/'val'/'test' to integer row positions in `df`.
    For a cross-domain evaluation set the whole dataset is the 'test' split.
    """

    dataset: str
    df: pd.DataFrame
    splits: dict
    cache_dir: Path
    cfg: dict
    _memo: dict = field(default_factory=dict, repr=False)

    @property
    def name(self) -> str:
        return self.cache_dir.name

    @property
    def y(self) -> np.ndarray:
        return self.df["label"].to_numpy()

    def idx(self, split: str) -> np.ndarray:
        return self.splits[split]

    def genuine_idx(self, split: str) -> np.ndarray:
        """Genuine-only rows of a split (anomaly detectors train/early-stop on these)."""
        return genuine_only(self.splits[split], self.y)

    def texts(self, idx, view: str = "text") -> list[str]:
        return self.df[view].iloc[idx].tolist()

    # Cached per-row features (no learned state -> safe to compute for every row)
    def sbert(self) -> np.ndarray:
        if "sbert" not in self._memo:
            self._memo["sbert"] = sbert_embeddings(self, self.cfg)
        return self._memo["sbert"]

    def style(self) -> pd.DataFrame:
        if "style" not in self._memo:
            path = self.cache_dir / "style.pkl"
            if path.exists():
                feats = pd.read_pickle(path)
            else:
                feats = style_features(self.df["text"].tolist())
                feats.to_pickle(path)
            self._memo["style"] = feats
        return self._memo["style"]

    def has_behaviour(self) -> bool:
        return self.df["user_id"].notna().all() and self.df["date"].notna().any()

    def behaviour(self) -> pd.DataFrame | None:
        if not self.has_behaviour():
            return None
        if "behaviour" not in self._memo:
            self._memo["behaviour"] = behaviour_features(self.df, self.sbert())
        return self._memo["behaviour"]


def load_bundle(dataset: str, cfg: dict, sample=None, as_eval_set: bool = False) -> DataBundle:
    """Prepare a dataset. `as_eval_set=True` treats every row as test (cross-domain use)."""
    df, splits, cache_dir = prepare_dataset(dataset, cfg, sample=sample)
    if as_eval_set:
        splits = {"test": np.arange(len(df))}
    return DataBundle(dataset=dataset, df=df, splits=splits, cache_dir=cache_dir, cfg=cfg)


# ============================================================================ style features

FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}
_PUNCT = set(string.punctuation)

STYLE_COLUMNS = [
    "n_chars", "n_words", "avg_word_len", "n_sentences", "words_per_sentence",
    "exclaim_rate", "question_rate", "punct_ratio", "first_person_rate",
    "caps_ratio", "allcaps_word_rate", "type_token_ratio", "digit_ratio",
]


def style_vector(text: str) -> list[float]:
    """Hand-crafted writing-style signals for one review (computed on case-preserving text).

    Rates are divided by the number of words so long and short reviews are comparable.
    """
    tokens = tokenize(text)
    words = [t for t in tokens if t[0].isalnum()]
    n_words = max(len(words), 1)
    letters = [c for c in text if c.isalpha()]
    n_letters = max(len(letters), 1)
    raw_words = text.split()
    n_sent = max(sum(text.count(p) for p in ".!?"), 1)
    return [
        len(text),
        len(words),
        float(np.mean([len(w) for w in words])) if words else 0.0,
        n_sent,
        len(words) / n_sent,
        text.count("!") / n_words,
        text.count("?") / n_words,
        sum(c in _PUNCT for c in text) / max(len(text), 1),
        sum(w in FIRST_PERSON for w in words) / n_words,
        sum(c.isupper() for c in letters) / n_letters,
        sum(w.isupper() and len(w) > 1 for w in raw_words) / max(len(raw_words), 1),
        len(set(words)) / n_words,                     # vocabulary richness
        sum(c.isdigit() for c in text) / max(len(text), 1),
    ]


def style_features(texts: list[str]) -> pd.DataFrame:
    return pd.DataFrame([style_vector(t) for t in texts], columns=STYLE_COLUMNS)


# ============================================================================ SBERT

def _rows_digest(ids, texts) -> str:
    """Fingerprint of (review_id, text) pairs, so a changed text invalidates the cache."""
    return hashlib.sha1("\n".join(f"{i}\t{t}" for i, t in zip(ids, texts)).encode("utf-8")).hexdigest()


_SBERT_MODELS: dict = {}


def get_sbert_model(model_name: str, device: str | None = None, max_seq_length: int = 256):
    """Load the SentenceTransformer once per process (it is reused by LIME and the app)."""
    key = (model_name, device)
    if key not in _SBERT_MODELS:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device=device)
        model.max_seq_length = max_seq_length
        _SBERT_MODELS[key] = model
    return _SBERT_MODELS[key]


def encode_texts(texts: list[str], cfg: dict, show_progress: bool = False) -> np.ndarray:
    """SBERT-encode raw texts (used for new reviews in the app / LIME / modern fakes)."""
    sb = cfg["features"]["sbert"]
    model = get_sbert_model(sb["model_name"], max_seq_length=sb["max_seq_length"])
    return model.encode(texts, batch_size=sb["batch_size"], convert_to_numpy=True,
                        show_progress_bar=show_progress).astype(np.float32)


def sbert_embeddings(bundle: DataBundle, cfg: dict) -> np.ndarray:
    """384-d all-MiniLM-L6-v2 embeddings for every row of the bundle, cached to disk.

    A --sample bundle reuses the full dataset's cache when it exists (same review_ids).
    """
    slug = cfg["features"]["sbert"]["model_name"].split("/")[-1]
    ids = bundle.df["review_id"].tolist()
    digest = _rows_digest(ids, bundle.df["text"].tolist())
    path = bundle.cache_dir / f"sbert_{slug}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if str(z["digest"]) == digest:
            return z["emb"]
    full_path = resolve_path(cfg["paths"]["processed"]) / bundle.dataset / f"sbert_{slug}.npz"
    if full_path.exists() and full_path != path:
        z = np.load(full_path, allow_pickle=False)
        full_ids = json.loads(str(z["ids"]))
        lookup = {rid: i for i, rid in enumerate(full_ids)}
        if all(r in lookup for r in ids):
            return z["emb"][[lookup[r] for r in ids]]
    log.info("Encoding %d texts with %s (cached afterwards)...", len(ids), slug)
    emb = encode_texts(bundle.df["text"].tolist(), cfg, show_progress=len(ids) > 2000)
    np.savez(path, emb=emb, digest=digest, ids=json.dumps(ids))
    return emb


# ============================================================================ TF-IDF

def make_tfidf(cfg: dict):
    """Unfitted TF-IDF vectoriser; the runner fits it on TRAIN text_tfidf only."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    t = cfg["features"]["tfidf"]
    return TfidfVectorizer(max_features=t["max_features"], ngram_range=(t["ngram_min"], t["ngram_max"]),
                           min_df=t["min_df"], sublinear_tf=t["sublinear_tf"])


# ============================================================================ GloVe sequences

class Vocab:
    """Word <-> id mapping built from TRAIN texts only. id 0 = <pad>, 1 = <unk>, 2 = <bos>."""

    PAD, UNK, BOS = 0, 1, 2

    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    def __len__(self):
        return len(self.itos)

    @classmethod
    def build(cls, texts: list[str], max_vocab: int, min_freq: int) -> "Vocab":
        counts = Counter(tok for t in texts for tok in t.split())
        common = [w for w, c in counts.most_common() if c >= min_freq][: max_vocab - 3]
        return cls(["<pad>", "<unk>", "<bos>"] + common)

    def encode(self, texts: list[str], max_len: int) -> tuple[np.ndarray, np.ndarray]:
        """Texts -> (ids [N, max_len] right-padded with 0, true lengths [N]). Long texts are cut."""
        ids = np.zeros((len(texts), max_len), dtype=np.int64)
        lengths = np.zeros(len(texts), dtype=np.int64)
        for i, t in enumerate(texts):
            seq = [self.stoi.get(w, self.UNK) for w in t.split()][:max_len] or [self.UNK]
            ids[i, : len(seq)] = seq
            lengths[i] = len(seq)
        return ids, lengths


def load_glove(path: str | Path, dim: int) -> tuple[list[str], np.ndarray]:
    """Read glove.6B.<dim>d.txt once and cache it as .npy for fast reloads (~1 s instead of ~30 s)."""
    path = resolve_path(path)
    if not path.exists():
        raise FileNotFoundError(f"GloVe file not found: {path}. Run `python -m src.download_data --glove`.")
    folder = resolve_path("data/processed/glove")
    npy, vocab_txt = folder / f"{path.stem}.npy", folder / f"{path.stem}.vocab.txt"   # e.g. glove.6B.100d.npy
    if npy.exists() and vocab_txt.exists():
        return vocab_txt.read_text(encoding="utf-8").split("\n"), np.load(npy)
    words, vecs = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            words.append(parts[0])
            vecs.append(np.asarray(parts[1:], dtype=np.float32))
    matrix = np.stack(vecs)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(npy, matrix)
    vocab_txt.write_text("\n".join(words), encoding="utf-8")
    return words, matrix


def glove_matrix(vocab: Vocab, cfg: dict, seed: int = 0) -> tuple[np.ndarray, float]:
    """Embedding matrix aligned with `vocab`. Words missing from GloVe get small random vectors.

    Returns (matrix, coverage) where coverage = share of vocab words found in GloVe.
    """
    g = cfg["features"]["glove"]
    words, vecs = load_glove(g["path"], g["dim"])
    lookup = {w: i for i, w in enumerate(words)}
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0, 0.1, size=(len(vocab), g["dim"])).astype(np.float32)
    matrix[Vocab.PAD] = 0.0
    hits = 0
    for i, w in enumerate(vocab.itos):
        if w in lookup:
            matrix[i] = vecs[lookup[w]]
            hits += 1
    return matrix, hits / len(vocab)


# ============================================================================ behavioural (Yelp only)

BEHAVIOUR_COLUMNS = [
    "rating_dev_from_product_mean", "user_max_reviews_per_day", "user_extreme_rating_share",
    "account_age_days", "user_n_reviews", "max_sim_same_user",
]


def behaviour_features(df: pd.DataFrame, emb: np.ndarray) -> pd.DataFrame:
    """Reviewer-behaviour features (Rayana & Akoglu 2015 style). Requires user_id/product_id/date.

    NO LABELS are used: aggregates only use ratings, dates and texts, which a platform would
    know for every review at prediction time. So using all rows (incl. test) does not leak labels.
    """
    d = df[["user_id", "product_id", "rating", "date"]].copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    prod_mean = d.groupby("product_id")["rating"].transform("mean")
    per_day = d.groupby(["user_id", d["date"].dt.date])["rating"].transform("size")
    out = pd.DataFrame(index=df.index)
    out["rating_dev_from_product_mean"] = (d["rating"] - prod_mean).abs()
    out["user_max_reviews_per_day"] = per_day.groupby(d["user_id"]).transform("max")
    out["user_extreme_rating_share"] = d["rating"].isin([1, 5]).groupby(d["user_id"]).transform("mean")
    first = d.groupby("user_id")["date"].transform("min")
    out["account_age_days"] = (d["date"] - first).dt.days.fillna(0)
    out["user_n_reviews"] = d.groupby("user_id")["rating"].transform("size")
    # Near-duplicate detection: max cosine similarity to the same user's other reviews.
    norm = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
    max_sim = np.zeros(len(df), dtype=np.float32)
    for _, rows in d.groupby("user_id").indices.items():
        if len(rows) < 2:
            continue
        sims = norm[rows] @ norm[rows].T
        np.fill_diagonal(sims, -1.0)
        max_sim[rows] = sims.max(axis=1)
    out["max_sim_same_user"] = max_sim
    return out[BEHAVIOUR_COLUMNS].astype(float)
