"""Text cleaning, tokenisation and the fixed train/val/test split.

Three "views" of every review are produced, because different models need different input:

  text        light clean only (HTML, URLs, emojis, whitespace) - case and stopwords kept.
              Used by SBERT, DistilBERT and the style features (caps ratio needs the case!).
  text_seq    lowercased tokens joined by spaces, stopwords KEPT.
              Used by GloVe sequence models (CNN, BiLSTM, LSTM-AE) - word order matters there.
  text_tfidf  lowercased, stopwords removed, lemmatised.
              Used ONLY by the TF-IDF baselines, as the project plan asks.

Splits are stratified 70/15/15 with a fixed `split_seed` and saved to disk as review_id lists,
so every model - and every repeated seed - is evaluated on exactly the same test reviews.
"""
from __future__ import annotations

import html
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import resolve_path
from .data_loading import load_dataset
from .utils import get_logger

log = get_logger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
# Main emoji / pictograph blocks. Good enough for review text; documented in the report.
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U00002B00-\U00002BFF]"
)
# Words, numbers and contractions ("don't") + sentence punctuation as separate tokens.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[!?.,]")


def _emoji_to_text(match: re.Match) -> str:
    """'😀' -> ' emoji_grinning_face ' so the emoji survives as a normal word."""
    name = unicodedata.name(match.group(0), "symbol").lower().replace(" ", "_").replace("-", "_")
    return f" emoji_{name} "


def light_clean(text: str, emoji: str = "text") -> str:
    """Minimal cleaning that keeps case and punctuation (style signals live there)."""
    text = html.unescape(str(text))
    text = _TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    if emoji == "text":
        text = _EMOJI_RE.sub(_emoji_to_text, text)
    elif emoji == "remove":
        text = _EMOJI_RE.sub(" ", text)
    else:
        raise ValueError(f"preprocess.emoji must be 'text' or 'remove', got {emoji!r}")
    return _WS_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Lowercase + regex tokeniser (no NLTK model download needed, and fully deterministic)."""
    return _TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=1)
def _nltk_resources():
    """Stopword list and lemmatiser. Downloads the small NLTK data files on first use."""
    import nltk

    for pkg, probe in [("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet"),
                       ("omw-1.4", "corpora/omw-1.4")]:
        try:
            nltk.data.find(probe)
        except LookupError:
            nltk.download(pkg, quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    # Keep negations: "not good" vs "good" matters for reviews.
    stops = set(stopwords.words("english")) - {"no", "not", "nor", "don't", "didn't", "wasn't"}
    return stops, WordNetLemmatizer()


def tfidf_clean(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> str:
    """Heavier normalisation used only for TF-IDF: drop stopwords/punctuation, lemmatise."""
    stops, lemmatizer = _nltk_resources()
    out = []
    for tok in tokenize(text):
        if not tok[0].isalnum():
            continue  # punctuation is captured by the style features instead
        if remove_stopwords and tok in stops:
            continue
        out.append(lemmatizer.lemmatize(tok) if lemmatize else tok)
    return " ".join(out)


def add_text_views(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add the text / text_seq / text_tfidf columns described in the module docstring."""
    pp = cfg["preprocess"]
    df = df.copy()
    df["text_raw"] = df["text"]
    df["text"] = [light_clean(t, pp["emoji"]) for t in df["text_raw"]]
    df["text_seq"] = [" ".join(tokenize(t)) for t in df["text"]]
    df["text_tfidf"] = [tfidf_clean(t, pp["remove_stopwords"], pp["lemmatize"]) for t in df["text"]]
    return df


# ----------------------------------------------------------------------------- sampling & splits

def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Class-balanced-as-original subsample used for quick CPU smoke tests (--sample 500)."""
    if n >= len(df):
        return df
    sample, _ = train_test_split(df, train_size=n, stratify=df["label"], random_state=seed)
    return sample.sort_index().reset_index(drop=True)


def make_splits(df: pd.DataFrame, ratios: dict, seed: int) -> dict[str, np.ndarray]:
    """Stratified train/val/test split. Returns integer positions into `df`."""
    positions = np.arange(len(df))
    test_frac = ratios["test"]
    val_frac_of_rest = ratios["val"] / (ratios["train"] + ratios["val"])
    rest, test = train_test_split(positions, test_size=test_frac, stratify=df["label"], random_state=seed)
    train, val = train_test_split(rest, test_size=val_frac_of_rest,
                                  stratify=df["label"].iloc[rest], random_state=seed)
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}


def bundle_name(dataset: str, sample) -> str:
    return f"{dataset}_s{sample}" if sample else dataset


def prepare_dataset(dataset: str, cfg: dict, sample=None, force: bool = False):
    """Load -> (optionally subsample) -> clean -> split, cached in data/processed/<name>/.

    Returns (df, splits, cache_dir). The split file stores review_ids, not row numbers, so it
    stays valid even if the cleaning code changes the row order in the future.
    """
    name = bundle_name(dataset, sample)
    cache_dir = resolve_path(cfg["paths"]["processed"]) / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    clean_path, split_path = cache_dir / "clean.pkl", cache_dir / "splits.json"

    # modern_fakes.csv is our own file and may be regenerated, so never trust its cache
    force = force or dataset == "modern_fakes"
    if clean_path.exists() and not force:
        df = pd.read_pickle(clean_path)
    else:
        df = load_dataset(dataset, cfg["paths"]["raw"])
        if sample:
            df = stratified_sample(df, int(sample), cfg["split_seed"])
        log.info("Cleaning %d %s reviews (first run only, then cached)...", len(df), dataset)
        df = add_text_views(df, cfg)
        df.to_pickle(clean_path)

    if split_path.exists() and not force:
        saved = json.loads(split_path.read_text())
        pos = {rid: i for i, rid in enumerate(df["review_id"])}
        splits = {k: np.array(sorted(pos[r] for r in ids)) for k, ids in saved["splits"].items()}
    else:
        splits = make_splits(df, cfg["split"], cfg["split_seed"])
        split_path.write_text(json.dumps({
            "split_seed": cfg["split_seed"], "ratios": cfg["split"],
            "splits": {k: df["review_id"].iloc[v].tolist() for k, v in splits.items()},
        }))
    return df, splits, cache_dir


def genuine_only(positions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Subset of positions whose label is genuine (0) - the anomaly detectors' training data.

    Called on the TRAIN split (and on VAL for early stopping), never on test.
    """
    return positions[labels[positions] == 0]
