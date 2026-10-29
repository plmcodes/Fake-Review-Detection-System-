"""One loader per dataset, all returning the same COMMON SCHEMA:

    review_id, text, label (1 = fake, 0 = genuine), rating, category,
    user_id, product_id, date, source_dataset

Columns a dataset does not have (e.g. user_id for Salminen) are filled with NaN, so the rest
of the pipeline never needs dataset-specific code. See data/README.md for where files go.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve_path

COMMON_COLUMNS = [
    "review_id", "text", "label", "rating", "category",
    "user_id", "product_id", "date", "source_dataset",
]


class DatasetNotFound(FileNotFoundError):
    """Raised when a dataset's raw files are missing; the message says how to get them."""


def _finalise(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Add missing schema columns, fix dtypes and column order."""
    for col in COMMON_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df["source_dataset"] = name
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str)
    df = df[COMMON_COLUMNS].reset_index(drop=True)
    assert df["review_id"].is_unique, f"{name}: review_id must be unique"
    return df


# ----------------------------------------------------------------------------- Salminen et al. 2022

def load_salminen(raw_dir: Path) -> pd.DataFrame:
    """Fake Reviews Dataset (Salminen et al., 2022): 40,432 Amazon reviews.

    Verified file layout (data/raw/salminen/fake reviews dataset.csv):
        category, rating, label, text_
    label: 'CG' = computer-generated (GPT-2)  -> 1 (fake)
           'OR' = original human-written      -> 0 (genuine)
    """
    folder = raw_dir / "salminen"
    candidates = sorted(folder.glob("*.csv")) if folder.exists() else []
    if not candidates:
        raise DatasetNotFound(
            f"Salminen dataset not found in {folder}. Run `python -m src.download_data` "
            "or download 'fake reviews dataset.csv' from https://osf.io/tyue9/ (also on Kaggle)."
        )
    raw = pd.read_csv(candidates[0])
    expected = {"category", "rating", "label", "text_"}
    if not expected.issubset(raw.columns):
        raise ValueError(f"Unexpected Salminen columns {list(raw.columns)}; expected {expected}")
    unknown = set(raw["label"].unique()) - {"CG", "OR"}
    if unknown:
        raise ValueError(f"Unexpected Salminen labels: {unknown}")
    df = pd.DataFrame({
        "review_id": [f"salminen_{i}" for i in range(len(raw))],
        "text": raw["text_"].fillna(""),
        "label": (raw["label"] == "CG").astype(int),
        "rating": raw["rating"].astype(float),
        # Categories look like 'Home_and_Kitchen_5' (the _5 is the Amazon 5-core subset)
        "category": raw["category"].str.replace(r"_5$", "", regex=True),
    })
    return _finalise(df, "salminen")


# ----------------------------------------------------------------------------- Ott et al. 2011 / 2013

def load_ott(raw_dir: Path) -> pd.DataFrame:
    """Deceptive Opinion Spam Corpus v1.4 (Ott et al. 2011; negative reviews added in 2013).

    1,600 hotel reviews (20 Chicago hotels) stored as one .txt file each:
        op_spam_v1.4/{positive,negative}_polarity/{deceptive_from_MTurk,truthful_from_*}/foldN/*.txt
    File names look like d_hilton_1.txt (d = deceptive, t = truthful).
    deceptive -> 1 (fake), truthful -> 0 (genuine). There is no star rating, so `rating` is NaN
    and the polarity is stored in `category`.
    """
    root = raw_dir / "ott" / "op_spam_v1.4"
    if not root.exists():
        raise DatasetNotFound(
            f"Ott corpus not found at {root}. Run `python -m src.download_data` or get "
            "op_spam_v1.4.zip from https://myleott.com/op_spam_v1.4.zip and unzip it into data/raw/ott/."
        )
    rows = []
    for path in sorted(root.glob("*_polarity/*/fold*/*.txt")):
        polarity = path.parts[-4].split("_")[0]            # positive | negative
        source_dir = path.parts[-3]                         # deceptive_from_MTurk | truthful_from_...
        hotel = re.sub(r"^[dt]_|_\d+$", "", path.stem)      # d_hilton_1 -> hilton
        rows.append({
            "review_id": f"ott_{polarity}_{path.parts[-2]}_{path.stem}",
            "text": path.read_text(encoding="utf-8", errors="replace").strip(),
            "label": int(source_dir.startswith("deceptive")),
            "category": f"hotel_{polarity}",
            "product_id": hotel,
        })
    if len(rows) != 1600:
        raise ValueError(f"Expected 1600 Ott reviews, found {len(rows)} - is the zip fully extracted?")
    return _finalise(pd.DataFrame(rows), "ott")


# ----------------------------------------------------------------------------- YelpChi (optional)

def load_yelpchi(raw_dir: Path) -> pd.DataFrame:
    """YelpChi (Rayana & Akoglu, KDD 2015) - OPTIONAL, obtained by emailing the authors.

    Expected files (the format the authors distribute for YelpChi/YelpNYC/YelpZip) [verify when
    the data arrives - inspect the real files before trusting this loader]:
        data/raw/yelpchi/metadata        tab-separated: user_id  prod_id  rating  label  date
        data/raw/yelpchi/reviewContent   tab-separated: user_id  prod_id  date    text
    label: -1 = filtered by Yelp (fake) -> 1,  1 = recommended (genuine) -> 0
    """
    folder = raw_dir / "yelpchi"
    meta_path, text_path = folder / "metadata", folder / "reviewContent"
    if not (meta_path.exists() and text_path.exists()):
        raise DatasetNotFound(
            f"YelpChi data not found in {folder} (expected 'metadata' and 'reviewContent'). "
            "YelpChi is optional: email the authors (Rayana & Akoglu) for access. "
            "Every step of this project runs without it."
        )
    meta = pd.read_csv(meta_path, sep="\t", header=None,
                       names=["user_id", "product_id", "rating", "label", "date"])
    text = pd.read_csv(text_path, sep="\t", header=None, quoting=3,
                       names=["user_id", "product_id", "date", "text"])
    if len(meta) != len(text):
        raise ValueError("YelpChi metadata and reviewContent have different lengths [verify format]")
    df = meta.copy()
    df["text"] = text["text"].fillna("").values
    df["label"] = (df["label"] == -1).astype(int)
    df["review_id"] = [f"yelpchi_{i}" for i in range(len(df))]
    df["category"] = "restaurant_or_hotel"
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return _finalise(df, "yelpchi")


# ----------------------------------------------------------------------------- Phase 8 (ours)

def load_modern_fakes(raw_dir: Path) -> pd.DataFrame:
    """Fake reviews from a recent LLM (Phase 8): data/raw/modern_fakes.csv, ALL label 1.

    Required column: text. Optional: category, rating, generator.
    Create it with `python -m src.modern_fakes generate` or by writing reviews yourself.
    """
    path = raw_dir / "modern_fakes.csv"
    if not path.exists():
        raise DatasetNotFound(f"{path} not found - run `python -m src.modern_fakes generate` or create it by hand")
    raw = pd.read_csv(path)
    df = pd.DataFrame({
        "review_id": raw["review_id"] if "review_id" in raw else [f"modern_{i}" for i in range(len(raw))],
        "text": raw["text"].fillna(""),
        "label": 1,
        "rating": raw.get("rating", np.nan),
        "category": raw.get("category", np.nan),
    })
    return _finalise(df, "modern_fakes")


LOADERS = {"salminen": load_salminen, "ott": load_ott, "yelpchi": load_yelpchi, "modern_fakes": load_modern_fakes}


def load_dataset(name: str, raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Load a dataset by name into the common schema."""
    if name not in LOADERS:
        raise KeyError(f"Unknown dataset '{name}'. Choose from {sorted(LOADERS)}")
    return LOADERS[name](resolve_path(raw_dir))


def dataset_available(name: str, raw_dir: str | Path = "data/raw") -> bool:
    """True if the dataset's raw files are present (used to skip optional Yelp steps)."""
    try:
        load_dataset(name, raw_dir)
        return True
    except DatasetNotFound:
        return False
