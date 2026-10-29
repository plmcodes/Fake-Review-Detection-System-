"""Tests for the text cleaner, tokeniser and the stratified splitter."""
import numpy as np
import pandas as pd
import pytest

from src.preprocess import light_clean, make_splits, stratified_sample, tfidf_clean, tokenize


def test_light_clean_strips_html_urls_and_whitespace():
    text = "Great&amp;cheap!  <br/>See   https://x.com/abc or www.shop.com\n\nNOW"
    assert light_clean(text) == "Great&cheap! See or NOW"


def test_light_clean_keeps_case_and_punctuation():
    # style features need capitals and "!" - the light cleaner must not remove them
    assert light_clean("AMAZING product!!!") == "AMAZING product!!!"


def test_emoji_modes():
    assert "emoji_smiling_face_with_smiling_eyes" in light_clean("love it 😊", emoji="text")
    assert light_clean("love it 😊", emoji="remove") == "love it"
    with pytest.raises(ValueError):
        light_clean("x", emoji="keep")


def test_tokenize_lowercases_and_splits_punctuation():
    assert tokenize("I don't LIKE it!") == ["i", "don't", "like", "it", "!"]


def test_tfidf_clean_removes_stopwords_and_lemmatises():
    out = tfidf_clean("The batteries were not working and the cables are broken!")
    assert "the" not in out.split()
    assert "battery" in out.split() and "cable" in out.split()
    assert "not" in out.split()          # negations are kept on purpose
    assert "!" not in out


def _toy_df(n=1000, fake_share=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"review_id": [f"r{i}" for i in range(n)],
                         "label": (rng.random(n) < fake_share).astype(int)})


def test_splits_are_disjoint_complete_and_stratified():
    df = _toy_df()
    s = make_splits(df, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)
    all_idx = np.concatenate([s["train"], s["val"], s["test"]])
    assert len(all_idx) == len(df) == len(set(all_idx))                     # no overlap, nothing lost
    assert abs(len(s["train"]) / len(df) - 0.70) < 0.01
    assert abs(len(s["test"]) / len(df) - 0.15) < 0.01
    overall = df["label"].mean()
    for part in s.values():
        assert abs(df["label"].iloc[part].mean() - overall) < 0.02          # stratified


def test_splits_are_deterministic_for_a_seed():
    df = _toy_df()
    a = make_splits(df, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)
    b = make_splits(df, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)
    c = make_splits(df, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=7)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["test"], c["test"])


def test_stratified_sample_keeps_class_ratio():
    df = _toy_df(n=2000)
    s = stratified_sample(df, 500, seed=42)
    assert len(s) == 500
    assert abs(s["label"].mean() - df["label"].mean()) < 0.01
