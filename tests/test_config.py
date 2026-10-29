"""Tests for config loading, seeding and the results/runs.csv logger."""
import csv

import numpy as np
import pytest

from src.config import config_hash, deep_update, load_config, parse_override, set_seed
from src.utils import RUN_COLUMNS, EarlyStopping, log_run


def test_default_config_has_expected_keys():
    cfg = load_config()
    assert cfg["seed"] == 42
    assert abs(sum(cfg["split"].values()) - 1.0) < 1e-9


def test_override_parsing_and_merge():
    assert parse_override("train.lr=0.0005") == {"train": {"lr": 0.0005}}
    cfg = load_config(overrides=["train.lr=0.0005", "sample=500"])
    assert cfg["train"]["lr"] == 0.0005
    assert cfg["sample"] == 500
    assert cfg["train"]["patience"] == 5  # sibling keys survive the merge


def test_deep_update_does_not_mutate_input():
    base = {"a": {"b": 1, "c": 2}}
    merged = deep_update(base, {"a": {"b": 9}})
    assert merged == {"a": {"b": 9, "c": 2}}
    assert base == {"a": {"b": 1, "c": 2}}


def test_config_hash_ignores_seed():
    cfg = load_config()
    other = dict(cfg, seed=7)
    assert config_hash(cfg) == config_hash(other)
    assert config_hash(cfg) != config_hash(deep_update(cfg, {"train": {"lr": 0.1}}))


def test_set_seed_is_reproducible():
    torch = pytest.importorskip("torch")
    set_seed(123)
    a = (np.random.rand(3), torch.rand(3))
    set_seed(123)
    b = (np.random.rand(3), torch.rand(3))
    assert np.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])


def test_log_run_appends_rows_with_fixed_columns(tmp_path):
    path = tmp_path / "runs.csv"
    log_run({"model": "m1", "f1": 0.5}, path)
    log_run({"model": "m2", "roc_auc": 0.9}, path)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert [r["model"] for r in rows] == ["m1", "m2"]
    assert list(rows[0].keys()) == RUN_COLUMNS
    assert rows[1]["f1"] == ""  # missing metric stays blank, never invented


def test_log_run_rejects_unknown_columns(tmp_path):
    with pytest.raises(KeyError):
        log_run({"modle": "typo"}, tmp_path / "runs.csv")


def test_early_stopping():
    es = EarlyStopping(patience=2, mode="min")
    assert not es.step(1.0, 0) and es.improved
    assert not es.step(1.1, 1) and not es.improved
    assert es.step(1.2, 2)  # two bad epochs -> stop
    assert es.best_epoch == 0
