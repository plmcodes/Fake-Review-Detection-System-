"""Copy the two checkpoints the Streamlit demo needs into app/model/, so a fresh clone can run the demo.

    python -m src.export_demo

checkpoints/ is git-ignored (it holds every run, hundreds of MB). The demo only needs the seed-42
hybrid (Model D) and the seed-42 dense autoencoder (Model A), a few MB in total. This script copies
them, plus the autoencoder's genuine validation errors (for the percentile shown in the app) and the
test metrics of both runs from results/runs.csv (shown in the app's sidebar).
"""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from .config import resolve_path
from .utils import get_logger, save_json

log = get_logger("export_demo")

RUNS = {"hybrid": "hybrid__salminen__seed42", "ae_dense": "ae_dense__salminen__seed42"}


def main():
    out = resolve_path("app/model")
    out.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(resolve_path("results/runs.csv"))
    info = {}
    for name, run_id in RUNS.items():
        src = resolve_path(f"checkpoints/{run_id}")
        if not (src / "meta.json").exists():
            raise SystemExit(f"{src} not found - train it first")
        dst = out / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("resume.pt"))
        test = runs[(runs["run_id"] == run_id) & (runs["eval_dataset"] == "salminen") & (runs["split"] == "test")]
        last = test.iloc[-1]
        info[name] = {"run_id": run_id, "test_f1": float(last["f1"]), "test_roc_auc": float(last["roc_auc"]),
                      "test_pr_auc": float(last["pr_auc"])}
        log.info("copied %s -> %s", run_id, dst)

    preds = resolve_path(f"results/predictions/{RUNS['ae_dense']}/salminen_val.csv")
    if preds.exists():
        v = pd.read_csv(preds)
        np.save(out / "ae_val_genuine_errors.npy", np.sort(v.loc[v["label"] == 0, "score"].to_numpy()))
    save_json(info, out / "demo_info.json")
    log.info("demo models written to %s", out)


if __name__ == "__main__":
    main()
