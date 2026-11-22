"""Train + evaluate one model with one command, and log the result to results/runs.csv.

Examples
    python -m src.train --model ae_dense --dataset salminen --sample 500     # CPU smoke test
    python -m src.train --model hybrid --dataset salminen --seeds 42 43 44   # 3-seed final run
    python -m src.train --model hybrid --dataset salminen --tag no_style --set model.features.style=false
    python -m src.train --model ae_dense --dataset salminen --tuned          # use Optuna's best params

What happens for each seed:
    1. load config, set the seed, load the cached cleaned data + the FIXED split
    2. runner.fit()           (anomaly detectors only ever see genuine training reviews)
    3. threshold: 0.5 for probabilities / 0 for SVM margins, or - for anomaly scores -
       the F1-maximising threshold on the VALIDATION split (then frozen)
    4. score the test split -> metrics -> one row in runs.csv, predictions to results/predictions/
    5. cross-domain: score Ott (and Yelp, if present) with the same frozen threshold
    6. save the checkpoint to checkpoints/<run_id>/ (on Colab this folder lives on Drive)
"""
from __future__ import annotations

import argparse
import shutil

from .config import config_hash, load_config, resolve_path, set_seed
from .data_loading import dataset_available
from .evaluate import best_f1_threshold, compute_metrics, save_predictions
from .features import load_bundle
from .models import build_runner
from .utils import Timer, get_logger, log_run, make_run_id, save_json

log = get_logger("train")


def run_one(model: str, dataset: str, seed: int, sample=None, tag: str = "", overrides=(),
            tuned: bool = False, cross_domain: bool = True) -> dict:
    extra = []
    if tuned:
        tuned_file = resolve_path(f"configs/models/tuned/{model}_{dataset}.yaml")
        if not tuned_file.exists():
            raise FileNotFoundError(f"{tuned_file} not found - run `python -m src.tune --model {model}` first")
        extra.append(tuned_file)
    cfg = load_config(model, overrides, extra)
    cfg["seed"], cfg["sample"] = seed, sample
    set_seed(seed)

    data = load_bundle(dataset, cfg, sample=sample)
    runner = build_runner(cfg, seed)
    run_id = make_run_id(cfg["model"]["name"], dataset, tag, seed, sample)
    ckpt_dir = resolve_path(cfg["paths"]["checkpoints"]) / run_id
    if ckpt_dir.exists():
        # keep a resumable transformer checkpoint; otherwise start clean
        for item in ckpt_dir.iterdir():
            if item.name != "resume.pt":
                shutil.rmtree(item) if item.is_dir() else item.unlink()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    runner.ckpt_dir = ckpt_dir
    log.info("=== %s | %d train / %d val / %d test | seed %d", run_id, len(data.idx("train")),
             len(data.idx("val")), len(data.idx("test")), seed)

    with Timer() as t:
        runner.fit(data)
    log.info("trained in %.1f s", t.seconds)

    val_idx, test_idx = data.idx("val"), data.idx("test")
    val_scores = runner.score(data, val_idx)
    threshold = best_f1_threshold(data.y[val_idx], val_scores) if runner.tune_threshold else runner.default_threshold
    val_m = compute_metrics(data.y[val_idx], val_scores, threshold)
    test_scores = runner.score(data, test_idx)
    test_m = compute_metrics(data.y[test_idx], test_scores, threshold)

    targets = []
    if cross_domain:
        targets = [d for d in cfg["eval"]["cross_domain"] if d != dataset and dataset_available(d, cfg["paths"]["raw"])]

    runner.save(ckpt_dir)
    runner.save_history(ckpt_dir)
    meta = {"run_id": run_id, "model": cfg["model"]["name"], "family": runner.family, "dataset": dataset,
            "sample": sample, "seed": seed, "tag": tag, "threshold": threshold, "config_hash": config_hash(cfg),
            "needs_labels": runner.needs_labels, "train_time_s": t.seconds, "cross_domain": targets,
            "notes": "; ".join(n for n in [getattr(runner, "notes", ""), "Optuna-tuned" if tuned else ""] if n),
            "config": cfg}
    save_json(meta, ckpt_dir / "meta.json")

    base_row = {"run_id": run_id, "model": cfg["model"]["name"], "family": runner.family,
                "needs_labels": runner.needs_labels, "train_dataset": dataset, "tag": tag,
                "config_hash": meta["config_hash"], "seed": seed, "sample": sample or "",
                "n_train": len(data.idx("train")), "train_time_s": t.seconds,
                "checkpoint": str(ckpt_dir.relative_to(resolve_path("."))), "notes": meta["notes"]}
    log_run({**base_row, **test_m, "eval_dataset": dataset, "split": "test", "n_eval": len(test_idx),
             "val_f1": val_m["f1"], "val_roc_auc": val_m["roc_auc"], "val_pr_auc": val_m["pr_auc"]},
            cfg["paths"]["runs_csv"])
    ids = data.df["review_id"]
    save_predictions(cfg, run_id, dataset, "val", ids.iloc[val_idx].tolist(), data.y[val_idx], val_scores, threshold)
    save_predictions(cfg, run_id, dataset, "test", ids.iloc[test_idx].tolist(), data.y[test_idx], test_scores, threshold)
    log.info("TEST  f1=%.4f roc_auc=%.4f pr_auc=%.4f (threshold %.4g)", test_m["f1"], test_m["roc_auc"],
             test_m["pr_auc"], threshold)

    results = {"test": test_m, "val": val_m}
    for target in targets:
        try:
            tdata = load_bundle(target, cfg, as_eval_set=True)
            tidx = tdata.idx("test")
            scores = runner.score(tdata, tidx)
        except ValueError as exc:   # e.g. hybrid trained with behaviour features, target has none
            log.warning("cross-domain %s skipped: %s", target, exc)
            continue
        m = compute_metrics(tdata.y[tidx], scores, threshold)
        log_run({**base_row, **m, "eval_dataset": target, "split": "test", "n_eval": len(tidx)}, cfg["paths"]["runs_csv"])
        save_predictions(cfg, run_id, target, "test", tdata.df["review_id"].iloc[tidx].tolist(), tdata.y[tidx], scores, threshold)
        log.info("CROSS %s  f1=%.4f roc_auc=%.4f pr_auc=%.4f", target, m["f1"], m["roc_auc"], m["pr_auc"])
        results[target] = m
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="name of a file in configs/models/ (e.g. ae_dense)")
    ap.add_argument("--dataset", default="salminen", choices=["salminen", "ott", "yelpchi"])
    ap.add_argument("--sample", type=int, default=None, help="stratified subsample for smoke tests, e.g. 500")
    ap.add_argument("--seed", type=int, default=None, help="single seed (default: config seed)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="several seeds, e.g. 42 43 44")
    ap.add_argument("--tag", default="", help="label for variants/ablations (becomes part of the run id)")
    ap.add_argument("--tuned", action="store_true", help="merge configs/models/tuned/<model>_<dataset>.yaml")
    ap.add_argument("--no-cross", action="store_true", help="skip cross-domain evaluation")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.lr=0.0005")
    args = ap.parse_args(argv)

    default_seed = load_config()["seed"]
    seeds = args.seeds or [args.seed if args.seed is not None else default_seed]
    if not dataset_available(args.dataset):
        raise SystemExit(f"Dataset '{args.dataset}' is not available - see data/README.md")
    for seed in seeds:
        run_one(args.model, args.dataset, seed, args.sample, args.tag, args.set, args.tuned, not args.no_cross)


if __name__ == "__main__":
    main()
