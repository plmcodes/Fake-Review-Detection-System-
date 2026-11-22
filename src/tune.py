"""Hyperparameter tuning with Optuna (small budget, VALIDATION split only - test is never touched).

    python -m src.tune --model ae_dense --trials 25
    python -m src.tune --model hybrid   --trials 20

Search space (from the project plan): latent size in {16, 32, 64}, learning rate, dropout, batch size.
Objective = validation PR-AUC. For the unsupervised AE this uses validation labels - the same small
labelled validation set that already picks the threshold - but TRAINING still never sees a label.

Output:
    configs/models/tuned/<model>_<dataset>.yaml   best params; use with  python -m src.train ... --tuned
    results/tuning/<model>_<dataset>_trials.csv   every trial (for the report's Experiments chapter)
Tuning trials are NOT final results, so they are not written to runs.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import yaml
from sklearn.metrics import average_precision_score

from .config import deep_update, load_config, resolve_path, set_seed
from .features import load_bundle
from .models import build_runner
from .utils import get_logger

log = get_logger("tune")


def suggest(trial, model: str) -> dict:
    """Return a nested override dict for one trial."""
    latent = trial.suggest_categorical("latent_dim", [16, 32, 64])
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    batch = trial.suggest_categorical("batch_size", [64, 128, 256])
    if model in {"ae_dense", "dae", "vae"}:
        return {"model": {"latent_dim": latent, "dropout": dropout}, "train": {"lr": lr, "batch_size": batch}}
    if model.startswith("hybrid"):
        return {"model": {"ae": {"latent_dim": latent}, "mlp": {"dropout": dropout}},
                "train": {"lr": lr, "batch_size": batch}}
    if model == "lstm_ae":
        return {"model": {"latent_dim": latent, "dropout": dropout}, "train": {"lr": lr, "batch_size": batch}}
    raise ValueError(f"No search space defined for {model}")


def main(argv=None):
    import optuna

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="salminen")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args(argv)

    base = load_config(args.model, args.set)
    seed = base["seed"]
    data = load_bundle(args.dataset, base, sample=args.sample)
    val = data.idx("val")

    def objective(trial):
        cfg = deep_update(base, suggest(trial, args.model))
        set_seed(seed)
        runner = build_runner(cfg, seed)
        runner.fit(data)
        return float(average_precision_score(data.y[val], runner.score(data, val)))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=args.trials,
                   callbacks=[lambda s, t: log.info("trial %d: val PR-AUC %.4f  %s  (best %.4f)",
                                                    t.number, t.value, t.params, s.best_value)])

    best = suggest(optuna.trial.FixedTrial(study.best_params), args.model)
    name = f"{args.model}_{args.dataset}" + (f"_s{args.sample}" if args.sample else "")
    out = resolve_path(f"configs/models/tuned/{name}.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# Optuna best params for {args.model} on {args.dataset} "
              f"({args.trials} trials, objective = validation PR-AUC = {study.best_value:.4f})\n")
    out.write_text(header + yaml.safe_dump(best, sort_keys=False))
    trials_dir = resolve_path("results/tuning")
    trials_dir.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(trials_dir / f"{name}_trials.csv", index=False)
    log.info("best val PR-AUC %.4f with %s -> %s", study.best_value, study.best_params, out)


if __name__ == "__main__":
    main()
