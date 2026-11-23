"""Metrics, thresholds, tables and plots.

    python -m src.evaluate --all            rescore every checkpoint, then rebuild all tables + plots
    python -m src.evaluate --tables --plots rebuild tables/plots from runs.csv + saved predictions only

Every table is GENERATED from results/runs.csv (never typed by hand). If a model has not been
run yet, it simply does not appear (or shows TBD) - nothing is estimated.

Metric notes (also for the viva):
  * PR-AUC is computed as average precision (sklearn `average_precision_score`).
  * Primary metrics: F1, ROC-AUC, PR-AUC. Accuracy is secondary because Yelp is imbalanced
    (87% genuine), where predicting "genuine" for everything already gives 87% accuracy.
  * Anomaly detectors: threshold = F1-maximising value on the VALIDATION split, frozen for test.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             precision_recall_curve, precision_recall_fscore_support, roc_auc_score, roc_curve)

from .config import load_config, resolve_path
from .utils import get_logger, load_json

log = get_logger(__name__)

MAIN_DATASET = "salminen"
FAMILY_ORDER = ["classical", "deep", "transformer", "autoencoder", "hybrid"]
MODEL_ORDER = ["logreg", "svm", "nb", "cnn", "bilstm", "distilbert_cpu", "distilbert", "roberta",
               "ae_dense", "lstm_ae", "dae", "vae", "hybrid", "hybrid_xgb", "hybrid_joint"]
PRETTY = {
    "logreg": "Logistic Regression (TF-IDF)", "svm": "Linear SVM (TF-IDF)", "nb": "Multinomial NB (TF-IDF)",
    "cnn": "1D-CNN (GloVe)", "bilstm": "BiLSTM (GloVe)", "distilbert": "DistilBERT (fine-tuned)",
    "distilbert_cpu": "DistilBERT (CPU: 2k-train subset, len 128)", "roberta": "RoBERTa (fine-tuned)",
    "ae_dense": "A: Dense AE (SBERT)", "lstm_ae": "B: LSTM seq2seq AE (GloVe)",
    "dae": "C1: Denoising AE (SBERT)", "vae": "C2: VAE (SBERT)",
    "hybrid": "D: Hybrid MLP (frozen) [proposed]", "hybrid_xgb": "D: Hybrid XGBoost (frozen)",
    "hybrid_joint": "D: Hybrid MLP (joint fine-tune)",
}
AE_MODELS = ["ae_dense", "lstm_ae", "dae", "vae"]


# ============================================================================ metrics

def best_f1_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Threshold that maximises F1 on (y, scores) - called on the VALIDATION split only."""
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[int(np.argmax(f1[:-1]))])   # last P/R point has no threshold


def compute_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(scores) >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    both = len(np.unique(y)) == 2
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, scores)) if both else float("nan"),
        "pr_auc": float(average_precision_score(y, scores)) if both else float("nan"),
        "threshold": float(threshold), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def predictions_path(cfg: dict, run_id: str, eval_dataset: str, split: str) -> Path:
    return resolve_path(cfg["paths"]["predictions"]) / run_id / f"{eval_dataset}_{split}.csv"


def save_predictions(cfg, run_id, eval_dataset, split, review_ids, y, scores, threshold) -> Path:
    path = predictions_path(cfg, run_id, eval_dataset, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"review_id": review_ids, "label": y, "score": scores,
                  "pred": (np.asarray(scores) >= threshold).astype(int)}).to_csv(path, index=False)
    return path


# ============================================================================ runs.csv -> tables

def load_runs(cfg: dict) -> pd.DataFrame:
    path = resolve_path(cfg["paths"]["runs_csv"])
    if not path.exists():
        return pd.DataFrame()
    runs = pd.read_csv(path, dtype={"tag": str, "sample": str, "notes": str})
    for col in ["tag", "sample", "notes"]:
        runs[col] = runs[col].fillna("")
    # If a run was repeated, keep only its latest result.
    return runs.sort_values("timestamp").drop_duplicates(["run_id", "eval_dataset", "split"], keep="last")


def _fmt(values: pd.Series, digits: int = 3) -> str:
    values = values.dropna()
    if values.empty:
        return "TBD"
    if len(values) == 1:
        return f"{values.iloc[0]:.{digits}f}"
    return f"{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}"


def _order_key(model: str) -> int:
    return MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)


def summarise(runs: pd.DataFrame, group_cols=("model",)) -> pd.DataFrame:
    """mean ± std over seeds for every metric, one row per group."""
    rows = []
    for key, g in runs.groupby(list(group_cols), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key))
        row.update({
            "Model": PRETTY.get(row.get("model"), row.get("model")),
            "Seeds": int(g["seed"].nunique()),
            "Precision": _fmt(g["precision"]), "Recall": _fmt(g["recall"]), "F1": _fmt(g["f1"]),
            "ROC-AUC": _fmt(g["roc_auc"]), "PR-AUC": _fmt(g["pr_auc"]), "Accuracy": _fmt(g["accuracy"]),
            "Needs labels?": "Y" if str(g["needs_labels"].iloc[0]).lower() in {"true", "1", "y"} else "N",
            "Train time (s)": f"{g['train_time_s'].mean():.0f}",
            "Notes": "; ".join(sorted(set(n for n in g["notes"] if n))),
        })
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty and "model" in out:
        out = out.iloc[sorted(range(len(out)), key=lambda i: (_order_key(out["model"].iloc[i]), str(out.iloc[i].get("tag", ""))))]
    return out.reset_index(drop=True)


def to_markdown(df: pd.DataFrame, title: str = "", note: str = "") -> str:
    if df.empty:
        return f"## {title}\n\nTBD - no runs logged yet.\n"
    cols = list(df.columns)
    lines = [f"## {title}", ""] if title else []
    if note:
        lines += [note, ""]
    lines += ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, name: str, cfg: dict, title: str, note: str = "") -> None:
    folder = resolve_path(cfg["paths"]["tables"])
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / f"{name}.csv", index=False)
    (folder / f"{name}.md").write_text(to_markdown(df, title, note), encoding="utf-8")
    log.info("wrote %s/%s.{csv,md}", folder.name, name)


def build_tables(cfg: dict) -> None:
    runs = load_runs(cfg)
    if runs.empty:
        log.warning("results/runs.csv is empty - nothing to tabulate")
        return
    full = runs[(runs["sample"] == "") & (runs["train_dataset"] == MAIN_DATASET)]
    in_domain = full[(full["eval_dataset"] == MAIN_DATASET) & (full["split"] == "test")]
    main = in_domain[in_domain["tag"] == ""]
    cols = ["Model", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Needs labels?", "Train time (s)", "Seeds", "Notes"]
    note = (f"Test split of {MAIN_DATASET} (15%, fixed split_seed). Mean ± std over seeds. "
            "Generated by `python -m src.evaluate` from results/runs.csv.")

    summary = summarise(main)
    if not summary.empty:
        write_table(summary[cols], "main_comparison", cfg, "Main comparison (all models)", note)
        base = summary[summary["model"].isin(["logreg", "svm", "nb", "cnn", "bilstm", "distilbert", "distilbert_cpu", "roberta"])]
        write_table(base[cols], "baselines", cfg, "Baselines", note)
        ae = summary[summary["model"].isin(AE_MODELS + ["hybrid", "hybrid_xgb", "hybrid_joint"])]
        write_table(ae[cols], "autoencoders", cfg, "Autoencoder models (A-D)", note)

    # Ablation: every hybrid-family run (tags mark the switched-off feature group).
    abl = in_domain[in_domain["model"].str.startswith("hybrid")]
    if not abl.empty:
        s = summarise(abl, ("model", "tag"))
        s.insert(1, "Variant", [t or "full (latent + error + style)" for t in s["tag"]])
        write_table(s[["Model", "Variant", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Seeds"]],
                    "ablation", cfg, "Ablation study (Model D)", note)

    # Cross-domain: train on Salminen (Amazon), test on Ott (hotels) / Yelp.
    cross = full[(full["split"] == "test") & (full["tag"] == "")]
    rows = []
    for model, g in cross.groupby("model"):
        src = g[g["eval_dataset"] == MAIN_DATASET]
        # modern_fakes is all-fake (only recall means anything there) and has its own table
        for target in sorted(set(g["eval_dataset"]) - {MAIN_DATASET, "modern_fakes"}):
            tgt = g[g["eval_dataset"] == target]
            row = {"model": model, "Model": PRETTY.get(model, model), "Target": target}
            for metric, label in [("f1", "F1"), ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]:
                a, b = src[metric].mean(), tgt[metric].mean()
                row[f"{label} in-domain"] = f"{a:.3f}" if pd.notna(a) else "TBD"
                row[f"{label} {target}"] = _fmt(tgt[metric])
                row[f"{label} drop"] = f"{a - b:+.3f}" if pd.notna(a) and pd.notna(b) else "TBD"
            rows.append(row)
    if rows:
        cd = pd.DataFrame(rows)
        cd = cd.iloc[sorted(range(len(cd)), key=lambda i: _order_key(cd["model"].iloc[i]))].drop(columns="model")
        write_table(cd, "cross_domain", cfg, "Cross-domain: train on Salminen (Amazon) → test elsewhere",
                    "Drop = in-domain − cross-domain (positive = worse). The threshold is frozen from the Salminen "
                    "validation split - no target-domain labels are used.")

    # Smoke tests on --sample subsets are listed separately so they never mix with real results.
    smoke = runs[(runs["sample"] != "") & (runs["split"] == "test") & (runs["eval_dataset"] == runs["train_dataset"])]
    if not smoke.empty:
        s = summarise(smoke, ("model", "sample"))
        write_table(s[["Model", "sample", "F1", "ROC-AUC", "PR-AUC"]], "smoke_tests", cfg,
                    "CPU smoke tests (--sample) - NOT results", "Tiny subsets, only proves the code runs.")


# ============================================================================ plots

def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    return plt


def _reference_runs(cfg: dict) -> dict[str, dict]:
    """For each model: the full-data, untagged run with seed == cfg['seed'] (one curve per model)."""
    ckpt_root = resolve_path(cfg["paths"]["checkpoints"])
    out = {}
    for meta_path in sorted(ckpt_root.glob("*/meta.json")):
        meta = load_json(meta_path)
        if meta.get("sample") or meta.get("tag") or meta["dataset"] != MAIN_DATASET or meta["seed"] != cfg["seed"]:
            continue
        meta["ckpt_dir"] = str(meta_path.parent)
        out[meta["model"]] = meta
    return dict(sorted(out.items(), key=lambda kv: _order_key(kv[0])))


def _load_preds(cfg, run_id, eval_dataset=MAIN_DATASET, split="test"):
    path = predictions_path(cfg, run_id, eval_dataset, split)
    return pd.read_csv(path) if path.exists() else None


def plot_curves(cfg: dict, refs: dict, out: Path, eval_dataset: str = MAIN_DATASET) -> None:
    plt = _plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    for model, meta in refs.items():
        p = _load_preds(cfg, meta["run_id"], eval_dataset)
        if p is None:
            continue
        fpr, tpr, _ = roc_curve(p["label"], p["score"])
        prec, rec, _ = precision_recall_curve(p["label"], p["score"])
        style = "-" if meta["family"] in {"autoencoder", "hybrid"} else "--"
        ax1.plot(fpr, tpr, style, lw=1.4, label=f"{model} ({roc_auc_score(p['label'], p['score']):.3f})")
        ax2.plot(rec, prec, style, lw=1.4, label=f"{model} ({average_precision_score(p['label'], p['score']):.3f})")
    ax1.plot([0, 1], [0, 1], ":", color="grey")
    ax1.set(xlabel="False positive rate", ylabel="True positive rate", title=f"ROC curves ({eval_dataset} test)")
    ax2.set(xlabel="Recall", ylabel="Precision", title=f"Precision-Recall curves ({eval_dataset} test)")
    ax1.legend(fontsize=7, title="ROC-AUC"); ax2.legend(fontsize=7, title="PR-AUC (AP)")
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)


def plot_confusions(cfg: dict, refs: dict, out: Path) -> None:
    plt = _plt()
    import seaborn as sns

    items = [(m, _load_preds(cfg, meta["run_id"])) for m, meta in refs.items()]
    items = [(m, p) for m, p in items if p is not None]
    if not items:
        return
    ncol = 4
    nrow = int(np.ceil(len(items) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3 * nrow))
    for ax in np.atleast_1d(axes).ravel()[len(items):]:
        ax.axis("off")
    for ax, (model, p) in zip(np.atleast_1d(axes).ravel(), items):
        cm = confusion_matrix(p["label"], p["pred"], labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cbar=False, cmap="Blues", ax=ax,
                    xticklabels=["genuine", "fake"], yticklabels=["genuine", "fake"])
        ax.set(title=model, xlabel="predicted", ylabel="true")
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)


def plot_recon_hists(cfg: dict, refs: dict, folder: Path) -> None:
    plt = _plt()
    for model in AE_MODELS:
        if model not in refs:
            continue
        p = _load_preds(cfg, refs[model]["run_id"])
        if p is None:
            continue
        thr = refs[model]["threshold"]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        bins = np.linspace(p["score"].quantile(0.001), p["score"].quantile(0.995), 80)
        for label, name, color in [(0, "genuine", "tab:blue"), (1, "fake", "tab:red")]:
            ax.hist(p.loc[p["label"] == label, "score"], bins=bins, alpha=0.55, density=True, label=name, color=color)
        ax.axvline(thr, color="black", ls="--", lw=1.2, label=f"threshold (val-F1) = {thr:.3g}")
        ax.set(xlabel="reconstruction error (anomaly score)", ylabel="density",
               title=f"{PRETTY.get(model, model)} - {MAIN_DATASET} test")
        ax.legend(); fig.tight_layout(); fig.savefig(folder / f"recon_error_hist_{model}.png", dpi=300); plt.close(fig)


def plot_training_curves(refs: dict, out: Path) -> None:
    plt = _plt()
    models = [m for m in refs if Path(refs[m]["ckpt_dir"], "history.json").exists()
              and refs[m]["family"] in {"autoencoder", "hybrid", "deep"}]
    if not models:
        return
    ncol = 4
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.8 * nrow))
    for ax in np.atleast_1d(axes).ravel()[len(models):]:
        ax.axis("off")
    for ax, model in zip(np.atleast_1d(axes).ravel(), models):
        hist = pd.DataFrame(load_json(Path(refs[model]["ckpt_dir"]) / "history.json"))
        if "epoch" not in hist:
            ax.axis("off"); continue
        ax.plot(hist["epoch"], hist["train_loss"], label="train loss")
        ax.plot(hist["epoch"], hist["val"], label="val (early-stop)")
        ax.set(title=model, xlabel="epoch"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out, dpi=300); plt.close(fig)


def plot_latent_space(cfg: dict, refs: dict, folder: Path, max_points: int = 4000) -> None:
    """t-SNE and UMAP of the autoencoder latent space on the test split, coloured by label."""
    from sklearn.manifold import TSNE

    from .features import load_bundle
    from .models import load_runner

    plt = _plt()
    for model in ["ae_dense", "vae", "lstm_ae"]:
        if model not in refs:
            continue
        runner, meta = load_runner(refs[model]["ckpt_dir"])
        data = load_bundle(MAIN_DATASET, meta["config"], sample=None)
        idx = data.idx("test")
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(idx, size=min(max_points, len(idx)), replace=False))
        z, y = runner.latent(data, idx), data.y[idx]
        embeds = {"tsne": TSNE(n_components=2, random_state=0, init="pca", perplexity=30).fit_transform(z)}
        try:
            import umap

            embeds["umap"] = umap.UMAP(n_components=2, random_state=0).fit_transform(z)
        except Exception as exc:  # umap is optional at runtime
            log.warning("UMAP skipped: %s", exc)
        for name, xy in embeds.items():
            fig, ax = plt.subplots(figsize=(6, 5.2))
            for label, lname, color in [(0, "genuine", "tab:blue"), (1, "fake", "tab:red")]:
                m = y == label
                ax.scatter(xy[m, 0], xy[m, 1], s=3, alpha=0.5, c=color, label=lname)
            ax.set(title=f"{name.upper()} of {model} latent space ({MAIN_DATASET} test, n={len(idx)})",
                   xticks=[], yticks=[])
            ax.legend(markerscale=4)
            fig.tight_layout(); fig.savefig(folder / f"latent_{name}_{model}.png", dpi=300); plt.close(fig)


def plot_cross_domain(cfg: dict, out: Path) -> None:
    runs = load_runs(cfg)
    if runs.empty:
        return
    full = runs[(runs["sample"] == "") & (runs["tag"] == "") & (runs["split"] == "test") &
                (runs["train_dataset"] == MAIN_DATASET)]
    targets = sorted(set(full["eval_dataset"]) - {MAIN_DATASET})
    if not targets:
        return
    plt = _plt()
    piv = full.groupby(["model", "eval_dataset"])["roc_auc"].mean().unstack()
    piv = piv.loc[sorted(piv.index, key=_order_key)]
    ax = piv[[MAIN_DATASET] + targets].plot.bar(figsize=(11, 4.5), width=0.8)
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set(ylabel="ROC-AUC", xlabel="", title="Cross-domain robustness: trained on Salminen (Amazon)", ylim=(0, 1))
    ax.legend(title="evaluated on")
    plt.xticks(rotation=35, ha="right"); plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()


def build_plots(cfg: dict, latent: bool = True) -> None:
    folder = resolve_path(cfg["paths"]["plots"])
    folder.mkdir(parents=True, exist_ok=True)
    refs = _reference_runs(cfg)
    if not refs:
        log.warning("No full-data checkpoints for seed %s yet - skipping plots", cfg["seed"])
        return
    plot_curves(cfg, refs, folder / "roc_pr_curves.png")
    if any(_load_preds(cfg, m["run_id"], "ott") is not None for m in refs.values()):
        plot_curves(cfg, refs, folder / "roc_pr_curves_ott.png", eval_dataset="ott")
    plot_confusions(cfg, refs, folder / "confusion_matrices.png")
    plot_recon_hists(cfg, refs, folder)
    plot_training_curves(refs, folder / "training_curves.png")
    plot_cross_domain(cfg, folder / "cross_domain_roc_auc.png")
    if latent:
        plot_latent_space(cfg, refs, folder)
    log.info("plots written to %s", folder)


# ============================================================================ rescoring from checkpoints

def rescore_all(cfg: dict, include_samples: bool = False) -> pd.DataFrame:
    """Reload every checkpoint, recompute its predictions and metrics, and compare with runs.csv.

    This is the reproducibility check: the rescored metrics should match the logged ones.
    """
    from .data_loading import dataset_available
    from .features import load_bundle
    from .models import load_runner

    rows = []
    for meta_path in sorted(resolve_path(cfg["paths"]["checkpoints"]).glob("*/meta.json")):
        meta = load_json(meta_path)
        if meta.get("sample") and not include_samples:
            continue
        log.info("rescoring %s", meta["run_id"])
        runner, meta = load_runner(meta_path.parent)
        rcfg = meta["config"]
        targets = [(meta["dataset"], meta.get("sample"), False)]
        targets += [(d, None, True) for d in meta.get("cross_domain", []) if dataset_available(d, rcfg["paths"]["raw"])]
        for ds, sample, as_eval in targets:
            data = load_bundle(ds, rcfg, sample=sample, as_eval_set=as_eval)
            idx = data.idx("test")
            scores = runner.score(data, idx)
            m = compute_metrics(data.y[idx], scores, meta["threshold"])
            save_predictions(cfg, meta["run_id"], ds, "test", data.df["review_id"].iloc[idx].tolist(),
                             data.y[idx], scores, meta["threshold"])
            rows.append({"run_id": meta["run_id"], "eval_dataset": ds, **m})
    rescored = pd.DataFrame(rows)
    if rescored.empty:
        return rescored
    out = resolve_path(cfg["paths"]["results"]) / "rescored_metrics.csv"
    logged = load_runs(cfg)
    if not logged.empty:
        merged = rescored.merge(logged[logged["split"] == "test"][["run_id", "eval_dataset", "f1", "roc_auc"]],
                                on=["run_id", "eval_dataset"], suffixes=("", "_logged"), how="left")
        merged["abs_diff_roc_auc"] = (merged["roc_auc"] - merged["roc_auc_logged"]).abs()
        merged.to_csv(out, index=False)
        log.info("Rescored %d evaluations; max |ROC-AUC difference| vs runs.csv = %.2e",
                 len(merged), merged["abs_diff_roc_auc"].max())
    else:
        rescored.to_csv(out, index=False)
    return rescored


def copy_figures_to_report(cfg: dict) -> None:
    src = resolve_path(cfg["paths"]["plots"])
    dst = resolve_path("report/figures")
    dst.mkdir(parents=True, exist_ok=True)
    for png in src.rglob("*.png"):
        rel = png.relative_to(src)
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(png, dst / rel)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="rescore checkpoints + tables + plots + copy figures")
    ap.add_argument("--rescore", action="store_true", help="reload checkpoints and recompute predictions")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--no-latent", action="store_true", help="skip the (slower) t-SNE/UMAP plots")
    ap.add_argument("--include-samples", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    args = ap.parse_args(argv)
    cfg = load_config(overrides=args.set)
    if args.all or args.rescore:
        rescore_all(cfg, args.include_samples)
    if args.all or args.tables:
        build_tables(cfg)
    if args.all or args.plots:
        build_plots(cfg, latent=not args.no_latent)
    if args.all:
        copy_figures_to_report(cfg)


if __name__ == "__main__":
    main()
