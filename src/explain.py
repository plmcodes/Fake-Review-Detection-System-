"""Explainability for the hybrid model (Model D).

    python -m src.explain                          # uses checkpoints/hybrid__salminen__seed42
    python -m src.explain --run hybrid_xgb__salminen__seed42

SHAP  - which hybrid INPUT FEATURES (latent dims, reconstruction error, style features) push a
        prediction towards "fake". Model-agnostic permutation SHAP on the MLP; TreeSHAP for XGBoost.
        Also aggregated per feature GROUP, which answers "does the autoencoder part matter?".
LIME  - which WORDS of a raw review push the decision: LIME deletes random words, re-scores the
        perturbed texts with the full pipeline (SBERT -> AE -> classifier) and fits a local linear model.

Outputs: results/plots/explain/*.png, *.html, *.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config, resolve_path
from .features import load_bundle
from .models import load_runner
from .utils import get_logger, save_json

log = get_logger("explain")


def feature_group(name: str) -> str:
    if name.startswith("z_"):
        return "latent z (AE encoder)"
    if name == "log_recon_error":
        return "reconstruction error"
    from .features import BEHAVIOUR_COLUMNS

    return "behavioural" if name in BEHAVIOUR_COLUMNS else "style"


def shap_analysis(runner, data, out: Path, n_explain: int = 300, n_background: int = 100, seed: int = 0) -> pd.DataFrame:
    import shap

    from .evaluate import _plt

    plt = _plt()
    rng = np.random.default_rng(seed)
    test, train = data.idx("test"), data.idx("train")
    ex_idx = np.sort(rng.choice(test, size=min(n_explain, len(test)), replace=False))
    bg_idx = np.sort(rng.choice(train, size=min(n_background, len(train)), replace=False))
    X, B = runner.feature_matrix(data, ex_idx), runner.feature_matrix(data, bg_idx)
    names = runner.feature_names()

    if runner.mcfg["classifier"] == "xgb":
        values = shap.TreeExplainer(runner.clf).shap_values(X)           # log-odds units
    else:
        explainer = shap.PermutationExplainer(runner.predict_matrix, shap.maskers.Independent(B), seed=seed)
        values = explainer(X, max_evals=10 * X.shape[1] + 1).values     # probability units (~5 permutations)
    values = np.asarray(values)

    imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(0)})
    imp["group"] = imp["feature"].map(feature_group)
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(out / "shap_feature_importance.csv", index=False)
    groups = imp.groupby("group")["mean_abs_shap"].sum().sort_values(ascending=False)
    groups.to_csv(out / "shap_group_importance.csv")

    shap.summary_plot(values, X, feature_names=names, max_display=20, show=False)
    plt.title("SHAP values - hybrid model (test sample)"); plt.tight_layout()
    plt.savefig(out / "shap_beeswarm.png", dpi=300); plt.close("all")

    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    groups[::-1].plot.barh(ax=ax, color="tab:purple")
    ax.set(xlabel="sum of mean |SHAP| over the group's features", ylabel="",
           title="Which feature group drives the hybrid model?")
    fig.tight_layout(); fig.savefig(out / "shap_group_importance.png", dpi=300); plt.close(fig)
    log.info("SHAP group importance:\n%s", groups.round(4).to_string())
    return imp


def lime_explainer_fn(runner):
    """LIME needs f(list_of_texts) -> [[P(genuine), P(fake)], ...]."""
    def predict(texts):
        p = runner.score_texts(list(texts))["proba"]
        return np.column_stack([1 - p, p])
    return predict


def explain_text(runner, text: str, num_features: int = 10, num_samples: int = 500, seed: int = 0):
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(class_names=["genuine", "fake"], random_state=seed)
    return explainer.explain_instance(text, lime_explainer_fn(runner), num_features=num_features,
                                      num_samples=num_samples, labels=(1,))


def lime_examples(runner, data, out: Path, n_per_class: int = 2, num_samples: int = 500, seed: int = 0) -> list[dict]:
    from .evaluate import _plt

    plt = _plt()
    rng = np.random.default_rng(seed)
    test = data.idx("test")
    rows = []
    for label in (1, 0):
        pool = test[data.y[test] == label]
        for k, i in enumerate(rng.choice(pool, size=n_per_class, replace=False)):
            text = data.df["text"].iloc[i]
            exp = explain_text(runner, text, num_samples=num_samples, seed=seed)
            proba = float(runner.score_texts([text])["proba"][0])
            name = f"lime_{'fake' if label else 'genuine'}_{k}"
            exp.save_to_file(str(out / f"{name}.html"))
            fig = exp.as_pyplot_figure(label=1)
            fig.set_size_inches(7, 3.8)
            plt.title(f"LIME: words pushing towards FAKE (true: {'fake' if label else 'genuine'}, P(fake)={proba:.2f})",
                      fontsize=9)
            fig.tight_layout(); fig.savefig(out / f"{name}.png", dpi=300); plt.close(fig)
            rows.append({"name": name, "review_id": data.df["review_id"].iloc[i], "true_label": int(label),
                         "p_fake": proba, "text": text, "weights": exp.as_list(label=1)})
    save_json(rows, out / "lime_examples.json")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="hybrid__salminen__seed42", help="checkpoint folder name")
    ap.add_argument("--no-lime", action="store_true")
    ap.add_argument("--lime-samples", type=int, default=500)
    ap.add_argument("--shap-n", type=int, default=300)
    args = ap.parse_args(argv)

    cfg = load_config()
    ckpt = resolve_path(cfg["paths"]["checkpoints"]) / args.run
    runner, meta = load_runner(ckpt)
    if meta["config"]["model"]["regime"] != "frozen":
        raise SystemExit("SHAP here explains the frozen-regime feature matrix; pick a frozen hybrid run")
    data = load_bundle(meta["dataset"], meta["config"], sample=meta.get("sample"))
    out = resolve_path(cfg["paths"]["plots"]) / "explain" / args.run
    out.mkdir(parents=True, exist_ok=True)
    shap_analysis(runner, data, out, n_explain=args.shap_n)
    if not args.no_lime:
        lime_examples(runner, data, out, num_samples=args.lime_samples)
    log.info("explanations written to %s", out)


if __name__ == "__main__":
    main()
