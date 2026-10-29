"""Exploratory data analysis. Used by notebooks/01_eda.ipynb, or run directly:

    python -m src.eda --dataset salminen
    python -m src.eda --dataset ott

Every plot is saved to results/plots/eda/ at 300 dpi. Summary numbers go to
results/eda_stats_<dataset>.json so results/eda_summary.md can quote real values.
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

from .config import load_config, resolve_path
from .features import style_features
from .preprocess import prepare_dataset
from .utils import get_logger, save_json

log = get_logger("eda")
LABELS = {0: "genuine", 1: "fake"}
PALETTE = {"genuine": "tab:blue", "fake": "tab:red"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    return plt, sns


def _save(fig, folder, name):
    path = folder / name
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    log.info("saved %s", path.relative_to(resolve_path(".")))
    return path


def top_ngrams(texts, ngram: int, k: int = 20) -> list[tuple[str, int]]:
    counts = Counter()
    for t in texts:
        toks = t.split()
        counts.update(" ".join(toks[i : i + ngram]) for i in range(len(toks) - ngram + 1))
    return counts.most_common(k)


def distinctive_words(df: pd.DataFrame, k: int = 20, min_count: int = 20) -> pd.DataFrame:
    """Words most over-represented in each class (log-odds ratio with add-1 smoothing)."""
    c = {lab: Counter(w for t in df.loc[df["label"] == lab, "text_tfidf"] for w in t.split()) for lab in (0, 1)}
    n = {lab: sum(c[lab].values()) for lab in (0, 1)}
    vocab = [w for w in set(c[0]) | set(c[1]) if c[0][w] + c[1][w] >= min_count]
    lor = {w: np.log((c[1][w] + 1) / (n[1] + 1)) - np.log((c[0][w] + 1) / (n[0] + 1)) for w in vocab}
    s = pd.Series(lor).sort_values()
    return pd.DataFrame({"fake_words": s.index[::-1][:k], "fake_log_odds": s.values[::-1][:k].round(2),
                         "genuine_words": s.index[:k], "genuine_log_odds": s.values[:k].round(2)})


def run_eda(dataset: str, cfg: dict) -> dict:
    plt, sns = _plt()
    folder = resolve_path(cfg["paths"]["plots"]) / "eda"
    folder.mkdir(parents=True, exist_ok=True)
    df, _, _ = prepare_dataset(dataset, cfg)
    df = df.assign(cls=df["label"].map(LABELS))
    df["n_words"] = [len(t.split()) for t in df["text_seq"]]
    style = style_features(df["text"].tolist())
    stats: dict = {"dataset": dataset, "n_reviews": len(df), "class_counts": df["cls"].value_counts().to_dict()}
    p = dataset

    # 1. Class balance
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    sns.countplot(data=df, x="cls", hue="cls", order=["genuine", "fake"], palette=PALETTE, ax=ax, legend=False)
    for cont in ax.containers:
        ax.bar_label(cont)
    ax.set(title=f"Class balance - {dataset}", xlabel="", ylabel="reviews")
    _save(fig, folder, f"{p}_class_balance.png"); plt.close(fig)

    # 2. Review length per class
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    clip = df["n_words"].quantile(0.99)
    sns.histplot(data=df[df["n_words"] <= clip], x="n_words", hue="cls", palette=PALETTE, bins=60,
                 stat="density", common_norm=False, element="step", ax=a1)
    a1.set(title=f"Review length (words, <= 99th pct) - {dataset}", xlabel="words")
    sns.boxplot(data=df, x="cls", y="n_words", hue="cls", order=["genuine", "fake"], palette=PALETTE, ax=a2,
                showfliers=False, legend=False)
    a2.set(title="Length by class (outliers hidden)", xlabel="", ylabel="words")
    _save(fig, folder, f"{p}_length_by_class.png"); plt.close(fig)
    stats["words_median"] = df.groupby("cls")["n_words"].median().to_dict()
    stats["words_mean"] = df.groupby("cls")["n_words"].mean().round(1).to_dict()
    stats["words_max"] = int(df["n_words"].max())
    stats["share_over_64_words"] = float((df["n_words"] > 64).mean().round(3))
    stats["share_over_200_words"] = float((df["n_words"] > 200).mean().round(3))

    # 3. Rating distribution per class (Ott has no ratings)
    if df["rating"].notna().any():
        fig, ax = plt.subplots(figsize=(6, 3.8))
        rt = df.groupby(["cls", "rating"]).size().rename("n").reset_index()
        rt["share"] = rt["n"] / rt.groupby("cls")["n"].transform("sum")
        sns.barplot(data=rt, x="rating", y="share", hue="cls", palette=PALETTE, ax=ax)
        ax.set(title=f"Star rating distribution per class - {dataset}", ylabel="share of class")
        _save(fig, folder, f"{p}_rating_by_class.png"); plt.close(fig)
        stats["rating_share"] = rt.pivot(index="rating", columns="cls", values="share").round(3).to_dict()

    # 4. Category / polarity per class
    fig, ax = plt.subplots(figsize=(8, 4))
    cat = df.groupby(["category", "cls"]).size().unstack(fill_value=0)
    cat.plot.barh(ax=ax, color=[PALETTE[c] for c in cat.columns])
    ax.set(title=f"Reviews per category and class - {dataset}", xlabel="reviews", ylabel="")
    _save(fig, folder, f"{p}_category_by_class.png"); plt.close(fig)
    stats["categories"] = cat.to_dict()

    # 5. Top unigrams / bigrams per class (stopwords removed, lemmatised)
    for n_gram, name in [(1, "unigrams"), (2, "bigrams")]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, lab in zip(axes, (0, 1)):
            top = top_ngrams(df.loc[df["label"] == lab, "text_tfidf"], n_gram)
            words, counts = zip(*top) if top else ([], [])
            ax.barh(list(words)[::-1], list(counts)[::-1], color=PALETTE[LABELS[lab]])
            ax.set(title=f"Top {name} - {LABELS[lab]}", xlabel="count")
            stats[f"top_{name}_{LABELS[lab]}"] = [w for w, _ in top[:10]]
        _save(fig, folder, f"{p}_top_{name}.png"); plt.close(fig)
    dist = distinctive_words(df)
    dist.to_csv(folder / f"{p}_distinctive_words.csv", index=False)
    stats["distinctive_fake"] = dist["fake_words"].tolist()[:10]
    stats["distinctive_genuine"] = dist["genuine_words"].tolist()[:10]

    # 6. Word clouds
    try:
        from wordcloud import WordCloud

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, lab in zip(axes, (0, 1)):
            text = " ".join(df.loc[df["label"] == lab, "text_tfidf"])
            wc = WordCloud(width=900, height=500, background_color="white", random_state=0,
                           colormap="Blues" if lab == 0 else "Reds", collocations=False).generate(text)
            ax.imshow(wc, interpolation="bilinear"); ax.axis("off"); ax.set_title(f"{LABELS[lab]} reviews")
        _save(fig, folder, f"{p}_wordclouds.png"); plt.close(fig)
    except ImportError:
        log.warning("wordcloud not installed - word clouds skipped")

    # 7. Style signals
    show = ["exclaim_rate", "first_person_rate", "caps_ratio", "type_token_ratio",
            "punct_ratio", "avg_word_len", "words_per_sentence", "allcaps_word_rate"]
    st = style.assign(cls=df["cls"].values)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for ax, col in zip(axes.ravel(), show):
        hi = st[col].quantile(0.99)
        sns.boxplot(data=st[st[col] <= hi], x="cls", y=col, hue="cls", order=["genuine", "fake"],
                    palette=PALETTE, ax=ax, showfliers=False, legend=False)
        ax.set(xlabel="", title=col)
    fig.suptitle(f"Style signals by class - {dataset} (values above 99th pct hidden)")
    _save(fig, folder, f"{p}_style_signals.png"); plt.close(fig)
    style_means = st.groupby("cls")[list(style.columns)].mean().T.round(4)
    style_means.to_csv(folder / f"{p}_style_means.csv")
    stats["style_means"] = style_means.to_dict()

    # 8. Exact-duplicate texts (a data-quality check worth reporting)
    dup = df.duplicated("text", keep=False)
    stats["duplicate_text_rows"] = int(dup.sum())
    stats["duplicate_text_rows_by_class"] = df.loc[dup, "cls"].value_counts().to_dict()
    # Does the review end with sentence punctuation? (GPT-2 output may be cut mid-sentence)
    ends = df["text"].str.strip().str[-1:].isin(list(".!?)\"'"))
    stats["share_ending_with_punctuation"] = ends.groupby(df["cls"]).mean().round(3).to_dict()

    save_json(stats, resolve_path(cfg["paths"]["results"]) / f"eda_stats_{dataset}.json")
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="salminen")
    args = ap.parse_args(argv)
    run_eda(args.dataset, load_config())


if __name__ == "__main__":
    main()
