"""Streamlit demo - Fake Review Detection with Autoencoders and Deep Learning.

Run locally (CPU is fine):
    streamlit run app/streamlit_app.py

What it shows for a pasted review:
  * P(fake) from the hybrid model (Model D)
  * the autoencoder reconstruction error (Model A) and where it sits relative to the frozen
    validation threshold and to genuine validation reviews
  * a verdict, and LIME word highlights (which words pushed the decision)
Optional: upload a CSV with a `text` column to score many reviews and download the results.

Models are loaded once per server process (st.cache_resource) from checkpoints/.
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import resolve_path  # noqa: E402

CKPT = resolve_path("checkpoints")
EXAMPLES = {
    "(type your own)": "",
    "Short, generic 5-star": "Love this! Well made, sturdy, and very comfortable. I love it! Very pretty",
    "Detailed human-style review": ("I bought this kettle to replace one that died after six years. It boils a full "
                                    "litre in about three minutes, the lid hinge feels a bit flimsy, and the blue light "
                                    "is brighter than I'd like at night, but for the price it does the job."),
}


def find_checkpoint(prefix: str) -> Path | None:
    """Prefer the full-data seed-42 run; fall back to any run of that model (e.g. a smoke test)."""
    preferred = CKPT / f"{prefix}__salminen__seed42"
    if (preferred / "meta.json").exists():
        return preferred
    found = sorted(p.parent for p in CKPT.glob(f"{prefix}__salminen*/meta.json"))
    return found[0] if found else None


@st.cache_resource(show_spinner="Loading models (first time only)...")
def load_models():
    from src.models import load_runner

    hybrid_dir, ae_dir = find_checkpoint("hybrid"), find_checkpoint("ae_dense")
    if hybrid_dir is None:
        return None
    hybrid, hmeta = load_runner(hybrid_dir)
    ae, ameta, val_genuine = None, None, None
    if ae_dir is not None:
        ae, ameta = load_runner(ae_dir)
        preds = resolve_path(f"results/predictions/{ameta['run_id']}/salminen_val.csv")
        if preds.exists():
            v = pd.read_csv(preds)
            val_genuine = np.sort(v.loc[v["label"] == 0, "score"].to_numpy())
    hybrid.score_texts(["warm-up"])   # loads SBERT now, so the first real click is fast
    return {"hybrid": hybrid, "hmeta": hmeta, "ae": ae, "ameta": ameta, "val_genuine": val_genuine}


def ae_scores(models, texts):
    """Model A reconstruction error for raw texts (same SBERT + scaler path as training)."""
    from src.features import encode_texts
    from src.preprocess import light_clean

    ae = models["ae"]
    clean = [light_clean(t, ae.cfg["preprocess"]["emoji"]) for t in texts]
    X = ae.transform_embeddings(encode_texts(clean, ae.cfg))
    import torch

    with torch.no_grad():
        return ae.net.recon_error(X.to(ae.device)).cpu().numpy()


def highlight(text: str, weights: list[tuple[str, float]]) -> str:
    """Colour words by LIME weight: red pushes towards fake, blue towards genuine."""
    w = {word.lower(): v for word, v in weights}
    peak = max([abs(v) for v in w.values()] + [1e-9])
    out = []
    for token in text.split():
        key = token.strip(".,!?;:\"'()").lower()
        if key in w:
            a = min(abs(w[key]) / peak, 1) * 0.75 + 0.15
            colour = f"rgba(220,40,40,{a:.2f})" if w[key] > 0 else f"rgba(40,90,220,{a:.2f})"
            out.append(f'<span style="background:{colour};padding:1px 3px;border-radius:3px">{html.escape(token)}</span>')
        else:
            out.append(html.escape(token))
    return " ".join(out)


st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️", layout="wide")
st.title("🕵️ Fake Review Detection with Autoencoders")
st.caption("Hybrid model (Model D): SBERT → autoencoder latent + reconstruction error + style features → MLP. "
           "Trained on the Salminen et al. (2022) Amazon dataset, where fakes are GPT-2 generated.")

models = load_models()
if models is None:
    st.error("No trained hybrid checkpoint found in checkpoints/. Train one first:\n\n"
             "`python -m src.train --model hybrid --dataset salminen`")
    st.stop()

with st.sidebar:
    st.subheader("Loaded models")
    st.write(f"Hybrid: `{models['hmeta']['run_id']}`")
    st.write(f"Autoencoder: `{models['ameta']['run_id'] if models['ameta'] else 'not found'}`")
    if models["hmeta"].get("sample"):
        st.warning("These are smoke-test models trained on a tiny sample - predictions are not meaningful.")
    lime_samples = st.slider("LIME perturbation samples", 100, 1000, 300, 100,
                             help="More samples = more stable word weights, but slower on CPU")
    st.markdown("**Limitations:** trained on GPT-2 Amazon fakes; may not transfer to other domains or to "
                "reviews written by modern LLMs or human spammers. Use as a decision aid, not a verdict.")

tab1, tab2 = st.tabs(["Single review", "Batch (CSV)"])

with tab1:
    choice = st.selectbox("Example", list(EXAMPLES))
    text = st.text_area("Paste a product review", value=EXAMPLES[choice], height=150)
    if st.button("Analyse", type="primary") and text.strip():
        out = models["hybrid"].score_texts([text])
        p = float(out["proba"][0])
        verdict = "FAKE (computer-generated style)" if p >= 0.5 else "GENUINE"
        c1, c2, c3 = st.columns(3)
        c1.metric("P(fake) - hybrid", f"{p:.1%}")
        c3.metric("Verdict", verdict)
        if models["ae"] is not None:
            err = float(ae_scores(models, [text])[0])
            thr = models["ameta"]["threshold"]
            c2.metric("AE reconstruction error", f"{err:.3f}", delta=f"{err - thr:+.3f} vs threshold {thr:.3f}",
                      delta_color="inverse")
            if models["val_genuine"] is not None:
                pct = 100 * np.searchsorted(models["val_genuine"], err) / len(models["val_genuine"])
                st.caption(f"Reconstruction error is higher than {pct:.0f}% of genuine validation reviews. "
                           f"Above the threshold ({thr:.3f}) the unsupervised Model A alone would flag it as fake.")
            st.progress(min(err / (2 * thr), 1.0), text="reconstruction error (threshold at the middle)")
        with st.spinner("Computing LIME explanation..."):
            from src.explain import explain_text

            exp = explain_text(models["hybrid"], text, num_features=10, num_samples=lime_samples)
            weights = exp.as_list(label=1)
        st.markdown("**Words that pushed the decision** (red → fake, blue → genuine):")
        st.markdown(f'<div style="line-height:2">{highlight(text, weights)}</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(weights, columns=["word", "LIME weight (+ = towards fake)"]),
                     hide_index=True, use_container_width=True)

with tab2:
    st.write("Upload a CSV with a **`text`** column. Each row is scored by the hybrid model and Model A.")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        col = "text" if "text" in df.columns else st.selectbox("Which column holds the review text?", df.columns)
        texts = df[col].fillna("").astype(str).tolist()
        if len(texts) > 5000:
            st.warning("Only the first 5,000 rows are scored in the demo.")
            texts, df = texts[:5000], df.iloc[:5000]
        with st.spinner(f"Scoring {len(texts)} reviews..."):
            out = models["hybrid"].score_texts(texts)
            df["p_fake"] = out["proba"].round(4)
            df["verdict"] = np.where(df["p_fake"] >= 0.5, "fake", "genuine")
            if models["ae"] is not None:
                df["ae_recon_error"] = ae_scores(models, texts).round(4)
                df["ae_flag"] = np.where(df["ae_recon_error"] >= models["ameta"]["threshold"], "fake", "genuine")
        st.dataframe(df, use_container_width=True)
        st.download_button("Download results CSV", df.to_csv(index=False).encode("utf-8"),
                           "scored_reviews.csv", "text/csv")
