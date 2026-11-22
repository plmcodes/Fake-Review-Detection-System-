# Slides outline (12-15 slides, ~12 minutes + demo)

Numbers on the result slides come from `results/tables/` (copy, never retype by hand). Figures: `report/figures/`.

1. **Title** - Fake Review Detection with Autoencoders and Deep Learning; names, guide, date.
2. **Problem** - fake reviews mislead buyers; GPT-written fakes fool humans (Salminen et al. 2022: humans near chance).
3. **Idea** - autoencoder trained on genuine reviews only → high reconstruction error = suspicious. Two uses:
   anomaly detector (no labels) and hybrid feature extractor (labels). Simple AE diagram.
4. **Related work** - Ott 2011 (hotel fakes), Dong 2018 (AE + decision forest), LSTM-AE spam detection, Salminen 2022.
5. **Data** - Salminen (40,432 Amazon, 50/50 GPT-2 vs. human), Ott (1,600 hotels) for cross-domain.
   EDA figure: `eda/salminen_length_by_class.png` + the punctuation artefact (35% vs. 91%).
6. **Pipeline** - cleaning → TF-IDF / GloVe / SBERT → fixed 70/15/15 split → models → runs.csv → tables.
   Leakage rules in one line each.
7. **Models** - baselines (LR, SVM, NB, CNN, BiLSTM, DistilBERT) and autoencoders A (dense), B (LSTM), C (DAE, VAE).
8. **Proposed hybrid (Model D)** - `figures/hybrid_architecture.png`; frozen vs. joint; MLP vs. XGBoost; out-of-fold errors.
9. **Main results** - `results/tables/main_comparison.md` (F1, ROC-AUC, PR-AUC, mean ± std over 3 seeds) + ROC/PR curves.
10. **Autoencoder finding** - reconstruction-error histograms (`figures/recon_error_hist_ae_dense.png`): what the
    anomaly detectors actually do on GPT-2 fakes, and why.
11. **Ablation** - `results/tables/ablation.md`: which feature group carries the hybrid.
12. **Robustness** - cross-domain to Ott (`figures/cross_domain_roc_auc.png`) and modern-LLM fakes (`modern_fakes.md`).
13. **Explainability** - SHAP feature-group importance + one LIME example (`figures/explain/`).
14. **Demo** - live Streamlit app: paste a genuine and a fake review; show the CSV upload.
15. **Limitations & future work** - GPT-2 artefacts, no Yelp, CPU-only DistilBERT; GPU runs, more generators,
    contrastive/one-class objectives.
