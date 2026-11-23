# Fake Review Detection with Autoencoders and Deep Learning

B.Tech CSE (AI & ML), 7th semester major project.

> Every number in this report is generated from `results/runs.csv` by `python -m src.evaluate`, then copied into
> this file by `python -m src.build_report` (the blocks between `<!-- TABLE:... -->` markers). Nothing is typed
> by hand. Figures are in `report/figures/`. References marked **[verify]** must be checked against the full text
> before submission.

## Abstract

Online shops depend on reviews, and fake reviews written by language models are now cheap to produce and hard for
people to spot. We study two ways of using autoencoders to detect them. (1) As an **anomaly detector**: an
autoencoder is trained only on genuine reviews, and a review it reconstructs poorly is flagged as fake. This needs
no labelled fakes. (2) As a **feature extractor in a hybrid model** (our proposed Model D): the encoder's latent
vector, the reconstruction error and simple writing-style features are combined by a supervised classifier. We
compare both against classical (TF-IDF + Logistic Regression / SVM / Naive Bayes), deep (CNN, BiLSTM on GloVe) and
transformer (DistilBERT) baselines on the Salminen et al. (2022) dataset of 40,432 Amazon reviews (half written by
GPT-2), test cross-domain transfer to the Ott et al. hotel reviews, run an ablation of the hybrid's feature groups,
explain its decisions with SHAP and LIME, and test whether the detectors catch reviews written by a modern
instruction-tuned model. The results and their interpretation are in Chapters 6-9.

## 1. Introduction

### 1.1 Problem
Customers read reviews before they buy, so fake reviews distort competition and mislead buyers. Text generators
make fakes fluent and cheap: Salminen et al. (2022) report that people detect GPT-2 generated reviews at about
chance level **[verify exact figure]**. Automatic detection is therefore needed.

### 1.2 Why autoencoders
Labelled fake reviews are scarce and quickly go out of date when generators change. An autoencoder trained only on
genuine reviews needs no fake examples at all: it learns what genuine reviews look like, and a high reconstruction
error signals "unlike the genuine reviews I was trained on". The same network also yields a compact latent
representation that a supervised classifier can use when labels do exist.

### 1.3 Objectives
1. Build autoencoder anomaly detectors on two input representations: sentence embeddings (Model A, dense AE;
   Model C, denoising and variational variants) and word sequences (Model B, LSTM sequence-to-sequence AE).
2. Propose a hybrid model (Model D) that combines latent features, reconstruction error and style features, and
   compare a frozen encoder with joint fine-tuning, and an MLP with XGBoost.
3. Compare everything with standard supervised baselines using F1, ROC-AUC and PR-AUC over three seeds.
4. Measure which parts of the hybrid matter (ablation), how well models transfer to another domain (Amazon → hotels),
   and whether they still catch fakes from a modern language model.
5. Explain predictions (SHAP, LIME) and deliver a demo application.

### 1.4 Contributions
- A leak-free, reproducible pipeline: one fixed split, train-only fitting of every vectoriser and scaler,
  validation-only threshold selection, out-of-fold reconstruction errors for the hybrid, every result logged.
- The hybrid Model D and its ablation.
- An honest negative-result analysis where autoencoders do not behave as the anomaly-detection assumption predicts
  (Chapter 6).
- A modern-fakes stress test and a Streamlit demo with word-level explanations.

## 2. Literature review

| Work | Approach | Relevance here |
|---|---|---|
| Jindal & Liu (2008) | First large study of opinion spam on Amazon; duplicate reviews as fakes | Problem framing |
| Ott et al. (2011, 2013) | Crowd-sourced deceptive hotel reviews; n-gram + LIWC classifiers reach about 90% accuracy | Cross-domain test set |
| Mukherjee et al. (2013) | Yelp filter labels; behavioural features beat linguistic ones on real data | Motivates behaviour features (Yelp) |
| Rayana & Akoglu (2015) | YelpChi/NYC/Zip; text + behaviour + review network | Optional Yelp data |
| Salminen et al. (2022) | GPT-2 generated Amazon reviews; humans detect them near chance, a fine-tuned RoBERTa detector does much better **[verify exact figures]** | Main dataset |
| Dong et al. (2018) | Neural autoencoder + decision forest for opinion fraud | Closest match to Model D |
| Sakurada & Yairi (2014) | Autoencoder reconstruction error for anomaly detection | Basis of Models A-C |
| LSTM autoencoder for spam reviews (unsupervised) **[verify authors/venue]** | Seq2seq AE trained on genuine reviews | Basis of Model B |
| Favuzzi, Deep-fake review detection, Politecnico di Milano thesis **[verify]** | VAE with attention on Amazon reviews | Basis of the VAE (Model C2) |
| Shahariar et al. **[verify venue/year]** | MLP/CNN/LSTM vs. NB/KNN/SVM on spam reviews | Baseline choice |
| Kim (2014); Hochreiter & Schmidhuber (1997) | Text CNN; LSTM | CNN / BiLSTM baselines |
| Reimers & Gurevych (2019); Pennington et al. (2014) | Sentence-BERT; GloVe | Input representations |
| Sanh et al. (2019) | DistilBERT | Transformer baseline |
| Vincent et al. (2008); Kingma & Welling (2014) | Denoising AE; VAE | Model C variants |

Gaps this project addresses: most autoencoder studies report a single run on one dataset without cross-domain or
generator-shift tests, and seldom separate what the autoencoder contributes from simple style cues. A data-quality
paper at RANLP 2023 **[verify]** also notes that the Amazon data behind Salminen is noisy, which we keep in mind
when interpreting very high scores.

## 3. Datasets and exploratory analysis

| Dataset | Size | Labels | Use |
|---|---|---|---|
| Salminen et al. (2022) | 40,432 Amazon reviews, 20,216 per class, 10 categories | GPT-2 generated (CG) vs. original (OR) | train / validation / test (70/15/15, stratified, fixed seed) |
| Ott et al. (2011, 2013), v1.4 | 1,600 hotel reviews, 800 per class | crowd-written deceptive vs. truthful | cross-domain test only |
| YelpChi (Rayana & Akoglu 2015) | 67,395 reviews | Yelp filter | optional, not available for this run |

Key EDA findings (`results/eda_summary.md`, figures in `report/figures/eda/`):
- Salminen fakes are shorter (median 39 vs. 47 words) and less lexically varied (type-token ratio 0.727 vs. 0.808);
  they use more first-person pronouns and fewer exclamation marks.
- **Generation artefacts:** only 35.2% of fakes end with sentence punctuation (vs. 90.6% of genuine reviews), and some
  fakes join two texts without a space. Models can learn these artefacts instead of "deceptiveness"; this is the main
  threat to validity and the reason for the cross-domain test.
- Ott reviews are about three times longer, and their style differences point in other directions (fakes use more
  exclamation marks there), so a model built on GPT-2 artefacts should not transfer.

## 4. Methodology

### 4.1 Pipeline
Raw text → cleaning (HTML, URLs, whitespace, emoji to words) → three views: lemmatised/stop-word-free text for
TF-IDF, lower-cased token sequences for GloVe models, lightly cleaned raw text for SBERT/DistilBERT → features
cached per dataset. The split (70/15/15, stratified, `split_seed=42`) is saved as review-id lists and never changes;
seeds 42/43/44 only change model initialisation and shuffling.

**Leakage rules.** TF-IDF, vocabularies and every scaler are fit on the training split only. Autoencoders see only
genuine training reviews. Anomaly thresholds are chosen on the validation split (maximum F1) and frozen for test and
for every cross-domain set; no target-domain labels are used.

### 4.2 Baselines
- TF-IDF (1-2 grams, 10k features) + Logistic Regression, linear SVM, multinomial Naive Bayes.
- 1D-CNN (filters 3/4/5) and BiLSTM (masked mean + max pooling) on trainable GloVe 100-d embeddings, 200 tokens.
- DistilBERT fine-tuned on raw text. On the CPU-only laptop it runs as `distilbert_cpu`: a stratified 2,000-review
  training subset, 128 tokens, 2 epochs, one seed. It is reported as its own row and is not comparable with a full run.

### 4.3 Autoencoder anomaly detectors (no labels)
- **Model A - dense AE** on standardised SBERT `all-MiniLM-L6-v2` embeddings (384 → 256 → 64 → z → 64 → 256 → 384),
  MSE loss, Adam, early stopping on genuine validation reconstruction error. Score = per-review MSE.
  Latent size, learning rate, dropout and batch size tuned with Optuna (validation PR-AUC).
- **Model B - LSTM seq2seq AE** on GloVe sequences (first 64 tokens): an LSTM encoder compresses the review into z,
  an LSTM decoder re-generates the tokens; score = mean token cross-entropy.
- **Model C1 - denoising AE:** Gaussian noise on the input during training. **Model C2 - VAE:** reconstruction +
  0.1·KL loss; score = reconstruction error of the mean.

### 4.4 Model D - hybrid (proposed)
![Hybrid architecture](figures/hybrid_architecture.png)

*Figure: Model D. The diagram shows the default 32-d latent vector; Optuna chose 16-d for the tuned frozen hybrid
(`configs/models/tuned/hybrid_salminen.yaml`), while the XGBoost and joint variants use 32-d.*

Inputs: the latent vector z of a Model-A autoencoder (size from Optuna), log reconstruction error, and 13 style
features (characters, words, average word length, sentences, words per sentence, exclamation and question rates,
punctuation ratio, first-person pronoun rate, capitals ratio, ALL-CAPS word rate, type-token ratio, digit ratio).
On Yelp, 6 behavioural features (rating deviation, reviews per day, extreme-rating share, account age, review count,
near-duplicate similarity) would be added; YelpChi was not available for this run.
- **Frozen regime:** the AE is pre-trained on genuine reviews and frozen; an MLP (64-32) or XGBoost classifies.
- **Joint regime:** encoder, decoder and MLP are fine-tuned together with loss = BCE + 0.5 · reconstruction loss on
  genuine rows.
- **Out-of-fold errors:** the AE has seen the genuine training reviews, so their errors are unrealistically low. We
  train 5 extra AEs, each without one fold, to give every genuine training review an error from an AE that never
  saw it. The ablation measures whether this matters.

### 4.5 Evaluation
Precision, recall, F1, ROC-AUC, PR-AUC and confusion matrices on the fixed test split; mean ± standard deviation over
3 seeds. Plots: ROC/PR curves, reconstruction-error histograms, training curves, t-SNE/UMAP of latent spaces.
`python -m src.evaluate --all` reloads every checkpoint and re-scores it to confirm the logged numbers.

## 5. Experimental setup
- Hardware: laptop CPU (Intel i5-1145G7, 16 GB RAM), no GPU. Software: Python 3.11, PyTorch, scikit-learn,
  sentence-transformers, transformers, XGBoost, Optuna, SHAP, LIME, Streamlit (exact versions in `requirements.txt`).
- Optuna: 25 trials for Model A, 20 for Model D, TPE sampler, objective = validation PR-AUC.
- Every run: `python -m src.train --model <name> --seeds 42 43 44`; full sequence in `run_all.sh`.

## 6. Results

### 6.1 Main comparison
<!-- TABLE:main_comparison -->
Test split of salminen (15%, fixed split_seed). Mean ± std over seeds. Generated by `python -m src.evaluate` from results/runs.csv.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Needs labels? | Train time (s) | Seeds | Notes |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression (TF-IDF) | 0.902 ± 0.000 | 0.897 ± 0.000 | 0.900 ± 0.000 | 0.967 ± 0.000 | 0.968 ± 0.000 | Y | 11 | 3 |  |
| Linear SVM (TF-IDF) | 0.911 ± 0.000 | 0.907 ± 0.000 | 0.909 ± 0.000 | 0.971 ± 0.000 | 0.971 ± 0.000 | Y | 11 | 3 |  |
| Multinomial NB (TF-IDF) | 0.888 ± 0.000 | 0.868 ± 0.000 | 0.878 ± 0.000 | 0.955 ± 0.000 | 0.960 ± 0.000 | Y | 10 | 3 |  |
| 1D-CNN (GloVe) | 0.958 ± 0.004 | 0.949 ± 0.007 | 0.953 ± 0.002 | 0.991 ± 0.000 | 0.992 ± 0.000 | Y | 1064 | 3 |  |
| BiLSTM (GloVe) | 0.957 | 0.953 | 0.955 | 0.992 | 0.992 | Y | 10285 | 1 |  |
| DistilBERT (CPU: 2k-train subset, len 128) | 0.893 | 0.977 | 0.933 | 0.985 | 0.984 | Y | 9604 | 1 | trained on stratified subset of 2000 train reviews (CPU budget) |
| A: Dense AE (SBERT) | 0.500 ± 0.000 | 1.000 ± 0.000 | 0.667 ± 0.000 | 0.379 ± 0.002 | 0.447 ± 0.002 | N | 188 | 3 | Optuna-tuned |
| B: LSTM seq2seq AE (GloVe) | 0.500 | 1.000 | 0.667 | 0.107 | 0.322 | N | 16942 | 1 |  |
| C1: Denoising AE (SBERT) | 0.500 ± 0.000 | 1.000 ± 0.000 | 0.667 ± 0.000 | 0.371 ± 0.002 | 0.440 ± 0.002 | N | 246 | 3 |  |
| C2: VAE (SBERT) | 0.500 ± 0.000 | 1.000 ± 0.000 | 0.667 ± 0.000 | 0.372 ± 0.001 | 0.442 ± 0.001 | N | 239 | 3 |  |
| D: Hybrid MLP (frozen) [proposed] | 0.842 ± 0.005 | 0.862 ± 0.007 | 0.852 ± 0.004 | 0.934 ± 0.002 | 0.935 ± 0.003 | Y | 66 | 3 | genuine-train recon errors out-of-fold (k=5); Optuna-tuned |
| D: Hybrid XGBoost (frozen) | 0.856 ± 0.001 | 0.871 ± 0.006 | 0.863 ± 0.003 | 0.944 ± 0.001 | 0.944 ± 0.001 | Y | 727 | 3 | genuine-train recon errors out-of-fold (k=5) |
| D: Hybrid MLP (joint fine-tune) | 0.875 ± 0.006 | 0.865 ± 0.008 | 0.870 ± 0.004 | 0.948 ± 0.001 | 0.949 ± 0.001 | Y | 79 | 3 | joint fine-tuning, lambda_recon=0.5 |
<!-- /TABLE -->

<!-- DISCUSSION:main -->
**Reading the table.** Supervised sequence models are best in-domain: the 1D-CNN (ROC-AUC 0.991 ± 0.000, F1 0.953)
and the BiLSTM (0.992, one seed) are almost perfect on GPT-2 fakes. The TF-IDF linear models follow closely
(SVM 0.971, Logistic Regression 0.967), which shows how much of the task is solvable from word and n-gram
statistics alone. DistilBERT, fine-tuned on only 2,000 reviews for 2 epochs on the CPU, still reaches 0.985; a full
GPU run would very likely match the CNN, but we did not run it, so we make no claim.

The hybrid model D is clearly useful (frozen MLP 0.934, XGBoost 0.944, joint fine-tuning 0.948 ROC-AUC) but it
does **not** beat the simpler supervised baselines in-domain. The autoencoders used as unsupervised anomaly detectors
(A, B, C1, C2) are below chance; Section 6.2 explains why.

Seed variation is tiny for every model (std ≤ 0.007 F1), so the differences between model families are real; with
three seeds the differences inside a family (e.g. hybrid XGBoost vs. joint) are small but consistent.

Figures: `figures/roc_pr_curves.png`, `figures/confusion_matrices.png`, `figures/training_curves.png`.

### 6.2 Autoencoders
<!-- TABLE:autoencoders -->
Test split of salminen (15%, fixed split_seed). Mean ± std over seeds. Generated by `python -m src.evaluate` from results/runs.csv.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Needs labels? | Train time (s) | Seeds | Notes |
|---|---|---|---|---|---|---|---|---|---|
| A: Dense AE (SBERT) | 0.500 ± 0.000 | 1.000 ± 0.000 | 0.667 ± 0.000 | 0.379 ± 0.002 | 0.447 ± 0.002 | N | 188 | 3 | Optuna-tuned |
| B: LSTM seq2seq AE (GloVe) | 0.500 | 1.000 | 0.667 | 0.107 | 0.322 | N | 16942 | 1 |  |
| C1: Denoising AE (SBERT) | 0.500 ± 0.000 | 1.000 ± 0.000 | 0.667 ± 0.000 | 0.371 ± 0.002 | 0.440 ± 0.002 | N | 246 | 3 |  |
| C2: VAE (SBERT) | 0.500 ± 0.000 | 1.000 ± 0.000 | 0.667 ± 0.000 | 0.372 ± 0.001 | 0.442 ± 0.001 | N | 239 | 3 |  |
| D: Hybrid MLP (frozen) [proposed] | 0.842 ± 0.005 | 0.862 ± 0.007 | 0.852 ± 0.004 | 0.934 ± 0.002 | 0.935 ± 0.003 | Y | 66 | 3 | genuine-train recon errors out-of-fold (k=5); Optuna-tuned |
| D: Hybrid XGBoost (frozen) | 0.856 ± 0.001 | 0.871 ± 0.006 | 0.863 ± 0.003 | 0.944 ± 0.001 | 0.944 ± 0.001 | Y | 727 | 3 | genuine-train recon errors out-of-fold (k=5) |
| D: Hybrid MLP (joint fine-tune) | 0.875 ± 0.006 | 0.865 ± 0.008 | 0.870 ± 0.004 | 0.948 ± 0.001 | 0.949 ± 0.001 | Y | 79 | 3 | joint fine-tuning, lambda_recon=0.5 |
<!-- /TABLE -->

<!-- DISCUSSION:autoencoders -->
**The anomaly-detection assumption fails on GPT-2 fakes - in the opposite direction.** All four autoencoders score
*below* chance: ROC-AUC 0.379 (A, Optuna-tuned), 0.371 (C1), 0.372 (C2) and 0.107 (B, LSTM seq2seq). A ROC-AUC below
0.5 means genuine reviews get *higher* reconstruction errors than fakes: the autoencoders, trained only on genuine
reviews, rebuild GPT-2 fakes *more easily*. Because the scores rank the classes the wrong way, the F1-maximising
validation threshold ends up flagging every review (precision 0.500, recall 1.000, F1 0.667 on this balanced set).

Why this happens (consistent with the EDA in Chapter 3):
- GPT-2 fakes are shorter, use a smaller vocabulary (type-token ratio 0.727 vs. 0.808) and repeat generic praise.
  Such "average" reviews sit near the centre of the genuine-review distribution in SBERT space, so a model that has
  learned that distribution reconstructs them well. Genuine reviews are more varied and specific, hence harder.
- The effect is strongest for the LSTM seq2seq AE (0.107). It reconstructs word sequences, so it effectively scores
  how *predictable* the text is, and text sampled from a language model is by construction high-probability text.
  This matches the idea behind machine-text detectors that look for unusually *low* perplexity.
- The validation split shows the same inversion (validation ROC-AUC 0.390 for A, 0.110 for B), so this is a property
  of the data, not a test-set accident.

What we did and did not do: we report the scores as designed ("high error = fake"). Negating a score turns ROC-AUC r
into 1 − r (for B: 1 − 0.107 = 0.893), and the validation split would have justified that choice, but a "too easy to
reconstruct = machine-written" detector is a different hypothesis from the one in the plan, so we leave it as future
work rather than switching after seeing results.

On the human-written Ott fakes the direction flips back: the SBERT autoencoders reach ROC-AUC 0.544-0.549 (Section 8.1),
weakly above chance. Reconstruction error does carry a little "unusual review" signal for human deception, just not for
GPT-2 text.

Figures: `figures/recon_error_hist_*.png`, `figures/latent_*.png`.

## 7. Ablation of the hybrid model
<!-- TABLE:ablation -->
Test split of salminen (15%, fixed split_seed). Mean ± std over seeds. Generated by `python -m src.evaluate` from results/runs.csv.

| Model | Variant | Precision | Recall | F1 | ROC-AUC | PR-AUC | Seeds |
|---|---|---|---|---|---|---|---|
| D: Hybrid MLP (frozen) [proposed] | full (latent + error + style) | 0.842 ± 0.005 | 0.862 ± 0.007 | 0.852 ± 0.004 | 0.934 ± 0.002 | 0.935 ± 0.003 | 3 |
| D: Hybrid MLP (frozen) [proposed] | no_latent | 0.837 ± 0.004 | 0.876 ± 0.003 | 0.856 ± 0.001 | 0.936 ± 0.000 | 0.936 ± 0.000 | 3 |
| D: Hybrid MLP (frozen) [proposed] | no_oof | 0.821 ± 0.006 | 0.868 ± 0.015 | 0.844 ± 0.009 | 0.928 ± 0.006 | 0.928 ± 0.006 | 3 |
| D: Hybrid MLP (frozen) [proposed] | no_recon | 0.841 ± 0.007 | 0.869 ± 0.007 | 0.855 ± 0.005 | 0.936 ± 0.002 | 0.936 ± 0.001 | 3 |
| D: Hybrid MLP (frozen) [proposed] | no_style | 0.677 ± 0.007 | 0.695 ± 0.025 | 0.686 ± 0.011 | 0.753 ± 0.006 | 0.751 ± 0.004 | 3 |
| D: Hybrid MLP (frozen) [proposed] | style_only | 0.837 ± 0.005 | 0.858 ± 0.004 | 0.847 ± 0.001 | 0.931 ± 0.001 | 0.931 ± 0.001 | 3 |
| D: Hybrid XGBoost (frozen) | full (latent + error + style) | 0.856 ± 0.001 | 0.871 ± 0.006 | 0.863 ± 0.003 | 0.944 ± 0.001 | 0.944 ± 0.001 | 3 |
| D: Hybrid MLP (joint fine-tune) | full (latent + error + style) | 0.875 ± 0.006 | 0.865 ± 0.008 | 0.870 ± 0.004 | 0.948 ± 0.001 | 0.949 ± 0.001 | 3 |
<!-- /TABLE -->

<!-- DISCUSSION:ablation -->
**What carries the hybrid?** (frozen MLP, tuned, 3 seeds; numbers from the table above)
- **Style features carry almost all of it.** Style features alone (`style_only`) reach ROC-AUC 0.931, against 0.934
  for the full frozen hybrid. Removing them (`no_style`, i.e. latent z + reconstruction error only) drops ROC-AUC to
  0.753.
- **The autoencoder features are informative but redundant with style.** On their own (`no_style`) they are far
  above chance (0.753), so the latent space does separate GPT-2 fakes from genuine reviews. But once the style
  features are present, removing either the latent vector (`no_latent`, 0.936) or the reconstruction error
  (`no_recon`, 0.936) changes nothing beyond seed noise. With a frozen encoder, the AE adds no information the
  13 style features do not already give.
- **Out-of-fold errors help a little.** Without them (`no_oof`) ROC-AUC falls to 0.928 ± 0.006 (more variance
  across seeds): the classifier over-trusts the unrealistically low errors of genuine training reviews.
- **Fine-tuning the encoder is what makes the AE useful.** The joint regime (0.948) and the XGBoost classifier
  (0.944) both beat the frozen MLP and `style_only`. When the encoder is trained with the labels, z becomes
  discriminative; when it is only trained to reconstruct genuine reviews, it is not.

Conclusion for the proposed model: in this dataset the hybrid's accuracy comes mainly from writing style, and the
autoencoder contributes only when it is fine-tuned jointly. We report this as it is rather than claiming the AE
features drive the result.

## 8. Robustness

### 8.1 Cross-domain: Amazon → hotels (Ott)
<!-- TABLE:cross_domain -->
Drop = in-domain − cross-domain (positive = worse). The threshold is frozen from the Salminen validation split - no target-domain labels are used.

| Model | Target | F1 in-domain | F1 ott | F1 drop | ROC-AUC in-domain | ROC-AUC ott | ROC-AUC drop | PR-AUC in-domain | PR-AUC ott | PR-AUC drop |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression (TF-IDF) | ott | 0.900 | 0.000 ± 0.000 | +0.900 | 0.967 | 0.493 ± 0.000 | +0.474 | 0.968 | 0.499 ± 0.000 | +0.469 |
| Linear SVM (TF-IDF) | ott | 0.909 | 0.000 ± 0.000 | +0.909 | 0.971 | 0.497 ± 0.000 | +0.474 | 0.971 | 0.498 ± 0.000 | +0.473 |
| Multinomial NB (TF-IDF) | ott | 0.878 | 0.000 ± 0.000 | +0.878 | 0.955 | 0.508 ± 0.000 | +0.448 | 0.960 | 0.509 ± 0.000 | +0.452 |
| 1D-CNN (GloVe) | ott | 0.953 | 0.005 ± 0.004 | +0.948 | 0.991 | 0.529 ± 0.009 | +0.462 | 0.992 | 0.532 ± 0.007 | +0.459 |
| BiLSTM (GloVe) | ott | 0.955 | 0.027 | +0.928 | 0.992 | 0.487 | +0.504 | 0.992 | 0.511 | +0.482 |
| DistilBERT (CPU: 2k-train subset, len 128) | ott | 0.933 | 0.089 | +0.844 | 0.985 | 0.613 | +0.372 | 0.984 | 0.600 | +0.384 |
| A: Dense AE (SBERT) | ott | 0.667 | 0.667 ± 0.000 | -0.000 | 0.379 | 0.549 ± 0.003 | -0.170 | 0.447 | 0.528 ± 0.006 | -0.081 |
| B: LSTM seq2seq AE (GloVe) | ott | 0.667 | 0.667 | -0.000 | 0.107 | 0.454 | -0.348 | 0.322 | 0.462 | -0.140 |
| C1: Denoising AE (SBERT) | ott | 0.667 | 0.667 ± 0.000 | -0.000 | 0.371 | 0.544 ± 0.006 | -0.173 | 0.440 | 0.523 ± 0.001 | -0.082 |
| C2: VAE (SBERT) | ott | 0.667 | 0.667 ± 0.000 | -0.000 | 0.372 | 0.547 ± 0.023 | -0.174 | 0.442 | 0.528 ± 0.020 | -0.086 |
| D: Hybrid MLP (frozen) [proposed] | ott | 0.852 | 0.081 ± 0.007 | +0.771 | 0.934 | 0.550 ± 0.010 | +0.384 | 0.935 | 0.552 ± 0.008 | +0.383 |
| D: Hybrid XGBoost (frozen) | ott | 0.863 | 0.066 ± 0.017 | +0.797 | 0.944 | 0.579 ± 0.007 | +0.365 | 0.944 | 0.583 ± 0.003 | +0.361 |
| D: Hybrid MLP (joint fine-tune) | ott | 0.870 | 0.226 ± 0.057 | +0.644 | 0.948 | 0.599 ± 0.003 | +0.349 | 0.949 | 0.602 ± 0.001 | +0.347 |
<!-- /TABLE -->

<!-- DISCUSSION:cross_domain -->
**Nothing transfers well from GPT-2 Amazon fakes to human-written hotel fakes.** Every supervised model trained on
Salminen falls to roughly chance on Ott: TF-IDF models 0.493-0.508, CNN 0.529, BiLSTM 0.487 ROC-AUC, and their F1 is
near 0 because, with the threshold frozen from Salminen, they label almost every hotel review "genuine". This was the
expectation from the EDA: the cues that separate GPT-2 text (missing final punctuation, short repetitive texts) do not
exist in the crowd-written Ott fakes, which are long and fluent and whose style differences point the other way.

The models that transfer *least badly* are the ones that rely less on surface statistics: DistilBERT (0.613 ROC-AUC,
one CPU seed), the jointly fine-tuned hybrid (0.599 ± 0.003) and the XGBoost hybrid (0.579 ± 0.007). The SBERT
autoencoders also sit slightly above chance (0.544-0.549). These are weak signals - none is usable as a hotel-review
detector - but the ordering suggests that sentence-level representations generalise better than n-grams.

### 8.2 Fakes from a modern language model
<!-- TABLE:modern_fakes -->
200 fake reviews generated by `Qwen/Qwen2.5-0.5B-Instruct`; seed-42 models, thresholds frozen from the Salminen validation split. Detection rate = share flagged as fake. Read it together with the share of GENUINE reviews flagged: a model that flags everything 'detects' 100%.

| Model | Variant | GPT-2 fakes recall (Salminen test) | Genuine flagged as fake (Salminen test) | Modern-LLM fakes detection rate |
|---|---|---|---|---|
| Logistic Regression (TF-IDF) | - | 0.897 | 0.097 | 0.015 |
| Linear SVM (TF-IDF) | - | 0.907 | 0.089 | 0.015 |
| Multinomial NB (TF-IDF) | - | 0.868 | 0.110 | 0.035 |
| 1D-CNN (GloVe) | - | 0.956 | 0.045 | 0.015 |
| BiLSTM (GloVe) | - | 0.953 | 0.043 | 0.015 |
| DistilBERT (CPU: 2k-train subset, len 128) | - | 0.977 | 0.116 | 0.075 |
| A: Dense AE (SBERT) | - | 1.000 | 1.000 | 1.000 |
| B: LSTM seq2seq AE (GloVe) | - | 1.000 | 1.000 | 1.000 |
| C1: Denoising AE (SBERT) | - | 1.000 | 1.000 | 1.000 |
| C2: VAE (SBERT) | - | 1.000 | 1.000 | 1.000 |
| D: Hybrid MLP (frozen) [proposed] | - | 0.869 | 0.159 | 0.015 |
| D: Hybrid MLP (frozen) [proposed] | no_latent | 0.873 | 0.164 | 0.000 |
| D: Hybrid MLP (frozen) [proposed] | no_oof | 0.885 | 0.189 | 0.015 |
| D: Hybrid MLP (frozen) [proposed] | no_recon | 0.870 | 0.155 | 0.015 |
| D: Hybrid MLP (frozen) [proposed] | no_style | 0.718 | 0.353 | 0.390 |
| D: Hybrid MLP (frozen) [proposed] | style_only | 0.859 | 0.165 | 0.000 |
| D: Hybrid XGBoost (frozen) | - | 0.868 | 0.145 | 0.000 |
| D: Hybrid MLP (joint fine-tune) | - | 0.866 | 0.117 | 0.020 |
<!-- /TABLE -->

<!-- DISCUSSION:modern_fakes -->
We generated 200 Amazon-style fake reviews with `Qwen/Qwen2.5-0.5B-Instruct` (same 10 categories, varied ratings,
tones and lengths; `data/raw/modern_fakes.csv`) and scored them with every seed-42 model, thresholds unchanged.
(The 1.5B model suggested in the code needs about 6 GB of RAM in float32 on a CPU, more than this laptop had free
alongside the experiments, so we used the 0.5B model of the same family.)

**The supervised detectors almost completely miss them.** Models that catch 87-98% of GPT-2 fakes flag only
1.5-7.5% of the Qwen fakes (TF-IDF models 1.5-3.5%, CNN and BiLSTM 1.5%, DistilBERT 7.5%, hybrid 0-2%). A modern
instruction-tuned model writes complete, fluent, varied reviews without GPT-2's artefacts, so detectors that learned
those artefacts see "genuine".

The autoencoders "detect" 100%, but only because their thresholds flag every review, genuine ones included (they
also flag 100% of genuine Salminen test reviews), so this is not detection. The one informative exception is the
hybrid **without style features** (`no_style`: autoencoder latent + reconstruction error only), which flags 39% of
the modern fakes, while flagging 35% of genuine reviews. Style features help in-domain but tie the hybrid to GPT-2's
habits; the autoencoder's latent space carries over somewhat better to a new generator. This is the strongest argument
in our results for the autoencoder part of the hybrid, and a direction for future work (training on several
generators, or without style features).

Caveat: 200 reviews from one small model; the detection rates are indicative, not precise (one review = 0.5%).

## 9. Explainability
<!-- DISCUSSION:explain -->
**SHAP (which inputs drive the hybrid).** Permutation SHAP on the seed-42 frozen hybrid (300 test reviews,
`figures/explain/hybrid__salminen__seed42/shap_group_importance.png`) gives mean |SHAP| per feature group:
style 0.686, autoencoder latent z 0.168, reconstruction error 0.024. The most important single features are the
type-token ratio (0.231) and review length in characters (0.131) and words (0.068); `log_recon_error` is 8th. This
agrees with the ablation: the hybrid mostly reads writing style (vocabulary richness and length), uses the latent
vector as a secondary signal, and barely uses the reconstruction error.

**LIME (which words drive one decision).** LIME removes random words and re-scores the texts with the full pipeline
(`figures/explain/hybrid__salminen__seed42/lime_*.html`). Two patterns appear in the examples:
- Function words and pronouns ("I", "a", "This", "My") often push towards *fake*. Removing them changes length,
  type-token ratio and first-person rate, i.e. the style features; the EDA found that GPT-2 fakes use more first-person
  pronouns.
- Specific, concrete words push towards *genuine*: "NetGalley", "exchange", "connection", "chemistry", "play"
  (as in "plug and play"). Genuine reviews mention concrete details, the same pattern Ott et al. found for truthful
  hotel reviews.
- The examples also show errors: a short generic genuine movie review ("This was a wonderful movie...") gets
  P(fake) = 0.89, and a fluent GPT-2 book review gets only 0.24. Short, generic texts are where the model is least
  reliable.

Figures: `figures/explain/`.

## 10. Limitations
<!-- DISCUSSION:limitations -->
- **Generator artefacts.** Salminen fakes come from GPT-2 and contain artefacts (35% end with sentence punctuation vs.
  91% of genuine reviews). In-domain scores therefore overstate real-world performance, as the Ott and modern-fakes
  results show.
- **CPU-only compute.** DistilBERT ran on a 2,000-review subset for 2 epochs (one seed); the LSTM-AE (~14 min/epoch)
  and BiLSTM ran with one seed instead of three. Their rows say so in the "Seeds" column. Training times of these
  runs also include periods when the laptop was in standby, so they overstate compute time. `run_all.sh` and the Colab
  notebook run the full 3-seed versions on a GPU.
- **No YelpChi.** Behavioural features are implemented (and skip cleanly) but were not evaluated, so the plan's
  "with vs. without behaviour" ablation is still open.
- **Truncation.** The LSTM-AE sees the first 64 tokens (36% of Salminen reviews and 93% of Ott reviews are longer),
  CNN/BiLSTM 200 tokens, DistilBERT 128 tokens.
- **Small modern-fakes set** (200 reviews, one small generator).
- **Labels.** Salminen labels say "generated by GPT-2", not "deceptive"; real fake reviews are often written by paid
  humans, which is closer to the Ott setting, where every model is weak.

## 11. Future work
- Run the full DistilBERT/RoBERTa baselines on a GPU (Colab notebook included) and YelpChi with behavioural features.
- Train autoencoders on representations that ignore surface artefacts (e.g. punctuation-normalised text), and test on
  fakes from several modern generators rather than one.
- Contrastive or one-class objectives (Deep SVDD) instead of reconstruction error.
- A second annotator-free check: human-written deceptive reviews from other domains (restaurants, apps).

## 12. Demo
`streamlit run app/streamlit_app.py`: paste a review → P(fake) from the hybrid model, the autoencoder's
reconstruction error against its frozen threshold, a verdict and LIME word highlights; or upload a CSV with a `text`
column and download the scores.

## References
- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. KDD.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD.
- Dong, M., Yao, L., Wang, X., Benatallah, B., Huang, C., & Ning, X. (2018). Opinion fraud detection via neural autoencoder decision forest. Pattern Recognition Letters. arXiv:1805.03379. **[verify volume/pages]**
- Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. Science, 313(5786).
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8).
- Jindal, N., & Liu, B. (2008). Opinion spam and analysis. WSDM.
- Kim, Y. (2014). Convolutional neural networks for sentence classification. EMNLP.
- Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. ICLR.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. ICCV.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS.
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold approximation and projection for dimension reduction. arXiv:1802.03426.
- Mukherjee, A., Venkataraman, V., Liu, B., & Glance, N. (2013). What Yelp fake review filter might be doing? ICWSM.
- Ott, M., Choi, Y., Cardie, C., & Hancock, J. T. (2011). Finding deceptive opinion spam by any stretch of the imagination. ACL-HLT.
- Ott, M., Cardie, C., & Hancock, J. T. (2013). Negative deceptive opinion spam. NAACL-HLT.
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. EMNLP.
- Qwen Team (2024). Qwen2.5 technical report. arXiv:2412.15115. **[verify]**
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI technical report.
- Rayana, S., & Akoglu, L. (2015). Collective opinion spam detection: Bridging review networks and metadata. KDD.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. EMNLP-IJCNLP.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. KDD.
- Sakurada, M., & Yairi, T. (2014). Anomaly detection using autoencoders with nonlinear dimensionality reduction. MLSDA workshop. **[verify]**
- Salminen, J., Kandpal, C., Kamel, A. M., Jung, S.-G., & Jansen, B. J. (2022). Creating and detecting fake reviews of online products. Journal of Retailing and Consumer Services, 64, 102771.
- Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT. arXiv:1910.01108.
- Shahariar, G. M., et al. Spam review detection using deep learning. arXiv:2211.01675. **[verify authors/venue/year]**
- Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P.-A. (2008). Extracting and composing robust features with denoising autoencoders. ICML.
- LSTM autoencoder for unsupervised spam review detection. **[verify: authors, venue, year]**
- Favuzzi. Deep-fake review detection. MSc thesis, Politecnico di Milano. **[verify: full name, year]**
- RANLP 2023 paper on the quality of Amazon fake-review data. **[verify: authors, title]**
