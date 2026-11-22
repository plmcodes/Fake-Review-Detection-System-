# Viva notes

Short answers to the questions an examiner is likely to ask, with the file to open if they want to see the code.
Result numbers are in `results/tables/` and `report/report.md` (Chapters 6-9); quote them from there.

## The idea

**What is an autoencoder?** A network that compresses its input into a small vector z (encoder) and rebuilds the
input from z (decoder). Trained to minimise reconstruction error, it learns the structure of its training data.
`src/models/ae_dense.py` (class `AE`).

**Why train only on genuine reviews?** Then the AE learns "what genuine reviews look like". A review unlike them
(possibly fake) should be rebuilt badly, so a high reconstruction error = anomaly = probably fake. No fake labels are
needed, which matters because labelled fakes are rare and go out of date when generators change.

**How is the threshold chosen without labels?** Training needs no labels, but choosing the cut-off does: we pick the
F1-maximising threshold on the validation split and freeze it for test and cross-domain sets (`src/train.py`,
`best_f1_threshold` in `src/evaluate.py`). In a real deployment a small labelled validation set, or a percentile of
genuine-review errors, would be used.

**Why SBERT and not TF-IDF for the AE?** A 384-d dense sentence embedding is a compact input that captures meaning;
TF-IDF is 10,000-d and sparse, which makes reconstruction error dominated by rare words.

**Why standardise SBERT dimensions?** SBERT vectors are unit-length, so each dimension has tiny variance and a few
dimensions dominate the MSE. Standardising (scaler fit on genuine train rows only) gives every dimension equal weight.

**What do the variants add?** DAE (C1) adds noise to the input, so it must learn robust features rather than copy
the input. VAE (C2) makes the latent space a smooth Gaussian (KL term), which tends to give better-behaved anomaly
scores. LSTM-AE (B) works on word order instead of a sentence embedding.

## The hybrid (Model D)

**What goes in?** Latent z, log reconstruction error, 13 style features (and 6 behavioural features on Yelp).
**Why log of the error?** Errors are skewed; the log makes the distribution closer to normal for the MLP.
**Frozen vs. joint?** Frozen: AE trained on genuine reviews, then fixed; only the classifier learns. Joint: the AE is
fine-tuned together with the classifier (loss = BCE + 0.5 × reconstruction on genuine rows), so z can become more
discriminative. `src/models/hybrid.py`.
**What is the out-of-fold trick?** The AE has seen the genuine training reviews, so their errors are too low compared
with unseen genuine reviews at test time; the classifier would learn a wrong rule ("low error = genuine" is too
strong). We train 5 extra AEs, each without one fold, and give each genuine training review the error from the AE
that did not see it. The `no_oof` ablation row measures the effect.
**Why the AE cache?** The AE part depends only on its own settings, the seed and the data, so the hybrid's variants
and ablation runs reuse one trained AE instead of retraining an identical one. Each AE is seeded explicitly, so the
cached AE is exactly the one that would be retrained.

## Methodology and honesty

**How do you avoid data leakage?** One fixed split saved as review ids; TF-IDF, vocabularies and scalers fit on
train only; AEs see genuine train rows only; thresholds chosen on validation; no oversampling of val/test; Ott is
used only for testing. (`CLAUDE.md` lists the rules; `tests/` checks the splits.)
**Why F1, ROC-AUC and PR-AUC instead of accuracy?** Accuracy hides errors on imbalanced data (Yelp is ~13% fake).
ROC-AUC measures ranking quality independent of the threshold; PR-AUC focuses on the fake class.
**Why three seeds?** One run can be lucky. Mean ± std over seeds 42/43/44 shows whether a difference is stable. The
split never changes, so only model randomness varies.
**Are the numbers reproducible?** Every run appends a row to `results/runs.csv` with its config hash and checkpoint;
`python -m src.evaluate --all` reloads every checkpoint and re-scores it (`results/rescored_metrics.csv`).
**Why is DistilBERT reported as `distilbert_cpu`?** No GPU: it was fine-tuned on a 2,000-review subset for 2 epochs,
so it is a lower bound for what a transformer achieves, not a fair full comparison. The Colab notebook runs the full
version on a free T4.

## Results questions to prepare (answer from the tables)

- Which model is best in-domain and by how much (mean ± std)? Is the hybrid better than its own parts?
- Do the anomaly detectors work on Salminen? If ROC-AUC < 0.5, explain: GPT-2 fakes are shorter, more repetitive and
  more generic, so they sit *closer* to the "average genuine review" and are rebuilt *better* - the anomaly
  assumption fails for this kind of fake. That is a finding, not a bug; we did not flip the score after seeing test
  results.
- What does the ablation say about reconstruction error vs. style features?
- Why does cross-domain performance on Ott drop? (GPT-2 artefacts such as missing final punctuation do not exist in
  human-written hotel fakes; Ott reviews are three times longer; the style differences point the other way.)
- Do the models catch fakes from a modern model (Qwen2.5-0.5B-Instruct)? Modern fakes are fluent and end their
  sentences properly, so artefact-based detectors should miss more of them.

## Limitations to state before being asked
- Salminen fakes come from GPT-2 and contain artefacts (35% end with punctuation vs. 91% of genuine reviews), so
  in-domain scores overstate real-world performance.
- No YelpChi data, so behavioural features were implemented but not evaluated.
- CPU-only: DistilBERT on a subset; LSTM models truncate long reviews (64 / 200 tokens).
- The modern-fakes set is small (200 reviews) and comes from one small model.
