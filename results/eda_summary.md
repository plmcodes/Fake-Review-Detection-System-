# EDA summary

Written from the plots in `results/plots/eda/` and the numbers in `results/eda_stats_salminen.json`,
`results/eda_stats_ott.json` and `results/plots/eda/*_style_means.csv` (produced by `python -m src.eda`).

## Salminen et al. (2022), Amazon, 40,432 reviews (main set)

- **Perfectly balanced:** 20,216 fake (GPT-2, label `CG`) and 20,216 genuine (`OR`) (`salminen_class_balance.png`).
  So accuracy is meaningful here, but we still report F1, ROC-AUC and PR-AUC for comparability with imbalanced Yelp.
- **Length:** fakes are shorter. Median 39 vs 47 words, mean 68.8 vs 82.1. The fake length distribution has a
  sharp peak at about 15-25 words (`salminen_length_by_class.png`). Length alone is a weak but real signal.
  Longest review: 441 words. 36.2% of reviews have more than 64 words (relevant to the LSTM-AE's 64-token limit)
  and 9.2% have more than 200 (relevant to the 200-token limit of the CNN/BiLSTM).
- **Ratings:** almost identical per class. 5 stars: 60.9% of fakes vs 60.5% of genuine; 1 star: 5.3% vs 5.4%
  (`salminen_rating_by_class.png`). The generator was conditioned on the original rating, so the rating carries no
  signal, and we do not use it as a feature.
- **Categories:** the 10 Amazon categories are balanced between classes (`salminen_category_by_class.png`).
- **Vocabulary:** the top unigrams and bigrams are largely shared ("great", "love", "book", "would recommend",
  "highly recommend", `salminen_top_*.png`). Fakes over-use generic praise bigrams such as "well developed",
  "well written" and "character well" (book reviews), and repeat rare product codes (e.g. "TRS-38") that GPT-2
  copied and repeated (`salminen_distinctive_words.csv`).
- **Style signals** (`salminen_style_signals.png`, class means from `salminen_style_means.csv`):
  - type-token ratio (vocabulary richness) is lower for fakes: 0.727 vs 0.808. GPT-2 repeats itself.
  - average word length is lower for fakes: 3.85 vs 4.19 characters.
  - first-person pronoun rate is higher for fakes: 0.060 vs 0.046 per word.
  - exclamation marks per word are lower for fakes: 0.0064 vs 0.0117. ALL-CAPS word rate is also lower: 0.0071 vs 0.0104.
    The medians of both are 0 for both classes, which is why their boxplots are flat.
  - caps ratio, punctuation ratio and words per sentence are nearly identical between classes.
- **Generation artefacts (important for the Limitations chapter):**
  - only **35.2%** of fakes end with sentence punctuation vs **90.6%** of genuine reviews. Many GPT-2 outputs stop
    mid-sentence ("...If you don't want your").
  - several fakes join two reviews without a space ("...the purchase.This is a great fan").
  - 71 rows are exact duplicate texts, 69 of them fakes.
  A model can partly learn these artefacts instead of "deceptiveness", which is one reason to test cross-domain on Ott.
  We did **not** add an "ends with punctuation" feature, so as not to build the artefact into the hybrid on purpose.
  The sequence and transformer models can still pick it up from the raw text.

## Ott et al. (2011/2013), hotels, 1,600 reviews (cross-domain test set)

- Balanced: 800 deceptive (written by Mechanical Turk workers) and 800 truthful. Half positive, half negative polarity.
- **Much longer** than Salminen reviews: median 137 (fake) vs 145 (genuine) words, and 92.9% have more than 64 words.
  Models with truncation (LSTM-AE: 64 tokens) see only the beginning of each review.
- **Style differences point in different directions from Salminen.** In Ott, fakes use more first-person pronouns
  (0.066 vs 0.049) and more exclamation marks (0.0082 vs 0.0071), but fewer digits (0.0014 vs 0.0044) and fewer
  sentences. TTR is nearly equal (0.664 vs 0.677), and fakes end with punctuation as often as genuine reviews
  (97% vs 93%).
- Distinctive words match Ott et al.'s own finding that deceptive reviews talk about the setting ("luxury",
  "vacation", "relax", "Chicago's"), while truthful ones mention concrete details ("priceline", "sofa", "fridge",
  numbers, "block").
- Expectation: models that rely on GPT-2 artefacts will **not** transfer well to Ott. This is the cross-domain
  question of Phase 6.
