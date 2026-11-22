#!/usr/bin/env bash
# Run every experiment of the project in order, from raw data to final tables and figures.
#
#   bash run_all.sh            # full runs (Colab T4: a few hours; CPU laptop: much longer)
#   bash run_all.sh --smoke    # every model on a 500-review sample, CPU, ~15 minutes
#
# Windows: run from Git Bash; it uses .venv/Scripts/python automatically if it exists.
# Every step appends to results/runs.csv; re-running a step simply logs a newer row.
set -euo pipefail
cd "$(dirname "$0")"

if [[ -x .venv/Scripts/python.exe ]]; then PY=.venv/Scripts/python.exe
elif [[ -x .venv/bin/python ]]; then PY=.venv/bin/python
else PY=python; fi

SEEDS="42 43 44"
HAS_GPU=$($PY -c "import torch; print(int(torch.cuda.is_available()))")

if [[ "${1:-}" == "--smoke" ]]; then
  for m in logreg svm nb cnn bilstm distilbert_cpu ae_dense lstm_ae dae vae hybrid hybrid_xgb hybrid_joint; do
    $PY -m src.train --model "$m" --dataset salminen --sample 500 --set train.epochs=3 model.max_train=200
  done
  $PY -m src.evaluate --tables
  exit 0
fi

echo "== Phase 0-2: tests, data, EDA"
$PY -m pytest
$PY -m src.download_data --glove
$PY -m src.eda --dataset salminen
$PY -m src.eda --dataset ott

echo "== Phase 3: baselines"
for m in logreg svm nb cnn bilstm; do
  $PY -m src.train --model "$m" --dataset salminen --seeds $SEEDS
done
if [[ "$HAS_GPU" == "1" ]]; then
  $PY -m src.train --model distilbert --dataset salminen --seeds $SEEDS
else
  $PY -m src.train --model distilbert_cpu --dataset salminen --seed 42   # CPU: subset, 1 seed
fi

echo "== Phase 5: Optuna tuning (validation split only)"
$PY -m src.tune --model ae_dense --dataset salminen --trials 25
$PY -m src.tune --model hybrid --dataset salminen --trials 20

echo "== Phase 4: autoencoders A-C"
$PY -m src.train --model ae_dense --dataset salminen --seeds $SEEDS --tuned
for m in lstm_ae dae vae; do
  $PY -m src.train --model "$m" --dataset salminen --seeds $SEEDS
done

echo "== Phase 4: hybrid D (+ comparisons)"
$PY -m src.train --model hybrid --dataset salminen --seeds $SEEDS --tuned
$PY -m src.train --model hybrid_xgb --dataset salminen --seeds $SEEDS
$PY -m src.train --model hybrid_joint --dataset salminen --seeds $SEEDS

echo "== Phase 6: ablation of the hybrid (frozen MLP, tuned)"
A="--model hybrid --dataset salminen --seeds $SEEDS --tuned --no-cross"
$PY -m src.train $A --tag no_style   --set model.features.style=false
$PY -m src.train $A --tag no_recon   --set model.features.recon_error=false
$PY -m src.train $A --tag no_latent  --set model.features.latent=false
$PY -m src.train $A --tag style_only --set model.features.latent=false model.features.recon_error=false
$PY -m src.train $A --tag no_oof     --set model.oof_folds=0
# With/without behavioural features needs YelpChi (optional):
if $PY -c "from src.data_loading import dataset_available as d; raise SystemExit(0 if d('yelpchi') else 1)"; then
  $PY -m src.train --model hybrid --dataset yelpchi --seeds $SEEDS
  $PY -m src.train --model hybrid --dataset yelpchi --seeds $SEEDS --tag no_behaviour --set model.features.behaviour=false
fi

echo "== Phase 6: evaluation, figures, explanations"
$PY -m src.diagram
$PY -m src.evaluate --all
$PY -m src.explain --run hybrid__salminen__seed42

echo "== Phase 8: modern-LLM fakes"
if [[ ! -f data/raw/modern_fakes.csv ]]; then
  $PY -m src.modern_fakes generate --n 200
fi
$PY -m src.modern_fakes evaluate --seed 42
$PY -m src.evaluate --tables
echo "Done. Tables: results/tables/  Figures: results/plots/ and report/figures/"
