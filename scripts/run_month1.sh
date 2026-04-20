#!/usr/bin/env bash
# run_month1.sh -- one-shot Month-1 pipeline:
#   1) Smoke-test recurrence features (no model download).
#   2) Compute recurrence features for all 8 dataset-model combos.
#   3) Run length-controlled evaluation per dataset and pooled.
#   4) Print summary tables and save JSON results.
#
# Requires: sentence-transformers, sklearn, pandas, numpy, tqdm
#   pip install sentence-transformers scikit-learn pandas numpy tqdm
#
# Usage:
#   bash scripts/run_month1.sh           # full run
#   bash scripts/run_month1.sh --quick   # 50 traces per file (debugging)

set -e

cd "$(dirname "$0")/.."
export PYTHONPATH=.

QUICK_FLAG=""
if [ "$1" == "--quick" ]; then
    QUICK_FLAG="--limit 50"
    echo "[QUICK] Using --limit 50 per file"
fi

echo "=========================================================="
echo "  STEP 0: Self-test recurrence feature module"
echo "=========================================================="
python3 src/features/recurrence_features.py

echo ""
echo "=========================================================="
echo "  STEP 1: Build recurrence features for all datasets"
echo "=========================================================="
python3 scripts/build_recurrence_features.py --all $QUICK_FLAG

echo ""
echo "=========================================================="
echo "  STEP 2: Per-dataset length-controlled evaluation"
echo "=========================================================="
mkdir -p results/month1
python3 src/analysis/length_controlled.py \
    --features data/features/*_features_rec.csv \
    --n-bins 5 --clf rf \
    --output results/month1/lengthctl_per_dataset.json

echo ""
echo "=========================================================="
echo "  STEP 3: Pooled length-controlled evaluation"
echo "=========================================================="
python3 src/analysis/length_controlled.py \
    --features data/features/*_features_rec.csv \
    --pool --n-bins 5 --clf rf \
    --output results/month1/lengthctl_pooled.json

echo ""
echo "=========================================================="
echo "  DONE. Results in results/month1/"
echo "=========================================================="
ls -la results/month1/
