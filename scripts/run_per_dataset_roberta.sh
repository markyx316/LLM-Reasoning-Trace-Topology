#!/bin/bash
# run_per_dataset_roberta.sh
#
# Train RoBERTa-base 5-fold CV on each of the 8 standard datasets separately.
# Produces results/month2_v2/roberta_per_dataset/roberta_<ds>.json + _oof.npz
# for each dataset. Idempotent — skips datasets whose output already exists.
#
# Why per-dataset: pooled CV averages across 8 datasets and hides the cells
# where structural features may beat the text encoder. Per-dataset gives us
# the right granularity to find the "structure wins" cases.
#
# Compute: ~2 hours total on RTX PRO 6000.
#   gpqa_diamond:  ~3 min/dataset (n=198)
#   math500:       ~10 min/dataset (n=500)
#   arc_challenge: ~20 min/dataset (n=1172)
#   gsm8k:         ~25 min/dataset (n=1319)
#
# Usage (on HPC, inside salloc + tmux + activated env):
#   nohup bash scripts/run_per_dataset_roberta.sh > logs/roberta_per_dataset.log 2>&1 &
#   tail -f logs/roberta_per_dataset.log

set -uo pipefail

PROJECT=/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology
PY=/home/cpsc4770_ym466/.conda/envs/torch311/bin/python

cd "$PROJECT"
export PYTHONPATH="$PROJECT"
export HF_HOME="$PROJECT/.cache/huggingface"

OUT_DIR="results/month2_v2/roberta_per_dataset"
mkdir -p "$OUT_DIR" logs

DATASETS=(math500_qwen7b math500_llama8b
          gsm8k_qwen7b gsm8k_llama8b
          gpqa_diamond_qwen7b gpqa_diamond_llama8b
          arc_challenge_qwen7b arc_challenge_llama8b)

echo "=== Per-dataset RoBERTa training ($(date)) ==="
echo "PROJECT: $PROJECT"
echo "OUTPUT:  $OUT_DIR"

for DS in "${DATASETS[@]}"; do
    OUT="$OUT_DIR/roberta_${DS}.json"
    if [ -f "$OUT" ]; then
        echo ""
        echo "$(date +%T) >>> $DS  (output exists, skip)"
        continue
    fi
    echo ""
    echo "$(date +%T) >>> START $DS"
    "$PY" src/modeling/deberta_baseline.py \
        --model roberta-base \
        --traces-glob "data/traces/${DS}_traces.jsonl" \
        --output "$OUT" \
        --epochs 3 --batch-size 16 --lr 2e-5 --seed 42 \
        > "logs/roberta_${DS}.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "$(date +%T) >>> DONE $DS"
        # Quick AUROC peek
        "$PY" -c "
import json
d = json.load(open('$OUT'))
s = d.get('summary') or d.get('overall') or d
print(f\"  AUROC: {s.get('auroc_mean', float('nan')):.4f} ± {s.get('auroc_std', 0):.4f}\")
"
    else
        echo "$(date +%T) >>> FAIL $DS  (rc=$rc; see logs/roberta_${DS}.log)"
    fi
done

echo ""
echo "=== Final inventory ==="
ls -la "$OUT_DIR"/
echo ""
echo "$(date +%T) >>> all done"
