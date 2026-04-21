#!/bin/bash
# run_per_dataset_hybrid.sh
#
# After per-dataset RoBERTa + per-dataset StepTF are both trained, run hybrid.py
# on each dataset INDIVIDUALLY (single-CSV features, single OOFs). Output: one
# hybrid_<ds>.json per dataset, each containing all variants × LR/RF/XGB.
#
# Prereqs:
#   - results/month2_v2/roberta_per_dataset/roberta_<ds>_oof.npz  (from 1.1)
#   - results/month2_v2/step_transformer_<ds>_oof.npz             (already exists)
#   - data/features/<ds>_features_rec.csv                          (already exists)
#
# CPU-only (~10 minutes total for all 8).

set -uo pipefail

PROJECT=/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology
PY=/home/cpsc4770_ym466/.conda/envs/torch311/bin/python

cd "$PROJECT"
export PYTHONPATH="$PROJECT"

OUT_DIR="results/month2_v2/roberta_per_dataset"
mkdir -p "$OUT_DIR" logs

DATASETS=(math500_qwen7b math500_llama8b
          gsm8k_qwen7b gsm8k_llama8b
          gpqa_diamond_qwen7b gpqa_diamond_llama8b
          arc_challenge_qwen7b arc_challenge_llama8b)

echo "=== Per-dataset hybrid ($(date)) ==="

for DS in "${DATASETS[@]}"; do
    HYB_OUT="$OUT_DIR/hybrid_${DS}.json"
    ROB_OOF="$OUT_DIR/roberta_${DS}_oof.npz"
    STP_OOF="results/month2_v2/step_transformer_${DS}_oof.npz"
    FEAT="data/features/${DS}_features_rec.csv"

    # Skip if output already exists
    if [ -f "$HYB_OUT" ]; then
        echo "$(date +%T) >>> $DS  (hybrid exists, skip)"
        continue
    fi

    # Verify prereqs
    missing=""
    [ -f "$ROB_OOF" ] || missing="$missing roberta_oof"
    [ -f "$STP_OOF" ] || missing="$missing steptf_oof"
    [ -f "$FEAT" ]    || missing="$missing features"
    if [ -n "$missing" ]; then
        echo "$(date +%T) >>> $DS  (MISSING:$missing — skip)"
        continue
    fi

    echo "$(date +%T) >>> $DS"
    "$PY" src/modeling/hybrid.py \
        --deberta-oof   "$ROB_OOF" \
        --step-oof      "$STP_OOF" \
        --features-glob "$FEAT" \
        --output        "$HYB_OUT" \
        --clf all --seed 42 \
        > "logs/hybrid_${DS}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "$(date +%T) >>> FAIL $DS (rc=$rc; see logs/hybrid_${DS}.log)"
    fi
done

echo ""
echo "=== Final inventory ==="
ls -la "$OUT_DIR"/hybrid_*.json
echo ""
echo "$(date +%T) >>> done"
