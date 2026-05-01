#!/bin/bash
# =============================================================================
# Phase 3 (T4) — Step-Transformer v2 with stronger sentence encoder.
#
# Phase-2's Step-TF used MiniLM-L6-v2 (384-d, distilled). This test rebuilds
# the step-embedding atlas with one of:
#
#   mpnet         sentence-transformers/all-mpnet-base-v2      (768-d)
#   e5-large      intfloat/e5-large-v2                         (1024-d)
#   bge-large     BAAI/bge-large-en-v1.5                       (1024-d)
#
# then retrains the existing StepTransformer (its `--emb-dim` is now
# auto-detected from the npz). We keep the sequence-length, type embedding,
# and transformer hyperparameters identical so the ONLY change is the
# sentence representation. This is the cleanest ablation.
#
# Stage outputs
# -------------
#   Stage A  data/step_embeddings_v2_{tag}/{group}.npz
#   Stage B  results/month3/step_tf_v2_{tag}.json
#            results/month3/step_tf_v2_{tag}_oof.npz
#
# Submit
# ------
#   # Default mpnet:
#   sbatch scripts/sbatch_phase3_step_tf_v2.sh
#
#   # Swap encoder:
#   ENCODER=e5-large sbatch scripts/sbatch_phase3_step_tf_v2.sh
# =============================================================================
#SBATCH --job-name=p3_steptfv2
#SBATCH --partition=gpu_b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=9:00:00
#SBATCH --chdir=/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology
#SBATCH --output=logs/p3_steptfv2_%j.out
#SBATCH --error=logs/p3_steptfv2_%j.err

trap - ERR
set +e

mkdir -p logs results/month3

ENCODER="${ENCODER:-mpnet}"
case "$ENCODER" in
    mpnet)    MODEL_ID="sentence-transformers/all-mpnet-base-v2" ;;
    e5-large) MODEL_ID="intfloat/e5-large-v2" ;;
    bge-large) MODEL_ID="BAAI/bge-large-en-v1.5" ;;
    *)        echo "[FATAL] Unknown ENCODER=$ENCODER"; exit 1 ;;
esac
TAG="$ENCODER"
OUT_DIR="data/step_embeddings_v2_${TAG}"
RESULT_JSON="results/month3/step_tf_v2_${TAG}.json"

echo "=== Phase 3 / T4 Step-Transformer v2  start $(date) on $(hostname) ==="
echo "encoder:     $ENCODER"
echo "model_id:    $MODEL_ID"
echo "out_dir:     $OUT_DIR"
echo "result_json: $RESULT_JSON"

module load miniconda || exit 2
conda activate torch311 || exit 3

export PYTHONPATH=$PWD

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 10
python -c "import sentence_transformers as st; print('sentence_transformers', st.__version__)" || exit 11
nvidia-smi -L | head -3

echo ""
echo "=== Stage A: Build step embeddings with $MODEL_ID ==="
mkdir -p "$OUT_DIR"
python scripts/build_step_embeddings.py \
    --all \
    --out-dir "$OUT_DIR" \
    --model "$MODEL_ID" \
    --batch-size 128
RCA=$?
if [ $RCA -ne 0 ]; then
    echo "[FATAL] build_step_embeddings failed with $RCA"
    exit 20
fi
du -sh "$OUT_DIR"
ls -la "$OUT_DIR"

echo ""
echo "=== Stage B: Train StepTransformer v2 (auto-detect emb_dim) ==="
python src/modeling/step_transformer.py \
    --npz-glob "${OUT_DIR}/*.npz" \
    --output "$RESULT_JSON" \
    --epochs 20 \
    --batch-size 16 \
    --lr 3e-4 \
    --n-splits 5 \
    --seed 42
RCB=$?
if [ $RCB -ne 0 ]; then
    echo "[FATAL] step_transformer failed with $RCB"
    exit 21
fi

echo ""
echo "=== Output summary ==="
ls -la "$RESULT_JSON" "${RESULT_JSON%.json}_oof.npz" 2>/dev/null

echo ""
echo "=== DONE at $(date) ==="
exit 0
