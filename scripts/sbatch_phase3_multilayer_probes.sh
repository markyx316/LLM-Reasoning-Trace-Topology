#!/bin/bash
# =============================================================================
# Phase 3 (T3) — Multi-layer hidden-state probe bank.
#
# Two-stage pipeline:
#   Stage 1  extract_layer_atlas.py --all          (fresh hidden-state atlas
#                                                   over 8 layers × 4 positions)
#   Stage 2  multi_layer_probe.py                  (OOF probes at 6 variants)
#   Stage 3  (optional) layer_probe_sweep.py       (per-cell AUROC heatmap)
#
# Both stages write OOF .npz artifacts that downstream stackers (super-hybrid,
# ultra-hybrid) can consume. This re-runs the atlas on the current cluster so
# that we have a clean, reproducible artifact set for Phase 3 stacking.
#
# If `data/hidden_atlas/*.npz` already exists and is the desired shape, you
# can submit the `MULTILAYER_STAGE=probes` variant to skip extraction.
#
# Submit
# ------
#   # Full pipeline (extract + probes + sweep):
#   sbatch scripts/sbatch_phase3_multilayer_probes.sh
#
#   # Just re-train probes on existing atlas:
#   MULTILAYER_STAGE=probes sbatch scripts/sbatch_phase3_multilayer_probes.sh
# =============================================================================
#SBATCH --job-name=p3_mlayer
#SBATCH --partition=gpu_b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=9:00:00
#SBATCH --chdir=/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology
#SBATCH --output=logs/p3_mlayer_%j.out
#SBATCH --error=logs/p3_mlayer_%j.err

trap - ERR
set +e

mkdir -p logs data/hidden_atlas results/month3

echo "=== Phase 3 / T3 multi-layer probe bank  start $(date) on $(hostname) ==="

module load miniconda || exit 2
conda activate torch311 || exit 3

export PYTHONPATH=$PWD

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 10
python -c "import transformers; print('transformers', transformers.__version__)" || true
nvidia-smi -L | head -3

STAGE="${MULTILAYER_STAGE:-all}"

if [ "$STAGE" = "all" ] || [ "$STAGE" = "extract" ]; then
    echo ""
    echo "=== Stage 1: Layer-atlas extraction (8 layers × 4 positions) ==="
    python scripts/extract_layer_atlas.py --all --n-layers 8
    RC1=$?
    if [ $RC1 -ne 0 ]; then
        echo "[FATAL] extract_layer_atlas failed with $RC1"
        exit 20
    fi
    ls -la data/hidden_atlas/
    du -sh data/hidden_atlas
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "probes" ]; then
    echo ""
    echo "=== Stage 2: Multi-layer probe OOF training ==="
    python src/modeling/multi_layer_probe.py \
        --npz-glob "data/hidden_atlas/*.npz" \
        --output   results/month3/multi_layer_probe.json
    RC2=$?
    if [ $RC2 -ne 0 ]; then
        echo "[FATAL] multi_layer_probe failed with $RC2"
        exit 21
    fi
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "sweep" ]; then
    echo ""
    echo "=== Stage 3: Per-cell layer-probe sweep (AUROC heatmap) ==="
    python src/modeling/layer_probe_sweep.py \
        --npz-glob "data/hidden_atlas/*.npz" \
        --output   results/month3/layer_atlas.json
    RC3=$?
    if [ $RC3 -ne 0 ]; then
        echo "[WARN] layer_probe_sweep failed with $RC3  (non-fatal)"
    fi
fi

echo ""
echo "=== Output summary ==="
ls -la results/month3/multi_layer_probe* 2>/dev/null | head -20
ls -la results/month3/layer_atlas* 2>/dev/null | head -10

echo ""
echo "=== DONE at $(date) ==="
exit 0
