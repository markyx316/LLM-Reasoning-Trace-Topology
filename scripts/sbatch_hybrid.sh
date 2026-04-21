#!/bin/bash
#SBATCH --job-name=hybrid_cpu
#SBATCH --partition=day
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/hybrid_%j.out
#SBATCH --error=logs/hybrid_%j.err

echo "=== Job start $(date) on $(hostname) ==="
mkdir -p logs || true
cd /gpfs/radev/home/pc838/LLM-Reasoning-Trace-Topology || exit 1

if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
if [ -f /etc/profile.d/lmod.sh ]; then source /etc/profile.d/lmod.sh; fi
module load PyTorch/2.1.2-foss-2022b || exit 2

echo "cores: $(nproc)"
export OMP_NUM_THREADS=16

echo ""
echo "=== Fixed hybrid (merge on item_id+group, n_jobs=-1 for XGB) ==="
python src/modeling/hybrid.py \
    --deberta-oof      results/month2/deberta_pooled_oof.npz \
    --step-oof         results/month2/step_transformer_pooled_oof.npz \
    --conditioned-oof  results/month2/deberta_conditioned_pooled_oof.npz \
    --features-glob    "data/features/*_features_align.csv" \
    --output           results/month2/hybrid_extended_fixed.json \
    --clf all || exit 10

echo ""
echo "=== DONE at $(date) ==="
ls -la results/month2/hybrid_extended_fixed.json
