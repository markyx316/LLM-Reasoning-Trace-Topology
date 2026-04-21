#!/bin/bash
#SBATCH --job-name=per_ds
#SBATCH --partition=day
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/perds_%j.out
#SBATCH --error=logs/perds_%j.err

echo "=== Job start $(date) on $(hostname) ==="
mkdir -p logs results/month3 || true
cd /gpfs/radev/home/pc838/LLM-Reasoning-Trace-Topology || exit 1

if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
if [ -f /etc/profile.d/lmod.sh ]; then source /etc/profile.d/lmod.sh; fi
module load PyTorch/2.1.2-foss-2022b || exit 2

export OMP_NUM_THREADS=16

echo ""
echo "=== Per-dataset breakdown ==="
python src/modeling/per_dataset_analysis.py \
    --deberta-oof     results/month2/deberta_pooled_oof.npz \
    --cond-oof        results/month2/deberta_conditioned_pooled_oof.npz \
    --probe-oof       results/month3/hidden_probe_pooled_mlp_h_answer_oof.npz \
    --features-glob   "data/features/*_features_align.csv" \
    --genunc-glob     "data/features/*_features_genunc.csv" \
    --output          results/month3/per_dataset_analysis.json || exit 10

echo ""
echo "=== DONE at $(date) ==="
