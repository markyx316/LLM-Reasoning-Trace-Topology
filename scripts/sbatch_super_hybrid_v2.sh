#!/bin/bash
#SBATCH --job-name=sh_v2
#SBATCH --partition=day
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/shv2_%j.out
#SBATCH --error=logs/shv2_%j.err

cd /gpfs/radev/home/pc838/LLM-Reasoning-Trace-Topology || exit 1
source /etc/profile.d/modules.sh 2>/dev/null || true
module load PyTorch/2.1.2-foss-2022b || exit 2
export OMP_NUM_THREADS=16

echo "=== Super-Hybrid v2 (with multi-layer probe) ==="
python src/modeling/super_hybrid.py \
    --deberta-oof     results/month2/deberta_pooled_oof.npz \
    --cond-oof        results/month2/deberta_conditioned_pooled_oof.npz \
    --probe-oof       results/month3/multi_layer_probe_L_spread_6_P_answer_oof.npz \
    --features-glob   "data/features/*_features_align.csv" \
    --genunc-glob     "data/features/*_features_genunc.csv" \
    --output          results/month3/super_hybrid_v2.json || exit 10

echo "DONE $(date)"
