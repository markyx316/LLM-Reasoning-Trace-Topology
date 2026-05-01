#!/bin/bash
#SBATCH --job-name=super_hybrid
#SBATCH --partition=day
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/superh_%j.out
#SBATCH --error=logs/superh_%j.err

echo "=== Job start $(date) on $(hostname) ==="
mkdir -p logs results/month3 || true
cd /gpfs/radev/home/pc838/LLM-Reasoning-Trace-Topology || exit 1

if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
if [ -f /etc/profile.d/lmod.sh ]; then source /etc/profile.d/lmod.sh; fi
module load PyTorch/2.1.2-foss-2022b || exit 2

python -c "import numpy, pandas, sklearn, xgboost; print('ok')" || {
    python -m pip install --user --quiet xgboost
}

export OMP_NUM_THREADS=16

echo ""
echo "=== Super Hybrid stacking (DeBERTa + Cond + Probe + all features) ==="
python src/modeling/super_hybrid.py \
    --deberta-oof     results/month2/deberta_pooled_oof.npz \
    --cond-oof        results/month2/deberta_conditioned_pooled_oof.npz \
    --probe-oof       results/month3/hidden_probe_pooled_mlp_h_answer_oof.npz \
    --features-glob   "data/features/*_features_align.csv" \
    --genunc-glob     "data/features/*_features_genunc.csv" \
    --output          results/month3/super_hybrid.json || exit 10

echo ""
echo "=== DONE at $(date) ==="
ls -la results/month3/
