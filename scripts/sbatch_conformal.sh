#!/bin/bash
#SBATCH --job-name=conformal
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/conformal_%j.out
#SBATCH --error=logs/conformal_%j.err

set -e
cd /gpfs/radev/home/pc838/LLM-Reasoning-Trace-Topology
source /etc/profile.d/modules.sh 2>/dev/null || true
module load PyTorch/2.1.2-foss-2022b

mkdir -p results/month3

# Step 1: Save SuperHybrid OOF
echo "=== Save SuperHybrid OOF ==="
python src/modeling/save_super_hybrid_oof.py

# Step 2: Run conformal coverage analysis on all OOF files
echo ""
echo "=== Run conformal coverage analysis ==="
python src/modeling/conformal_wrapper.py \
    --oof results/month2/deberta_pooled_oof.npz:DeBERTa \
    --oof results/month2/deberta_conditioned_pooled_oof.npz:Cond_DeBERTa \
    --oof results/month3/hidden_probe_pooled_mlp_h_answer_oof.npz:Probe_MLP \
    --oof results/month3/superhybrid_DeBERTa_Cond_oof.npz:DeBERTa_plus_Cond \
    --oof results/month3/superhybrid_ThreeProbs_oof.npz:ThreeProbs \
    --oof results/month3/superhybrid_SuperHybrid_LR_oof.npz:SuperHybrid_LR \
    --oof results/month3/superhybrid_SuperHybrid_RF_oof.npz:SuperHybrid_RF \
    --output results/month3/conformal_coverage.json
echo "DONE $(date)"
