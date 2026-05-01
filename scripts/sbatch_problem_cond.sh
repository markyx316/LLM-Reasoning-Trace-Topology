#!/bin/bash
#SBATCH --job-name=prob_cond
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/probcond_%j.out
#SBATCH --error=logs/probcond_%j.err

echo "=== Job start $(date) on $(hostname) ==="
mkdir -p logs results/month2 || true

cd /gpfs/radev/home/pc838/LLM-Reasoning-Trace-Topology || { echo "cd failed"; exit 1; }
echo "PWD: $PWD"

if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
if [ -f /etc/profile.d/lmod.sh ]; then source /etc/profile.d/lmod.sh; fi
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1 || { echo "module load failed"; exit 2; }

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 3
nvidia-smi -L || true

echo ""
echo "=== Step 0: self-test alignment features ==="
python src/features/problem_alignment.py || exit 10

echo ""
echo "=== Step 1: Build problem-alignment features for all datasets ==="
python scripts/build_problem_alignment.py --all || exit 11
ls -la data/features/*_align.csv 2>/dev/null

echo ""
echo "=== Step 2: Train Problem-Conditioned DeBERTa (pooled, 5-fold) ==="
python src/modeling/deberta_conditioned.py \
    --traces-glob "data/traces/*_traces.jsonl" \
    --output results/month2/deberta_conditioned_pooled.json \
    --epochs 3 --batch-size 8 --lr 2e-5 --problem-budget 128 || exit 12

echo ""
echo "=== Step 3: Extended hybrid with conditioned DeBERTa + alignment ==="
python src/modeling/hybrid.py \
    --deberta-oof      results/month2/deberta_pooled_oof.npz \
    --step-oof         results/month2/step_transformer_pooled_oof.npz \
    --conditioned-oof  results/month2/deberta_conditioned_pooled_oof.npz \
    --features-glob    "data/features/*_features_align.csv" \
    --output           results/month2/hybrid_extended.json \
    --clf all || exit 13

echo ""
echo "=== DONE at $(date) ==="
ls -la results/month2/
