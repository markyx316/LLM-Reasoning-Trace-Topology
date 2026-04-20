#!/bin/bash
#SBATCH --job-name=recurrence_m1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/month1_%j.out
#SBATCH --error=logs/month1_%j.err

# -- Shell setup (robust) --
echo "=== Job start $(date) on $(hostname) ==="
mkdir -p logs results/month1 || true

cd /gpfs/radev/home/pc838/LLM-Reasoning-Trace-Topology || { echo "cd failed"; exit 1; }
echo "PWD: $PWD"

# Source module system; don't die if file missing
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi
if [ -f /etc/profile.d/lmod.sh ]; then
    source /etc/profile.d/lmod.sh
fi

module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1 2>&1 || { echo "module load failed"; exit 2; }

echo "Python: $(which python)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 3
nvidia-smi -L || true

echo ""
echo "=== Step 0: self-test recurrence features ==="
python src/features/recurrence_features.py || exit 10

echo ""
echo "=== Step 1: build recurrence features for all 8 datasets ==="
python scripts/build_recurrence_features.py --all || exit 11

echo ""
echo "=== Step 2: per-dataset length-controlled ==="
python src/analysis/length_controlled.py \
    --features data/features/*_features_rec.csv \
    --n-bins 5 --clf rf \
    --output results/month1/lengthctl_per_dataset.json || exit 12

echo ""
echo "=== Step 3: pooled length-controlled ==="
python src/analysis/length_controlled.py \
    --features data/features/*_features_rec.csv \
    --pool --n-bins 5 --clf rf \
    --output results/month1/lengthctl_pooled.json || exit 13

echo ""
echo "=== DONE at $(date) ==="
ls -la results/month1/
