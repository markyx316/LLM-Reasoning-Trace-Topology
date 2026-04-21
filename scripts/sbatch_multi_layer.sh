#!/bin/bash
#SBATCH --job-name=multi_layer
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/mlayer_%j.out
#SBATCH --error=logs/mlayer_%j.err

echo "=== Job start $(date) on $(hostname) ==="
mkdir -p logs results/month3
cd /gpfs/radev/home/pc838/LLM-Reasoning-Trace-Topology || exit 1

source /etc/profile.d/modules.sh 2>/dev/null || true
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1 || exit 2

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi -L | head -3

echo ""
echo "=== Multi-layer probe sweep (reuses data/hidden_atlas/) ==="
python src/modeling/multi_layer_probe.py \
    --npz-glob "data/hidden_atlas/*.npz" \
    --output   results/month3/multi_layer_probe.json || exit 10

echo ""
echo "=== DONE at $(date) ==="
ls -la results/month3/multi_layer_probe*
