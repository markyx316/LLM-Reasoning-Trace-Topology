#!/bin/bash
#SBATCH --job-name=probe_qwen
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/probeqwen_%j.out
#SBATCH --error=logs/probeqwen_%j.err

echo "=== Job start $(date) on $(hostname) ==="
mkdir -p logs results/month3 || true
cd /gpfs/radev/home/pc838/LLM-Reasoning-Trace-Topology || exit 1

if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
if [ -f /etc/profile.d/lmod.sh ]; then source /etc/profile.d/lmod.sh; fi
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1 || exit 2

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo ""
echo "=== Probe on Qwen-only data (4 datasets, ~3189 samples) ==="
python src/modeling/hidden_state_probe.py \
    --npz-glob     "data/hidden_states/*_qwen7b.npz" \
    --genunc-glob  "data/features/*_qwen7b_features_genunc.csv" \
    --output       results/month3/hidden_probe_qwen.json || exit 10

echo ""
echo "=== DONE at $(date) ==="
