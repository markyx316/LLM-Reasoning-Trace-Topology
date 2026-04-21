#!/bin/bash
#SBATCH --job-name=probe_pool
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/probepool_%j.out
#SBATCH --error=logs/probepool_%j.err

echo "=== Job start $(date) on $(hostname) ==="
mkdir -p logs results/month3 || true
cd /gpfs/radev/home/pc838/LLM-Reasoning-Trace-Topology || exit 1

if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
if [ -f /etc/profile.d/lmod.sh ]; then source /etc/profile.d/lmod.sh; fi
module load PyTorch/2.1.2-foss-2022b || exit 2

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo ""
echo "=== Per-model probes, pooled OOF across Qwen + Llama ==="
python src/modeling/hidden_state_probe.py \
    --npz-glob     "data/hidden_states/*.npz" \
    --genunc-glob  "data/features/*_features_genunc.csv" \
    --output       results/month3/hidden_probe_pooled.json || exit 10

echo ""
echo "=== DONE at $(date) ==="
ls -la results/month3/
