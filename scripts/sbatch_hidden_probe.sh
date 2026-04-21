#!/bin/bash
#SBATCH --job-name=hidden_probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/hprobe_%j.out
#SBATCH --error=logs/hprobe_%j.err

echo "=== Job start $(date) on $(hostname) ==="
mkdir -p logs data/hidden_states results/month3 || true
cd /gpfs/radev/home/pc838/LLM-Reasoning-Trace-Topology || exit 1

if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
if [ -f /etc/profile.d/lmod.sh ]; then source /etc/profile.d/lmod.sh; fi
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1 || exit 2

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 3
nvidia-smi -L | head -3

echo ""
echo "=== Step 0: Install dependencies ==="
python -m pip install --user --quiet accelerate 'bitsandbytes<0.45' || true
python -c "import accelerate; print('accelerate', accelerate.__version__)"

echo ""
echo "=== Step 1: Extract hidden states + generation uncertainty for all 8 datasets ==="
python scripts/extract_hidden_states.py --all || exit 10
ls -la data/hidden_states/
du -sh data/hidden_states

echo ""
echo "=== Step 2: Train probes (Direction A + B) pooled across 8 ==="
python src/modeling/hidden_state_probe.py \
    --npz-glob     "data/hidden_states/*.npz" \
    --genunc-glob  "data/features/*_features_genunc.csv" \
    --output       results/month3/hidden_probe_pooled.json || exit 11

echo ""
echo "=== DONE at $(date) ==="
ls -la results/month3/
