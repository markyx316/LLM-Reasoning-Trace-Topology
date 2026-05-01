#!/bin/bash
#SBATCH --job-name=layer_atlas
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/atlas_%j.out
#SBATCH --error=logs/atlas_%j.err

echo "=== Job start $(date) on $(hostname) ==="
mkdir -p logs data/hidden_atlas results/month3
cd /gpfs/radev/home/pc838/LLM-Reasoning-Trace-Topology || exit 1

source /etc/profile.d/modules.sh 2>/dev/null || true
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1 || exit 2

# Need transformers >= 4.43 for Llama-3 rope_scaling
python -m pip install --user --quiet --upgrade "transformers==4.44.2" "tokenizers>=0.19" "accelerate>=0.30" 2>&1 | tail -3
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
nvidia-smi -L | head -3

echo ""
echo "=== Step 1: Extract layer-atlas hidden states (8 layers x 4 positions) ==="
python scripts/extract_layer_atlas.py --all --n-layers 8 || exit 10
ls -la data/hidden_atlas/
du -sh data/hidden_atlas

echo ""
echo "=== Step 2: Layer probe sweep (train MLP per cell, ~64 cells) ==="
python src/modeling/layer_probe_sweep.py \
    --npz-glob "data/hidden_atlas/*.npz" \
    --output   results/month3/layer_atlas.json || exit 11

echo ""
echo "=== DONE at $(date) ==="
ls -la results/month3/layer_atlas*
