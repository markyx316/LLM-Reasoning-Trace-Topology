#!/bin/bash
#SBATCH --job-name=extract_llama
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/llama_%j.out
#SBATCH --error=logs/llama_%j.err

echo "=== Job start $(date) on $(hostname) ==="
mkdir -p logs data/hidden_states || true
cd /gpfs/radev/home/pc838/LLM-Reasoning-Trace-Topology || exit 1

if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
if [ -f /etc/profile.d/lmod.sh ]; then source /etc/profile.d/lmod.sh; fi
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1 || exit 2

# Upgrade transformers to support Llama-3 rope_scaling (>= 4.43)
python -m pip install --user --quiet --upgrade "transformers==4.44.2" "tokenizers>=0.19" "accelerate>=0.30" || true
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo ""
echo "=== Extract hidden states + gen-unc for 4 Llama datasets ==="
for ds in math500_llama8b gsm8k_llama8b gpqa_diamond_llama8b arc_challenge_llama8b; do
    python scripts/extract_hidden_states.py \
        --traces data/traces/${ds}_traces.jsonl \
        --model  deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --output data/hidden_states/${ds}.npz || exit 10
done

echo ""
echo "=== Train probes on full 8-dataset pool ==="
python src/modeling/hidden_state_probe.py \
    --npz-glob     "data/hidden_states/*.npz" \
    --genunc-glob  "data/features/*_features_genunc.csv" \
    --output       results/month3/hidden_probe_pooled.json || exit 11

echo ""
echo "=== DONE at $(date) ==="
ls -la data/hidden_states/ results/month3/
