#!/bin/bash
#SBATCH --job-name=month2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/month2_%j.out
#SBATCH --error=logs/month2_%j.err

echo "=== Job start $(date) on $(hostname) ==="
mkdir -p logs results/month2 data/step_embeddings || true

cd /gpfs/radev/home/pc838/LLM-Reasoning-Trace-Topology || { echo "cd failed"; exit 1; }
echo "PWD: $PWD"

if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
if [ -f /etc/profile.d/lmod.sh ]; then source /etc/profile.d/lmod.sh; fi
module load PyTorch/2.1.2-foss-2022b-CUDA-12.1.1 || { echo "module load failed"; exit 2; }

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 3
nvidia-smi -L || true

echo ""
echo "=== Step 0: install / confirm dependencies ==="
python -m pip install --user --quiet sentencepiece 'protobuf<5' || true
python -c "import sentence_transformers, transformers, sentencepiece; print('st', sentence_transformers.__version__, 'tx', transformers.__version__, 'sp', sentencepiece.__version__)"

echo ""
echo "=== Step 1: Pre-compute step embeddings for all 8 datasets ==="
python scripts/build_step_embeddings.py --all --batch-size 256 || exit 11
ls -la data/step_embeddings/

echo ""
echo "=== Step 2: Train Step Transformer (pooled, 5-fold) ==="
python src/modeling/step_transformer.py \
    --npz-glob "data/step_embeddings/*.npz" \
    --output results/month2/step_transformer_pooled.json \
    --epochs 15 --batch-size 32 --lr 3e-4 || exit 12

echo ""
echo "=== Step 3: DeBERTa fine-tune (pooled, 5-fold) ==="
python src/modeling/deberta_baseline.py \
    --traces-glob "data/traces/*_traces.jsonl" \
    --output results/month2/deberta_pooled.json \
    --epochs 3 --batch-size 8 --lr 2e-5 || exit 13

echo ""
echo "=== Step 4: Stacking hybrid (DeBERTa + StepTF + handcrafted+recurrence) ==="
python -m pip install --user --quiet xgboost || true
python src/modeling/hybrid.py \
    --deberta-oof   results/month2/deberta_pooled_oof.npz \
    --step-oof      results/month2/step_transformer_pooled_oof.npz \
    --features-glob "data/features/*_features_rec.csv" \
    --output        results/month2/hybrid_pooled.json \
    --clf all || exit 14

echo ""
echo "=== DONE at $(date) ==="
ls -la results/month2/
