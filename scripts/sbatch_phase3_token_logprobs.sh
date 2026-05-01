#!/bin/bash
# =============================================================================
# Phase 3 (T2) — Token-level logprob features via HPC teacher-forcing.
#
# For each of the 8 dataset-model combos, run one teacher-forcing pass of the
# generator (Qwen-7B or Llama-8B) over every trace. Produces:
#   - data/features/{group}_features_steplp.csv       (24 new features)
#   - data/token_logprobs/{group}_raw.npz             (padded raw arrays)
#
# These features are novel relative to the 10-feature `_features_genunc.csv`
# already in the stack: they segment the trace into quartiles, zoom in on
# the answer moment, and detect local NLL spikes.
#
# Submit:
#   sbatch scripts/sbatch_phase3_token_logprobs.sh
# =============================================================================
#SBATCH --job-name=p3_steplp
#SBATCH --partition=gpu_b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=9:00:00
#SBATCH --chdir=/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology
#SBATCH --output=logs/p3_steplp_%j.out
#SBATCH --error=logs/p3_steplp_%j.err

trap - ERR
set +e

mkdir -p logs data/features data/token_logprobs

echo "=== Phase 3 / T2 token-logprob features  start $(date) on $(hostname) ==="

module load miniconda || exit 2
conda activate torch311 || exit 3

export PYTHONPATH=$PWD

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 10
nvidia-smi -L | head -3

echo ""
echo "=== Running scripts/phase3/extract_token_logprob_features.py --all ==="
python scripts/phase3/extract_token_logprob_features.py --all \
    --max-len 4096
STATUS=$?

echo ""
echo "=== Output summary ==="
ls -la data/features/*_features_steplp.csv 2>/dev/null | head -20
echo ""
ls -la data/token_logprobs/*_raw.npz 2>/dev/null | head -20
du -sh data/token_logprobs 2>/dev/null

echo ""
echo "=== DONE at $(date)  status=$STATUS ==="
exit $STATUS
