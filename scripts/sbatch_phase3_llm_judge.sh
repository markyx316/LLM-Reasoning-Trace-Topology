#!/bin/bash
# =============================================================================
# Phase 3 (T1) — LLM-as-judge OOF via external API.
#
# The judge is network-I/O-bound (HTTP to DeepSeek-V3 / Claude / GPT-4o-mini),
# so no GPU is strictly required. We still target gpu_b200 because that is
# the verified-working partition for this cluster; feel free to switch to a
# CPU partition (e.g. day) if available.
#
# Prereqs
# -------
#   1. `DEEPSEEK_API_KEY` set in .env (repo root) OR exported in the
#      submitting shell. This script `export`s whatever is in .env.
#   2. Output directory `results/month3/` and `data/traces/*_traces.jsonl`
#      already on disk.
#
# Submit:
#   sbatch scripts/sbatch_phase3_llm_judge.sh
#
# Resume:
#   The driver script writes a JSONL cache next to the output npz
#   (`<output>.raw.jsonl`). Re-running picks up from where it stopped.
# =============================================================================
#SBATCH --job-name=p3_judge
#SBATCH --partition=gpu_b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --chdir=/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology
#SBATCH --output=logs/p3_judge_%j.out
#SBATCH --error=logs/p3_judge_%j.err

# --- Safety: don't let a stray ERR trap kill the job on module load ---
trap - ERR
set +e

mkdir -p logs results/month3

echo "=== Phase 3 / T1 LLM-as-judge  start $(date) on $(hostname) ==="

module load miniconda || exit 2
conda activate torch311 || exit 3

export PYTHONPATH=$PWD

# Load .env (needed for DEEPSEEK_API_KEY) — the Python driver also reads it,
# but we echo it here for sanity.
if [ -f .env ]; then
    # Export only keys we care about, ignoring blank / comment lines.
    set -a
    # shellcheck disable=SC1091
    source <(grep -E '^[A-Z_][A-Z0-9_]*=' .env | sed 's/\r$//')
    set +a
fi

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "[FATAL] DEEPSEEK_API_KEY not set in env or .env — cannot run LLM-judge"
    exit 4
fi
echo "DEEPSEEK_API_KEY prefix: ${DEEPSEEK_API_KEY:0:6}..."

python -c "import sys; print('python:', sys.version)"

JUDGE_MODEL="${JUDGE_MODEL:-deepseek-chat}"
OUTPUT="${OUTPUT:-results/month3/llm_judge_deepseek_v3_oof.npz}"
WORKERS="${WORKERS:-16}"
TRACES_GLOB="${TRACES_GLOB:-data/traces/*_traces.jsonl}"

echo ""
echo "=== Driver: scripts/phase3/build_llm_judge_oof.py ==="
echo "  judge model: $JUDGE_MODEL"
echo "  traces glob: $TRACES_GLOB"
echo "  workers:     $WORKERS"
echo "  output:      $OUTPUT"
echo ""

python scripts/phase3/build_llm_judge_oof.py \
    --traces-glob "$TRACES_GLOB" \
    --output "$OUTPUT" \
    --judge-backend openai \
    --judge-model "$JUDGE_MODEL" \
    --workers "$WORKERS" \
    --log-level INFO
STATUS=$?

echo ""
echo "=== DONE at $(date)  status=$STATUS ==="
ls -la "$OUTPUT" 2>/dev/null || true

exit $STATUS
