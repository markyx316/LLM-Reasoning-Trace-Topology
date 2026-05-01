#!/usr/bin/env bash
# =============================================================================
# submit_hpc.sh - SLURM job submission for HPC clusters
# =============================================================================
# Submits trace generation jobs to your school's HPC cluster.
#
# Usage:
#   ./scripts/submit_hpc.sh pilot                    # Quick pilot study
#   ./scripts/submit_hpc.sh generate qwen7b math500  # Single dataset+model
#   ./scripts/submit_hpc.sh generate llama8b math500  # Llama-8B model
#   ./scripts/submit_hpc.sh all-datasets qwen7b      # All datasets, one model
#   ./scripts/submit_hpc.sh cross-model math500       # Both models, one dataset
#   ./scripts/submit_hpc.sh sc math500                # Self-consistency
#   ./scripts/submit_hpc.sh api math500               # API-based generation
#
# BEFORE FIRST USE:
#   1. Edit the SBATCH directives below for your cluster
#      (partition name, account, GPU type, module names)
#   2. Set up your conda/venv environment
#   3. Set DEEPSEEK_API_KEY in your ~/.bashrc if using API backend
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACES_DIR="$PROJECT_DIR/data/traces"
LOGS_DIR="$PROJECT_DIR/logs"
mkdir -p "$TRACES_DIR" "$LOGS_DIR"

# ============================================================
# CLUSTER CONFIGURATION — EDIT THESE FOR YOUR HPC
# ============================================================
PARTITION="gpu"            # Your GPU partition name
ACCOUNT=""                 # Your account/allocation (leave empty if none)
GPU_TYPE="a100"            # GPU type: a100, v100, a6000, etc.
NUM_GPUS=1
TIME_PILOT="02:00:00"     # 2 hours for pilot
TIME_SINGLE="12:00:00"    # 12 hours for single dataset
TIME_SC="24:00:00"        # 24 hours for self-consistency
CONDA_ENV="trace-uq"      # Your conda environment name
# ============================================================

# Model mapping
declare -A MODEL_MAP
MODEL_MAP[qwen7b]="r1-distill-qwen-7b"
MODEL_MAP[llama8b]="r1-distill-llama-8b"
MODEL_MAP[qwen14b]="r1-distill-qwen-14b"

ACCOUNT_FLAG=""
if [ -n "$ACCOUNT" ]; then
    ACCOUNT_FLAG="#SBATCH --account=$ACCOUNT"
fi

submit_job() {
    local JOB_NAME="$1"
    local TIME="$2"
    local CMD="$3"
    local EXTRA_SBATCH="${4:-}"

    local SCRIPT="$LOGS_DIR/job_${JOB_NAME}.sh"

    cat > "$SCRIPT" << SLURM_EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$PARTITION
$ACCOUNT_FLAG
#SBATCH --gres=gpu:${GPU_TYPE}:${NUM_GPUS}
#SBATCH --time=$TIME
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=$LOGS_DIR/${JOB_NAME}_%j.out
#SBATCH --error=$LOGS_DIR/${JOB_NAME}_%j.err
$EXTRA_SBATCH

echo "========================================"
echo "Job: $JOB_NAME"
echo "Node: \$(hostname)"
echo "GPUs: \$(nvidia-smi -L 2>/dev/null || echo 'N/A')"
echo "Started: \$(date)"
echo "========================================"

# Load modules (edit for your cluster)
# module load cuda/12.1
# module load anaconda3

# Activate environment
source activate $CONDA_ENV 2>/dev/null || conda activate $CONDA_ENV 2>/dev/null || true

cd $PROJECT_DIR
export PYTHONPATH=$PROJECT_DIR

# Load .env (API keys, HF_HOME, etc.)
if [ -f "$PROJECT_DIR/.env" ]; then
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# CRITICAL: Point HuggingFace to project dir cache (not home dir)
# Edit this path for your cluster:
export HF_HOME=\${HF_HOME:-$PROJECT_DIR/.cache/huggingface}
mkdir -p "\$HF_HOME"

echo "HF_HOME: \$HF_HOME"
echo "PYTHONPATH: \$PYTHONPATH"
echo "Python: \$(which python)"
echo ""

$CMD

echo "Completed: \$(date)"
SLURM_EOF

    chmod +x "$SCRIPT"
    echo "Submitting: $JOB_NAME"
    sbatch "$SCRIPT"
}

case "${1:-help}" in
    pilot)
        submit_job "pilot_math500" "$TIME_PILOT" \
            "python3 src/generation/generate_traces.py --dataset math500 --output $TRACES_DIR/pilot_math500_qwen7b.jsonl --backend hf --model r1-distill-qwen-7b --limit 50 --checkpoint-interval 10"
        ;;

    generate)
        MODEL_KEY="${2:?Usage: $0 generate <model> <dataset>}"
        DATASET="${3:?Usage: $0 generate <model> <dataset>}"
        MODEL="${MODEL_MAP[$MODEL_KEY]:-$MODEL_KEY}"
        submit_job "gen_${MODEL_KEY}_${DATASET}" "$TIME_SINGLE" \
            "python3 src/generation/generate_traces.py --dataset $DATASET --output $TRACES_DIR/${DATASET}_${MODEL_KEY}.jsonl --backend hf --model $MODEL --checkpoint-interval 50"
        ;;

    all-datasets)
        MODEL_KEY="${2:?Usage: $0 all-datasets <model>}"
        MODEL="${MODEL_MAP[$MODEL_KEY]:-$MODEL_KEY}"
        for DS in math500 gsm8k gpqa_diamond arc_challenge; do
            submit_job "gen_${MODEL_KEY}_${DS}" "$TIME_SINGLE" \
                "python3 src/generation/generate_traces.py --dataset $DS --output $TRACES_DIR/${DS}_${MODEL_KEY}.jsonl --backend hf --model $MODEL --checkpoint-interval 50"
        done
        ;;

    cross-model)
        DATASET="${2:?Usage: $0 cross-model <dataset>}"
        for MK in qwen7b llama8b; do
            MODEL="${MODEL_MAP[$MK]}"
            submit_job "gen_${MK}_${DATASET}" "$TIME_SINGLE" \
                "python3 src/generation/generate_traces.py --dataset $DATASET --output $TRACES_DIR/${DATASET}_${MK}.jsonl --backend hf --model $MODEL --checkpoint-interval 50"
        done
        ;;

    sc)
        DATASET="${2:?Usage: $0 sc <dataset>}"
        MODEL_KEY="${3:-qwen7b}"
        MODEL="${MODEL_MAP[$MODEL_KEY]:-$MODEL_KEY}"
        submit_job "sc_${MODEL_KEY}_${DATASET}" "$TIME_SC" \
            "python3 src/generation/generate_traces.py --dataset $DATASET --output $TRACES_DIR/${DATASET}_${MODEL_KEY}_sc.jsonl --backend hf --model $MODEL --self-consistency --num-samples 8 --checkpoint-interval 10"
        ;;

    api)
        DATASET="${2:?Usage: $0 api <dataset>}"
        # API jobs don't need GPU — run on CPU partition
        submit_job "api_r1_${DATASET}" "$TIME_SINGLE" \
            "python3 src/generation/generate_traces.py --dataset $DATASET --output $TRACES_DIR/${DATASET}_deepseek_r1.jsonl --backend api --model deepseek-r1 --delay 0.5 --checkpoint-interval 10" \
            "#SBATCH --partition=cpu
#SBATCH --gres="
        ;;

    status)
        echo "Current jobs:"
        squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %.6D %R" 2>/dev/null || echo "(squeue not available)"
        echo ""
        echo "Completed traces:"
        for f in "$TRACES_DIR"/*.jsonl; do
            [ -f "$f" ] || continue
            n=$(wc -l < "$f")
            errs=$(grep -c '"error"' "$f" 2>/dev/null || echo 0)
            echo "  $(basename $f): $n records ($errs errors)"
        done
        ;;

    *)
        echo "Usage: $0 {pilot|generate|all-datasets|cross-model|sc|api|status}"
        echo ""
        echo "  pilot                    50-item test run (Qwen-7B)"
        echo "  generate <model> <ds>    Single model+dataset"
        echo "  all-datasets <model>     All datasets, one model"
        echo "  cross-model <ds>         Both models (Qwen-7B + Llama-8B)"
        echo "  sc <ds> [model]          Self-consistency samples"
        echo "  api <ds>                 API-based (DeepSeek-R1 full)"
        echo "  status                   Check job status + trace counts"
        echo ""
        echo "Models: qwen7b, llama8b, qwen14b"
        echo "Datasets: math500, gsm8k, gpqa_diamond, arc_challenge"
        ;;
esac
