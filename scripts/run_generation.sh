#!/usr/bin/env bash
# =============================================================================
# run_generation.sh - Trace generation runner (API + HPC)
# =============================================================================
# Usage:
#   ./scripts/run_generation.sh pilot                  # 50-item API pilot
#   ./scripts/run_generation.sh pilot-hpc              # 50-item local pilot
#   ./scripts/run_generation.sh api-all                 # All datasets via API
#   ./scripts/run_generation.sh hpc-all qwen7b         # All datasets, local
#   ./scripts/run_generation.sh hpc-all llama8b         # Llama-8B on HPC
#   ./scripts/run_generation.sh cross-model math500     # Both models
#   ./scripts/run_generation.sh sc math500              # Self-consistency
#
# Environment:
#   DEEPSEEK_API_KEY=sk-xxx  (required for API backend)
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"

TRACES_DIR="data/traces"
mkdir -p "$TRACES_DIR"

echo "================================================"
echo "Reasoning Trace Generation Pipeline"
echo "================================================"

case "${1:-help}" in
    pilot)
        echo ">>> Pilot study: 50 items from MATH500 via DeepSeek API"
        python src/generation/generate_traces.py \
            --dataset math500 \
            --output "$TRACES_DIR/pilot_math500_r1.jsonl" \
            --backend api --model deepseek-r1 \
            --limit 50 --checkpoint-interval 10 --delay 0.5
        ;;

    pilot-hpc)
        echo ">>> Pilot study: 50 items from MATH500 via local HF (Qwen-7B)"
        python src/generation/generate_traces.py \
            --dataset math500 \
            --output "$TRACES_DIR/pilot_math500_qwen7b.jsonl" \
            --backend hf --model r1-distill-qwen-7b \
            --limit 50 --checkpoint-interval 10
        ;;

    api-all)
        echo ">>> Full generation via DeepSeek API (all datasets)"
        for DS in math500 gsm8k gpqa_diamond arc_challenge; do
            echo ""; echo "--- $DS ---"
            python src/generation/generate_traces.py \
                --dataset "$DS" \
                --output "$TRACES_DIR/${DS}_deepseek_r1.jsonl" \
                --backend api --model deepseek-r1 \
                --checkpoint-interval 20 --delay 0.5
        done
        ;;

    hpc-all)
        MODEL_KEY="${2:-qwen7b}"
        declare -A MODELS=(
            [qwen7b]="r1-distill-qwen-7b"
            [llama8b]="r1-distill-llama-8b"
            [qwen14b]="r1-distill-qwen-14b"
        )
        MODEL="${MODELS[$MODEL_KEY]:-$MODEL_KEY}"
        echo ">>> Full generation via HPC ($MODEL_KEY) - all datasets"
        for DS in math500 gsm8k gpqa_diamond arc_challenge; do
            echo ""; echo "--- $DS ($MODEL_KEY) ---"
            python src/generation/generate_traces.py \
                --dataset "$DS" \
                --output "$TRACES_DIR/${DS}_${MODEL_KEY}.jsonl" \
                --backend hf --model "$MODEL" \
                --checkpoint-interval 50
        done
        ;;

    cross-model)
        DS="${2:?Usage: $0 cross-model <dataset>}"
        echo ">>> Cross-model: $DS with Qwen-7B AND Llama-8B"
        for MK in qwen7b llama8b; do
            declare -A MODELS=([qwen7b]="r1-distill-qwen-7b" [llama8b]="r1-distill-llama-8b")
            echo ""; echo "--- $MK ---"
            python src/generation/generate_traces.py \
                --dataset "$DS" \
                --output "$TRACES_DIR/${DS}_${MK}.jsonl" \
                --backend hf --model "${MODELS[$MK]}" \
                --checkpoint-interval 50
        done
        ;;

    sc)
        DS="${2:?Usage: $0 sc <dataset> [backend] [model]}"
        BACKEND="${3:-api}"
        MODEL="${4:-deepseek-r1}"
        MK=$(echo "$MODEL" | sed 's/r1-distill-//;s/deepseek-//')
        echo ">>> Self-consistency: $DS (${BACKEND}, N=8)"
        python src/generation/generate_traces.py \
            --dataset "$DS" \
            --output "$TRACES_DIR/${DS}_${MK}_sc.jsonl" \
            --backend "$BACKEND" --model "$MODEL" \
            --self-consistency --num-samples 8 \
            --checkpoint-interval 10
        ;;

    status)
        echo ">>> Trace file status:"
        for f in "$TRACES_DIR"/*.jsonl; do
            [ -f "$f" ] || continue
            total=$(wc -l < "$f")
            errs=$(grep -c '"error"' "$f" 2>/dev/null || echo 0)
            correct=$(grep -c '"is_correct": true' "$f" 2>/dev/null || echo 0)
            model=$(grep -o '"model_short_name": "[^"]*"' "$f" | head -1 | cut -d'"' -f4)
            printf "  %-45s %4d records (%d correct, %d errors) [%s]\n" \
                "$(basename $f)" "$total" "$correct" "$errs" "$model"
        done
        ;;

    *)
        echo "Usage: $0 {pilot|pilot-hpc|api-all|hpc-all|cross-model|sc|status}"
        echo ""
        echo "  pilot                   50-item pilot via DeepSeek API"
        echo "  pilot-hpc               50-item pilot via local GPU"
        echo "  api-all                 All datasets via DeepSeek API"
        echo "  hpc-all <model>         All datasets via local GPU"
        echo "  cross-model <dataset>   Both models (Qwen-7B + Llama-8B)"
        echo "  sc <dataset> [api|hf]   Self-consistency samples"
        echo "  status                  Show trace file summary"
        echo ""
        echo "Models: qwen7b, llama8b, qwen14b (for hpc-all/cross-model)"
        echo "Requires: DEEPSEEK_API_KEY env var for API backend"
        exit 1
        ;;
esac
