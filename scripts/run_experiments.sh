#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh - Run the complete experimental pipeline
# =============================================================================
# Prerequisites: Trace generation must be complete (run_generation.sh full)
#
# Usage:
#   ./scripts/run_experiments.sh all       # Full pipeline
#   ./scripts/run_experiments.sh parse     # Parse + extract features only
#   ./scripts/run_experiments.sh train     # Train classifiers + evaluate
#   ./scripts/run_experiments.sh transfer  # Cross-domain transfer experiments
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"

TRACES_DIR="data/traces"
PARSED_DIR="data/parsed"
FEATURES_DIR="data/features"
RESULTS_DIR="results"

mkdir -p "$PARSED_DIR" "$FEATURES_DIR" "$RESULTS_DIR"

echo "================================================"
echo "Experimental Pipeline"
echo "================================================"
echo ""

run_parse() {
    echo ">>> STEP 1: Parse traces and extract features"
    for TRACE_FILE in "$TRACES_DIR"/*.jsonl; do
        [ -f "$TRACE_FILE" ] || continue
        FILENAME=$(basename "$TRACE_FILE")

        # Skip pilot, temp, and self-consistency files
        case "$FILENAME" in
            pilot_*|_*|*_sc.jsonl|*_sc_*.jsonl) continue ;;
        esac

        # Derive clean base name
        BASENAME="${FILENAME%.jsonl}"
        BASENAME="${BASENAME%_traces}"

        echo "  Parsing: $BASENAME"

        # Parse behavior sequences
        python3 -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
from src.parsing.behavior_classifier import parse_trace_file
parse_trace_file('$TRACE_FILE', '$PARSED_DIR/${BASENAME}_parsed.jsonl')
"

        # Extract features
        echo "  Extracting features: $BASENAME"
        python3 -c "
import logging
logging.basicConfig(level=logging.INFO)
from src.features.feature_pipeline import extract_features_from_file
extract_features_from_file('$PARSED_DIR/${BASENAME}_parsed.jsonl', '$FEATURES_DIR/${BASENAME}_features.csv')
"
    done
    echo ">>> Parse + feature extraction complete."
}

run_train() {
    echo ">>> STEP 2: Train classifiers and evaluate"
    for FEATURES_FILE in "$FEATURES_DIR"/*_features.csv; do
        [ -f "$FEATURES_FILE" ] || continue
        BASENAME=$(basename "$FEATURES_FILE" _features.csv)
        echo ""
        echo "  === Dataset: $BASENAME ==="

        python3 src/modeling/train_and_evaluate.py \
            --features "$FEATURES_FILE" \
            --output "$RESULTS_DIR/${BASENAME}_results.json"
    done
    echo ">>> Training and evaluation complete."
}

run_transfer() {
    echo ">>> STEP 3: Cross-domain transfer experiments"

    # Auto-discover feature files and run transfers between datasets
    # We look for any math500 file as source and any gpqa/arc file as target

    # Find all feature files grouped by dataset prefix
    MATH_FILES=$(ls "$FEATURES_DIR"/math500*_features.csv 2>/dev/null || true)
    GPQA_FILES=$(ls "$FEATURES_DIR"/gpqa_diamond*_features.csv 2>/dev/null || true)
    ARC_FILES=$(ls "$FEATURES_DIR"/arc_challenge*_features.csv 2>/dev/null || true)
    GSM_FILES=$(ls "$FEATURES_DIR"/gsm8k*_features.csv 2>/dev/null || true)

    FOUND=0

    # MATH500 → GPQA (for each model that has both)
    for MATH_F in $MATH_FILES; do
        [ -f "$MATH_F" ] || continue
        MATH_BASE=$(basename "$MATH_F" _features.csv)
        # Extract model suffix (e.g., "qwen7b", "llama8b", "deepseek_r1")
        MODEL_SUFFIX="${MATH_BASE#math500_}"

        for GPQA_F in $GPQA_FILES; do
            [ -f "$GPQA_F" ] || continue
            GPQA_BASE=$(basename "$GPQA_F" _features.csv)
            GPQA_SUFFIX="${GPQA_BASE#gpqa_diamond_}"
            # Match same model
            if [ "$MODEL_SUFFIX" = "$GPQA_SUFFIX" ]; then
                echo "  Transfer: $MATH_BASE → $GPQA_BASE"
                FOUND=$((FOUND + 1))
                python3 src/modeling/train_and_evaluate.py \
                    --features "$MATH_F" \
                    --test-features "$GPQA_F" \
                    --output "$RESULTS_DIR/transfer_${MATH_BASE}_to_gpqa_${MODEL_SUFFIX}.json"
            fi
        done

        for ARC_F in $ARC_FILES; do
            [ -f "$ARC_F" ] || continue
            ARC_BASE=$(basename "$ARC_F" _features.csv)
            ARC_SUFFIX="${ARC_BASE#arc_challenge_}"
            if [ "$MODEL_SUFFIX" = "$ARC_SUFFIX" ]; then
                echo "  Transfer: $MATH_BASE → $ARC_BASE"
                FOUND=$((FOUND + 1))
                python3 src/modeling/train_and_evaluate.py \
                    --features "$MATH_F" \
                    --test-features "$ARC_F" \
                    --output "$RESULTS_DIR/transfer_${MATH_BASE}_to_arc_${MODEL_SUFFIX}.json"
            fi
        done
    done

    # GSM8K → MATH500 (difficulty transfer, same model)
    for GSM_F in $GSM_FILES; do
        [ -f "$GSM_F" ] || continue
        GSM_BASE=$(basename "$GSM_F" _features.csv)
        MODEL_SUFFIX="${GSM_BASE#gsm8k_}"

        for MATH_F in $MATH_FILES; do
            [ -f "$MATH_F" ] || continue
            MATH_BASE=$(basename "$MATH_F" _features.csv)
            MATH_SUFFIX="${MATH_BASE#math500_}"
            if [ "$MODEL_SUFFIX" = "$MATH_SUFFIX" ]; then
                echo "  Transfer: $GSM_BASE → $MATH_BASE (difficulty)"
                FOUND=$((FOUND + 1))
                python3 src/modeling/train_and_evaluate.py \
                    --features "$GSM_F" \
                    --test-features "$MATH_F" \
                    --output "$RESULTS_DIR/transfer_${GSM_BASE}_to_math500_${MODEL_SUFFIX}.json"
            fi
        done
    done

    if [ "$FOUND" -eq 0 ]; then
        echo "  No matching feature file pairs found for transfer experiments."
        echo "  Need feature files for multiple datasets with the same model suffix."
    fi

    echo ">>> Transfer experiments complete ($FOUND experiments run)."
}

case "${1:-help}" in
    parse)    run_parse ;;
    train)    run_train ;;
    transfer) run_transfer ;;
    all)
        run_parse
        echo ""
        run_train
        echo ""
        run_transfer
        echo ""
        echo ">>> All experiments complete. Results in: $RESULTS_DIR/"
        ;;
    *)
        echo "Usage: $0 {parse|train|transfer|all}"
        echo ""
        echo "  parse     - Parse traces and extract features"
        echo "  train     - Train classifiers on each dataset"
        echo "  transfer  - Cross-domain transfer experiments"
        echo "  all       - Run complete pipeline"
        exit 1
        ;;
esac
