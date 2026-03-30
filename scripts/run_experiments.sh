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
    for TRACE_FILE in "$TRACES_DIR"/*_traces.jsonl; do
        [ -f "$TRACE_FILE" ] || continue
        BASENAME=$(basename "$TRACE_FILE" _traces.jsonl)
        echo "  Parsing: $BASENAME"

        # Parse behavior sequences
        python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
from src.parsing.behavior_classifier import parse_trace_file
parse_trace_file('$TRACE_FILE', '$PARSED_DIR/${BASENAME}_parsed.jsonl')
"

        # Extract features
        echo "  Extracting features: $BASENAME"
        python -c "
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

        python src/modeling/train_and_evaluate.py \
            --features "$FEATURES_FILE" \
            --output "$RESULTS_DIR/${BASENAME}_results.json"
    done
    echo ">>> Training and evaluation complete."
}

run_transfer() {
    echo ">>> STEP 3: Cross-domain transfer experiments"

    # MATH500 → GPQA Diamond
    if [ -f "$FEATURES_DIR/math500_features.csv" ] && [ -f "$FEATURES_DIR/gpqa_diamond_features.csv" ]; then
        echo "  Transfer: math500 → gpqa_diamond"
        python src/modeling/train_and_evaluate.py \
            --features "$FEATURES_DIR/math500_features.csv" \
            --test-features "$FEATURES_DIR/gpqa_diamond_features.csv" \
            --output "$RESULTS_DIR/transfer_math500_to_gpqa.json"
    fi

    # MATH500 → ARC-Challenge
    if [ -f "$FEATURES_DIR/math500_features.csv" ] && [ -f "$FEATURES_DIR/arc_challenge_features.csv" ]; then
        echo "  Transfer: math500 → arc_challenge"
        python src/modeling/train_and_evaluate.py \
            --features "$FEATURES_DIR/math500_features.csv" \
            --test-features "$FEATURES_DIR/arc_challenge_features.csv" \
            --output "$RESULTS_DIR/transfer_math500_to_arc.json"
    fi

    # GSM8K → MATH500 (difficulty transfer)
    if [ -f "$FEATURES_DIR/gsm8k_features.csv" ] && [ -f "$FEATURES_DIR/math500_features.csv" ]; then
        echo "  Transfer: gsm8k → math500 (difficulty transfer)"
        python src/modeling/train_and_evaluate.py \
            --features "$FEATURES_DIR/gsm8k_features.csv" \
            --test-features "$FEATURES_DIR/math500_features.csv" \
            --output "$RESULTS_DIR/transfer_gsm8k_to_math500.json"
    fi

    echo ">>> Transfer experiments complete."
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
