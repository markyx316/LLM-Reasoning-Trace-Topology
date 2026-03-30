#!/usr/bin/env bash
# =============================================================================
# run_parsing.sh - Parse traces and extract features
# =============================================================================
# Usage:
#   ./scripts/run_parsing.sh parse            # Parse all generated traces
#   ./scripts/run_parsing.sh features         # Extract features from parsed
#   ./scripts/run_parsing.sh annotate         # Create annotation template
#   ./scripts/run_parsing.sh all              # Parse + Features (full pipeline)
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR"

TRACES_DIR="data/traces"
PARSED_DIR="data/parsed"
FEATURES_DIR="data/features"
ANNOTATIONS_DIR="data/annotations"

mkdir -p "$PARSED_DIR" "$FEATURES_DIR" "$ANNOTATIONS_DIR"

echo "================================================"
echo "Trace Parsing & Feature Extraction Pipeline"
echo "================================================"

case "${1:-help}" in
    parse)
        echo ">>> Parsing all generated traces..."
        for TRACE_FILE in "$TRACES_DIR"/*_traces.jsonl; do
            if [ -f "$TRACE_FILE" ]; then
                BASENAME=$(basename "$TRACE_FILE" _traces.jsonl)
                echo "  Parsing: $BASENAME"
                python -c "
import logging, sys
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
from src.parsing.behavior_classifier import parse_trace_file
parse_trace_file('$TRACE_FILE', '$PARSED_DIR/${BASENAME}_parsed.jsonl')
"
            fi
        done
        echo ">>> Parsing complete."
        ;;

    features)
        echo ">>> Extracting features from parsed traces..."
        for PARSED_FILE in "$PARSED_DIR"/*_parsed.jsonl; do
            if [ -f "$PARSED_FILE" ]; then
                BASENAME=$(basename "$PARSED_FILE" _parsed.jsonl)
                echo "  Features: $BASENAME"
                python -c "
import logging
logging.basicConfig(level=logging.INFO)
from src.features.feature_pipeline import extract_features_from_file
extract_features_from_file('$PARSED_FILE', '$FEATURES_DIR/${BASENAME}_features.csv')
"
            fi
        done
        echo ">>> Feature extraction complete."
        ;;

    annotate)
        echo ">>> Creating annotation template..."
        # Combine all trace files and sample
        COMBINED="$TRACES_DIR/_combined_for_annotation.jsonl"
        cat "$TRACES_DIR"/*_traces.jsonl > "$COMBINED" 2>/dev/null || true
        if [ -s "$COMBINED" ]; then
            PYTHONPATH="$PROJECT_DIR" python src/parsing/parser_evaluation.py \
                create \
                --traces "$COMBINED" \
                --output "$ANNOTATIONS_DIR/annotation_template.csv" \
                --n-samples 100 \
                --per-dataset 25
            rm -f "$COMBINED"
            echo ">>> Template created: $ANNOTATIONS_DIR/annotation_template.csv"
            echo "    Fill in the 'manual_label' column, then run:"
            echo "    ./scripts/run_parsing.sh evaluate"
        else
            echo "ERROR: No trace files found in $TRACES_DIR"
            exit 1
        fi
        ;;

    evaluate)
        echo ">>> Evaluating parser accuracy..."
        PYTHONPATH="$PROJECT_DIR" python src/parsing/parser_evaluation.py \
            evaluate \
            --annotations "$ANNOTATIONS_DIR/annotation_template.csv" \
            --report "$ANNOTATIONS_DIR/parser_evaluation_report.json"
        ;;

    all)
        echo ">>> Running full pipeline: parse → features"
        $0 parse
        echo ""
        $0 features
        ;;

    *)
        echo "Usage: $0 {parse|features|annotate|evaluate|all}"
        echo ""
        echo "  parse     - Parse behavior sequences from trace files"
        echo "  features  - Extract features from parsed traces"
        echo "  annotate  - Create annotation template for manual validation"
        echo "  evaluate  - Evaluate parser vs. manual annotations"
        echo "  all       - Run parse + features (complete pipeline)"
        exit 1
        ;;
esac
