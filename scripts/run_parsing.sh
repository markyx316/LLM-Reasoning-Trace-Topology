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
        FOUND=0
        for TRACE_FILE in "$TRACES_DIR"/*.jsonl; do
            [ -f "$TRACE_FILE" ] || continue

            FILENAME=$(basename "$TRACE_FILE")

            # Skip pilot files and temp files
            case "$FILENAME" in
                pilot_*|_*|*_sc.jsonl|*_sc_*.jsonl) continue ;;
            esac

            # Derive a clean base name by stripping known suffixes
            BASENAME="${FILENAME%.jsonl}"
            BASENAME="${BASENAME%_traces}"  # strip _traces if present

            # Skip if already parsed
            if [ -f "$PARSED_DIR/${BASENAME}_parsed.jsonl" ]; then
                echo "  Skipping (already parsed): $BASENAME"
                continue
            fi

            echo "  Parsing: $BASENAME"
            FOUND=$((FOUND + 1))
            python3 -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
from src.parsing.behavior_classifier import parse_trace_file
parse_trace_file('$TRACE_FILE', '$PARSED_DIR/${BASENAME}_parsed.jsonl')
"
        done
        if [ "$FOUND" -eq 0 ]; then
            echo "  No new trace files to parse."
            echo "  (To re-parse, delete files in $PARSED_DIR/)"
        fi
        echo ">>> Parsing complete."
        ;;

    features)
        echo ">>> Extracting features from parsed traces..."
        for PARSED_FILE in "$PARSED_DIR"/*_parsed.jsonl; do
            [ -f "$PARSED_FILE" ] || continue
            BASENAME=$(basename "$PARSED_FILE" _parsed.jsonl)

            # Skip if already extracted
            if [ -f "$FEATURES_DIR/${BASENAME}_features.csv" ]; then
                echo "  Skipping (already extracted): $BASENAME"
                continue
            fi

            echo "  Features: $BASENAME"
            python3 -c "
import logging
logging.basicConfig(level=logging.INFO)
from src.features.feature_pipeline import extract_features_from_file
extract_features_from_file('$PARSED_FILE', '$FEATURES_DIR/${BASENAME}_features.csv')
"
        done
        echo ">>> Feature extraction complete."
        ;;

    annotate)
        echo ">>> Creating annotation template..."
        # Combine all trace files (excluding pilot and SC files)
        COMBINED="$TRACES_DIR/_combined_for_annotation.jsonl"
        rm -f "$COMBINED"
        for f in "$TRACES_DIR"/*.jsonl; do
            [ -f "$f" ] || continue
            case "$(basename "$f")" in
                pilot_*|_*|*_sc.jsonl|*_sc_*.jsonl) continue ;;
            esac
            cat "$f" >> "$COMBINED"
        done
        if [ -s "$COMBINED" ]; then
            python3 src/parsing/parser_evaluation.py \
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
            rm -f "$COMBINED"
            exit 1
        fi
        ;;

    evaluate)
        echo ">>> Evaluating parser accuracy..."
        PYTHONPATH="$PROJECT_DIR" python3 src/parsing/parser_evaluation.py \
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
