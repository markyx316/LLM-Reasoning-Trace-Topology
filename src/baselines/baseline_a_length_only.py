"""
baseline_a_length_only.py - Baseline A: Length-Only Classifier.

Uses only two features to predict answer correctness:
  - trace_token_count : actual tokenizer-count of the reasoning trace
  - answer_length     : word count of answer_text (proxy for answer tokens)

Trained with logistic regression + 5-fold stratified CV, evaluated with the
same metrics as the main experiments (AUROC, AUPRC, ECE, Acc@80, Acc@90).

Usage:
    # Single dataset
    PYTHONPATH=. python src/baselines/baseline_a_length_only.py \\
        --traces data/traces/math500_qwen7b_traces.jsonl \\
        --output results/baseline_a_math500_qwen7b.json

    # All 8 datasets at once
    PYTHONPATH=. python src/baselines/baseline_a_length_only.py --all
"""

import argparse
import json
import logging
import os

import numpy as np

from src.modeling.train_and_evaluate import train_cv, evaluate_predictions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset × Model manifest (relative to repo root)
# ---------------------------------------------------------------------------

ALL_DATASETS = [
    ("data/traces/math500_qwen7b_traces.jsonl",       "results/baseline_a_math500_qwen7b.json"),
    ("data/traces/math500_llama8b_traces.jsonl",      "results/baseline_a_math500_llama8b.json"),
    ("data/traces/gsm8k_qwen7b_traces.jsonl",         "results/baseline_a_gsm8k_qwen7b.json"),
    ("data/traces/gsm8k_llama8b_traces.jsonl",        "results/baseline_a_gsm8k_llama8b.json"),
    ("data/traces/gpqa_diamond_qwen7b_traces.jsonl",  "results/baseline_a_gpqa_diamond_qwen7b.json"),
    ("data/traces/gpqa_diamond_llama8b_traces.jsonl", "results/baseline_a_gpqa_diamond_llama8b.json"),
    ("data/traces/arc_challenge_qwen7b_traces.jsonl", "results/baseline_a_arc_challenge_qwen7b.json"),
    ("data/traces/arc_challenge_llama8b_traces.jsonl","results/baseline_a_arc_challenge_llama8b.json"),
]

FEATURE_NAMES = ["trace_token_count", "answer_length"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_length_features(jsonl_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Read a JSONL trace file and return (X, y, feature_names).

    X shape: (N, 2) — [trace_token_count, answer_length]
    y shape: (N,)   — is_correct (0/1)
    """
    rows = []
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)

            # Trace length: prefer pre-computed token count, fall back to word count
            trace_len = d.get("trace_token_count")
            if trace_len is None:
                trace_len = len(d.get("reasoning_trace", "").split())

            # Answer length: word count of the final answer text
            answer_len = len(d.get("answer_text", "").split())

            is_correct = int(bool(d["is_correct"]))

            rows.append((float(trace_len), float(answer_len), is_correct))

    if not rows:
        raise ValueError(f"No records loaded from {jsonl_path}")

    data = np.array(rows, dtype=float)
    X = data[:, :2]
    y = data[:, 2].astype(int)

    # Safety: replace any NaN / inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, FEATURE_NAMES


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_baseline_a(jsonl_path: str, output_path: str) -> dict:
    """
    Run Baseline A on a single JSONL trace file and save results.

    Returns the results dict.
    """
    logger.info(f"Loading traces: {jsonl_path}")
    X, y, feature_names = load_length_features(jsonl_path)
    logger.info(f"  Samples: {len(y)}  |  Positive rate: {y.mean():.1%}")

    dataset_name = os.path.basename(jsonl_path).replace("_traces.jsonl", "")

    # 5-fold CV with logistic regression
    cv_result = train_cv(X, y, feature_names, classifier_name="logistic_regression", n_splits=5)

    s = cv_result["summary"]
    logger.info(
        f"  AUROC: {s['auroc_mean']:.3f} ± {s['auroc_std']:.3f}  "
        f"AUPRC: {s['auprc_mean']:.3f} ± {s['auprc_std']:.3f}  "
        f"ECE: {s['ece_mean']:.3f} ± {s['ece_std']:.3f}  "
        f"Acc@80: {s['accuracy_at_80_mean']:.3f}"
    )

    results = {
        "dataset": dataset_name,
        "n_samples": int(len(y)),
        "n_correct": int(y.sum()),
        "n_incorrect": int((1 - y).sum()),
        "base_accuracy": float(y.mean()),
        "baseline_a": {
            "features": feature_names,
            "classifier": "logistic_regression",
            "summary": cv_result["summary"],
            "aggregated_metrics": cv_result["aggregated_metrics"],
            "fold_metrics": cv_result["fold_metrics"],
            "feature_importance": cv_result["feature_importance"],
            "predictions": cv_result["predictions"],
        },
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info(f"  Saved: {output_path}")

    return results


def print_summary_table(all_results: list[dict]):
    """Print a compact comparison table across all datasets."""
    print("\n" + "=" * 75)
    print("BASELINE A — LENGTH ONLY SUMMARY")
    print("=" * 75)
    print(f"{'Dataset':40s} {'AUROC':>12s} {'AUPRC':>12s} {'Acc@80':>8s}")
    print("-" * 75)
    for r in all_results:
        s = r["baseline_a"]["summary"]
        print(
            f"{r['dataset']:40s} "
            f"{s['auroc_mean']:.3f}±{s['auroc_std']:.3f}  "
            f"{s['auprc_mean']:.3f}±{s['auprc_std']:.3f}  "
            f"{s['accuracy_at_80_mean']:.3f}"
        )
    print("=" * 75)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Baseline A: Length-Only Classifier")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--traces", metavar="PATH",
        help="Path to a single JSONL trace file",
    )
    group.add_argument(
        "--all", action="store_true",
        help="Run on all 8 standard dataset × model pairs",
    )
    parser.add_argument(
        "--output", metavar="PATH",
        help="Output JSON path (required when --traces is used)",
    )
    args = parser.parse_args()

    if args.all:
        all_results = []
        for traces_path, out_path in ALL_DATASETS:
            if not os.path.exists(traces_path):
                logger.warning(f"Skipping (not found): {traces_path}")
                continue
            result = run_baseline_a(traces_path, out_path)
            all_results.append(result)
        if all_results:
            print_summary_table(all_results)
    else:
        if not args.output:
            parser.error("--output is required when --traces is used")
        run_baseline_a(args.traces, args.output)


if __name__ == "__main__":
    main()
