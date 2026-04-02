"""
baseline_b_lexical.py - Baseline B: Lexical Cue Classifier.

Uses only surface-level lexical signals from the raw reasoning trace text
to predict answer correctness. No parsing of cognitive structure required.

Features (all normalized per 100 tokens unless otherwise noted):
  - wait_ratio       : "wait", "hmm", "uh", "um" family
  - maybe_ratio      : "maybe", "perhaps", "possibly", "might"
  - verify_ratio     : "verify", "check", "let me verify", "double-check", "confirm"
  - actually_ratio   : "actually", "but actually", "wait actually"
  - negation_ratio   : "no", "not", "wrong", "incorrect", "error", "mistake", contractions
  - question_mark_rate: question marks per token
  - repetition_rate  : fraction of 4-grams that appear more than once

Trained with logistic regression + 5-fold stratified CV.

Usage:
    # Single dataset
    PYTHONPATH=. python src/baselines/baseline_b_lexical.py \\
        --traces data/traces/math500_qwen7b_traces.jsonl \\
        --output results/baseline_b_math500_qwen7b.json

    # All datasets at once
    PYTHONPATH=. python src/baselines/baseline_b_lexical.py --all
"""

import argparse
import json
import logging
import os
import re

import numpy as np

from src.modeling.train_and_evaluate import train_cv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset × Model manifest
# ---------------------------------------------------------------------------

ALL_DATASETS = [
    ("data/traces/math500_qwen7b_traces.jsonl",       "results/baseline_b_math500_qwen7b.json"),
    ("data/traces/math500_llama8b_traces.jsonl",      "results/baseline_b_math500_llama8b.json"),
    ("data/traces/gsm8k_qwen7b_traces.jsonl",         "results/baseline_b_gsm8k_qwen7b.json"),
    ("data/traces/gsm8k_llama8b_traces.jsonl",        "results/baseline_b_gsm8k_llama8b.json"),
    ("data/traces/gpqa_diamond_qwen7b_traces.jsonl",  "results/baseline_b_gpqa_diamond_qwen7b.json"),
    ("data/traces/gpqa_diamond_llama8b_traces.jsonl", "results/baseline_b_gpqa_diamond_llama8b.json"),
    ("data/traces/arc_challenge_qwen7b_traces.jsonl", "results/baseline_b_arc_challenge_qwen7b.json"),
    ("data/traces/arc_challenge_llama8b_traces.jsonl","results/baseline_b_arc_challenge_llama8b.json"),
]

FEATURE_NAMES = [
    "wait_ratio",
    "maybe_ratio",
    "verify_ratio",
    "actually_ratio",
    "negation_ratio",
    "question_mark_rate",
    "repetition_rate_4gram",
]

# ---------------------------------------------------------------------------
# Lexical pattern definitions
# ---------------------------------------------------------------------------

_WAIT_WORDS = frozenset({
    "wait", "hmm", "hmmm", "hmmmm", "uh", "um", "well",
})

_MAYBE_PATTERNS = re.compile(
    r'\b(maybe|perhaps|possibly|might|could be|probably)\b', re.IGNORECASE
)

_VERIFY_PATTERNS = re.compile(
    r'\b(verify|verified|verif(?:ying|ication)|'
    r'check|checked|checking|'
    r'let me (?:verify|check|confirm|re-?check)|'
    r'double.?check(?:ed|ing)?|'
    r'confirm(?:ed|ing)?)\b',
    re.IGNORECASE,
)

_ACTUALLY_PATTERNS = re.compile(
    r'\b(actually|but actually|wait actually|in fact|on second thought)\b',
    re.IGNORECASE,
)

_NEGATION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'\bno\b', r'\bnot\b', r'\bwrong\b', r'\bincorrect\b',
        r'\berror\b', r'\bmistake\b',
        r"\bcan't\b", r"\bcannot\b", r"\bdon't\b", r"\bdoesn't\b",
        r"\bisn't\b", r"\bwon't\b", r"\bwouldn't\b", r"\bshouldn't\b",
    ]
]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_lexical_features(trace_text: str) -> list[float]:
    """
    Extract 7 lexical cue features from raw trace text.

    Returns a list aligned with FEATURE_NAMES.
    """
    if not trace_text or not trace_text.strip():
        return [0.0] * len(FEATURE_NAMES)

    text_lower = trace_text.lower()
    tokens = text_lower.split()
    total = max(len(tokens), 1)

    # 1. wait_ratio
    wait_count = sum(
        1 for t in tokens if t.strip(".,!?;:'\"") in _WAIT_WORDS
    )
    wait_ratio = wait_count / total

    # 2. maybe_ratio
    maybe_count = len(_MAYBE_PATTERNS.findall(trace_text))
    maybe_ratio = maybe_count / total

    # 3. verify_ratio
    verify_count = len(_VERIFY_PATTERNS.findall(trace_text))
    verify_ratio = verify_count / total

    # 4. actually_ratio
    actually_count = len(_ACTUALLY_PATTERNS.findall(trace_text))
    actually_ratio = actually_count / total

    # 5. negation_ratio
    negation_count = sum(len(p.findall(trace_text)) for p in _NEGATION_PATTERNS)
    negation_ratio = negation_count / total

    # 6. question_mark_rate
    question_mark_rate = trace_text.count("?") / total

    # 7. repetition_rate_4gram
    if len(tokens) >= 4:
        four_grams = [tuple(tokens[i:i + 4]) for i in range(len(tokens) - 3)]
        unique = len(set(four_grams))
        repetition_rate = 1.0 - (unique / max(len(four_grams), 1))
    else:
        repetition_rate = 0.0

    return [
        wait_ratio,
        maybe_ratio,
        verify_ratio,
        actually_ratio,
        negation_ratio,
        question_mark_rate,
        repetition_rate,
    ]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_lexical_features(jsonl_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Read a JSONL trace file and return (X, y, feature_names).

    X shape: (N, 7)
    y shape: (N,)
    """
    X_rows = []
    y_rows = []

    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            trace_text = d.get("reasoning_trace", "")
            feats = extract_lexical_features(trace_text)
            X_rows.append(feats)
            y_rows.append(int(bool(d["is_correct"])))

    if not X_rows:
        raise ValueError(f"No records loaded from {jsonl_path}")

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=int)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, FEATURE_NAMES


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_baseline_b(jsonl_path: str, output_path: str) -> dict:
    """Run Baseline B on a single JSONL trace file and save results."""
    logger.info(f"Loading traces: {jsonl_path}")
    X, y, feature_names = load_lexical_features(jsonl_path)
    logger.info(f"  Samples: {len(y)}  |  Positive rate: {y.mean():.1%}")

    dataset_name = os.path.basename(jsonl_path).replace("_traces.jsonl", "")

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
        "baseline_b": {
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
    print("\n" + "=" * 75)
    print("BASELINE B — LEXICAL CUE SUMMARY")
    print("=" * 75)
    print(f"{'Dataset':40s} {'AUROC':>12s} {'AUPRC':>12s} {'Acc@80':>8s}")
    print("-" * 75)
    for r in all_results:
        s = r["baseline_b"]["summary"]
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

    parser = argparse.ArgumentParser(description="Baseline B: Lexical Cue Classifier")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--traces", metavar="PATH", help="Path to a single JSONL trace file")
    group.add_argument("--all", action="store_true", help="Run on all 8 standard dataset × model pairs")
    parser.add_argument("--output", metavar="PATH", help="Output JSON path (required with --traces)")
    args = parser.parse_args()

    if args.all:
        all_results = []
        for traces_path, out_path in ALL_DATASETS:
            if not os.path.exists(traces_path):
                logger.warning(f"Skipping (not found): {traces_path}")
                continue
            result = run_baseline_b(traces_path, out_path)
            all_results.append(result)
        if all_results:
            print_summary_table(all_results)
    else:
        if not args.output:
            parser.error("--output is required when --traces is used")
        run_baseline_b(args.traces, args.output)


if __name__ == "__main__":
    main()
