"""
baseline_g_hvr.py - Baseline G: Hedge-to-Verify Ratio (HVR).

Direct port of the SELFDOUBT signal (arXiv:2505.23845): the closest published
text-only, single-pass, black-box behavioral UQ score. SELFDOUBT reports that
traces with **zero** hedging markers are correct ~96% of the time across 7
models / 3 multi-step benchmarks; the residual is captured by a
hedge-to-verify ratio.

Three deliverables in one file:
  1. The raw HVR scalar score (`hvr = hedge_count / (verify_count + 1)`).
     Used as a calibration-free score directly: confidence = 1 / (1 + hvr).
  2. A "zero-hedge" rule: precision/recall when we predict correct iff
     hedge_count == 0. Reports SELFDOUBT-style accept-set numbers.
  3. A 4-feature LR (`hedge_count`, `verify_count`, `hvr`, `zero_hedge_flag`)
     trained 5-fold so we get an apples-to-apples AUROC against the other
     baselines.

Hedge markers (lowercased, word-boundary regex):
  maybe, perhaps, possibly, i think, i guess, i believe, not sure, might,
  could be, somewhat, probably, presumably, seems, appears.

Verify count: from the existing 6-class parser (`rule_based_parser.parse_trace`).
We count episodes labelled VERIFY (the canonical structural signal of self-
checking). This is the cleanest combination of "hedge from raw text" + "verify
from parsed structure"; it is faithful to SELFDOUBT while avoiding double-
counting "let me check" as both a hedge and a verify.

Usage:
    PYTHONPATH=. python src/baselines/baseline_g_hvr.py \\
        --traces data/traces/math500_qwen7b_traces.jsonl \\
        --output results/baseline_g_math500_qwen7b.json

    PYTHONPATH=. python src/baselines/baseline_g_hvr.py --all
"""

import argparse
import json
import logging
import os
import re

import numpy as np

from src.modeling.train_and_evaluate import train_cv
from src.modeling.cv_utils import evaluate

logger = logging.getLogger(__name__)


ALL_DATASETS = [
    ("data/traces/math500_qwen7b_traces.jsonl",       "results/baseline_g_math500_qwen7b.json"),
    ("data/traces/math500_llama8b_traces.jsonl",      "results/baseline_g_math500_llama8b.json"),
    ("data/traces/gsm8k_qwen7b_traces.jsonl",         "results/baseline_g_gsm8k_qwen7b.json"),
    ("data/traces/gsm8k_llama8b_traces.jsonl",        "results/baseline_g_gsm8k_llama8b.json"),
    ("data/traces/gpqa_diamond_qwen7b_traces.jsonl",  "results/baseline_g_gpqa_diamond_qwen7b.json"),
    ("data/traces/gpqa_diamond_llama8b_traces.jsonl", "results/baseline_g_gpqa_diamond_llama8b.json"),
    ("data/traces/arc_challenge_qwen7b_traces.jsonl", "results/baseline_g_arc_challenge_qwen7b.json"),
    ("data/traces/arc_challenge_llama8b_traces.jsonl","results/baseline_g_arc_challenge_llama8b.json"),
]

HEDGE_PATTERNS = [
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bpossibly\b",
    r"\bi\s+think\b",
    r"\bi\s+guess\b",
    r"\bi\s+believe\b",
    r"\bnot\s+sure\b",
    r"\bmight(?:\s+be)?\b",
    r"\bcould\s+be\b",
    r"\bsomewhat\b",
    r"\bprobably\b",
    r"\bpresumably\b",
    r"\bseems?\s+(?:like|to|that)\b",
    r"\bappears?\s+(?:to|that)\b",
]
HEDGE_RE = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)

FEATURE_NAMES = ["hedge_count", "verify_count", "hvr", "zero_hedge_flag"]


def _count_verifies(trace: str) -> int:
    """Use the canonical 6-class parser; count VERIFY episodes."""
    from src.parsing.rule_based_parser import parse_trace, BehaviorType
    try:
        eps = parse_trace(trace) if trace else []
    except Exception:
        eps = []
    return sum(1 for ep in eps if ep.behavior == BehaviorType.VERIFY)


def load_hvr_features(jsonl_path: str):
    """Returns (X, y, hedge_counts, verify_counts) as numpy arrays."""
    rows = []
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            trace = d.get("reasoning_trace") or d.get("full_response") or ""
            hedges = len(HEDGE_RE.findall(trace))
            verifies = _count_verifies(trace)
            hvr = hedges / (verifies + 1.0)
            zero_hedge = 1.0 if hedges == 0 else 0.0
            rows.append((float(hedges), float(verifies), float(hvr), zero_hedge,
                         int(bool(d["is_correct"]))))

    if not rows:
        raise ValueError(f"No records loaded from {jsonl_path}")

    data = np.array(rows, dtype=float)
    X = data[:, :4]
    y = data[:, 4].astype(int)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


def zero_hedge_stats(X: np.ndarray, y: np.ndarray) -> dict:
    """Precision / recall / coverage of the SELFDOUBT-style 'zero hedge' rule."""
    zero_hedge = (X[:, 0] == 0)
    n_total = len(y)
    n_accept = int(zero_hedge.sum())
    if n_accept == 0:
        return {"n_accept": 0, "coverage": 0.0, "precision": float("nan"),
                "lift_over_base_acc": float("nan"), "base_accuracy": float(y.mean())}
    precision = float(y[zero_hedge].mean())
    base = float(y.mean())
    return {
        "n_accept": n_accept,
        "coverage": float(n_accept / n_total),
        "precision": precision,
        "base_accuracy": base,
        "lift_over_base_acc": precision - base,
    }


def hvr_score_metric(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Use the raw HVR as a calibration-free confidence score (no training).
    Confidence := 1 / (1 + hvr) so low HVR -> high confidence.
    """
    hvr = X[:, 2]
    confidence = 1.0 / (1.0 + hvr)
    return evaluate(y, confidence, name="hvr_raw_score")


def run_baseline_g(jsonl_path: str, output_path: str) -> dict:
    logger.info(f"Loading + parsing traces: {jsonl_path}")
    X, y = load_hvr_features(jsonl_path)
    logger.info(
        f"  Samples: {len(y)}  pos_rate: {y.mean():.1%}  "
        f"mean_hedges: {X[:, 0].mean():.2f}  mean_verifies: {X[:, 1].mean():.2f}  "
        f"mean_hvr: {X[:, 2].mean():.3f}"
    )

    dataset_name = os.path.basename(jsonl_path).replace("_traces.jsonl", "").replace(".jsonl", "")

    # 1. Raw HVR as a score
    raw = hvr_score_metric(X, y)

    # 2. Zero-hedge rule (SELFDOUBT 96% claim)
    zh = zero_hedge_stats(X, y)
    logger.info(
        f"  Zero-hedge rule: accept={zh['n_accept']}/{len(y)} "
        f"(coverage {zh['coverage']:.1%})  "
        f"precision={zh['precision']:.3f}  vs base_acc {zh['base_accuracy']:.3f}  "
        f"lift={zh['lift_over_base_acc']:+.3f}"
    )

    # 3. 4-feature LR
    cv_result = train_cv(X, y, FEATURE_NAMES, classifier_name="logistic_regression", n_splits=5)
    s = cv_result["summary"]
    logger.info(
        f"  LR-4feat: AUROC: {s['auroc_mean']:.3f}±{s['auroc_std']:.3f}  "
        f"AUPRC: {s['auprc_mean']:.3f}±{s['auprc_std']:.3f}  "
        f"ECE: {s['ece_mean']:.3f}  Acc@80: {s['accuracy_at_80_mean']:.3f}  "
        f"raw_HVR_AUROC: {raw['auroc']:.3f}  raw_HVR_PRR: {raw['prr']:+.3f}"
    )

    results = {
        "dataset": dataset_name,
        "n_samples": int(len(y)),
        "n_correct": int(y.sum()),
        "n_incorrect": int((1 - y).sum()),
        "base_accuracy": float(y.mean()),
        "baseline_g": {
            "features": FEATURE_NAMES,
            "raw_hvr_score": raw,
            "zero_hedge_rule": zh,
            "lr_classifier": {
                "summary": cv_result["summary"],
                "aggregated_metrics": cv_result["aggregated_metrics"],
                "fold_metrics": cv_result["fold_metrics"],
                "feature_importance": cv_result["feature_importance"],
                "predictions": cv_result["predictions"],
            },
        },
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info(f"  Saved: {output_path}")
    return results


def print_summary_table(all_results: list[dict]):
    print("\n" + "=" * 95)
    print("BASELINE G — HEDGE-TO-VERIFY RATIO (SELFDOUBT-port)")
    print("=" * 95)
    print(f"{'Dataset':<35s} {'LR_AUROC':>10s} {'raw_HVR_AUROC':>14s} "
          f"{'zero-hedge_prec':>15s} {'cov':>6s} {'lift':>7s}")
    print("-" * 95)
    for r in all_results:
        s = r["baseline_g"]["lr_classifier"]["summary"]
        raw = r["baseline_g"]["raw_hvr_score"]
        zh = r["baseline_g"]["zero_hedge_rule"]
        print(
            f"{r['dataset']:<35s} "
            f"{s['auroc_mean']:>5.3f}±{s['auroc_std']:.3f}  "
            f"{raw['auroc']:>13.3f}  "
            f"{zh['precision']:>14.3f}  "
            f"{zh['coverage']:>6.2%}  "
            f"{zh['lift_over_base_acc']:>+7.3f}"
        )
    print("=" * 95)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Baseline G: Hedge-to-Verify Ratio")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--traces", metavar="PATH",
                       help="Path to a single JSONL trace file")
    group.add_argument("--all", action="store_true",
                       help="Run on all 8 standard dataset × model pairs")
    parser.add_argument("--output", metavar="PATH",
                        help="Output JSON path (required when --traces is used)")
    args = parser.parse_args()

    if args.all:
        all_results = []
        for traces_path, out_path in ALL_DATASETS:
            if not os.path.exists(traces_path):
                logger.warning(f"Skipping (not found): {traces_path}")
                continue
            try:
                result = run_baseline_g(traces_path, out_path)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Failed on {traces_path}: {e}")
        if all_results:
            print_summary_table(all_results)
    else:
        if not args.output:
            parser.error("--output is required when --traces is used")
        run_baseline_g(args.traces, args.output)


if __name__ == "__main__":
    main()
