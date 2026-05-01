"""
baseline_e_perplexity.py - Baseline E: Logit-confidence (perplexity).

Uses the per-trace mean log-probability that the HF generation pipeline already
stores in every trace record (`mean_log_prob`). Trains LR + 5-fold CV with the
same metric stack as the other baselines so PRR / AUROC / ECE are comparable.

Features:
  - mean_log_prob   : per-token mean log-probability under the generating model
                      (higher = more confident; available for HF backend only)
  - perplexity      : exp(-mean_log_prob), monotone transform of mean_log_prob
                      kept as a redundant feature so a tree learner can split on
                      it without inverting the sign
  - seq_log_prob    : mean_log_prob * total_generated_tokens, the total
                      log-likelihood. Highly correlated with length but feeds
                      the LR bias term cleanly.

Note on DeepConf proper (Zhao et al., 2025, arXiv:2508.15260):
  DeepConf takes the sliding-window minimum of per-token log-probabilities and
  uses *that* as the confidence score. We cannot compute it here because the
  trace JSONLs only store the scalar mean — the per-token sequence was discarded
  at generation time. A faithful DeepConf-text baseline requires regenerating
  traces with `output_scores=True`, which is a separate work item.

R1 (DeepSeek API) traces have mean_log_prob == 0 because the API doesn't return
per-token log probs. They are filtered out here; the HF (Qwen-7B / Llama-8B)
traces are the population this baseline applies to.

Usage:
    PYTHONPATH=. python src/baselines/baseline_e_perplexity.py \\
        --traces data/traces/math500_qwen7b_traces.jsonl \\
        --output results/baseline_e_math500_qwen7b.json

    PYTHONPATH=. python src/baselines/baseline_e_perplexity.py --all
"""

import argparse
import json
import logging
import math
import os

import numpy as np

from src.modeling.train_and_evaluate import train_cv

logger = logging.getLogger(__name__)


ALL_DATASETS = [
    ("data/traces/math500_qwen7b_traces.jsonl",       "results/baseline_e_math500_qwen7b.json"),
    ("data/traces/math500_llama8b_traces.jsonl",      "results/baseline_e_math500_llama8b.json"),
    ("data/traces/gsm8k_qwen7b_traces.jsonl",         "results/baseline_e_gsm8k_qwen7b.json"),
    ("data/traces/gsm8k_llama8b_traces.jsonl",        "results/baseline_e_gsm8k_llama8b.json"),
    ("data/traces/gpqa_diamond_qwen7b_traces.jsonl",  "results/baseline_e_gpqa_diamond_qwen7b.json"),
    ("data/traces/gpqa_diamond_llama8b_traces.jsonl", "results/baseline_e_gpqa_diamond_llama8b.json"),
    ("data/traces/arc_challenge_qwen7b_traces.jsonl", "results/baseline_e_arc_challenge_qwen7b.json"),
    ("data/traces/arc_challenge_llama8b_traces.jsonl","results/baseline_e_arc_challenge_llama8b.json"),
]

FEATURE_NAMES = ["mean_log_prob", "perplexity", "seq_log_prob"]


def load_perplexity_features(jsonl_path: str) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """
    Returns (X, y, feature_names, n_skipped).

    n_skipped counts records with mean_log_prob == 0 (API traces or missing).
    """
    rows = []
    skipped = 0
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)

            mlp = d.get("mean_log_prob", 0.0) or 0.0
            n_tok = d.get("total_generated_tokens") or d.get("trace_token_count") or 0

            if mlp == 0.0:
                skipped += 1
                continue

            ppl = math.exp(-mlp) if mlp > -50 else math.exp(50)  # clamp overflow
            seq_lp = mlp * n_tok

            rows.append((float(mlp), float(ppl), float(seq_lp),
                         int(bool(d["is_correct"]))))

    if not rows:
        raise ValueError(f"No usable records in {jsonl_path} (all had mean_log_prob == 0)")

    data = np.array(rows, dtype=float)
    X = data[:, :3]
    y = data[:, 3].astype(int)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y, FEATURE_NAMES, skipped


def run_baseline_e(jsonl_path: str, output_path: str) -> dict:
    logger.info(f"Loading traces: {jsonl_path}")
    X, y, feature_names, skipped = load_perplexity_features(jsonl_path)
    logger.info(f"  Samples: {len(y)} (skipped {skipped} with mean_log_prob==0)  "
                f"|  Positive rate: {y.mean():.1%}")

    dataset_name = os.path.basename(jsonl_path).replace("_traces.jsonl", "").replace(".jsonl", "")

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
        "n_skipped_zero_mlp": int(skipped),
        "n_correct": int(y.sum()),
        "n_incorrect": int((1 - y).sum()),
        "base_accuracy": float(y.mean()),
        "baseline_e": {
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
    print("\n" + "=" * 80)
    print("BASELINE E — PERPLEXITY (mean_log_prob, HF backend only)")
    print("=" * 80)
    print(f"{'Dataset':40s} {'AUROC':>14s} {'AUPRC':>14s} {'Acc@80':>8s}")
    print("-" * 80)
    for r in all_results:
        s = r["baseline_e"]["summary"]
        print(
            f"{r['dataset']:40s} "
            f"{s['auroc_mean']:.3f}±{s['auroc_std']:.3f}  "
            f"{s['auprc_mean']:.3f}±{s['auprc_std']:.3f}  "
            f"{s['accuracy_at_80_mean']:.3f}"
        )
    print("=" * 80)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Baseline E: Perplexity (mean_log_prob)")
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
                result = run_baseline_e(traces_path, out_path)
                all_results.append(result)
            except ValueError as e:
                logger.warning(f"Skipping {traces_path}: {e}")
        if all_results:
            print_summary_table(all_results)
    else:
        if not args.output:
            parser.error("--output is required when --traces is used")
        run_baseline_e(args.traces, args.output)


if __name__ == "__main__":
    main()
