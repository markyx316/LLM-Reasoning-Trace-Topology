"""
baseline_c_handcrafted.py - Baseline C: Full Handcrafted Feature Classifier.

Uses all 23 handcrafted features from the main feature pipeline:

  Group 1 — Length & Proportion (9):
    total_tokens, total_episodes,
    prop_forward, prop_verification, prop_backtrack, prop_restart,
    prop_hesitation, prop_subgoal, prop_conclusion

  Group 2 — Structural / Topological (10):
    backtrack_count, verification_count, restart_count,
    vf_ratio, bt_position_mean, first_conclusion_pos,
    v_clustering, max_forward_run, transition_entropy, cycle_count

  Group 3 — Content-Free Meta (4):
    wait_ratio, question_mark_count, negation_count,
    repetition_rate_4gram

Trained with Logistic Regression, Random Forest, and XGBoost (if available)
via 5-fold stratified CV.

Reads from pre-extracted feature CSVs in data/features/.

Usage:
    # Single dataset
    PYTHONPATH=. python src/baselines/baseline_c_handcrafted.py \\
        --features data/features/math500_qwen7b_features.csv \\
        --output results/baseline_c_math500_qwen7b.json

    # All datasets at once
    PYTHONPATH=. python src/baselines/baseline_c_handcrafted.py --all
"""

import argparse
import json
import logging
import os

import numpy as np

from src.modeling.train_and_evaluate import (
    load_features,
    train_cv,
    get_classifiers,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset × Model manifest
# ---------------------------------------------------------------------------

ALL_DATASETS = [
    ("data/features/math500_qwen7b_features.csv",       "results/baseline_c_math500_qwen7b.json"),
    ("data/features/math500_llama8b_features.csv",      "results/baseline_c_math500_llama8b.json"),
    ("data/features/gsm8k_qwen7b_features.csv",         "results/baseline_c_gsm8k_qwen7b.json"),
    ("data/features/gsm8k_llama8b_features.csv",        "results/baseline_c_gsm8k_llama8b.json"),
    ("data/features/gpqa_diamond_qwen7b_features.csv",  "results/baseline_c_gpqa_diamond_qwen7b.json"),
    ("data/features/gpqa_diamond_llama8b_features.csv", "results/baseline_c_gpqa_diamond_llama8b.json"),
    ("data/features/arc_challenge_qwen7b_features.csv", "results/baseline_c_arc_challenge_qwen7b.json"),
    ("data/features/arc_challenge_llama8b_features.csv","results/baseline_c_arc_challenge_llama8b.json"),
]


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_baseline_c(csv_path: str, output_path: str) -> dict:
    """Run Baseline C on a single feature CSV and save results."""
    logger.info(f"Loading features: {csv_path}")
    X, y, feature_names, _ = load_features(csv_path)
    logger.info(f"  Samples: {len(y)}  |  Positive rate: {y.mean():.1%}  |  Features: {len(feature_names)}")

    dataset_name = os.path.basename(csv_path).replace("_features.csv", "")

    cv_results = {}
    for clf_name in get_classifiers():
        result = train_cv(X, y, feature_names, classifier_name=clf_name, n_splits=5)
        s = result["summary"]
        logger.info(
            f"  [{clf_name}]  AUROC: {s['auroc_mean']:.3f} ± {s['auroc_std']:.3f}  "
            f"AUPRC: {s['auprc_mean']:.3f} ± {s['auprc_std']:.3f}  "
            f"ECE: {s['ece_mean']:.3f} ± {s['ece_std']:.3f}  "
            f"Acc@80: {s['accuracy_at_80_mean']:.3f}"
        )
        cv_results[clf_name] = {
            "summary": result["summary"],
            "aggregated_metrics": result["aggregated_metrics"],
            "fold_metrics": result["fold_metrics"],
            "feature_importance": result["feature_importance"],
            "predictions": result["predictions"],
        }

    results = {
        "dataset": dataset_name,
        "n_samples": int(len(y)),
        "n_correct": int(y.sum()),
        "n_incorrect": int((1 - y).sum()),
        "base_accuracy": float(y.mean()),
        "baseline_c": {
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "classifiers": cv_results,
        },
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info(f"  Saved: {output_path}")

    return results


def print_summary_table(all_results: list[dict]):
    clfs = list(next(iter(all_results))["baseline_c"]["classifiers"].keys())
    width = 75 + 18 * (len(clfs) - 1)
    print("\n" + "=" * width)
    print("BASELINE C — HANDCRAFTED FEATURES SUMMARY")
    print("=" * width)
    header = f"{'Dataset':40s}"
    for clf in clfs:
        short = clf.replace("logistic_regression", "LR").replace("random_forest", "RF").replace("xgboost", "XGB")
        header += f" {'AUROC['+short+']':>16s}"
    print(header)
    print("-" * width)
    for r in all_results:
        row = f"{r['dataset']:40s}"
        for clf in clfs:
            s = r["baseline_c"]["classifiers"][clf]["summary"]
            row += f" {s['auroc_mean']:.3f}±{s['auroc_std']:.3f}  "
        print(row)
    print("=" * width)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Baseline C: Full Handcrafted Feature Classifier")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--features", metavar="PATH", help="Path to a single feature CSV")
    group.add_argument("--all", action="store_true", help="Run on all 8 standard dataset × model pairs")
    parser.add_argument("--output", metavar="PATH", help="Output JSON path (required with --features)")
    args = parser.parse_args()

    if args.all:
        all_results = []
        for csv_path, out_path in ALL_DATASETS:
            if not os.path.exists(csv_path):
                logger.warning(f"Skipping (not found): {csv_path}")
                continue
            result = run_baseline_c(csv_path, out_path)
            all_results.append(result)
        if all_results:
            print_summary_table(all_results)
    else:
        if not args.output:
            parser.error("--output is required when --features is used")
        run_baseline_c(args.features, args.output)


if __name__ == "__main__":
    main()
