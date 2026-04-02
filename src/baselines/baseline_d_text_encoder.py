"""
baseline_d_text_encoder.py - Baseline D: Raw Trace Text Encoder.

Directly encodes the full reasoning trace with a TF-IDF vectorizer and
feeds the resulting sparse representation into a logistic regression
classifier. No hand-engineered features, no parsing — pure text.

Pipeline:
  reasoning_trace  →  TF-IDF (unigrams + bigrams, top 20 000 terms)
                   →  LogisticRegression (5-fold stratified CV)

Because traces can be arbitrarily long, TF-IDF naturally handles
variable-length input via its bag-of-words representation.

Usage:
    # Single dataset
    PYTHONPATH=. python src/baselines/baseline_d_text_encoder.py \\
        --traces data/traces/math500_qwen7b_traces.jsonl \\
        --output results/baseline_d_math500_qwen7b.json

    # All datasets at once
    PYTHONPATH=. python src/baselines/baseline_d_text_encoder.py --all
"""

import argparse
import json
import logging
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

from src.modeling.train_and_evaluate import compute_ece, compute_selective_generation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset × Model manifest
# ---------------------------------------------------------------------------

ALL_DATASETS = [
    ("data/traces/math500_qwen7b_traces.jsonl",       "results/baseline_d_math500_qwen7b.json"),
    ("data/traces/math500_llama8b_traces.jsonl",      "results/baseline_d_math500_llama8b.json"),
    ("data/traces/gsm8k_qwen7b_traces.jsonl",         "results/baseline_d_gsm8k_qwen7b.json"),
    ("data/traces/gsm8k_llama8b_traces.jsonl",        "results/baseline_d_gsm8k_llama8b.json"),
    ("data/traces/gpqa_diamond_qwen7b_traces.jsonl",  "results/baseline_d_gpqa_diamond_qwen7b.json"),
    ("data/traces/gpqa_diamond_llama8b_traces.jsonl", "results/baseline_d_gpqa_diamond_llama8b.json"),
    ("data/traces/arc_challenge_qwen7b_traces.jsonl", "results/baseline_d_arc_challenge_qwen7b.json"),
    ("data/traces/arc_challenge_llama8b_traces.jsonl","results/baseline_d_arc_challenge_llama8b.json"),
]

# TF-IDF settings
TFIDF_PARAMS = dict(
    ngram_range=(1, 2),
    max_features=20_000,
    sublinear_tf=True,      # Apply log(1+tf) — helps with very long traces
    min_df=2,               # Ignore terms appearing in only 1 document
    strip_accents="unicode",
    lowercase=True,
)

# Logistic Regression settings (same hyperparams as other baselines)
LR_PARAMS = dict(
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_traces(jsonl_path: str) -> tuple[list[str], np.ndarray]:
    """
    Read a JSONL trace file.

    Returns:
        texts : list of raw reasoning_trace strings (length N)
        y     : binary correctness labels, shape (N,)
    """
    texts, labels = [], []
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            texts.append(d.get("reasoning_trace", "") or "")
            labels.append(int(bool(d["is_correct"])))

    if not texts:
        raise ValueError(f"No records loaded from {jsonl_path}")

    return texts, np.array(labels, dtype=int)


# ---------------------------------------------------------------------------
# Evaluation helpers (mirrors train_and_evaluate.evaluate_predictions)
# ---------------------------------------------------------------------------

def _evaluate(y_true: np.ndarray, y_prob: np.ndarray, name: str = "") -> dict:
    auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) >= 2 else 0.5
    auprc = float(average_precision_score(y_true, y_prob))
    ece   = compute_ece(y_true, y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    acc   = float(accuracy_score(y_true, y_pred))
    f1    = float(f1_score(y_true, y_pred, zero_division=0))
    sel   = compute_selective_generation(y_true, y_prob)
    return {
        "method": name,
        "auroc": auroc,
        "auprc": auprc,
        "ece": ece,
        "accuracy": acc,
        "f1": f1,
        "accuracy_at_80": sel["accuracy_at_80"],
        "accuracy_at_90": sel["accuracy_at_90"],
        "au_acc_cov": sel["au_acc_cov"],
    }


# ---------------------------------------------------------------------------
# Cross-validation with TF-IDF fit inside each fold
# ---------------------------------------------------------------------------

def train_cv_tfidf(
    texts: list[str],
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """
    5-fold stratified CV where TF-IDF is fit only on training folds
    (no data leakage from test documents).

    Returns result dict compatible with train_cv output format.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    texts_arr = np.array(texts, dtype=object)

    fold_metrics = []
    all_y_true, all_y_prob, all_indices = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(texts_arr, y)):
        texts_train = texts_arr[train_idx].tolist()
        texts_test  = texts_arr[test_idx].tolist()
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit TF-IDF on training split only
        vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
        X_train = vectorizer.fit_transform(texts_train)
        X_test  = vectorizer.transform(texts_test)

        # Train logistic regression
        clf = LogisticRegression(**LR_PARAMS)
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_test)[:, 1]

        fold_result = _evaluate(y_test, y_prob, f"fold_{fold_idx}")
        fold_metrics.append(fold_result)

        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob)
        all_indices.extend(test_idx)

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)

    agg = _evaluate(all_y_true, all_y_prob, "tfidf_logreg")

    metric_keys = ["auroc", "auprc", "ece", "accuracy_at_80", "accuracy_at_90"]
    summary = {}
    for key in metric_keys:
        vals = [fm[key] for fm in fold_metrics]
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"]  = float(np.std(vals))

    return {
        "classifier": "tfidf_logreg",
        "n_folds": n_splits,
        "tfidf_params": TFIDF_PARAMS,
        "lr_params": LR_PARAMS,
        "aggregated_metrics": agg,
        "summary": summary,
        "fold_metrics": fold_metrics,
        "predictions": {
            "indices": [int(i) for i in all_indices],
            "y_true":  [int(v) for v in all_y_true],
            "y_prob":  [float(p) for p in all_y_prob],
        },
    }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_baseline_d(jsonl_path: str, output_path: str) -> dict:
    """Run Baseline D on a single JSONL trace file and save results."""
    logger.info(f"Loading traces: {jsonl_path}")
    texts, y = load_traces(jsonl_path)
    logger.info(f"  Samples: {len(y)}  |  Positive rate: {y.mean():.1%}")
    logger.info(f"  Avg trace length: {np.mean([len(t.split()) for t in texts]):.0f} words")

    dataset_name = os.path.basename(jsonl_path).replace("_traces.jsonl", "")

    logger.info("  Fitting TF-IDF + LogReg (5-fold CV)...")
    cv_result = train_cv_tfidf(texts, y, n_splits=5)

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
        "baseline_d": cv_result,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info(f"  Saved: {output_path}")

    return results


def print_summary_table(all_results: list[dict]):
    print("\n" + "=" * 75)
    print("BASELINE D — TF-IDF TEXT ENCODER SUMMARY")
    print("=" * 75)
    print(f"{'Dataset':40s} {'AUROC':>12s} {'AUPRC':>12s} {'Acc@80':>8s}")
    print("-" * 75)
    for r in all_results:
        s = r["baseline_d"]["summary"]
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

    parser = argparse.ArgumentParser(description="Baseline D: TF-IDF Text Encoder Classifier")
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
            result = run_baseline_d(traces_path, out_path)
            all_results.append(result)
        if all_results:
            print_summary_table(all_results)
    else:
        if not args.output:
            parser.error("--output is required when --traces is used")
        run_baseline_d(args.traces, args.output)


if __name__ == "__main__":
    main()
