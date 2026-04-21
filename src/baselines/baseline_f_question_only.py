"""
baseline_f_question_only.py - Baseline F: Question-Only Classifier (sanity floor).

Encodes the *question text alone* (not the trace, not the answer) with
all-MiniLM-L6-v2 and trains an LR + 5-fold CV on the resulting 384-d embedding
to predict `is_correct`.

Why this baseline matters (Xiao et al. 2025, "Generalized Correctness Model"
+ arXiv:2509.10625): a non-trivial fraction of "correctness UQ" signal is
question difficulty leakage, not introspection. If question-only AUROC ≥
trace-structural AUROC on a dataset, claims about "trace structure carries
signal" are inflated. This baseline is the *sanity floor* the field now demands.

Cached embeddings are written to a sibling .npz so reruns are instant; delete
the cache to force re-embedding after a parser change.

Usage:
    PYTHONPATH=. python src/baselines/baseline_f_question_only.py \\
        --traces data/traces/math500_qwen7b_traces.jsonl \\
        --output results/baseline_f_math500_qwen7b.json

    PYTHONPATH=. python src/baselines/baseline_f_question_only.py --all
"""

import argparse
import hashlib
import json
import logging
import os

import numpy as np

from src.modeling.train_and_evaluate import train_cv

logger = logging.getLogger(__name__)


ALL_DATASETS = [
    ("data/traces/math500_qwen7b_traces.jsonl",       "results/baseline_f_math500_qwen7b.json"),
    ("data/traces/math500_llama8b_traces.jsonl",      "results/baseline_f_math500_llama8b.json"),
    ("data/traces/gsm8k_qwen7b_traces.jsonl",         "results/baseline_f_gsm8k_qwen7b.json"),
    ("data/traces/gsm8k_llama8b_traces.jsonl",        "results/baseline_f_gsm8k_llama8b.json"),
    ("data/traces/gpqa_diamond_qwen7b_traces.jsonl",  "results/baseline_f_gpqa_diamond_qwen7b.json"),
    ("data/traces/gpqa_diamond_llama8b_traces.jsonl", "results/baseline_f_gpqa_diamond_llama8b.json"),
    ("data/traces/arc_challenge_qwen7b_traces.jsonl", "results/baseline_f_arc_challenge_qwen7b.json"),
    ("data/traces/arc_challenge_llama8b_traces.jsonl","results/baseline_f_arc_challenge_llama8b.json"),
]

CACHE_DIR = "data/question_embeddings"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _cache_path(jsonl_path: str, model_name: str) -> str:
    base = os.path.basename(jsonl_path).replace("_traces.jsonl", "")
    tag = hashlib.sha1(model_name.encode()).hexdigest()[:8]
    return os.path.join(CACHE_DIR, f"{base}_q_{tag}.npz")


def load_question_features(jsonl_path: str, model_name: str = EMBED_MODEL):
    """Returns (X, y, item_ids). Caches embeddings on disk for instant reruns."""
    cache = _cache_path(jsonl_path, model_name)
    if os.path.exists(cache):
        logger.info(f"  Cache hit: {cache}")
        z = np.load(cache, allow_pickle=True)
        return z["X"], z["y"].astype(int), z["item_ids"]

    questions, labels, item_ids = [], [], []
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            q = d.get("problem") or d.get("prompt") or ""
            if not q:
                continue
            questions.append(str(q))
            labels.append(int(bool(d["is_correct"])))
            item_ids.append(str(d.get("item_id", "")))

    if not questions:
        raise ValueError(f"No questions extracted from {jsonl_path}")

    logger.info(f"  Encoding {len(questions)} questions with {model_name}")
    from src.features.recurrence_features import load_embedder
    embedder = load_embedder(model_name)
    X = embedder.encode(
        questions, batch_size=64, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    ).astype(np.float32)
    y = np.array(labels, dtype=int)
    ids = np.array(item_ids, dtype=object)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache, X=X, y=y, item_ids=ids)
    logger.info(f"  Cached: {cache}  shape={X.shape}")
    return X, y, ids


def run_baseline_f(jsonl_path: str, output_path: str) -> dict:
    logger.info(f"Loading traces: {jsonl_path}")
    X, y, item_ids = load_question_features(jsonl_path)
    logger.info(f"  Samples: {len(y)}  embed_dim={X.shape[1]}  pos_rate={y.mean():.1%}")

    dataset_name = os.path.basename(jsonl_path).replace("_traces.jsonl", "").replace(".jsonl", "")
    feature_names = [f"q_emb_{i}" for i in range(X.shape[1])]

    cv_result = train_cv(X, y, feature_names, classifier_name="logistic_regression", n_splits=5)
    s = cv_result["summary"]
    logger.info(
        f"  AUROC: {s['auroc_mean']:.3f} ± {s['auroc_std']:.3f}  "
        f"AUPRC: {s['auprc_mean']:.3f} ± {s['auprc_std']:.3f}  "
        f"ECE: {s['ece_mean']:.3f} ± {s['ece_std']:.3f}  "
        f"Acc@80: {s['accuracy_at_80_mean']:.3f}"
    )

    # Drop per-feature importance — 384 is too noisy to be informative.
    summary_clean = {
        "summary": cv_result["summary"],
        "aggregated_metrics": cv_result["aggregated_metrics"],
        "fold_metrics": cv_result["fold_metrics"],
        "predictions": cv_result["predictions"],
    }

    results = {
        "dataset": dataset_name,
        "n_samples": int(len(y)),
        "n_correct": int(y.sum()),
        "n_incorrect": int((1 - y).sum()),
        "base_accuracy": float(y.mean()),
        "embedder": EMBED_MODEL,
        "embed_dim": int(X.shape[1]),
        "baseline_f": {
            "features": "all-MiniLM-L6-v2 sentence embedding of the *question* (no trace, no answer)",
            "classifier": "logistic_regression",
            **summary_clean,
        },
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info(f"  Saved: {output_path}")
    return results


def print_summary_table(all_results: list[dict]):
    print("\n" + "=" * 80)
    print("BASELINE F — QUESTION ONLY  (sanity floor; predicts correctness from")
    print("                             question text alone, no trace, no answer)")
    print("=" * 80)
    print(f"{'Dataset':40s} {'AUROC':>14s} {'AUPRC':>14s} {'Acc@80':>8s}")
    print("-" * 80)
    for r in all_results:
        s = r["baseline_f"]["summary"]
        print(
            f"{r['dataset']:40s} "
            f"{s['auroc_mean']:.3f}±{s['auroc_std']:.3f}  "
            f"{s['auprc_mean']:.3f}±{s['auprc_std']:.3f}  "
            f"{s['accuracy_at_80_mean']:.3f}"
        )
    print("=" * 80)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Baseline F: Question-Only Classifier")
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
                result = run_baseline_f(traces_path, out_path)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Failed on {traces_path}: {e}")
        if all_results:
            print_summary_table(all_results)
    else:
        if not args.output:
            parser.error("--output is required when --traces is used")
        run_baseline_f(args.traces, args.output)


if __name__ == "__main__":
    main()
