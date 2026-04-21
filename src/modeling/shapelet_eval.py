"""
shapelet_eval.py - Fold-aware shapelet evaluator with OOF probability output.

Why this file exists: shapelet *mining* (selecting the top-K candidates by
information gain) must happen inside each CV fold's training split to avoid
label leakage. We cannot emit a fixed "best shapelet" CSV before CV runs.
Instead this module:

  1. Loads pre-computed per-dataset shapelet distance matrices from
     `data/features/{dataset}_{model}_shapelet_distmat.npz` (built by
     src/features/shapelet_features.py — those distances are a function
     of the raw traces alone, not labels, so they are leakage-safe).
  2. Concatenates across all supplied dataset/model combos -> pooled
     (N, M) distance matrix + (N,) labels + (N,) group labels.
  3. For each stratified 5-fold split:
     a. Restrict to training-fold indices.
     b. For each of the M candidate shapelets, find the best binary split
        threshold tau* on d(t, s) that maximizes information gain on the
        training labels.
     c. Rank candidates by info gain; keep the top-K (default 40).
     d. Compute the (n_tr, K) feature matrix by looking up the train rows'
        distances to those K shapelets. Fit a standard-scaled logistic
        regression.
     e. For each held-out test item, compute its K distances, transform,
        predict P(correct).
  4. Emit a pooled OOF .npz:
        results/route_ab/shapelet_oof.npz
        keys:
          item_ids (N,)  str,  y_true (N,) int, oof_prob (N,) float,
          groups (N,) str

Design choices:
  - Info gain is computed from a binary partition at threshold tau; tau is
    swept over the unique train distances per candidate. This is the
    Ye-Keogh (2009) shapelet scoring, restricted to a single threshold.
  - We refit the meta-LR inside each fold — the candidate selection AND the
    classifier are both fold-local. No train data leaks into test.
  - Default K=40 is a compromise between feature count and overfit risk;
    configurable via --top-k.
  - If multiple datasets/groups are pooled, stratification uses the
    (group, label) joint key (matches cv_utils.stratified_split pattern).

Usage:
    PYTHONPATH=. python src/modeling/shapelet_eval.py \\
        --distmat-glob "data/features/*_shapelet_distmat.npz" \\
        --output results/route_ab/shapelet_oof.npz \\
        --top-k 40
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.modeling.cv_utils import (  # noqa: E402
    aggregate_folds, evaluate, save_results, stratified_split,
)

logger = logging.getLogger(__name__)


# =============================================================================
# INFO-GAIN SHAPELET SCORING
# =============================================================================

def _entropy_binary(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def _best_split_gain(dist_col: np.ndarray, y: np.ndarray) -> float:
    """
    For a single shapelet's distance column (n_tr,) and label vector
    (n_tr,), compute the best info-gain over all candidate thresholds.
    Returns the gain; ignores the tau itself because we retrain in the
    LR stage.
    """
    n = len(y)
    if n == 0 or len(np.unique(y)) < 2:
        return 0.0
    p_parent = y.mean()
    H_parent = _entropy_binary(p_parent)

    # Candidate thresholds = midpoints between sorted unique distances
    order = np.argsort(dist_col, kind="stable")
    d_sorted = dist_col[order]
    y_sorted = y[order]

    # Sweep by prefix cum sum: left = items with d <= tau
    cum_y = np.cumsum(y_sorted)
    ks = np.arange(1, n)  # split after item ks-1 (items [0..ks-1] vs [ks..n-1])
    n_left = ks
    n_right = n - ks
    p_left = cum_y[ks - 1] / n_left
    p_right = (cum_y[-1] - cum_y[ks - 1]) / np.maximum(n_right, 1)

    # Only consider distinct thresholds
    d_left = d_sorted[:-1]
    d_right = d_sorted[1:]
    distinct = d_left != d_right
    if not distinct.any():
        return 0.0

    H_left = np.array([_entropy_binary(float(p)) for p in p_left])
    H_right = np.array([_entropy_binary(float(p)) for p in p_right])
    H_after = (n_left / n) * H_left + (n_right / n) * H_right
    gains = H_parent - H_after
    gains_filtered = np.where(distinct, gains, -np.inf)
    best = float(gains_filtered.max())
    return max(best, 0.0)


def rank_shapelets_by_gain(distances_tr: np.ndarray,
                           y_tr: np.ndarray,
                           top_k: int = 40) -> np.ndarray:
    """
    Score every shapelet (column of distances_tr) by best-split info gain
    on the training labels. Return the indices of the top-k.
    """
    n_tr, M = distances_tr.shape
    gains = np.zeros(M, dtype=np.float32)
    for j in range(M):
        gains[j] = _best_split_gain(distances_tr[:, j], y_tr)
    # Top-k by descending gain; ties broken by column index (stable).
    top_idx = np.argsort(-gains, kind="stable")[:top_k]
    return top_idx


# =============================================================================
# DATA LOADING
# =============================================================================

def load_pooled_distmat(distmat_paths: list[str]
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray, list[str], np.ndarray]:
    """
    Load multiple per-dataset distance matrices, concatenate on axis 0.
    Because candidate sets differ per dataset (each built from its own
    corpus), we align to the *union* of candidate strings and fill
    missing entries with distance 1.0 (maximum, i.e. shapelet never
    appears in this trace).

    Returns:
        distances:  (N_total, M_union) float32
        labels:     (N_total,) int
        groups:     (N_total,) str
        item_ids:   (N_total,) object
        candidates: list[str]
        cand_lens:  (M_union,) int8
    """
    if not distmat_paths:
        raise ValueError("no distmat paths supplied")

    logger.info(f"Loading {len(distmat_paths)} distmats...")
    # First pass: union of candidates
    union: dict[str, int] = {}  # candidate -> length
    total_N = 0
    per_file = []
    for p in distmat_paths:
        z = np.load(p, allow_pickle=True)
        cands = list(z["candidates"])
        lens = list(z["cand_lens"])
        for s, k in zip(cands, lens):
            union.setdefault(str(s), int(k))
        total_N += len(z["item_ids"])
        per_file.append((p, cands, lens, z))

    all_cands = sorted(union.keys())
    cand_to_col = {c: i for i, c in enumerate(all_cands)}
    cand_lens = np.array([union[c] for c in all_cands], dtype=np.int8)
    M_union = len(all_cands)
    logger.info(f"  Pooled N={total_N}  M_union={M_union}")

    # Second pass: populate
    distances = np.ones((total_N, M_union), dtype=np.float32)
    labels = np.zeros(total_N, dtype=np.int32)
    groups = np.empty(total_N, dtype=object)
    item_ids = np.empty(total_N, dtype=object)

    row = 0
    for p, cands, lens, z in per_file:
        n = len(z["item_ids"])
        d_file = z["distances"]   # (n, M_file)
        # Build column indices in the union matrix for this file's candidates
        col_idx = np.array([cand_to_col[str(c)] for c in cands], dtype=np.int64)
        distances[row:row + n, col_idx] = d_file
        labels[row:row + n] = z["is_correct"].astype(np.int32)
        item_ids[row:row + n] = z["item_ids"]
        group_name = str(z["dataset"]) if "dataset" in z.files else \
            os.path.basename(p).replace("_shapelet_distmat.npz", "")
        groups[row:row + n] = group_name
        row += n

    assert row == total_N
    return distances, labels, groups, item_ids, all_cands, cand_lens


# =============================================================================
# META-LR OOF CV
# =============================================================================

def run_shapelet_oof_cv(distances: np.ndarray,
                        labels: np.ndarray,
                        groups: np.ndarray,
                        top_k: int = 40,
                        n_splits: int = 5,
                        seed: int = 42) -> dict:
    """
    5-fold CV with fold-local shapelet selection + LR meta-classifier.
    Returns:
        {
          "oof_prob":    (N,) float32,
          "y_true":      (N,) int,
          "groups":      (N,) str,
          "per_fold":    [ { fold, selected_shapelets, auroc, ... }, ... ],
          "summary":     aggregate metrics,
          "overall":     pooled evaluate(y, prob)
        }
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    N = len(labels)
    oof_prob = np.zeros(N, dtype=np.float32)
    # Track the fold in which each item served as a held-out example. Used by
    # downstream meta-learners (hybrid stacker, HP tuning) to align folds
    # across base models. -1 sentinel means "never held out"; expected to be
    # fully overwritten by the fold loop since stratified_split is a partition.
    oof_fold = np.full(N, -1, dtype=np.int8)
    fold_metrics = []
    per_fold_info = []

    splitter = stratified_split(
        labels, group_id=groups if len(set(groups)) > 1 else None,
        n_splits=n_splits, seed=seed)

    for fold, (tr, te) in enumerate(splitter):
        # (a) Rank shapelets by info gain on training split
        top_idx = rank_shapelets_by_gain(distances[tr], labels[tr], top_k=top_k)

        X_tr = distances[np.ix_(tr, top_idx)]
        X_te = distances[np.ix_(te, top_idx)]
        y_tr = labels[tr]
        y_te = labels[te]

        scaler = StandardScaler().fit(X_tr)
        Xt_tr = scaler.transform(X_tr)
        Xt_te = scaler.transform(X_te)

        # class_weight balanced: data can be imbalanced per-dataset
        m = LogisticRegression(max_iter=3000, C=1.0,
                               class_weight="balanced", random_state=seed)
        m.fit(Xt_tr, y_tr)
        prob = m.predict_proba(Xt_te)[:, 1]
        oof_prob[te] = prob.astype(np.float32)
        oof_fold[te] = fold

        fm = evaluate(y_te, prob, name=f"fold_{fold + 1}")
        fold_metrics.append(fm)
        per_fold_info.append({
            "fold": fold + 1,
            "top_idx": top_idx.tolist(),
            "metrics": fm,
        })
        logger.info(f"  fold {fold + 1}/{n_splits}: AUROC={fm['auroc']:.4f}  "
                    f"AUPRC={fm['auprc']:.4f}")

    if (oof_fold < 0).any():
        n_missed = int((oof_fold < 0).sum())
        logger.warning(f"  {n_missed} items were never held out (oof_fold=-1); "
                       "stratified_split may not be a complete partition")

    overall = evaluate(labels, oof_prob, name="shapelet_oof")
    summary = aggregate_folds(fold_metrics)
    logger.info(f"[shapelet_oof]  AUROC={summary.get('auroc_mean',0):.4f} "
                f"± {summary.get('auroc_std',0):.4f}  "
                f"pooled_AUROC={overall['auroc']:.4f}")
    return {
        "oof_prob": oof_prob,
        "oof_fold": oof_fold,
        "y_true": labels,
        "groups": groups,
        "per_fold": per_fold_info,
        "summary": summary,
        "overall": overall,
        "seed": seed,
        "n_splits": n_splits,
    }


# =============================================================================
# DRIVER
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--distmat-glob",
                    default="data/features/*_shapelet_distmat.npz")
    ap.add_argument("--distmat", nargs="+", default=None,
                    help="Explicit distmat paths (overrides --distmat-glob)")
    ap.add_argument("--output",
                    default="results/route_ab/shapelet_oof.npz",
                    help="Output .npz with OOF probs (hybrid-compatible schema)")
    ap.add_argument("--metrics-json",
                    default="results/route_ab/shapelet_oof_metrics.json",
                    help="Path to JSON summary of per-fold + pooled metrics")
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    paths = args.distmat if args.distmat else sorted(glob.glob(args.distmat_glob))
    paths = [p for p in paths
             if not os.path.basename(p).startswith(("pilot_", "_"))
             and "_sc" not in os.path.basename(p)]
    if not paths:
        logger.error("No shapelet distmats found")
        sys.exit(1)

    distances, labels, groups, item_ids, cands, cand_lens = \
        load_pooled_distmat(paths)
    result = run_shapelet_oof_cv(
        distances, labels, groups,
        top_k=args.top_k, n_splits=args.n_splits, seed=args.seed,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(
        args.output,
        item_ids=item_ids,
        y_true=labels.astype(np.int32),
        oof_prob=result["oof_prob"],
        oof_fold=result["oof_fold"],
        groups=groups,
        seed=np.int32(result["seed"]),
        n_splits=np.int32(result["n_splits"]),
    )
    logger.info(f"Wrote OOF: {args.output}  "
                f"(keys: item_ids, y_true, oof_prob, oof_fold, groups, seed, n_splits)")

    # Drop distmat indices from JSON summary to keep it small
    summary = {
        "summary": result["summary"],
        "overall": result["overall"],
        "per_fold": [{k: v for k, v in pf.items() if k != "top_idx"}
                     for pf in result["per_fold"]],
        "top_k": args.top_k,
        "n_samples": int(len(labels)),
        "n_candidates": int(distances.shape[1]),
    }
    save_results(args.metrics_json, summary)
    logger.info(f"Wrote metrics: {args.metrics_json}")


# =============================================================================
# SELF-TEST
# =============================================================================

def _run_self_test():
    print("Running shapelet_eval self-test...")
    rng = np.random.default_rng(0)

    # Synthetic: 60 items, 200 candidates. Items with label=1 have very low
    # distance (close to 0) to candidate 0; items with label=0 have low
    # distance to candidate 1. Other candidates are noise.
    N = 60
    M = 200
    y = rng.integers(0, 2, size=N)
    dist = rng.uniform(0.3, 1.0, size=(N, M)).astype(np.float32)
    for i in range(N):
        if y[i] == 1:
            dist[i, 0] = rng.uniform(0.0, 0.05)
            dist[i, 1] = rng.uniform(0.8, 1.0)
        else:
            dist[i, 0] = rng.uniform(0.8, 1.0)
            dist[i, 1] = rng.uniform(0.0, 0.05)
    groups = np.array(["synth"] * N)
    ids = np.array([f"x_{i:03d}" for i in range(N)], dtype=object)

    # Info-gain ranking should put candidate 0 or 1 in the top
    gains = np.array([
        _best_split_gain(dist[:, j], y) for j in range(M)
    ])
    top = np.argsort(-gains)[:5]
    assert 0 in top.tolist() or 1 in top.tolist(), \
        f"expected candidate 0 or 1 among top-5 gains, got {top}"

    # Full CV loop
    res = run_shapelet_oof_cv(dist, y, groups, top_k=20, n_splits=5)
    auroc = res["overall"]["auroc"]
    print(f"  pooled OOF AUROC: {auroc:.4f}")
    assert auroc > 0.85, f"synthetic OOF AUROC too low: {auroc:.4f}"
    print("All shapelet_eval tests passed.")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _run_self_test()
    else:
        main()
