"""
cross_model_transfer.py - Cross-model transfer evaluation for the super-hybrid.

Tests whether the meta-learner's learned combination rules generalize
across LLM generator families (Qwen vs Llama).

Two directions:
    Q -> L: Train meta on all Qwen items, evaluate on all Llama items.
    L -> Q: Train meta on all Llama items, evaluate on all Qwen items.

Also reports:
    - "oracle within" (train + eval on same family, 5-fold CV) for comparison.
    - Transfer gap: oracle_within - transfer_auroc (smaller gap = better transfer).

Note on rigor: the base OOF predictions (DeBERTa, Cond, Probe) were trained
in a pooled 5-fold CV that mixed Qwen + Llama items, so those base models
already saw both families during training. This script therefore evaluates
*meta-learner generalization* only. Full base-model retraining would be more
rigorous but costs ~6 additional GPU runs; we include this as a limitation
in the paper.

Usage:
    PYTHONPATH=. python src/modeling/cross_model_transfer.py \
        --deberta-oof    results/month2/deberta_pooled_oof.npz \
        --cond-oof       results/month2/deberta_conditioned_pooled_oof.npz \
        --probe-oof      results/month3/hidden_probe_pooled_mlp_h_answer_oof.npz \
        --features-glob  "data/features/*_features_align.csv" \
        --genunc-glob    "data/features/*_features_genunc.csv" \
        --output         results/month3/cross_model_transfer.json
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features.generation_uncertainty import GENERATION_UNC_FEATURE_NAMES
from src.modeling.cv_utils import evaluate, save_results, stratified_split
from src.modeling.super_hybrid import (
    ALIGNMENT_FEATS, RECURRENCE_FEATS,
    load_features_csv, load_oof,
)

logger = logging.getLogger(__name__)


# =============================================================================
# FAMILY INFERENCE
# =============================================================================

def infer_family(group_name: str) -> str:
    """Map a group label like 'math500_qwen7b' -> 'qwen' or 'llama'."""
    g = group_name.lower()
    if "llama" in g:
        return "llama"
    if "qwen" in g:
        return "qwen"
    return "other"


# =============================================================================
# CLFs
# =============================================================================

def fit_clf(clf_name: str, X: np.ndarray, y: np.ndarray, seed: int = 42):
    if clf_name == "lr":
        return LogisticRegression(C=1.0, max_iter=2000,
                                  class_weight="balanced",
                                  random_state=seed).fit(X, y)
    elif clf_name == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                      class_weight="balanced", n_jobs=-1,
                                      random_state=seed).fit(X, y)
    else:
        raise ValueError(clf_name)


def cv_same_family(X, y, group, clf_name, n_splits=5, seed=42):
    """Standard within-family 5-fold CV."""
    oof = np.full(len(y), np.nan, dtype=np.float32)
    for tr, te in stratified_split(
            y, group_id=group if len(set(group)) > 1 else None,
            n_splits=n_splits, seed=seed):
        scaler = StandardScaler().fit(X[tr])
        m = fit_clf(clf_name, scaler.transform(X[tr]), y[tr], seed=seed)
        oof[te] = m.predict_proba(scaler.transform(X[te]))[:, 1]
    return oof


def transfer_one_direction(X_tr, y_tr, X_te, clf_name, seed=42):
    """Train on (X_tr, y_tr), predict on X_te."""
    scaler = StandardScaler().fit(X_tr)
    m = fit_clf(clf_name, scaler.transform(X_tr), y_tr, seed=seed)
    return m.predict_proba(scaler.transform(X_te))[:, 1]


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deberta-oof", required=True)
    p.add_argument("--cond-oof", required=True)
    p.add_argument("--probe-oof", required=True)
    p.add_argument("--features-glob", required=True)
    p.add_argument("--genunc-glob", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-splits", type=int, default=5)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Load everything
    deb_df   = load_oof(args.deberta_oof, "deberta_prob")
    cond_df  = load_oof(args.cond_oof,    "cond_prob")
    probe_df = load_oof(args.probe_oof,   "probe_prob")
    feat_df  = load_features_csv(args.features_glob)
    gu_df    = load_features_csv(args.genunc_glob)

    key = ["item_id", "group"]
    merged = (deb_df[["item_id","group","y_true","deberta_prob"]]
              .merge(cond_df[["item_id","group","cond_prob"]], on=key)
              .merge(probe_df[["item_id","group","probe_prob"]], on=key)
              .merge(feat_df, on=key, suffixes=("","_feat"))
              .merge(gu_df[key + GENERATION_UNC_FEATURE_NAMES],
                     on=key, suffixes=("","_gu")))
    logger.info(f"After join: {len(merged)} rows")

    merged["family"] = merged["group"].apply(infer_family)
    y = merged["y_true"].to_numpy(dtype=int)
    group = merged["group"].to_numpy()
    family = merged["family"].to_numpy()

    exclude = ({"item_id","dataset","is_correct","y_true","group","family",
                "deberta_prob","cond_prob","probe_prob"}
               | set(RECURRENCE_FEATS) | set(ALIGNMENT_FEATS)
               | set(GENERATION_UNC_FEATURE_NAMES))
    handcrafted_cols = [c for c in merged.columns
                        if c not in exclude and merged[c].dtype != object]
    rec_cols   = [c for c in RECURRENCE_FEATS if c in merged.columns]
    align_cols = [c for c in ALIGNMENT_FEATS if c in merged.columns]
    gu_cols    = [c for c in GENERATION_UNC_FEATURE_NAMES if c in merged.columns]

    def X_of(cols):
        X = merged[cols].to_numpy(dtype=float)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    methods = {
        "DeBERTa":          ("lr", ["deberta_prob"]),
        "DeBERTa+Cond":     ("lr", ["deberta_prob","cond_prob"]),
        "3probs":           ("lr", ["deberta_prob","cond_prob","probe_prob"]),
        "SuperHybrid_LR":   ("lr", ["deberta_prob","cond_prob","probe_prob"] + gu_cols),
        "SuperHybrid_RF":   ("rf", ["deberta_prob","cond_prob","probe_prob"]
                                    + handcrafted_cols + rec_cols + align_cols + gu_cols),
    }

    qwen_mask  = (family == "qwen")
    llama_mask = (family == "llama")
    n_q, n_l = int(qwen_mask.sum()), int(llama_mask.sum())
    logger.info(f"Qwen items: {n_q}, Llama items: {n_l}")

    results = {"n_qwen": n_q, "n_llama": n_l, "methods": {}}

    for mname, (clf, cols) in methods.items():
        X = X_of(cols)
        logger.info(f"\n--- {mname}  clf={clf}  d={X.shape[1]} ---")

        # Oracle "within" via 5-fold CV inside each family
        oof_within_q = cv_same_family(
            X[qwen_mask],  y[qwen_mask],  group[qwen_mask],
            clf, n_splits=args.n_splits, seed=args.seed)
        oof_within_l = cv_same_family(
            X[llama_mask], y[llama_mask], group[llama_mask],
            clf, n_splits=args.n_splits, seed=args.seed)

        # Transfer: train on one family, predict on the other
        p_q_to_l = transfer_one_direction(
            X[qwen_mask], y[qwen_mask], X[llama_mask], clf, seed=args.seed)
        p_l_to_q = transfer_one_direction(
            X[llama_mask], y[llama_mask], X[qwen_mask], clf, seed=args.seed)

        mr = {
            "within_qwen":  evaluate(y[qwen_mask],  oof_within_q, f"{mname}_q"),
            "within_llama": evaluate(y[llama_mask], oof_within_l, f"{mname}_l"),
            "q_to_l":       evaluate(y[llama_mask], p_q_to_l,     f"{mname}_Q2L"),
            "l_to_q":       evaluate(y[qwen_mask],  p_l_to_q,     f"{mname}_L2Q"),
        }
        # Transfer gap (within - transfer)
        mr["gap_Q2L"] = mr["within_llama"]["auroc"] - mr["q_to_l"]["auroc"]
        mr["gap_L2Q"] = mr["within_qwen"]["auroc"]  - mr["l_to_q"]["auroc"]

        results["methods"][mname] = mr
        logger.info(f"  within_qwen AUROC={mr['within_qwen']['auroc']:.4f}  "
                    f"within_llama={mr['within_llama']['auroc']:.4f}")
        logger.info(f"  Q->L AUROC={mr['q_to_l']['auroc']:.4f}  "
                    f"(gap {mr['gap_Q2L']:+.4f})")
        logger.info(f"  L->Q AUROC={mr['l_to_q']['auroc']:.4f}  "
                    f"(gap {mr['gap_L2Q']:+.4f})")

    save_results(args.output, results)

    # Pretty table
    print("\n" + "=" * 92)
    print(f"{'method':18s}  {'within_Q':>9s}  {'within_L':>9s}  "
          f"{'Q->L':>9s}  {'gap':>7s}  {'L->Q':>9s}  {'gap':>7s}")
    print("-" * 92)
    for mname, mr in results["methods"].items():
        print(f"{mname:18s}  "
              f"{mr['within_qwen']['auroc']:>9.3f}  "
              f"{mr['within_llama']['auroc']:>9.3f}  "
              f"{mr['q_to_l']['auroc']:>9.3f}  "
              f"{mr['gap_Q2L']:>+7.3f}  "
              f"{mr['l_to_q']['auroc']:>9.3f}  "
              f"{mr['gap_L2Q']:>+7.3f}")


if __name__ == "__main__":
    main()
