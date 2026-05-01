"""
per_dataset_analysis.py - Per-dataset breakdown of the Super-Hybrid method.

Runs the same stacking as super_hybrid but slices out-of-fold predictions
by `group` (i.e., the 8 dataset-model combinations) to report AUROC/ECE
on each subset. Uses the same pooled 5-fold CV for consistent comparison.

Output: a JSON with per-group metrics for the key variants, plus a printed
pivot table.

Usage:
    PYTHONPATH=. python src/modeling/per_dataset_analysis.py \
        --deberta-oof    results/month2/deberta_pooled_oof.npz \
        --cond-oof       results/month2/deberta_conditioned_pooled_oof.npz \
        --probe-oof      results/month3/hidden_probe_pooled_mlp_h_answer_oof.npz \
        --features-glob  "data/features/*_features_align.csv" \
        --genunc-glob    "data/features/*_features_genunc.csv" \
        --output         results/month3/per_dataset_analysis.json
"""

from __future__ import annotations

import argparse
import glob
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
# META CV that returns full OOF prediction arrays
# =============================================================================

def run_cv_save_oof(X, y, group, clf_name="rf", n_splits=5, seed=42):
    """Same CV as super_hybrid but returns OOF prob aligned to input order."""
    oof = np.full(len(y), np.nan, dtype=np.float32)
    for fold, (tr, te) in enumerate(stratified_split(
            y, group_id=group if len(set(group)) > 1 else None,
            n_splits=n_splits, seed=seed)):
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        ytr = y[tr]
        if clf_name == "lr":
            m = LogisticRegression(C=1.0, max_iter=2000,
                                   class_weight="balanced", random_state=seed)
        elif clf_name == "rf":
            m = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                       class_weight="balanced", n_jobs=-1,
                                       random_state=seed)
        elif clf_name == "xgb":
            from xgboost import XGBClassifier
            n_pos = max(int(ytr.sum()), 1); n_neg = max(len(ytr) - n_pos, 1)
            m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                              scale_pos_weight=n_neg / n_pos,
                              random_state=seed, eval_metric="logloss",
                              n_jobs=-1, tree_method="hist")
        else:
            raise ValueError(clf_name)
        m.fit(Xtr, ytr)
        oof[te] = m.predict_proba(Xte)[:, 1]
    return oof


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

    y = merged["y_true"].to_numpy(dtype=int)
    group = merged["group"].to_numpy()

    exclude = ({"item_id","dataset","is_correct","y_true","group",
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

    # Methods to evaluate per-group
    method_specs = {
        "DeBERTa":            ("lr", ["deberta_prob"]),
        "DeBERTa+Cond":       ("lr", ["deberta_prob","cond_prob"]),
        "Cond+Probe":         ("lr", ["cond_prob","probe_prob"]),
        "3probs":             ("lr", ["deberta_prob","cond_prob","probe_prob"]),
        "SuperHybrid_LR":     ("lr", ["deberta_prob","cond_prob","probe_prob"]
                                      + gu_cols),
        "SuperHybrid_RF":     ("rf", ["deberta_prob","cond_prob","probe_prob"]
                                      + handcrafted_cols + rec_cols + align_cols + gu_cols),
    }

    # --- Run all methods once (pooled CV), save OOF ---
    oof_cache: dict[str, np.ndarray] = {}
    for m_name, (clf, cols) in method_specs.items():
        X = X_of(cols)
        logger.info(f"\n--- Pooled CV: {m_name}  clf={clf}  d={X.shape[1]} ---")
        oof_cache[m_name] = run_cv_save_oof(X, y, group,
                                            clf_name=clf,
                                            n_splits=args.n_splits,
                                            seed=args.seed)

    # --- Slice by group, compute metrics ---
    groups_sorted = sorted(set(group))
    per_group_results = {}

    for g in groups_sorted:
        mask = (group == g)
        y_g = y[mask]
        per_group_results[g] = {
            "n": int(mask.sum()),
            "base_acc": float(y_g.mean()),
            "methods": {},
        }
        for m_name, _ in method_specs.items():
            p_g = oof_cache[m_name][mask]
            metrics = evaluate(y_g, p_g, name=f"{m_name}_{g}")
            per_group_results[g]["methods"][m_name] = {
                k: metrics.get(k) for k in
                ["auroc","auprc","ece","accuracy_at_80","accuracy_at_90"]
            }

    # Also compute pooled for reference
    pooled_results = {}
    for m_name, _ in method_specs.items():
        metrics = evaluate(y, oof_cache[m_name], name=f"{m_name}_pooled")
        pooled_results[m_name] = {
            k: metrics.get(k) for k in
            ["auroc","auprc","ece","accuracy_at_80","accuracy_at_90"]
        }

    all_results = {
        "n_samples": int(len(y)),
        "base_acc": float(y.mean()),
        "methods": list(method_specs.keys()),
        "per_group": per_group_results,
        "pooled": pooled_results,
    }
    save_results(args.output, all_results)

    # --- Pretty table: groups x methods (AUROC) ---
    method_list = list(method_specs.keys())
    print("\n" + "=" * (22 + 14 * len(method_list)))
    print(f"Per-dataset AUROC (out-of-fold, {len(y)} total samples)")
    print("=" * (22 + 14 * len(method_list)))
    header = f"{'group':22s}  {'n':>5s}  {'base':>5s}  " + "  ".join(
        f"{m:>12s}" for m in method_list)
    print(header)
    print("-" * len(header))
    for g in groups_sorted:
        gr = per_group_results[g]
        row = f"{g:22s}  {gr['n']:>5d}  {gr['base_acc']:.3f}  "
        row += "  ".join(f"{gr['methods'][m]['auroc']:>12.3f}" for m in method_list)
        print(row)
    print("-" * len(header))
    row = f"{'POOLED':22s}  {len(y):>5d}  {y.mean():.3f}  "
    row += "  ".join(f"{pooled_results[m]['auroc']:>12.3f}" for m in method_list)
    print(row)

    # ECE table
    print("\n" + "=" * (22 + 14 * len(method_list)))
    print("Per-dataset ECE (lower is better)")
    print("=" * (22 + 14 * len(method_list)))
    print(header)
    print("-" * len(header))
    for g in groups_sorted:
        gr = per_group_results[g]
        row = f"{g:22s}  {gr['n']:>5d}  {gr['base_acc']:.3f}  "
        row += "  ".join(f"{gr['methods'][m]['ece']:>12.3f}" for m in method_list)
        print(row)


if __name__ == "__main__":
    main()
