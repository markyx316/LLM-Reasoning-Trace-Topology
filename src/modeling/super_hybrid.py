"""
super_hybrid.py - Final stacking meta-learner combining every OOF signal.

This merges:
  * DeBERTa OOF prob                 (trace-only fine-tune)     -- 1-d
  * DeBERTa-Conditioned OOF prob     (problem+trace fine-tune)  -- 1-d
  * Hidden-state probe OOF prob      (MLP on h_answer)           -- 1-d
  * Handcrafted 25 + recurrence 5 + alignment 4 features        -- 34-d
  * Generation-uncertainty 10 features                           -- 10-d

It produces a table over multiple variant subsets and 3 meta-learners
(LR / RF / XGB) under the corrected `item_id + group` merge (no
cartesian-product leakage).

Usage:
    PYTHONPATH=. python src/modeling/super_hybrid.py \
        --deberta-oof     results/month2/deberta_pooled_oof.npz \
        --cond-oof        results/month2/deberta_conditioned_pooled_oof.npz \
        --probe-oof       results/month3/hidden_probe_pooled_mlp_h_answer_oof.npz \
        --features-glob   "data/features/*_features_align.csv" \
        --genunc-glob     "data/features/*_features_genunc.csv" \
        --output          results/month3/super_hybrid.json
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
from src.modeling.cv_utils import (
    aggregate_folds, evaluate, save_results, stratified_split,
)

logger = logging.getLogger(__name__)

RECURRENCE_FEATS = [
    "semantic_recurrence_rate", "max_semantic_cycle_span",
    "progress_repetition", "termination_recycle", "revision_ineffectiveness",
]
ALIGNMENT_FEATS = [
    "problem_conclusion_sim", "problem_trace_max_sim",
    "problem_drift", "problem_keyword_coverage",
]


# =============================================================================
# LOAD
# =============================================================================

def load_oof(npz_path: str, col_name: str) -> pd.DataFrame:
    z = np.load(npz_path, allow_pickle=True)
    return pd.DataFrame({
        "item_id":    z["item_ids"].astype(str),
        "group":      z["groups"].astype(str),
        "y_true":     z["y_true"].astype(int),
        col_name:     z["oof_prob"].astype(float),
    })


def load_features_csv(glob_pat: str) -> pd.DataFrame:
    paths = sorted(glob.glob(glob_pat))
    dfs = []
    for p in paths:
        d = pd.read_csv(p)
        group = (os.path.basename(p)
                 .replace("_features_align.csv", "")
                 .replace("_features_rec.csv", "")
                 .replace("_features_genunc.csv", "")
                 .replace("_features.csv", ""))
        d["group"] = group
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df["item_id"] = df["item_id"].astype(str)
    df["group"] = df["group"].astype(str)
    return df


# =============================================================================
# META CV
# =============================================================================

def run_meta_cv(X, y, group, name, clf_name="rf", n_splits=5, seed=42) -> dict:
    fold_metrics = []
    all_y, all_p = [], []
    for fold, (tr, te) in enumerate(stratified_split(
            y, group_id=group if len(set(group)) > 1 else None,
            n_splits=n_splits, seed=seed)):
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X[te])
        ytr, yte = y[tr], y[te]
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
        p = m.predict_proba(Xte)[:, 1]
        fm = evaluate(yte, p, name=f"fold_{fold + 1}")
        fold_metrics.append(fm)
        all_y.append(yte); all_p.append(p)
    all_y = np.concatenate(all_y); all_p = np.concatenate(all_p)
    overall = evaluate(all_y, all_p, name=name)
    summary = aggregate_folds(fold_metrics)
    logger.info(f"  [{name:32s} clf={clf_name:3s}]  "
                f"AUROC={summary.get('auroc_mean',0):.4f} ± {summary.get('auroc_std',0):.4f}  "
                f"AUPRC={summary.get('auprc_mean',0):.3f}  "
                f"ECE={summary.get('ece_mean',0):.3f}")
    return {"variant": name, "clf": clf_name,
            "fold_metrics": fold_metrics,
            "summary": summary, "overall": overall}


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

    feat_df = load_features_csv(args.features_glob)
    gu_df   = load_features_csv(args.genunc_glob)

    logger.info(f"deberta_oof: {len(deb_df)}, cond_oof: {len(cond_df)}, "
                f"probe_oof: {len(probe_df)}, feat: {len(feat_df)}, gu: {len(gu_df)}")

    # Merge on (item_id, group)
    key = ["item_id", "group"]
    merged = (deb_df[["item_id","group","y_true","deberta_prob"]]
              .merge(cond_df[["item_id","group","cond_prob"]], on=key, how="inner")
              .merge(probe_df[["item_id","group","probe_prob"]], on=key, how="inner")
              .merge(feat_df, on=key, how="inner", suffixes=("","_feat"))
              .merge(gu_df[key + GENERATION_UNC_FEATURE_NAMES],
                     on=key, how="inner", suffixes=("","_gu")))

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

    logger.info(f"Handcrafted: {len(handcrafted_cols)}, "
                f"Recurrence: {len(rec_cols)}, Alignment: {len(align_cols)}, "
                f"GenUnc: {len(gu_cols)}")

    def X_of(cols):
        X = merged[cols].to_numpy(dtype=float)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # -------- Variants --------
    variants = {
        # Singles
        "deberta_only":       ["deberta_prob"],
        "cond_only":          ["cond_prob"],
        "probe_only":         ["probe_prob"],

        # Pairs
        "deberta+cond":       ["deberta_prob","cond_prob"],
        "deberta+probe":      ["deberta_prob","probe_prob"],
        "cond+probe":         ["cond_prob","probe_prob"],
        "three_probs":        ["deberta_prob","cond_prob","probe_prob"],

        # With features
        "three+feats":        ["deberta_prob","cond_prob","probe_prob"] + handcrafted_cols + rec_cols + align_cols,
        "three+genunc":       ["deberta_prob","cond_prob","probe_prob"] + gu_cols,
        "three+all_features": ["deberta_prob","cond_prob","probe_prob"] + handcrafted_cols + rec_cols + align_cols + gu_cols,

        # Feature-only ablations for context
        "features_only":      handcrafted_cols + rec_cols + align_cols,
        "features+genunc":    handcrafted_cols + rec_cols + align_cols + gu_cols,
    }

    all_results = {"n_samples": int(len(y)),
                   "base_acc": float(y.mean()),
                   "variants": {}}

    clf_list = ["lr", "rf", "xgb"]

    for vname, cols in variants.items():
        X = X_of(cols)
        logger.info(f"\n--- {vname}  (d={X.shape[1]}) ---")
        all_results["variants"][vname] = {
            "n_features": X.shape[1],
            "clfs": {},
        }
        for c in clf_list:
            try:
                if c == "xgb":
                    import xgboost  # noqa: F401
            except ImportError:
                continue
            all_results["variants"][vname]["clfs"][c] = run_meta_cv(
                X, y, group, vname, clf_name=c,
                n_splits=args.n_splits, seed=args.seed)

    save_results(args.output, all_results)

    # -------- Summary --------
    print("\n" + "=" * 92)
    print(f"{'variant':28s}  {'d':>4s}  {'LR':>13s}  {'RF':>13s}  {'XGB':>13s}")
    print("-" * 92)
    for vname, vb in all_results["variants"].items():
        row = f"{vname:28s}  {vb['n_features']:>4d}"
        for c in ["lr","rf","xgb"]:
            if c in vb["clfs"]:
                s = vb["clfs"][c]["summary"]
                row += f"  {s.get('auroc_mean',0):.3f}±{s.get('auroc_std',0):.3f}"
            else:
                row += f"  {'--':>13s}"
        print(row)


if __name__ == "__main__":
    main()
