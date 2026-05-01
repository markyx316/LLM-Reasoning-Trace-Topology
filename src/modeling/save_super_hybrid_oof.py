"""save_super_hybrid_oof.py - Compute and save SuperHybrid OOF probabilities.

Reuses the same logic as super_hybrid but saves OOF arrays so they can be
fed to the conformal wrapper.

Usage:
    PYTHONPATH=. python src/modeling/save_super_hybrid_oof.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features.generation_uncertainty import GENERATION_UNC_FEATURE_NAMES
from src.modeling.cv_utils import stratified_split
from src.modeling.super_hybrid import (
    ALIGNMENT_FEATS, RECURRENCE_FEATS,
    load_features_csv, load_oof,
)

logger = logging.getLogger(__name__)


def cv_oof(X, y, group, clf_name="lr", n_splits=5, seed=42):
    oof = np.full(len(y), np.nan, dtype=np.float32)
    for tr, te in stratified_split(
            y, group_id=group if len(set(group)) > 1 else None,
            n_splits=n_splits, seed=seed):
        scaler = StandardScaler().fit(X[tr])
        if clf_name == "lr":
            m = LogisticRegression(C=1.0, max_iter=2000,
                                   class_weight="balanced", random_state=seed)
        else:
            m = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                       class_weight="balanced", n_jobs=-1,
                                       random_state=seed)
        m.fit(scaler.transform(X[tr]), y[tr])
        oof[te] = m.predict_proba(scaler.transform(X[te]))[:, 1]
    return oof


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    deb_df  = load_oof("results/month2/deberta_pooled_oof.npz", "deberta_prob")
    cond_df = load_oof("results/month2/deberta_conditioned_pooled_oof.npz", "cond_prob")
    pr_df   = load_oof("results/month3/hidden_probe_pooled_mlp_h_answer_oof.npz", "probe_prob")
    feat_df = load_features_csv("data/features/*_features_align.csv")
    gu_df   = load_features_csv("data/features/*_features_genunc.csv")

    key = ["item_id", "group"]
    merged = (deb_df[["item_id","group","y_true","deberta_prob"]]
              .merge(cond_df[["item_id","group","cond_prob"]], on=key)
              .merge(pr_df[["item_id","group","probe_prob"]], on=key)
              .merge(feat_df, on=key, suffixes=("","_feat"))
              .merge(gu_df[key + GENERATION_UNC_FEATURE_NAMES],
                     on=key, suffixes=("","_gu")))
    logger.info(f"merged: {len(merged)}")

    y = merged["y_true"].to_numpy(dtype=int)
    group = merged["group"].to_numpy()
    item_ids = merged["item_id"].to_numpy()

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

    methods = {
        "DeBERTa":         ("lr", ["deberta_prob"]),
        "DeBERTa_Cond":    ("lr", ["deberta_prob","cond_prob"]),
        "ThreeProbs":      ("lr", ["deberta_prob","cond_prob","probe_prob"]),
        "SuperHybrid_LR":  ("lr", ["deberta_prob","cond_prob","probe_prob"] + gu_cols),
        "SuperHybrid_RF":  ("rf", ["deberta_prob","cond_prob","probe_prob"]
                                   + handcrafted_cols + rec_cols + align_cols + gu_cols),
    }

    out_dir = "results/month3"
    for name, (clf, cols) in methods.items():
        X = X_of(cols)
        logger.info(f"-- {name}  d={X.shape[1]}  clf={clf} --")
        oof = cv_oof(X, y, group, clf_name=clf, n_splits=5, seed=42)
        out_path = f"{out_dir}/superhybrid_{name}_oof.npz"
        np.savez_compressed(out_path,
            item_ids=item_ids, groups=group,
            y_true=y, oof_prob=oof.astype(np.float32),
            seed=np.array([42]), n_splits=np.array([5]))
        logger.info(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
