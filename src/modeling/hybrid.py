"""
hybrid.py - Stacking meta-learner combining DeBERTa + Step Transformer + handcrafted+recurrence features.

Input signals (all at the item level):
  1. DeBERTa OOF probability               (1-d)     -- text signal
  2. Step Transformer OOF probability      (1-d)     -- structure signal
  3. Handcrafted 25 + recurrence 5 = 30 features     -- interpretable signal

Meta-learner: LR / RF / XGB, trained in a fresh 5-fold CV.
  Inputs: 32-d feature vector per item.
  Output: P(correct).

Note on leakage:
  The OOF predictions from the base models are already "held out" (each item
  was in the validation set of one fold when the base model was trained).
  So we can safely run a new 5-fold CV on top without label leakage.

Usage:
    PYTHONPATH=. python src/modeling/hybrid.py \
        --deberta-oof    results/month2/deberta_pooled_oof.npz \
        --step-oof       results/month2/step_transformer_pooled_oof.npz \
        --features-glob  "data/features/*_features_rec.csv" \
        --output         results/month2/hybrid_pooled.json
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

from src.modeling.cv_utils import (
    aggregate_folds, evaluate, save_results, stratified_split,
)

logger = logging.getLogger(__name__)


# =============================================================================
# LOAD + ALIGN
# =============================================================================

RECURRENCE_FEATS = [
    "semantic_recurrence_rate", "max_semantic_cycle_span",
    "progress_repetition", "termination_recycle", "revision_ineffectiveness",
]

ALIGNMENT_FEATS = [
    "problem_conclusion_sim", "problem_trace_max_sim",
    "problem_drift", "problem_keyword_coverage",
]


def load_oof(npz_path: str) -> pd.DataFrame:
    """Load an OOF prediction file into a DataFrame keyed by item_id."""
    z = np.load(npz_path, allow_pickle=True)
    df = pd.DataFrame({
        "item_id": z["item_ids"].astype(str),
        "y_true": z["y_true"].astype(int),
        "prob": z["oof_prob"].astype(float),
        "group": z["groups"].astype(str),
    })
    return df


def load_features_csv(glob_pat: str) -> pd.DataFrame:
    """Concatenate all per-dataset feature CSVs, tagging each row with `group`
    derived from the filename. `group` disambiguates rows that share item_id
    across different (dataset, model) combinations (e.g., math500_qwen7b and
    math500_llama8b both use item_id 'math500_0000')."""
    paths = sorted(glob.glob(glob_pat))
    logger.info(f"Loading {len(paths)} feature CSVs")
    dfs = []
    for p in paths:
        d = pd.read_csv(p)
        group = (os.path.basename(p)
                 .replace("_features_align.csv", "")
                 .replace("_features_rec.csv", "")
                 .replace("_features.csv", ""))
        d["group"] = group
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df["item_id"] = df["item_id"].astype(str)
    df["group"] = df["group"].astype(str)
    return df


def build_matrix(deberta_df: pd.DataFrame, step_df: pd.DataFrame,
                 feat_df: pd.DataFrame,
                 cond_df: pd.DataFrame | None = None,
                 include_deberta: bool = True,
                 include_step: bool = True,
                 include_conditioned: bool = False,
                 include_feats: bool = True,
                 include_recurrence: bool = True,
                 include_alignment: bool = False,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Inner-join on (item_id, group). Returns X, y, group, col_names.

    Joining on item_id alone would cartesian-product rows that share
    item_id across (dataset, model) pairs (e.g. math500_0000 appears in
    both Qwen and Llama traces). We disambiguate with `group`."""
    d = deberta_df.rename(columns={"prob": "deberta_prob"})[["item_id", "group", "deberta_prob", "y_true"]]
    s = step_df.rename(columns={"prob": "step_prob"})[["item_id", "group", "step_prob"]]
    merged = d.merge(s, on=["item_id", "group"], how="inner")
    if cond_df is not None:
        c = cond_df.rename(columns={"prob": "cond_prob"})[["item_id", "group", "cond_prob"]]
        merged = merged.merge(c, on=["item_id", "group"], how="inner")
    merged = merged.merge(feat_df, on=["item_id", "group"], how="inner",
                          suffixes=("", "_feat"))

    logger.info(f"After join: {len(merged)} items")

    y = merged["y_true"].to_numpy().astype(int)
    group = merged["group"].to_numpy()

    cols = []
    if include_deberta:
        cols.append("deberta_prob")
    if include_step:
        cols.append("step_prob")
    if include_conditioned and "cond_prob" in merged.columns:
        cols.append("cond_prob")

    # Handcrafted 25 (excluding labels/ids/recurrence/alignment/group)
    exclude = ({"item_id", "dataset", "is_correct", "y_true",
                "deberta_prob", "step_prob", "cond_prob", "group"}
               | set(RECURRENCE_FEATS) | set(ALIGNMENT_FEATS))
    handcrafted_cols = [c for c in merged.columns
                        if c not in exclude and merged[c].dtype != object]
    if include_feats:
        cols.extend(handcrafted_cols)
    if include_recurrence:
        cols.extend([c for c in RECURRENCE_FEATS if c in merged.columns])
    if include_alignment:
        cols.extend([c for c in ALIGNMENT_FEATS if c in merged.columns])

    X = merged[cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    logger.info(f"X shape: {X.shape}  (cols: {len(cols)})")
    return X, y, group, cols


# =============================================================================
# META-LEARNER CV
# =============================================================================

def run_meta_cv(X: np.ndarray, y: np.ndarray, group: np.ndarray,
                variant_name: str, clf_name: str = "rf",
                n_splits: int = 5, seed: int = 42) -> dict:
    fold_metrics = []
    all_y, all_p = [], []
    for fold, (tr, te) in enumerate(stratified_split(
            y, group_id=group if len(set(group)) > 1 else None,
            n_splits=n_splits, seed=seed)):
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr]); Xte = scaler.transform(X[te])
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
        prob = m.predict_proba(Xte)[:, 1]
        fm = evaluate(yte, prob, name=f"fold_{fold + 1}")
        fold_metrics.append(fm)
        all_y.append(yte); all_p.append(prob)

    all_y = np.concatenate(all_y); all_p = np.concatenate(all_p)
    overall = evaluate(all_y, all_p, name=variant_name)
    summary = aggregate_folds(fold_metrics)

    logger.info(f"[{variant_name:30s}  clf={clf_name:3s}]  "
                f"AUROC={summary.get('auroc_mean',0):.4f} "
                f"± {summary.get('auroc_std',0):.4f}  "
                f"ECE={summary.get('ece_mean',0):.4f}")
    return {
        "variant": variant_name, "clf": clf_name,
        "fold_metrics": fold_metrics,
        "summary": summary,
        "overall": overall,
    }


# =============================================================================
# DRIVER
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deberta-oof", required=True)
    p.add_argument("--step-oof", required=True)
    p.add_argument("--conditioned-oof", default=None,
                   help="OOF .npz from problem-conditioned DeBERTa (optional)")
    p.add_argument("--features-glob", required=True,
                   help="Feature CSVs (use *_features_align.csv when available)")
    p.add_argument("--output", required=True)
    p.add_argument("--clf", default="all", choices=["lr", "rf", "xgb", "all"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-splits", type=int, default=5)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    deberta_df = load_oof(args.deberta_oof)
    step_df = load_oof(args.step_oof)
    cond_df = load_oof(args.conditioned_oof) if args.conditioned_oof else None
    feat_df = load_features_csv(args.features_glob)

    logger.info(f"DeBERTa OOF rows:        {len(deberta_df)}")
    logger.info(f"Step OOF rows:           {len(step_df)}")
    if cond_df is not None:
        logger.info(f"Conditioned OOF rows:    {len(cond_df)}")
    logger.info(f"Feature rows:            {len(feat_df)}")

    # ---------- Variants ----------
    base_variants = {
        "deberta_only":        dict(include_deberta=True,  include_step=False, include_feats=False, include_recurrence=False, include_alignment=False),
        "step_only":           dict(include_deberta=False, include_step=True,  include_feats=False, include_recurrence=False, include_alignment=False),
        "handcrafted25_only":  dict(include_deberta=False, include_step=False, include_feats=True,  include_recurrence=False, include_alignment=False),
        "handcrafted+rec":     dict(include_deberta=False, include_step=False, include_feats=True,  include_recurrence=True,  include_alignment=False),
        "handcrafted+rec+align": dict(include_deberta=False, include_step=False, include_feats=True, include_recurrence=True, include_alignment=True),
        "deberta+feats":       dict(include_deberta=True,  include_step=False, include_feats=True,  include_recurrence=True,  include_alignment=False),
        "deberta+step":        dict(include_deberta=True,  include_step=True,  include_feats=False, include_recurrence=False, include_alignment=False),
        "deberta+feats+align": dict(include_deberta=True,  include_step=False, include_feats=True,  include_recurrence=True,  include_alignment=True),
        "FULL_HYBRID":         dict(include_deberta=True,  include_step=True,  include_feats=True,  include_recurrence=True,  include_alignment=True),
    }
    if cond_df is not None:
        base_variants["conditioned_only"] = dict(include_deberta=False, include_step=False, include_conditioned=True, include_feats=False, include_recurrence=False, include_alignment=False)
        base_variants["deberta+cond"] = dict(include_deberta=True, include_step=False, include_conditioned=True, include_feats=False, include_recurrence=False, include_alignment=False)
        base_variants["conditioned+feats+align"] = dict(include_deberta=False, include_step=False, include_conditioned=True, include_feats=True, include_recurrence=True, include_alignment=True)
        base_variants["deberta+cond+feats+align"] = dict(include_deberta=True, include_step=False, include_conditioned=True, include_feats=True, include_recurrence=True, include_alignment=True)
        base_variants["ULTRA_HYBRID"] = dict(include_deberta=True, include_step=True, include_conditioned=True, include_feats=True, include_recurrence=True, include_alignment=True)

    variants = base_variants

    clf_list = [args.clf] if args.clf != "all" else ["lr", "rf", "xgb"]

    all_results = {"variants": {}}
    for vname, cfg in variants.items():
        X, y, group, cols = build_matrix(deberta_df, step_df, feat_df,
                                          cond_df=cond_df, **cfg)
        variant_block = {"n_features": X.shape[1],
                         "n_samples": X.shape[0],
                         "feature_names": cols,
                         "base_accuracy": float(y.mean()),
                         "clfs": {}}
        for c in clf_list:
            if c == "xgb":
                try:
                    import xgboost  # noqa: F401
                except ImportError:
                    logger.warning("xgboost not installed, skip")
                    continue
            variant_block["clfs"][c] = run_meta_cv(
                X, y, group, vname, clf_name=c,
                n_splits=args.n_splits, seed=args.seed,
            )
        all_results["variants"][vname] = variant_block

    save_results(args.output, all_results)

    # Pretty summary table
    print("\n" + "=" * 88)
    print(f"{'variant':28s}  {'n_feat':>6s}  {'LR_auroc':>12s}  {'RF_auroc':>12s}  {'XGB_auroc':>12s}")
    print("-" * 88)
    for vname, vblock in all_results["variants"].items():
        row = f"{vname:28s}  {vblock['n_features']:6d}"
        for c in ["lr", "rf", "xgb"]:
            if c in vblock["clfs"]:
                s = vblock["clfs"][c]["summary"]
                row += f"  {s.get('auroc_mean', 0):.3f}±{s.get('auroc_std', 0):.3f}"
            else:
                row += f"  {'--':>12s}"
        print(row)


if __name__ == "__main__":
    main()
