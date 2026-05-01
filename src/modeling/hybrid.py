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
from typing import Optional

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

PH_FEATS = [
    "h0_total_persistence", "h0_max_persistence", "h0_n_bars",
    "h0_persistence_entropy",
    "h1_total_persistence", "h1_max_persistence", "h1_n_bars",
]

# Persistence-image features from topology_features_v2.py:
#  4 length-normalized scalars + 2 * 4*4 grid cells = 36 features
PHIMG_PI_RES = 4
PHIMG_FEATS = (
    ["h0_total_per_step", "h0_n_per_step",
     "h1_total_per_step", "h1_n_per_step"]
    + [f"h{h}_pi_{r}_{c}"
       for h in (0, 1) for r in range(PHIMG_PI_RES) for c in range(PHIMG_PI_RES)]
)


def load_oof(npz_path: str) -> pd.DataFrame:
    """Load an OOF prediction file into a DataFrame keyed by (item_id, group)."""
    z = np.load(npz_path, allow_pickle=True)
    df = pd.DataFrame({
        "item_id": z["item_ids"].astype(str),
        "y_true": z["y_true"].astype(int),
        "prob": z["oof_prob"].astype(float),
        "group": z["groups"].astype(str),
    })
    return df


def _normalize_feat_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every per-dataset feature CSV has a 'dataset' column carrying the
    model-disambiguated key (e.g. 'math500_qwen7b'). Some older CSVs stored
    only the bare dataset ('math500'), causing the subsequent (item_id,
    dataset) join against StepTF's 'group' column to return zero matches.
    This helper accepts either form and normalizes in-place."""
    if "dataset" not in df.columns:
        raise ValueError("feature CSV missing 'dataset' column — cannot disambiguate "
                         "qwen/llama traces with the same item_id")
    return df


def _group_name_from_path(path: str) -> str:
    """Derive canonical (dataset_model) group name from a feature CSV path.

    Accepts:  .../<group>_features_rec.csv
              .../<group>_features_ph.csv
              .../<group>_features.csv
    Returns e.g. 'math500_qwen7b' — matches the step_transformer OOF 'group'
    column, which is derived the same way from the .npz basename in
    cv_utils.load_pooled_npz.
    """
    base = os.path.basename(path)
    for suffix in ("_features_rec.csv", "_features_ph.csv", "_features.csv"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base.replace(".csv", "")


def load_features_csv(glob_pat: str) -> pd.DataFrame:
    """Load per-dataset *_features_rec.csv files.

    We re-derive the 'dataset' column from the filename (e.g.
    'math500_qwen7b_features_rec.csv' -> 'math500_qwen7b') because some
    legacy CSVs wrote only the bare dataset name ('math500'), which would
    cause a 0-row join against StepTF's 'group' column. Filename is the
    single source of truth for the disambiguated group key."""
    paths = sorted(glob.glob(glob_pat))
    logger.info(f"Loading {len(paths)} feature CSVs")
    dfs = []
    for p in paths:
        d = pd.read_csv(p)
        d["dataset"] = _group_name_from_path(p)   # authoritative; overwrite
        dfs.append(_normalize_feat_df(d))
    df = pd.concat(dfs, ignore_index=True)
    df["item_id"] = df["item_id"].astype(str)
    df["dataset"] = df["dataset"].astype(str)
    n_rows, n_unique_item = len(df), df["item_id"].nunique()
    n_unique_pair = df.groupby(["item_id", "dataset"]).ngroups
    logger.info(f"  rows={n_rows}  unique_item_id={n_unique_item}  "
                f"unique_(item_id,dataset)={n_unique_pair}")
    logger.info(f"  dataset values: {sorted(df['dataset'].unique())[:4]}...")
    if n_unique_pair != n_rows:
        logger.warning("  duplicate (item_id,dataset) pairs found — "
                       "consider checking your CSV generation pipeline")
    return df


def load_ph_csv(glob_pat: str) -> Optional[pd.DataFrame]:
    """Optional: load per-dataset *_features_ph.csv (PH features, v1).
    Returns None if no files match. Same filename-derived dataset stamping."""
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        return None
    logger.info(f"Loading {len(paths)} PH feature CSVs")
    dfs = []
    for p in paths:
        d = pd.read_csv(p)
        d["dataset"] = _group_name_from_path(p)
        d["item_id"] = d["item_id"].astype(str)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df["dataset"] = df["dataset"].astype(str)
    keep = ["item_id", "dataset"] + [c for c in PH_FEATS if c in df.columns]
    logger.info(f"  PH dataset values: {sorted(df['dataset'].unique())[:4]}...")
    return df[keep].copy()


def load_phimg_csv(glob_pat: str) -> Optional[pd.DataFrame]:
    """Optional: load per-dataset *_features_phimg.csv (PH-image features, v2).
    Returns None if no files match. Same filename-derived dataset stamping
    pattern as load_ph_csv."""
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        return None
    logger.info(f"Loading {len(paths)} PH-image feature CSVs")
    dfs = []
    for p in paths:
        d = pd.read_csv(p)
        d["dataset"] = _group_name_from_path(p)
        d["item_id"] = d["item_id"].astype(str)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df["dataset"] = df["dataset"].astype(str)
    keep = ["item_id", "dataset"] + [c for c in PHIMG_FEATS if c in df.columns]
    logger.info(f"  PH-image dataset values: {sorted(df['dataset'].unique())[:4]}... "
                f"({len(keep) - 2} feature cols)")
    return df[keep].copy()


def build_matrix(deberta_df: Optional[pd.DataFrame], step_df: pd.DataFrame,
                 feat_df: pd.DataFrame,
                 ph_df: Optional[pd.DataFrame] = None,
                 phimg_df: Optional[pd.DataFrame] = None,
                 include_deberta: bool = False,       # CHANGED: all signals
                 include_step: bool = False,          # opt-IN now. Variant
                 include_feats: bool = False,         # dicts must explicitly
                 include_recurrence: bool = False,    # set True for every
                 include_ph: bool = False,            # signal they want.
                 include_phimg: bool = False,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Inner-join on (item_id, dataset/group) pair. Returns X, y, group, col_names.

    CRITICAL BUG FIX: item_ids in feature CSVs are NOT model-disambiguated
    (e.g. 'math500_0001' exists in both math500_qwen7b and math500_llama8b
    CSVs). Joining on item_id alone caused each item to appear twice, which
    inflated hybrid AUROC via fold-shuffle leakage. We now join on the
    (item_id, dataset) pair, where step_df.group ≡ feat_df.dataset, e.g.
    'math500_qwen7b'.

    When `deberta_df is None`, labels/groups are anchored on StepTF's OOF
    instead (which carries the same y_true and group arrays)."""
    s = step_df.rename(columns={"prob": "step_prob"})[
        ["item_id", "step_prob", "y_true", "group"]
    ]

    # Early diagnostic — show group values from both sides before any merge
    step_groups = sorted(set(s["group"].unique()))
    feat_groups = sorted(set(feat_df["dataset"].unique()))
    overlap = sorted(set(step_groups) & set(feat_groups))
    if not overlap:
        logger.error(f"  step_df.group:  {step_groups}")
        logger.error(f"  feat_df.dataset: {feat_groups}")
        logger.error("  NO OVERLAP — the join will return 0 rows. "
                     "Check that your *_features_rec.csv filenames encode the "
                     "model suffix (e.g. math500_qwen7b_features_rec.csv).")

    if deberta_df is not None:
        d = deberta_df.rename(columns={"prob": "deberta_prob"})[
            ["item_id", "deberta_prob", "y_true", "group"]
        ]
        # Join DeBERTa + Step on (item_id, group) — same items in both OOF files
        merged = d.merge(s.drop(columns=["y_true"]),
                         on=["item_id", "group"], how="inner")
    else:
        merged = s.copy()

    # Join with features on (item_id, dataset/group). We drop is_correct from
    # feat_df before merging to avoid a column collision with y_true.
    feat_to_merge = feat_df.rename(columns={"dataset": "group"})
    if "is_correct" in feat_to_merge.columns:
        feat_to_merge = feat_to_merge.drop(columns=["is_correct"])
    merged = merged.merge(feat_to_merge, on=["item_id", "group"], how="inner",
                          suffixes=("", "_feat"))

    # Optional: merge PH features (v1 — 7 summary stats)
    if ph_df is not None and include_ph:
        ph_to_merge = ph_df.rename(columns={"dataset": "group"})
        merged = merged.merge(ph_to_merge, on=["item_id", "group"], how="inner",
                              suffixes=("", "_ph"))

    # Optional: merge PH-image features (v2 — length-normalized + 32 image cells)
    if phimg_df is not None and include_phimg:
        phimg_to_merge = phimg_df.rename(columns={"dataset": "group"})
        merged = merged.merge(phimg_to_merge, on=["item_id", "group"], how="inner",
                              suffixes=("", "_phimg"))

    logger.info(f"After join: {len(merged)} rows  (unique item_ids: "
                f"{merged['item_id'].nunique()}, "
                f"unique (item_id,group) pairs: "
                f"{merged.groupby(['item_id', 'group']).ngroups})")
    if merged["item_id"].nunique() == 0 or len(merged) == 0:
        raise ValueError("Join produced zero rows — likely dataset-column "
                         "mismatch between feature CSVs ('dataset') and "
                         "OOF files ('group'). Run a diagnostic:\n"
                         "  head -1 data/features/math500_qwen7b_features_rec.csv\n"
                         "  python -c \"import numpy as np; "
                         "z=np.load('results/month2_v2/step_transformer_pooled_oof.npz', "
                         "allow_pickle=True); print(sorted(set(z['groups'].astype(str))))\"")

    y = merged["y_true"].to_numpy().astype(int)
    group = merged["group"].to_numpy()

    cols = []
    if include_deberta:
        cols.append("deberta_prob")
    if include_step:
        cols.append("step_prob")

    # Handcrafted 25 (excluding labels/ids/recurrence/PH/PHimg/group)
    exclude = ({"item_id", "dataset", "is_correct", "y_true",
                "deberta_prob", "step_prob", "group"}
               | set(RECURRENCE_FEATS) | set(PH_FEATS) | set(PHIMG_FEATS))
    handcrafted_cols = [c for c in merged.columns
                        if c not in exclude and merged[c].dtype != object]
    if include_feats:
        cols.extend(handcrafted_cols)
    if include_recurrence:
        cols.extend([c for c in RECURRENCE_FEATS if c in merged.columns])
    if include_ph:
        cols.extend([c for c in PH_FEATS if c in merged.columns])
    if include_phimg:
        cols.extend([c for c in PHIMG_FEATS if c in merged.columns])

    if not cols:
        raise ValueError(
            "No feature columns selected (all include_* flags are False or "
            "optional dataframes missing). Check variant config."
        )
    X = merged[cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    logger.info(f"X shape: {X.shape}  (cols: {len(cols)})")
    # Print the first few and last few column names so each variant's feature
    # set is visible in the log — catches regressions like "every variant is
    # actually using the same columns" that would otherwise go unnoticed.
    preview = cols if len(cols) <= 6 else cols[:3] + ["..."] + cols[-2:]
    logger.info(f"  cols preview: {preview}")
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
                              use_label_encoder=False)
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
    p.add_argument("--deberta-oof", default=None,
                   help="Path to DeBERTa pooled OOF .npz. Omit to run "
                        "structural-only hybrid (no text-encoder signal).")
    p.add_argument("--step-oof", required=True)
    p.add_argument("--features-glob", required=True,
                   help="Glob for *_features_rec.csv (handcrafted-25 + recurrence-5)")
    p.add_argument("--ph-glob", default=None,
                   help="Optional glob for *_features_ph.csv (persistent-homology-7). "
                        "When supplied, unlocks PH-augmented variants.")
    p.add_argument("--phimg-glob", default=None,
                   help="Optional glob for *_features_phimg.csv (length-normalized PH "
                        "+ 32 persistence-image cells). Unlocks PHimg variants.")
    p.add_argument("--output", required=True)
    p.add_argument("--clf", default="all", choices=["lr", "rf", "xgb", "all"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-splits", type=int, default=5)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    have_deberta = args.deberta_oof is not None
    deberta_df = load_oof(args.deberta_oof) if have_deberta else None
    step_df = load_oof(args.step_oof)
    feat_df = load_features_csv(args.features_glob)
    ph_df = load_ph_csv(args.ph_glob) if args.ph_glob else None
    have_ph = ph_df is not None
    phimg_df = load_phimg_csv(args.phimg_glob) if args.phimg_glob else None
    have_phimg = phimg_df is not None

    if have_deberta:
        logger.info(f"DeBERTa OOF rows: {len(deberta_df)}")
    else:
        logger.info("DeBERTa OOF: (not supplied — structural-only hybrid)")
    logger.info(f"Step OOF rows:    {len(step_df)}")
    logger.info(f"Feature rows:     {len(feat_df)}")
    if have_ph:
        logger.info(f"PH feature rows:  {len(ph_df)}")
    else:
        logger.info("PH features: (not supplied)")
    if have_phimg:
        logger.info(f"PHimg feature rows: {len(phimg_df)}")
    else:
        logger.info("PH-image features: (not supplied)")

    # ---------- Variants to try ----------
    # Each config dict gets splatted into build_matrix(**cfg).
    # Defaults for any flag not specified = False (handled below by .get).
    full_variants = {
        # Single-signal baselines
        "deberta_only":         dict(include_deberta=True),
        "step_only":             dict(include_step=True),
        "handcrafted25_only":    dict(include_feats=True),
        "ph_only":               dict(include_ph=True),
        "phimg_only":            dict(include_phimg=True),
        # 2-3 signal combos
        "handcrafted+rec":       dict(include_feats=True, include_recurrence=True),
        "handcrafted+rec+ph":    dict(include_feats=True, include_recurrence=True, include_ph=True),
        "handcrafted+rec+phimg": dict(include_feats=True, include_recurrence=True, include_phimg=True),
        "deberta+feats":         dict(include_deberta=True, include_feats=True, include_recurrence=True),
        "deberta+step":          dict(include_deberta=True, include_step=True),
        "step+ph":               dict(include_step=True, include_ph=True),
        "step+phimg":            dict(include_step=True, include_phimg=True),
        # All-structural variants
        "STRUCTURAL_FULL":       dict(include_step=True, include_feats=True, include_recurrence=True),
        "STRUCTURAL_FULL+ph":    dict(include_step=True, include_feats=True, include_recurrence=True, include_ph=True),
        "STRUCTURAL_FULL+phimg": dict(include_step=True, include_feats=True, include_recurrence=True, include_phimg=True),
        # Full hybrids (need text encoder)
        "FULL_HYBRID":           dict(include_deberta=True, include_step=True, include_feats=True, include_recurrence=True),
        "FULL_HYBRID+ph":        dict(include_deberta=True, include_step=True, include_feats=True, include_recurrence=True, include_ph=True),
        "FULL_HYBRID+phimg":     dict(include_deberta=True, include_step=True, include_feats=True, include_recurrence=True, include_phimg=True),
    }
    # Drop variants that need signals we don't have
    variants = {
        v: cfg for v, cfg in full_variants.items()
        if (have_deberta or not cfg.get("include_deberta", False))
        and (have_ph    or not cfg.get("include_ph", False))
        and (have_phimg or not cfg.get("include_phimg", False))
    }

    clf_list = [args.clf] if args.clf != "all" else ["lr", "rf", "xgb"]

    all_results = {"variants": {}}
    for vname, cfg in variants.items():
        X, y, group, cols = build_matrix(
            deberta_df, step_df, feat_df,
            ph_df=ph_df, phimg_df=phimg_df, **cfg,
        )
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
