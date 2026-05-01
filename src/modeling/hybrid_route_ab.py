"""
hybrid_route_ab.py - Stacking meta-learner for the Route A + B1 ablation.

Extends src/modeling/hybrid.py with the new content-free feature families and
OOF predictors:

  Route A CSVs (content-free):
    A1 n-gram motifs           data/features/{group}_ngram.csv
    A3 trace-graph descriptors data/features/{group}_graph.csv
    A5 inter-event timing      data/features/{group}_timing.csv
    A4 structural PH           data/features/{group}_structural_ph.csv

  Route B1 OOFs (learned-model probs from their own 5-fold CV):
    Shapelet OOF       results/route_ab/shapelet_oof.npz
    GNN structural OOF results/route_ab/trace_gnn_structural_oof.npz
    GNN hybrid OOF     results/route_ab/trace_gnn_hybrid_oof.npz   (optional)

  Legacy base-model OOFs (required for falsifier variants only):
    DeBERTa   results/month2*/deberta_pooled_oof.npz
    StepTF    results/month2*/step_transformer_pooled_oof.npz

  Legacy handcrafted CSVs:
    data/features/{group}_features_rec.csv   (handcrafted-25 + recurrence-5)

Variants evaluated (see VARIANTS dict below). The two falsifier-critical ones:
  * ROUTE_AB_TOTAL    handcrafted + rec + A1 + A3 + A5 + A4 + shapelet +
                      GNN-structural       (purely structural stack)
  * ROUTE_AB_DEBERTA  above + DeBERTa OOF  (does structure add to text?)

Outputs:
  <output>.json                    all variants x {lr, rf, xgb}
  <output>_per_dataset.csv         per (variant, group, clf) -> AUROC etc
  <output>_falsifier.csv           per group: best-DeBERTa vs ROUTE_AB_TOTAL

Usage:
  PYTHONPATH=. python src/modeling/hybrid_route_ab.py \\
      --features-glob      "data/features/*_features_rec.csv" \\
      --ngram-glob         "data/features/*_ngram.csv" \\
      --graph-glob         "data/features/*_graph.csv" \\
      --timing-glob        "data/features/*_timing.csv" \\
      --structural-ph-glob "data/features/*_structural_ph.csv" \\
      --shapelet-oof       results/route_ab/shapelet_oof.npz \\
      --gnn-structural-oof results/route_ab/trace_gnn_structural_oof.npz \\
      --deberta-oof        results/month2_v2/deberta_pooled_oof.npz \\
      --step-oof           results/month2_v2/step_transformer_pooled_oof.npz \\
      --output             results/route_ab/route_ab_pooled.json \\
      --clf all
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
# CONSTANTS — column groups
# =============================================================================

RECURRENCE_FEATS = [
    "semantic_recurrence_rate", "max_semantic_cycle_span",
    "progress_repetition", "termination_recycle", "revision_ineffectiveness",
]

# These are the "handcrafted 25" columns once we strip labels/ids/recurrence.
BASE_EXCLUDE = {
    "item_id", "dataset", "is_correct", "y_true", "group", "prob",
    "deberta_prob", "step_prob", "shapelet_prob",
    "gnn_struct_prob", "gnn_hybrid_prob",
} | set(RECURRENCE_FEATS)


# =============================================================================
# LOADERS
# =============================================================================

def _group_name_from_path(path: str) -> str:
    """Strip suffix like _features_rec.csv, _ngram.csv, etc."""
    base = os.path.basename(path)
    for suffix in (
        "_features_rec.csv", "_features_ph.csv", "_features_phimg.csv",
        "_features.csv",
        "_ngram.csv", "_graph.csv", "_timing.csv", "_structural_ph.csv",
    ):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base.replace(".csv", "")


def _load_family_csv(glob_pat: Optional[str], family_name: str,
                     prefixes: tuple = ()) -> Optional[pd.DataFrame]:
    """Generic per-family CSV loader.

    Every family CSV must carry item_id + dataset + features. Pass one or more
    column name prefixes (e.g. ("ng2_", "ng3_", "ng_") for n-grams) to keep
    only that family's columns; pass an empty tuple to keep every non-id,
    non-label numeric column (used when the family has no unified prefix, e.g.
    the structural PH features reuse names from topology_features_v2).

    Returns a DataFrame with (item_id, dataset, [is_correct]) + family feature
    cols. Returns None if no files match.
    """
    if not glob_pat:
        return None
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        logger.warning(f"  [{family_name}] no CSVs match {glob_pat}")
        return None
    logger.info(f"  [{family_name}] loading {len(paths)} CSVs")
    dfs = []
    for p in paths:
        d = pd.read_csv(p)
        d["dataset"] = _group_name_from_path(p)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df["item_id"] = df["item_id"].astype(str)
    df["dataset"] = df["dataset"].astype(str)
    drop = {"item_id", "dataset", "is_correct"}
    if prefixes:
        feat_cols = [c for c in df.columns
                     if c not in drop
                     and any(c.startswith(pref) for pref in prefixes)]
    else:
        feat_cols = [c for c in df.columns
                     if c not in drop and df[c].dtype != object]
    keep = ["item_id", "dataset"]
    if "is_correct" in df.columns:
        keep.append("is_correct")
    keep.extend(feat_cols)
    out = df[keep].copy()
    logger.info(f"    -> {len(out)} rows, {len(feat_cols)} feature cols  "
                f"(datasets: {sorted(out['dataset'].unique())})")
    return out


def load_handcrafted(glob_pat: str) -> pd.DataFrame:
    """*_features_rec.csv — handcrafted-25 + recurrence-5."""
    df = _load_family_csv(glob_pat, "handcrafted+rec", prefixes=())
    if df is None:
        raise SystemExit(
            f"No handcrafted feature CSVs matched {glob_pat!r}. Run "
            "src/features/feature_pipeline.py (or feature_extractor.py) to "
            "regenerate them.")
    return df


def load_ngram(glob_pat: Optional[str]) -> Optional[pd.DataFrame]:
    """A1 — n-gram motif features (cols prefixed ng2_, ng3_, ng_)."""
    return _load_family_csv(glob_pat, "ngram",
                            prefixes=("ng2_", "ng3_", "ng_"))


def load_graph(glob_pat: Optional[str]) -> Optional[pd.DataFrame]:
    """A3 — graph descriptor features (cols prefixed g_)."""
    return _load_family_csv(glob_pat, "graph", prefixes=("g_",))


def load_timing(glob_pat: Optional[str]) -> Optional[pd.DataFrame]:
    """A5 — timing features (cols prefixed t_)."""
    return _load_family_csv(glob_pat, "timing", prefixes=("t_",))


def load_structural_ph(glob_pat: Optional[str]) -> Optional[pd.DataFrame]:
    """A4 — structural persistent-homology features.

    structural_ph_features.py reuses topology_features_v2 column names
    (h0_total_per_step, h0_pi_r_c, ...) so we can't filter by a simple prefix;
    accept the prefix set that covers its outputs.
    """
    return _load_family_csv(glob_pat, "structural_ph",
                            prefixes=("h0_", "h1_"))


def load_oof(path: Optional[str], prob_col: str) -> Optional[pd.DataFrame]:
    """Load an OOF .npz -> DataFrame with cols (item_id, dataset, y_true,
    <prob_col>)."""
    if not path:
        return None
    z = np.load(path, allow_pickle=True)
    df = pd.DataFrame({
        "item_id": z["item_ids"].astype(str),
        "y_true": z["y_true"].astype(int),
        prob_col: z["oof_prob"].astype(float),
        "dataset": z["groups"].astype(str),
    })
    logger.info(f"  [{prob_col}] loaded {len(df)} rows from {path}  "
                f"datasets={sorted(df['dataset'].unique())}")
    return df


# =============================================================================
# JOIN + FEATURE-MATRIX BUILD
# =============================================================================

def _feat_cols(df: pd.DataFrame) -> list[str]:
    """Numeric feature columns only, excluding ids/labels/oof cols."""
    return [c for c in df.columns
            if c not in BASE_EXCLUDE and df[c].dtype != object]


def build_merged_table(
    hand_df: pd.DataFrame,
    ngram_df: Optional[pd.DataFrame],
    graph_df: Optional[pd.DataFrame],
    timing_df: Optional[pd.DataFrame],
    struct_ph_df: Optional[pd.DataFrame],
    deberta_df: Optional[pd.DataFrame],
    step_df: Optional[pd.DataFrame],
    shapelet_df: Optional[pd.DataFrame],
    gnn_struct_df: Optional[pd.DataFrame],
    gnn_hybrid_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Inner-join everything we have on (item_id, dataset).

    The variant selectors below choose which cols go into X, but they *all*
    operate on the same merged table so the label and fold assignments match
    across variants. Items that are missing from any *required* family just
    drop out — callers should compare variant n_samples to catch attrition.
    """
    # Anchor = handcrafted (which we always have).
    merged = hand_df.rename(columns={"is_correct": "y_true"}).copy()
    if "y_true" not in merged.columns:
        raise ValueError(
            "handcrafted CSV does not carry an 'is_correct' column; cannot "
            "derive labels. Check data/features/*_features_rec.csv."
        )
    logger.info(f"Anchor (handcrafted+rec): {len(merged)} rows, "
                f"{merged['item_id'].nunique()} unique items, "
                f"{merged.groupby(['item_id','dataset']).ngroups} unique pairs")

    def _merge(base, extra, suffix):
        if extra is None:
            return base
        # Drop extras' is_correct (anchor owns y_true) + any shared feature
        # cols to avoid column-collision suffixes muddying later selection.
        extra = extra.drop(columns=["is_correct"], errors="ignore")
        shared = set(base.columns) & set(extra.columns) - {"item_id", "dataset"}
        if shared:
            extra = extra.drop(columns=list(shared))
        return base.merge(extra, on=["item_id", "dataset"], how="inner",
                          suffixes=("", suffix))

    merged = _merge(merged, ngram_df, "_ngram")
    merged = _merge(merged, graph_df, "_graph")
    merged = _merge(merged, timing_df, "_timing")
    merged = _merge(merged, struct_ph_df, "_sph")

    # OOF probs carry their own y_true column; we drop theirs and keep ours.
    for oof in (deberta_df, step_df, shapelet_df, gnn_struct_df, gnn_hybrid_df):
        if oof is None:
            continue
        to_merge = oof.drop(columns=["y_true"], errors="ignore")
        merged = merged.merge(to_merge, on=["item_id", "dataset"], how="inner")

    logger.info(f"Merged table: {len(merged)} rows, "
                f"{len(merged.columns)} cols")
    return merged


# =============================================================================
# VARIANT CONFIG
# =============================================================================

# Each variant is a set of flags describing what goes into X.
# include_feats -> handcrafted-25 (everything non-excluded in hand_df)
# include_rec   -> recurrence-5 (RECURRENCE_FEATS)
# include_<fam> -> the corresponding family CSV's feature cols
# include_<oof> -> add the scalar prob as a feature
VARIANT_FLAGS_DEFAULT = dict(
    include_feats=False, include_rec=False,
    include_ngram=False, include_graph=False, include_timing=False,
    include_struct_ph=False,
    include_shapelet_oof=False,
    include_gnn_struct_oof=False, include_gnn_hybrid_oof=False,
    include_deberta_oof=False, include_step_oof=False,
)


def _mk(**overrides) -> dict:
    v = dict(VARIANT_FLAGS_DEFAULT)
    v.update(overrides)
    return v


VARIANTS: dict[str, dict] = {
    # Baselines
    "baselineC":             _mk(include_feats=True),
    "baselineC+rec":         _mk(include_feats=True, include_rec=True),

    # Single-family additions to C+rec
    "C+rec+ngram":           _mk(include_feats=True, include_rec=True, include_ngram=True),
    "C+rec+graph":           _mk(include_feats=True, include_rec=True, include_graph=True),
    "C+rec+timing":          _mk(include_feats=True, include_rec=True, include_timing=True),
    "C+rec+structural_ph":   _mk(include_feats=True, include_rec=True, include_struct_ph=True),

    # All Route-A CSV-based families
    "ROUTE_A_FULL":          _mk(include_feats=True, include_rec=True,
                                 include_ngram=True, include_graph=True,
                                 include_timing=True, include_struct_ph=True),

    # Route A + shapelet OOF
    "ROUTE_A_FULL+shapelet": _mk(include_feats=True, include_rec=True,
                                 include_ngram=True, include_graph=True,
                                 include_timing=True, include_struct_ph=True,
                                 include_shapelet_oof=True),

    # Route A + GNN structural (B1 content-free)
    "ROUTE_A_FULL+gnn_s":    _mk(include_feats=True, include_rec=True,
                                 include_ngram=True, include_graph=True,
                                 include_timing=True, include_struct_ph=True,
                                 include_gnn_struct_oof=True),

    # Route A + GNN hybrid (B1 with content)
    "ROUTE_A_FULL+gnn_h":    _mk(include_feats=True, include_rec=True,
                                 include_ngram=True, include_graph=True,
                                 include_timing=True, include_struct_ph=True,
                                 include_gnn_hybrid_oof=True),

    # The headline structural stack — this is what must beat DeBERTa on >= 5/8
    "ROUTE_AB_TOTAL":        _mk(include_feats=True, include_rec=True,
                                 include_ngram=True, include_graph=True,
                                 include_timing=True, include_struct_ph=True,
                                 include_shapelet_oof=True,
                                 include_gnn_struct_oof=True),

    # Complementarity tests (needs DeBERTa or StepTF)
    "deberta_only":          _mk(include_deberta_oof=True),
    "step_only":             _mk(include_step_oof=True),
    "ROUTE_AB+deberta":      _mk(include_feats=True, include_rec=True,
                                 include_ngram=True, include_graph=True,
                                 include_timing=True, include_struct_ph=True,
                                 include_shapelet_oof=True,
                                 include_gnn_struct_oof=True,
                                 include_deberta_oof=True),
    "ROUTE_AB+deberta+step": _mk(include_feats=True, include_rec=True,
                                 include_ngram=True, include_graph=True,
                                 include_timing=True, include_struct_ph=True,
                                 include_shapelet_oof=True,
                                 include_gnn_struct_oof=True,
                                 include_deberta_oof=True,
                                 include_step_oof=True),
}


def variant_needs(cfg: dict) -> dict[str, bool]:
    """Which input dataframes are needed by a variant."""
    return {
        "ngram":      cfg["include_ngram"],
        "graph":      cfg["include_graph"],
        "timing":     cfg["include_timing"],
        "struct_ph":  cfg["include_struct_ph"],
        "shapelet":   cfg["include_shapelet_oof"],
        "gnn_struct": cfg["include_gnn_struct_oof"],
        "gnn_hybrid": cfg["include_gnn_hybrid_oof"],
        "deberta":    cfg["include_deberta_oof"],
        "step":       cfg["include_step_oof"],
    }


# =============================================================================
# MATRIX BUILD + META-LEARNER
# =============================================================================

def _collect_cols(
    merged: pd.DataFrame, cfg: dict,
    hand_df: pd.DataFrame,
    ngram_df: Optional[pd.DataFrame],
    graph_df: Optional[pd.DataFrame],
    timing_df: Optional[pd.DataFrame],
    struct_ph_df: Optional[pd.DataFrame],
) -> list[str]:
    """Select feature columns for one variant. Defensive against missing
    columns (e.g. when a CSV family is absent -> its include flag is skipped)."""
    cols: list[str] = []
    # Handcrafted-25 and recurrence-5 both live on hand_df. Handcrafted = the
    # non-recurrence, non-id numeric cols.
    hand_numeric = [c for c in hand_df.columns
                    if c not in ({"item_id", "dataset", "is_correct"}
                                 | set(RECURRENCE_FEATS))
                    and hand_df[c].dtype != object]
    if cfg["include_feats"]:
        cols.extend([c for c in hand_numeric if c in merged.columns])
    if cfg["include_rec"]:
        cols.extend([c for c in RECURRENCE_FEATS if c in merged.columns])

    def _add_if_present(flag, df):
        if flag and df is not None:
            for c in _feat_cols(df):
                if c in merged.columns:
                    cols.append(c)

    _add_if_present(cfg["include_ngram"], ngram_df)
    _add_if_present(cfg["include_graph"], graph_df)
    _add_if_present(cfg["include_timing"], timing_df)
    _add_if_present(cfg["include_struct_ph"], struct_ph_df)

    for flag_key, col_name in [
        ("include_shapelet_oof", "shapelet_prob"),
        ("include_gnn_struct_oof", "gnn_struct_prob"),
        ("include_gnn_hybrid_oof", "gnn_hybrid_prob"),
        ("include_deberta_oof", "deberta_prob"),
        ("include_step_oof", "step_prob"),
    ]:
        if cfg[flag_key] and col_name in merged.columns:
            cols.append(col_name)

    # De-dup while preserving order (different families can collide on
    # generic-sounding names; prefer first occurrence).
    seen = set()
    ordered = []
    for c in cols:
        if c in seen:
            continue
        seen.add(c); ordered.append(c)
    return ordered


def run_meta_cv(X: np.ndarray, y: np.ndarray, group: np.ndarray,
                variant_name: str, clf_name: str = "rf",
                n_splits: int = 5, seed: int = 42,
                return_oof: bool = False) -> dict:
    fold_metrics = []
    oof_prob = np.full(len(y), np.nan, dtype=np.float32)
    oof_fold = np.full(len(y), -1, dtype=np.int32)
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
        oof_prob[te] = prob
        oof_fold[te] = fold
        fm = evaluate(yte, prob, name=f"fold_{fold + 1}")
        fold_metrics.append(fm)
        all_y.append(yte); all_p.append(prob)

    all_y = np.concatenate(all_y); all_p = np.concatenate(all_p)
    overall = evaluate(all_y, all_p, name=variant_name)
    summary = aggregate_folds(fold_metrics)
    logger.info(f"[{variant_name:28s}  clf={clf_name:3s}]  "
                f"AUROC={summary.get('auroc_mean',0):.4f} "
                f"± {summary.get('auroc_std',0):.4f}  "
                f"ECE={summary.get('ece_mean',0):.4f}")
    out = {
        "variant": variant_name, "clf": clf_name,
        "fold_metrics": fold_metrics,
        "summary": summary,
        "overall": overall,
    }
    if return_oof:
        out["oof_prob"] = oof_prob.tolist()
        out["oof_fold"] = oof_fold.tolist()
    return out


# =============================================================================
# PER-DATASET + FALSIFIER TABLES
# =============================================================================

def per_dataset_metrics(y: np.ndarray, prob: np.ndarray, group: np.ndarray
                        ) -> dict[str, dict]:
    out = {}
    for g in sorted(set(group)):
        mk = group == g
        if mk.sum() == 0:
            continue
        out[g] = evaluate(y[mk], prob[mk], name=g)
    return out


# =============================================================================
# DRIVER
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-glob", required=True,
                   help="Glob for *_features_rec.csv (handcrafted + rec)")
    p.add_argument("--ngram-glob", default=None)
    p.add_argument("--graph-glob", default=None)
    p.add_argument("--timing-glob", default=None)
    p.add_argument("--structural-ph-glob", default=None)

    p.add_argument("--shapelet-oof", default=None)
    p.add_argument("--gnn-structural-oof", default=None)
    p.add_argument("--gnn-hybrid-oof", default=None)
    p.add_argument("--deberta-oof", default=None)
    p.add_argument("--step-oof", default=None)

    p.add_argument("--output", required=True)
    p.add_argument("--clf", default="all", choices=["lr", "rf", "xgb", "all"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--variants", nargs="+", default=None,
                   help="Subset of variants to run; default = all applicable.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # ----- Load everything -----
    logger.info("Loading feature CSVs...")
    hand_df = load_handcrafted(args.features_glob)
    ngram_df = load_ngram(args.ngram_glob)
    graph_df = load_graph(args.graph_glob)
    timing_df = load_timing(args.timing_glob)
    struct_ph_df = load_structural_ph(args.structural_ph_glob)

    logger.info("Loading OOF predictors...")
    deberta_df = load_oof(args.deberta_oof, "deberta_prob")
    step_df = load_oof(args.step_oof, "step_prob")
    shapelet_df = load_oof(args.shapelet_oof, "shapelet_prob")
    gnn_s_df = load_oof(args.gnn_structural_oof, "gnn_struct_prob")
    gnn_h_df = load_oof(args.gnn_hybrid_oof, "gnn_hybrid_prob")

    merged = build_merged_table(
        hand_df, ngram_df, graph_df, timing_df, struct_ph_df,
        deberta_df, step_df, shapelet_df, gnn_s_df, gnn_h_df,
    )

    y_all = merged["y_true"].to_numpy().astype(int)
    group_all = merged["dataset"].to_numpy()

    # ----- Pick variants -----
    # Drop variants whose required inputs are absent.
    have = {
        "ngram": ngram_df is not None,
        "graph": graph_df is not None,
        "timing": timing_df is not None,
        "struct_ph": struct_ph_df is not None,
        "shapelet": shapelet_df is not None,
        "gnn_struct": gnn_s_df is not None,
        "gnn_hybrid": gnn_h_df is not None,
        "deberta": deberta_df is not None,
        "step": step_df is not None,
    }
    runnable = {}
    for vname, cfg in VARIANTS.items():
        needs = variant_needs(cfg)
        if all(have[k] or (not needs[k]) for k in needs):
            runnable[vname] = cfg
        else:
            missing = [k for k in needs if needs[k] and not have[k]]
            logger.info(f"  [skip] {vname} — missing {missing}")
    if args.variants:
        runnable = {v: runnable[v] for v in args.variants if v in runnable}
    logger.info(f"Running {len(runnable)} variants")

    clf_list = [args.clf] if args.clf != "all" else ["lr", "rf", "xgb"]

    all_results = {"variants": {}}
    per_dataset_rows = []
    oof_store: dict[str, dict] = {}   # keyed by variant_clf

    for vname, cfg in runnable.items():
        cols = _collect_cols(merged, cfg, hand_df, ngram_df, graph_df,
                             timing_df, struct_ph_df)
        if not cols:
            logger.warning(f"  [skip] {vname} — no columns selected")
            continue
        X = merged[cols].to_numpy(dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        vblock = {"n_features": X.shape[1], "n_samples": X.shape[0],
                  "feature_names": cols, "base_accuracy": float(y_all.mean()),
                  "clfs": {}}
        for c in clf_list:
            if c == "xgb":
                try:
                    import xgboost  # noqa: F401
                except ImportError:
                    logger.warning("  xgboost not installed, skip")
                    continue
            res = run_meta_cv(
                X, y_all, group_all, vname, clf_name=c,
                n_splits=args.n_splits, seed=args.seed,
                return_oof=True,
            )
            vblock["clfs"][c] = res
            # per-dataset decomposition (on concatenated OOF probs)
            pd_metrics = per_dataset_metrics(
                y_all, np.asarray(res["oof_prob"]), group_all,
            )
            for g, m in pd_metrics.items():
                per_dataset_rows.append({
                    "variant": vname, "clf": c, "group": g,
                    "auroc": m["auroc"], "auprc": m["auprc"],
                    "ece": m["ece"], "accuracy": m["accuracy"],
                    "accuracy_at_80": m["accuracy_at_80"],
                    "accuracy_at_90": m["accuracy_at_90"],
                    "prr": m["prr"],
                    "n_samples": m["n_samples"],
                })
            oof_store[f"{vname}__{c}"] = {
                "oof_prob": np.asarray(res["oof_prob"]),
                "item_id": merged["item_id"].to_numpy(),
                "group": group_all,
                "y_true": y_all,
            }
            # Drop the verbose oof arrays from the JSON to keep it lean.
            res.pop("oof_prob", None); res.pop("oof_fold", None)
        all_results["variants"][vname] = vblock

    save_results(args.output, all_results)

    # ----- Per-dataset CSV -----
    per_dataset_path = args.output.replace(".json", "_per_dataset.csv")
    pd.DataFrame(per_dataset_rows).to_csv(per_dataset_path, index=False)
    logger.info(f"Saved: {per_dataset_path}")

    # ----- Falsifier table (DeBERTa vs ROUTE_AB_TOTAL) -----
    if "ROUTE_AB_TOTAL" in all_results["variants"] \
            and "deberta_only" in all_results["variants"]:
        falsifier = []
        pdd = pd.DataFrame(per_dataset_rows)
        # Pick the best classifier per variant (by pooled AUROC)
        def _best_clf(v):
            clfs = all_results["variants"][v]["clfs"]
            return max(clfs, key=lambda c: clfs[c]["summary"].get("auroc_mean", 0))
        best_D_clf = _best_clf("deberta_only")
        best_R_clf = _best_clf("ROUTE_AB_TOTAL")
        for g in sorted(set(group_all)):
            d_row = pdd[(pdd["variant"] == "deberta_only") &
                        (pdd["clf"] == best_D_clf) &
                        (pdd["group"] == g)]
            r_row = pdd[(pdd["variant"] == "ROUTE_AB_TOTAL") &
                        (pdd["clf"] == best_R_clf) &
                        (pdd["group"] == g)]
            if d_row.empty or r_row.empty:
                continue
            d_auroc = float(d_row["auroc"].iloc[0])
            r_auroc = float(r_row["auroc"].iloc[0])
            falsifier.append({
                "group": g,
                "deberta_auroc": d_auroc,
                "route_ab_total_auroc": r_auroc,
                "delta": r_auroc - d_auroc,
                "passes": r_auroc >= d_auroc,
            })
        if falsifier:
            fal_path = args.output.replace(".json", "_falsifier.csv")
            pd.DataFrame(falsifier).to_csv(fal_path, index=False)
            n_pass = sum(1 for x in falsifier if x["passes"])
            logger.info(f"Falsifier: {n_pass}/{len(falsifier)} datasets ROUTE_AB_TOTAL >= DeBERTa")
            logger.info(f"Saved: {fal_path}")

    # ----- Pretty summary -----
    print("\n" + "=" * 92)
    print(f"{'variant':28s}  {'n_feat':>6s}  "
          f"{'LR':>14s}  {'RF':>14s}  {'XGB':>14s}")
    print("-" * 92)
    for vname, vblock in all_results["variants"].items():
        row = f"{vname:28s}  {vblock['n_features']:6d}"
        for c in ["lr", "rf", "xgb"]:
            if c in vblock["clfs"]:
                s = vblock["clfs"][c]["summary"]
                row += f"  {s.get('auroc_mean', 0):.3f}±{s.get('auroc_std', 0):.3f}"
            else:
                row += f"  {'--':>14s}"
        print(row)


if __name__ == "__main__":
    main()
