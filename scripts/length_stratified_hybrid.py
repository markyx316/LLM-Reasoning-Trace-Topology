#!/usr/bin/env python3
"""
length_stratified_hybrid.py

Inside POOLED data, bin items by trace length (in tokens) and re-evaluate each
hybrid variant within each bin separately. Tests whether structure beats text
in specific length regimes — short MCQ traces where text is sparse, or very
long traces where the text encoder loses early context to truncation.

Differs from src/analysis/length_controlled.py (which works on feature CSVs
only) — this works on the OOF prediction arrays directly, so it can include
the RoBERTa and StepTF channels.

Inputs (all pooled, n=6378):
  - results/month2_v2/roberta_pooled_oof.npz      (RoBERTa text-encoder)
  - results/month2_v2/step_transformer_pooled_oof.npz  (StepTF structural)
  - data/features/*_features_rec.csv              (handcrafted-25 + recurrence-5)

Output:
  - prints per-bin AUROC table for each variant
  - writes results/month2_v2/length_stratified_summary.csv

Usage:
    PYTHONPATH=. python scripts/length_stratified_hybrid.py
    PYTHONPATH=. python scripts/length_stratified_hybrid.py --n-bins 5 --clf rf
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.modeling.cv_utils import evaluate, stratified_split

logger = logging.getLogger(__name__)


def load_oof(npz_path: str) -> pd.DataFrame:
    z = np.load(npz_path, allow_pickle=True)
    return pd.DataFrame({
        "item_id": z["item_ids"].astype(str),
        "y_true":  z["y_true"].astype(int),
        "prob":    z["oof_prob"].astype(float),
        "group":   z["groups"].astype(str),
    })


def _group_from_path(path: str) -> str:
    base = os.path.basename(path)
    for suffix in ("_features_rec.csv", "_features_ph.csv", "_features.csv"):
        if base.endswith(suffix):
            return base[:-len(suffix)]
    return base.replace(".csv", "")


def load_features(glob_pat: str) -> pd.DataFrame:
    paths = sorted(glob.glob(glob_pat))
    dfs = []
    for p in paths:
        d = pd.read_csv(p)
        d["dataset"] = _group_from_path(p)
        d["item_id"] = d["item_id"].astype(str)
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def build_pooled_frame(roberta_oof: str, step_oof: str, features_glob: str) -> pd.DataFrame:
    """Inner-join everything on (item_id, dataset/group). Returns one row per
    item with text_prob, struct_prob, and all handcrafted+recurrence features.
    The trace_token_count column comes from the feature CSV (= total_tokens)."""
    rob = load_oof(roberta_oof).rename(columns={"prob": "text_prob"})
    stp = load_oof(step_oof).rename(columns={"prob": "step_prob"})
    feat = load_features(features_glob)

    merged = rob[["item_id", "group", "y_true", "text_prob"]].merge(
        stp[["item_id", "group", "step_prob"]],
        on=["item_id", "group"], how="inner",
    )
    feat_to_merge = feat.rename(columns={"dataset": "group"})
    if "is_correct" in feat_to_merge.columns:
        feat_to_merge = feat_to_merge.drop(columns=["is_correct"])
    merged = merged.merge(feat_to_merge, on=["item_id", "group"],
                          how="inner", suffixes=("", "_f"))
    return merged


def make_bins(df: pd.DataFrame, n_bins: int, length_col: str) -> pd.Series:
    edges = np.quantile(df[length_col], np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    bin_ids = np.digitize(df[length_col], edges[1:-1])
    labels = []
    for i in range(n_bins):
        lo, hi = int(edges[i]), int(edges[i + 1])
        labels.append(f"bin{i+1}_[{lo},{hi}]")
    return pd.Series([labels[b] for b in bin_ids], index=df.index, name="length_bin")


def fit_clf(X_tr, y_tr, X_te, clf_name: str, seed: int):
    sc = StandardScaler().fit(X_tr)
    X_tr = sc.transform(X_tr); X_te = sc.transform(X_te)
    if clf_name == "lr":
        m = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                               random_state=seed)
    elif clf_name == "rf":
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                   class_weight="balanced", n_jobs=-1,
                                   random_state=seed)
    else:
        raise ValueError(clf_name)
    m.fit(X_tr, y_tr)
    return m.predict_proba(X_te)[:, 1]


def cv_eval(X: np.ndarray, y: np.ndarray, group: np.ndarray,
            clf_name: str, n_splits: int, seed: int) -> dict:
    if X.shape[1] == 0 or len(np.unique(y)) < 2 or len(y) < n_splits * 4:
        return {"auroc": float("nan"), "n": int(len(y)),
                "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}
    all_y, all_p = [], []
    for tr, te in stratified_split(y, group_id=group if len(set(group)) > 1 else None,
                                   n_splits=n_splits, seed=seed):
        try:
            p = fit_clf(X[tr], y[tr], X[te], clf_name, seed)
            all_y.append(y[te]); all_p.append(p)
        except Exception as e:
            logger.warning(f"  fold failed: {e}")
            return {"auroc": float("nan"), "n": int(len(y)),
                    "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}
    yy = np.concatenate(all_y); pp = np.concatenate(all_p)
    m = evaluate(yy, pp, name="bin_eval")
    return {"auroc": float(m["auroc"]), "n": int(len(y)),
            "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}


HANDCRAFTED_25 = [
    "total_tokens", "total_episodes",
    "prop_forward", "prop_verification", "prop_backtrack",
    "prop_restart", "prop_hesitation", "prop_subgoal", "prop_conclusion",
    "backtrack_count", "verification_count", "restart_count",
    "vf_ratio", "bt_position_mean", "first_conclusion_pos",
    "v_clustering", "max_forward_run", "transition_entropy",
    "cycle_count",
    "wait_ratio", "question_mark_count", "negation_count",
    "repetition_rate_4gram",
]
RECURRENCE = [
    "semantic_recurrence_rate", "max_semantic_cycle_span",
    "progress_repetition", "termination_recycle", "revision_ineffectiveness",
]


def variant_columns(df: pd.DataFrame, variant: str) -> list[str]:
    """Return the column names this variant uses in the pooled DataFrame."""
    cfg = {
        "step_only":         (False, True,  False, False),
        "handcrafted+rec":   (False, False, True,  True),
        "STRUCTURAL_FULL":   (False, True,  True,  True),
        "text_only":         (True,  False, False, False),
        "text+struct":       (True,  True,  False, False),
        "FULL_HYBRID":       (True,  True,  True,  True),
        "length_only":       (False, False, False, False),  # special-case: just total_tokens
    }
    inc_text, inc_step, inc_hc, inc_rec = cfg[variant]
    cols = []
    if variant == "length_only":
        return ["total_tokens"] if "total_tokens" in df.columns else []
    if inc_text and "text_prob" in df.columns: cols.append("text_prob")
    if inc_step and "step_prob" in df.columns: cols.append("step_prob")
    if inc_hc:
        cols += [c for c in HANDCRAFTED_25 if c in df.columns]
    if inc_rec:
        cols += [c for c in RECURRENCE if c in df.columns]
    return cols


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roberta-oof", default="results/month2_v2/roberta_pooled_oof.npz")
    p.add_argument("--step-oof",    default="results/month2_v2/step_transformer_pooled_oof.npz")
    p.add_argument("--features-glob", default="data/features/*_features_rec.csv")
    p.add_argument("--length-col", default="total_tokens",
                   help="Column to bin on (default: total_tokens from feature CSVs)")
    p.add_argument("--n-bins", type=int, default=5)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--clf", default="rf", choices=["lr", "rf"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", default="results/month2_v2/length_stratified_summary.csv")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info("Loading pooled OOF + features ...")
    df = build_pooled_frame(args.roberta_oof, args.step_oof, args.features_glob)
    logger.info(f"  joined: n={len(df)}  unique_(item_id,group)="
                f"{df.groupby(['item_id','group']).ngroups}")

    if args.length_col not in df.columns:
        logger.error(f"length-col '{args.length_col}' missing; available: "
                     f"{[c for c in df.columns if 'token' in c or 'len' in c]}")
        sys.exit(2)

    df["length_bin"] = make_bins(df, args.n_bins, args.length_col)

    # Print bin summary
    logger.info("Bins:")
    for bn, sub in df.groupby("length_bin"):
        logger.info(f"  {bn:<22s}  n={len(sub):4d}  "
                    f"pos_rate={sub['y_true'].mean():.3f}  "
                    f"min_len={int(sub[args.length_col].min())}  "
                    f"max_len={int(sub[args.length_col].max())}")

    VARIANTS = ["length_only", "step_only", "handcrafted+rec", "STRUCTURAL_FULL",
                "text_only", "text+struct", "FULL_HYBRID"]

    results = []
    for bn, sub in df.groupby("length_bin"):
        y = sub["y_true"].to_numpy()
        group = sub["group"].to_numpy()
        for v in VARIANTS:
            cols = variant_columns(sub, v)
            X = sub[cols].to_numpy(dtype=float) if cols else np.zeros((len(sub), 0))
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            res = cv_eval(X, y, group, args.clf, args.n_splits, args.seed)
            results.append({
                "length_bin": bn, "variant": v, "n_features": len(cols),
                "n_items": res["n"], "pos_rate": res["n_pos"] / max(res["n"], 1),
                "auroc": res["auroc"],
            })
            logger.info(f"  bin={bn:<22s} {v:<18s} (n_feat={len(cols):2d})  "
                        f"AUROC={res['auroc']:.4f}")

    out = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv}")

    # Pivot: rows=length_bin, cols=variant, values=AUROC
    print()
    print("=" * 110)
    print(f"AUROC by length bin × variant  (clf={args.clf})")
    print("=" * 110)
    pivot = out.pivot_table(index="length_bin", columns="variant", values="auroc")
    pivot = pivot.reindex(columns=VARIANTS)
    print(pivot.round(4).to_string(na_rep="   .  "))

    # Highlight: where does text+struct or FULL_HYBRID beat text_only by ≥ 0.01?
    print()
    print("Δ(FULL_HYBRID - text_only) per bin — positive means structure adds in that length regime:")
    text_col = pivot["text_only"]
    full_col = pivot["FULL_HYBRID"]
    for bn in pivot.index:
        d = full_col[bn] - text_col[bn]
        marker = "  <-- structure wins" if d > 0.01 else ("  <-- text wins" if d < -0.01 else "")
        print(f"  {bn:<22s}  FULL_HYBRID={full_col[bn]:.4f}  text_only={text_col[bn]:.4f}  "
              f"Δ={d:+.4f}{marker}")


if __name__ == "__main__":
    main()
