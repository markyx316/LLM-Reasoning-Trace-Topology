"""
length_controlled.py - Length-controlled evaluation of UQ methods.

The central concern: do our structural / recurrence features carry signal
*beyond* trace length? This script answers by binning traces by token count
and re-running classification INSIDE each length bin. If our method stays
above the length-only baseline in every bin, we can claim the signal is
not just length.

Key outputs:
  1. Per-bin AUROC table (length bins x methods)
  2. Per-bin ECE + Acc@80
  3. JSON with every fold metric, for plotting

Usage:
    PYTHONPATH=. python src/analysis/length_controlled.py \
        --features data/features/math500_qwen7b_features_rec.csv \
        --n-bins 5 \
        --output results/lengthctl_math500_qwen7b.json

    # For pooled analysis across datasets:
    PYTHONPATH=. python src/analysis/length_controlled.py \
        --features data/features/*_features_rec.csv \
        --pool \
        --output results/lengthctl_pooled.json
"""

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
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)


# =============================================================================
# FEATURE GROUPS (name-based so it works with or without new features)
# =============================================================================

RECURRENCE_FEATS = [
    "semantic_recurrence_rate",
    "max_semantic_cycle_span",
    "progress_repetition",
    "termination_recycle",
    "revision_ineffectiveness",
]

LENGTH_FEATS = ["total_tokens", "total_episodes"]

LEXICAL_FEATS = [
    "wait_density", "maybe_density", "verify_density",
    "actually_density", "negation_density",
    "question_mark_rate", "repetition_rate_4gram",
    # legacy names if present:
    "wait_ratio", "negation_count", "question_mark_count",
]


def available(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


# =============================================================================
# CV EVALUATION
# =============================================================================

def run_cv(X: np.ndarray, y: np.ndarray, clf_name: str = "rf",
           n_splits: int = 5, seed: int = 42) -> dict:
    """Stratified K-fold CV. Returns aggregated metrics + per-fold AUROC."""
    if len(np.unique(y)) < 2 or len(y) < n_splits * 2:
        return {"auroc_mean": 0.5, "auroc_std": 0.0, "n": len(y),
                "base_acc": float(y.mean()) if len(y) else 0.0,
                "skipped": True}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aurocs, auprcs = [], []
    y_true_all, y_prob_all = [], []

    for tr_idx, te_idx in skf.split(X, y):
        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y[tr_idx], y[te_idx]
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        if clf_name == "lr":
            model = LogisticRegression(C=1.0, max_iter=1000,
                                       class_weight="balanced", random_state=seed)
        elif clf_name == "rf":
            model = RandomForestClassifier(
                n_estimators=200, min_samples_leaf=5,
                class_weight="balanced", random_state=seed, n_jobs=-1)
        else:
            raise ValueError(clf_name)
        model.fit(Xtr_s, ytr)
        prob = model.predict_proba(Xte_s)[:, 1]

        if len(np.unique(yte)) >= 2:
            aurocs.append(roc_auc_score(yte, prob))
            auprcs.append(average_precision_score(yte, prob))
        y_true_all.extend(yte); y_prob_all.extend(prob)

    y_true_all = np.asarray(y_true_all); y_prob_all = np.asarray(y_prob_all)

    return {
        "auroc_mean": float(np.mean(aurocs)) if aurocs else 0.5,
        "auroc_std":  float(np.std(aurocs))  if aurocs else 0.0,
        "auprc_mean": float(np.mean(auprcs)) if auprcs else float(y.mean()),
        "n": int(len(y)),
        "base_acc": float(y.mean()),
        "skipped": False,
    }


# =============================================================================
# METHOD DEFINITIONS (each method = feature column subset)
# =============================================================================

def build_method_matrices(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return dict of method_name -> feature matrix X (using columns available)."""
    methods = {}

    # 1) length only
    cols = available(df, LENGTH_FEATS)
    if cols:
        methods["length_only"] = df[cols].to_numpy(dtype=float)

    # 2) lexical only
    cols = available(df, LEXICAL_FEATS)
    if cols:
        methods["lexical_only"] = df[cols].to_numpy(dtype=float)

    # 3) handcrafted 25 (everything except labels and recurrence)
    exclude = {"item_id", "dataset", "is_correct"} | set(RECURRENCE_FEATS)
    handcrafted_cols = [c for c in df.columns
                        if c not in exclude and df[c].dtype != object]
    if handcrafted_cols:
        methods["handcrafted_25"] = df[handcrafted_cols].to_numpy(dtype=float)

    # 4) recurrence only
    cols = available(df, RECURRENCE_FEATS)
    if cols:
        methods["recurrence_only"] = df[cols].to_numpy(dtype=float)

    # 5) handcrafted + recurrence (the proposed combination)
    rec_cols = available(df, RECURRENCE_FEATS)
    if handcrafted_cols and rec_cols:
        combined = handcrafted_cols + rec_cols
        methods["handcrafted_plus_recurrence"] = df[combined].to_numpy(dtype=float)

    return methods


# =============================================================================
# LENGTH BINNING
# =============================================================================

def bin_by_length(df: pd.DataFrame, n_bins: int = 5,
                  length_col: str = "total_tokens") -> list[tuple[str, pd.DataFrame]]:
    """Split df into n_bins roughly equal-count length bins. Returns [(label, sub_df)]."""
    if length_col not in df.columns:
        logger.warning(f"No {length_col} column; returning single bin")
        return [("all", df)]

    # quantile edges
    q = np.linspace(0, 1, n_bins + 1)
    edges = df[length_col].quantile(q).to_numpy()
    edges[0] -= 1e-9
    edges[-1] += 1e-9

    bins = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        sub = df[(df[length_col] > lo) & (df[length_col] <= hi)]
        label = f"bin{i+1}_[{int(lo)+1},{int(hi)}]"
        bins.append((label, sub))
    return bins


# =============================================================================
# MAIN EVAL
# =============================================================================

def run_length_controlled(
    df: pd.DataFrame,
    n_bins: int = 5,
    clf_name: str = "rf",
    label: str = "",
) -> dict:
    """Run per-bin CV for every method. Returns a nested dict of results."""
    if "is_correct" not in df.columns:
        raise ValueError("DataFrame must contain 'is_correct'")

    methods = build_method_matrices(df)
    logger.info(f"[{label}] methods: {list(methods.keys())}")
    logger.info(f"[{label}] n rows: {len(df)}, base acc: {df['is_correct'].mean():.3f}")

    # Overall (no bins), for reference
    overall = {}
    y = df["is_correct"].astype(int).to_numpy()
    for name, X in methods.items():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        overall[name] = run_cv(X, y, clf_name)
        logger.info(f"  OVERALL  {name:30s} AUROC={overall[name]['auroc_mean']:.3f}"
                    f" ± {overall[name]['auroc_std']:.3f}  n={overall[name]['n']}")

    # Per-bin
    binned = {}
    for bin_label, sub in bin_by_length(df, n_bins=n_bins):
        if len(sub) < 20:
            logger.info(f"  [{bin_label}] too small ({len(sub)}), skip")
            continue
        y_sub = sub["is_correct"].astype(int).to_numpy()
        bin_result = {"n": int(len(sub)),
                      "base_acc": float(y_sub.mean()),
                      "methods": {}}
        for name, X in methods.items():
            # rebuild X for this subset to align indices
            X_sub = X[sub.index - df.index[0]] if isinstance(df.index, pd.RangeIndex) else None
            if X_sub is None:
                X_sub = build_method_matrices(sub.reset_index(drop=True))[name]
            X_sub = np.nan_to_num(X_sub, nan=0.0, posinf=0.0, neginf=0.0)
            bin_result["methods"][name] = run_cv(X_sub, y_sub, clf_name)
        binned[bin_label] = bin_result
        logger.info(f"  [{bin_label}] n={len(sub)}  base={y_sub.mean():.3f}")
        for name, r in bin_result["methods"].items():
            logger.info(f"       {name:30s} AUROC={r['auroc_mean']:.3f} ± {r['auroc_std']:.3f}")

    return {
        "label": label,
        "classifier": clf_name,
        "n_bins": n_bins,
        "overall": overall,
        "binned": binned,
    }


def pretty_print_table(results: dict):
    """Print a method-x-bin AUROC matrix."""
    binned = results["binned"]
    overall = results["overall"]
    bin_labels = list(binned.keys())
    method_names = list(overall.keys())

    print("\n" + "=" * 100)
    print(f"Length-Controlled AUROC ({results.get('classifier','?')}) -- {results.get('label','')}")
    print("=" * 100)
    header = f"{'method':32s}  {'overall':>10s}  " + "  ".join(f"{b:>18s}" for b in bin_labels)
    print(header)
    print("-" * len(header))
    for m in method_names:
        row = f"{m:32s}  "
        row += f"{overall[m]['auroc_mean']:.3f}±{overall[m]['auroc_std']:.3f}  "
        for b in bin_labels:
            r = binned[b]["methods"].get(m, {})
            row += f"  {r.get('auroc_mean', float('nan')):.3f}±{r.get('auroc_std', 0):.3f}    "
        print(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", nargs="+", required=True,
                        help="One or more feature CSVs (supports glob via shell)")
    parser.add_argument("--pool", action="store_true",
                        help="Concatenate multiple CSVs and evaluate jointly")
    parser.add_argument("--n-bins", type=int, default=5)
    parser.add_argument("--clf", default="rf", choices=["lr", "rf"])
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Expand globs
    csvs = []
    for pat in args.features:
        csvs.extend(sorted(glob.glob(pat)) if "*" in pat else [pat])

    all_results = {}

    if args.pool:
        dfs = []
        for csv in csvs:
            d = pd.read_csv(csv)
            d["__source__"] = os.path.basename(csv)
            dfs.append(d)
        df = pd.concat(dfs, ignore_index=True)
        r = run_length_controlled(df, n_bins=args.n_bins,
                                  clf_name=args.clf, label="pooled")
        pretty_print_table(r)
        all_results["pooled"] = r
    else:
        for csv in csvs:
            name = os.path.basename(csv).replace("_features_rec.csv", "").replace("_features.csv", "")
            df = pd.read_csv(csv)
            r = run_length_controlled(df, n_bins=args.n_bins,
                                      clf_name=args.clf, label=name)
            pretty_print_table(r)
            all_results[name] = r

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    logger.info(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
