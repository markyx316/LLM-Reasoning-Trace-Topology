"""
train.py — Train and evaluate correctness-prediction classifiers.

Trains LR, RF, XGBoost on extracted features (20-feature CSV).
Runs 5-fold stratified cross-validation. Reports AUROC, AUPRC, ECE.

Usage:
  python src/train.py --features data/features/math500.csv
  python src/train.py --features data/features/math500.csv \
                      --test data/features/gpqa.csv \
                      --cross-domain
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "g1_token_count", "g1_episode_count",
    "g1_prop_F", "g1_prop_V", "g1_prop_B", "g1_prop_R", "g1_prop_S", "g1_prop_H",
    "g2_backtrack_count", "g2_verification_count", "g2_restart_count",
    "g2_vf_ratio", "g2_backtrack_pos_mean", "g2_first_conclusion_pos",
    "g2_verification_cluster_coeff", "g2_longest_forward_run",
    "g2_transition_entropy", "g2_cycle_count",
    "g3_wait_ratio", "g3_question_mark_count", "g3_negation_count",
    "g3_fourgram_repetition_rate",
]

# Feature groups for ablation
FEAT_GROUPS = {
    "G1_length": [c for c in FEATURE_COLS if c.startswith("g1_")],
    "G2_structural": [c for c in FEATURE_COLS if c.startswith("g2_")],
    "G3_meta": [c for c in FEATURE_COLS if c.startswith("g3_")],
    "G1+G2": [c for c in FEATURE_COLS if c.startswith("g1_") or c.startswith("g2_")],
    "All": FEATURE_COLS,
}


def make_classifiers() -> dict:
    return {
        "LR": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ]),
        "RF": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGB": xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, eval_metric="logloss", verbosity=0,
        ),
    }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute ECE (lower is better)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def cv_evaluate(X: np.ndarray, y: np.ndarray, clf_name: str, clf, n_splits: int = 5) -> dict:
    """Run stratified k-fold CV. Return mean ± std metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aurocs, auprcs, eces = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        clf_clone = make_classifiers()[clf_name]
        clf_clone.fit(X_tr, y_tr)
        probs = clf_clone.predict_proba(X_val)[:, 1]

        aurocs.append(roc_auc_score(y_val, probs))
        auprcs.append(average_precision_score(y_val, probs))
        eces.append(expected_calibration_error(y_val, probs))

    return {
        "AUROC_mean": float(np.mean(aurocs)),
        "AUROC_std": float(np.std(aurocs)),
        "AUPRC_mean": float(np.mean(auprcs)),
        "AUPRC_std": float(np.std(auprcs)),
        "ECE_mean": float(np.mean(eces)),
        "ECE_std": float(np.std(eces)),
    }


def cross_domain_evaluate(
    train_df: pd.DataFrame, test_df: pd.DataFrame,
    feat_cols: list[str], clf_name: str
) -> dict:
    """Train on train_df, evaluate on test_df."""
    X_tr = train_df[feat_cols].values
    y_tr = train_df["correct"].values
    X_te = test_df[feat_cols].values
    y_te = test_df["correct"].values

    clf = make_classifiers()[clf_name]
    clf.fit(X_tr, y_tr)
    probs = clf.predict_proba(X_te)[:, 1]

    return {
        "AUROC": float(roc_auc_score(y_te, probs)),
        "AUPRC": float(average_precision_score(y_te, probs)),
        "ECE": float(expected_calibration_error(y_te, probs)),
        "n_train": len(train_df),
        "n_test": len(test_df),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, help="Feature CSV (training data)")
    parser.add_argument("--test", default=None, help="Feature CSV for cross-domain test")
    parser.add_argument("--cross-domain", action="store_true")
    parser.add_argument("--ablation", action="store_true", help="Run feature group ablation")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    train_df = pd.read_csv(args.features).dropna()
    feat_name = Path(args.features).stem

    results = {}

    if args.cross_domain and args.test:
        # Cross-domain: train on train_df, test on test_df
        test_df = pd.read_csv(args.test).dropna()
        test_name = Path(args.test).stem
        print(f"\n--- Cross-domain: train={feat_name}, test={test_name} ---")
        for clf_name in ["LR", "RF", "XGB"]:
            res = cross_domain_evaluate(train_df, test_df, FEATURE_COLS, clf_name)
            key = f"{feat_name}→{test_name}/{clf_name}"
            results[key] = res
            print(f"  {clf_name}: AUROC={res['AUROC']:.4f}, AUPRC={res['AUPRC']:.4f}, ECE={res['ECE']:.4f}")
    else:
        # Within-dataset CV
        X = train_df[FEATURE_COLS].values
        y = train_df["correct"].values
        print(f"\n--- Within-dataset CV: {feat_name} (n={len(train_df)}, pos={y.mean():.1%}) ---")

        feat_sets = FEAT_GROUPS if args.ablation else {"All": FEATURE_COLS}
        for feat_group, feat_cols in feat_sets.items():
            X_g = train_df[feat_cols].values
            for clf_name in ["LR", "RF", "XGB"]:
                res = cv_evaluate(X_g, y, clf_name)
                key = f"{feat_name}/{feat_group}/{clf_name}"
                results[key] = res
                print(
                    f"  [{feat_group}] {clf_name}: "
                    f"AUROC={res['AUROC_mean']:.4f}±{res['AUROC_std']:.4f}, "
                    f"AUPRC={res['AUPRC_mean']:.4f}±{res['AUPRC_std']:.4f}, "
                    f"ECE={res['ECE_mean']:.4f}±{res['ECE_std']:.4f}"
                )

    out_path = args.output or (RESULTS_DIR / f"{feat_name}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
