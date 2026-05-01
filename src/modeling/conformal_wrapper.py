"""
conformal_wrapper.py - Split-Conformal Prediction wrapper for our UQ methods.

Given any UQ method's continuous output P(correct in [0,1]), we use Split
Conformal Prediction (SCP) to provide a finite-sample coverage guarantee:

    Pr( y_test ∈ prediction_set ) >= 1 - α     for any α ∈ (0,1)

This holds *exchangeably*: assumes calibration and test are drawn from the
same distribution. The interesting empirical question is whether coverage
SURVIVES distribution shift (cross-model, cross-dataset). We answer this by
computing coverage when calibrating on one slice and testing on another.

Algorithm (per α):
    1. Non-conformity score: s(x, y) = 1 - p(y | x)
       (we treat correctness as binary: y ∈ {0, 1})
    2. Compute scores on the calibration set: {s_i}.
    3. Threshold q = ceil((n+1)(1-α))/n empirical quantile of {s_i}.
    4. Prediction set on test: { y' : 1 - p(y' | x_test) <= q }.

Practical reporting:
    - Empirical coverage on test set (should be >= 1 - α).
    - Average prediction set size (smaller = more informative).
    - Singleton fraction (fraction of test items where the set has size 1).
    - Selective accuracy: among singleton predictions, accuracy.

Usage:
    PYTHONPATH=. python src/modeling/conformal_wrapper.py \
        --oof-glob "results/month3/super_hybrid_oof_*.npz"  # any OOF .npz
        --output   results/month3/conformal_coverage.json
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

from src.modeling.cv_utils import save_results

logger = logging.getLogger(__name__)


# =============================================================================
# CORE CONFORMAL FUNCTIONS
# =============================================================================

def _scores(prob: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Non-conformity score: 1 - p(y_i | x_i)."""
    p_y = np.where(y == 1, prob, 1.0 - prob)
    return 1.0 - p_y


def _quantile_threshold(scores: np.ndarray, alpha: float) -> float:
    """Conformal quantile (slightly inflated for finite-sample correction)."""
    n = len(scores)
    q_level = np.minimum(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(scores, q_level))


def _prediction_set(prob: np.ndarray, threshold: float
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Return per-item flags (in_set_0, in_set_1)."""
    in_set_1 = (1.0 - prob) <= threshold
    in_set_0 = prob <= threshold
    return in_set_0, in_set_1


def conformal_metrics(prob_cal: np.ndarray, y_cal: np.ndarray,
                      prob_te: np.ndarray, y_te: np.ndarray,
                      alpha: float) -> dict:
    """Calibrate on (prob_cal, y_cal), evaluate on test set."""
    scores_cal = _scores(prob_cal, y_cal)
    threshold = _quantile_threshold(scores_cal, alpha)

    in_set_0, in_set_1 = _prediction_set(prob_te, threshold)
    set_size = in_set_0.astype(int) + in_set_1.astype(int)

    # 1) Coverage: did we include true label?
    correct = np.where(y_te == 1, in_set_1, in_set_0)
    coverage = float(correct.mean())

    # 2) Average set size
    avg_size = float(set_size.mean())
    singleton_frac = float((set_size == 1).mean())
    empty_frac = float((set_size == 0).mean())     # should be 0

    # 3) Among singleton predictions, accuracy
    singleton_mask = (set_size == 1)
    if singleton_mask.sum() > 0:
        # the predicted singleton is "1" if in_set_1 else "0"
        sing_pred = in_set_1[singleton_mask].astype(int)
        sing_acc = float((sing_pred == y_te[singleton_mask]).mean())
    else:
        sing_acc = float("nan")

    return {
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "empirical_coverage": coverage,
        "threshold": threshold,
        "avg_set_size": avg_size,
        "singleton_fraction": singleton_frac,
        "empty_fraction": empty_frac,
        "singleton_accuracy": sing_acc,
        "n_cal": int(len(y_cal)),
        "n_test": int(len(y_te)),
    }


# =============================================================================
# DATA LOAD HELPERS
# =============================================================================

def load_oof_npz(path: str) -> pd.DataFrame:
    z = np.load(path, allow_pickle=True)
    return pd.DataFrame({
        "item_id": z["item_ids"].astype(str),
        "group":   z["groups"].astype(str),
        "y_true":  z["y_true"].astype(int),
        "prob":    z["oof_prob"].astype(float),
    })


def family_of(group: str) -> str:
    g = group.lower()
    if "llama" in g: return "llama"
    if "qwen" in g:  return "qwen"
    return "other"


# =============================================================================
# EXPERIMENTS
# =============================================================================

def run_one_method(df: pd.DataFrame, method_name: str,
                   alphas=(0.20, 0.10, 0.05),
                   seed: int = 42) -> dict:
    """Run several conformal experiments for one OOF method."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["family"] = df["group"].apply(family_of)
    n = len(df)

    results = {"method": method_name, "n_items": n, "experiments": {}}

    # ---------- 1. IID split (random 50/50) ----------
    for alpha in alphas:
        idx = np.arange(n); rng.shuffle(idx)
        half = n // 2
        cal_idx, te_idx = idx[:half], idx[half:]
        m = conformal_metrics(
            df["prob"].to_numpy()[cal_idx], df["y_true"].to_numpy()[cal_idx],
            df["prob"].to_numpy()[te_idx],  df["y_true"].to_numpy()[te_idx],
            alpha=alpha,
        )
        results["experiments"][f"iid_alpha={alpha}"] = m

    # ---------- 2. Cross-family transfer ----------
    for alpha in alphas:
        for cal_fam, te_fam in [("qwen", "llama"), ("llama", "qwen")]:
            cal_mask = (df["family"] == cal_fam)
            te_mask  = (df["family"] == te_fam)
            if cal_mask.sum() < 50 or te_mask.sum() < 50:
                continue
            m = conformal_metrics(
                df.loc[cal_mask, "prob"].to_numpy(),
                df.loc[cal_mask, "y_true"].to_numpy(),
                df.loc[te_mask,  "prob"].to_numpy(),
                df.loc[te_mask,  "y_true"].to_numpy(),
                alpha=alpha,
            )
            results["experiments"][f"cal={cal_fam}_test={te_fam}_alpha={alpha}"] = m

    # ---------- 3. Leave-one-dataset-out (LODO) ----------
    datasets = sorted(df["group"].unique())
    for ds in datasets:
        cal_mask = (df["group"] != ds)
        te_mask  = (df["group"] == ds)
        if te_mask.sum() < 30:
            continue
        m = conformal_metrics(
            df.loc[cal_mask, "prob"].to_numpy(),
            df.loc[cal_mask, "y_true"].to_numpy(),
            df.loc[te_mask,  "prob"].to_numpy(),
            df.loc[te_mask,  "y_true"].to_numpy(),
            alpha=0.10,   # use 90% target for LODO
        )
        results["experiments"][f"lodo_test={ds}_alpha=0.10"] = m

    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oof", action="append", default=[],
                   help="Path:Name pair, e.g. results/foo.npz:DeBERTa. "
                        "Repeat for multiple methods.")
    p.add_argument("--output", required=True)
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.20, 0.10, 0.05])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if not args.oof:
        p.error("Provide at least one --oof path:name")

    all_results = {"methods": {}}
    for spec in args.oof:
        if ":" in spec:
            path, name = spec.split(":", 1)
        else:
            path = spec
            name = os.path.basename(spec).replace(".npz", "")
        df = load_oof_npz(path)
        logger.info(f"\n=== {name}  ({len(df)} items)  base={df['y_true'].mean():.3f} ===")
        r = run_one_method(df, method_name=name, alphas=tuple(args.alphas),
                           seed=args.seed)
        all_results["methods"][name] = r

    save_results(args.output, all_results)

    # --- Pretty print ---
    print("\n" + "=" * 100)
    print("Conformal coverage summary (target = 1-α)")
    print("=" * 100)
    for mname, mres in all_results["methods"].items():
        print(f"\n--- {mname} ---")
        print(f"{'experiment':50s} {'target':>7s} {'empir':>7s} {'set_sz':>7s} {'sing%':>6s} {'sing_acc':>8s}")
        print("-" * 100)
        for ename, m in mres["experiments"].items():
            print(f"{ename:50s} "
                  f"{m['target_coverage']:>7.2f} "
                  f"{m['empirical_coverage']:>7.3f} "
                  f"{m['avg_set_size']:>7.2f} "
                  f"{m['singleton_fraction']*100:>5.1f}% "
                  f"{m['singleton_accuracy']:>8.3f}")


if __name__ == "__main__":
    main()
