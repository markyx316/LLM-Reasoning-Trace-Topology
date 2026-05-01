"""RoBERTa-only stacker ablation with DeLong 95% CIs + paired tests.

The full hybrid stacker achieves pooled AUROC = 0.8054 using the config
(L1-LR, C=0.0455, robust scaler, group dummies) on the 'oof+hand'
subset (6 neural OOFs + 100 handcrafted/graph/etc. features).  The
paired DeLong test (scripts/compute_per_group_ci.py) revealed that
RoBERTa alone gives AUROC = 0.7969 on the same items and the
stacker-vs-RoBERTa gap is only +0.0085 (p=5.8e-5).

This ablation makes the "how much do we really need the extras?"
question explicit by re-fitting the SAME L1-LR stacker on restricted
feature subsets and measuring the pooled AUROC, its DeLong CI, and
the paired difference vs. the full stacker.  We cover:

    0. full                 : oof+hand (reproducing 0.8054 as a sanity check)
    1. roberta_only         : just oof_roberta  (+ group dummies)
    2. roberta+hand         : oof_roberta + hand_* columns
    3. hand_only            : hand_* columns (structural features alone,
                              no neural OOFs)
    4. oof_minus_roberta    : 5 other OOFs  (can we recover the result
                              without RoBERTa content signal?)
    5. all_oofs             : all 6 OOFs, no hand (upper bound with
                              content-heavy inputs only)

The stacker uses the table's own `fold` column so the AUROCs are
directly comparable across subsets.  Outputs:

    reports/route_ab/roberta_only_ablation.csv
    reports/route_ab/roberta_only_ablation.json

Usage:
    PYTHONPATH=. python scripts/roberta_only_ablation.py
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.analysis.delong_ci import delong_auroc_ci, delong_paired_test  # noqa: E402

# Reuse the tuner's exact CV logic so AUROCs are directly comparable to
# the published 0.8054 number.
from scripts.tune_hybrid import (  # noqa: E402
    cv_train_predict,
    load_hybrid_table,
    column_groups,
    pooled_auroc,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# The winning config from the 2000-trial Optuna study
# (results/route_ab/hybrid_tuned_best_hybrid_v1_clean.json).
BEST_PARAMS = {
    "family": "lr",
    "scaler_kind": "robust",
    "add_group_dummies": True,
    "isotonic": False,
    "model_params": {
        "C": 0.0454710975896273,
        "penalty": "l1",
        "class_weight_balanced": False,
    },
}


def _feature_subsets(df: pd.DataFrame) -> dict[str, list[str]]:
    """Build the ablation feature subsets from the table columns."""
    oof_cols = [c for c in df.columns if c.startswith("oof_")]
    hand_cols = [c for c in df.columns if c.startswith("hand_")]
    if "oof_roberta" not in oof_cols:
        raise RuntimeError(
            f"hybrid_table.parquet is missing oof_roberta; have {oof_cols}"
        )

    return {
        "full": oof_cols + hand_cols,
        "roberta_only": ["oof_roberta"],
        "roberta+hand": ["oof_roberta"] + hand_cols,
        "hand_only": hand_cols,
        "oof_minus_roberta": [c for c in oof_cols if c != "oof_roberta"],
        "all_oofs": oof_cols,
    }


def _fit_subset(df: pd.DataFrame, feat_cols: list[str], seed: int = 42) -> np.ndarray:
    """Run the full CV refit for one subset, return OOF probabilities."""
    return cv_train_predict(
        df=df,
        feat_cols=feat_cols,
        family=BEST_PARAMS["family"],
        model_params=BEST_PARAMS["model_params"],
        scaler_kind=BEST_PARAMS["scaler_kind"],
        add_group_dummies=BEST_PARAMS["add_group_dummies"],
        isotonic=BEST_PARAMS["isotonic"],
        seed=seed,
        n_jobs=1,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--parquet", default="data/hybrid_table.parquet",
        help="Path to hybrid_table.parquet.",
    )
    ap.add_argument(
        "--meta", default="data/hybrid_table.META.json",
        help="Path to hybrid_table.META.json.",
    )
    ap.add_argument(
        "--out-dir", default="reports/route_ab",
        help="Output directory.",
    )
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    parquet_path = (_REPO_ROOT / args.parquet).resolve()
    meta_path = (_REPO_ROOT / args.meta).resolve()
    out_dir = (_REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df, meta = load_hybrid_table(parquet_path, meta_path, leaky_policy="warn")
    print(
        f"loaded {parquet_path.name}  shape={df.shape}  "
        f"any_leaky={meta.get('any_leaky', 'unknown')}"
    )

    subsets = _feature_subsets(df)
    print(
        "feature subsets to ablate (all use "
        f"family=lr, penalty=l1, C={BEST_PARAMS['model_params']['C']:.4f}, "
        f"scaler=robust, group_dummies=True):"
    )
    for name, cols in subsets.items():
        n_oof = sum(1 for c in cols if c.startswith("oof_"))
        n_hand = sum(1 for c in cols if c.startswith("hand_"))
        print(
            f"  {name:20s} n_features={len(cols):3d}  "
            f"({n_oof} OOFs + {n_hand} hand)"
        )

    y = df["label"].values.astype(int)
    groups = df["group"].values.astype(str)

    # Fit each subset, stash OOF preds.
    oof_by_subset = {}
    print("\nfitting each subset on the table's existing fold assignments...")
    for name, cols in subsets.items():
        print(f"  fitting {name} ...", flush=True, end=" ")
        oof = _fit_subset(df, cols, seed=args.seed)
        oof_by_subset[name] = oof
        pooled = pooled_auroc(df, oof)
        print(f"pooled AUROC = {pooled:.4f}")

    # Per-subset AUROC + DeLong CI + paired test vs `full`.
    full_oof = oof_by_subset["full"]
    rows = []
    for name, oof in oof_by_subset.items():
        ci = delong_auroc_ci(y, oof, alpha=args.alpha, method="logit")
        row = {
            "subset": name,
            "n_features": len(subsets[name]),
            "pooled_auroc": ci.auroc,
            "ci_low": ci.ci_low,
            "ci_high": ci.ci_high,
            "ci_width": ci.ci_high - ci.ci_low,
            "var_auroc": ci.var_auroc,
        }
        if name == "full":
            row.update(
                {
                    "diff_vs_full": 0.0,
                    "diff_ci_low": 0.0,
                    "diff_ci_high": 0.0,
                    "z_vs_full": 0.0,
                    "p_vs_full": 1.0,
                }
            )
        else:
            pr = delong_paired_test(y, oof, full_oof, alpha=args.alpha)
            row.update(
                {
                    "diff_vs_full": pr.diff,  # subset - full
                    "diff_ci_low": pr.ci_low,
                    "diff_ci_high": pr.ci_high,
                    "z_vs_full": pr.z,
                    "p_vs_full": pr.p_two_sided,
                }
            )
        rows.append(row)

    df_rows = pd.DataFrame(rows)
    df_rows = df_rows.sort_values("pooled_auroc", ascending=False).reset_index(
        drop=True
    )
    csv_path = out_dir / "roberta_only_ablation.csv"
    df_rows.to_csv(csv_path, index=False)

    json_path = out_dir / "roberta_only_ablation.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "parquet": str(parquet_path),
                "any_leaky_in_table": meta.get("any_leaky"),
                "best_params": BEST_PARAMS,
                "alpha": args.alpha,
                "seed": args.seed,
                "rows": df_rows.to_dict(orient="records"),
            },
            f,
            indent=2,
            default=float,
        )

    # Pretty summary.
    print("\n" + "=" * 88)
    print("RoBERTa-only ablation (all using best hybrid_v1_clean LR config)")
    print("=" * 88)
    print(
        f"{'subset':22s} {'n_feat':>6s}  {'AUROC':>7s}  "
        f"{'95% CI':17s}  {'Δ vs full':>9s}  {'p_paired':>10s}"
    )
    print("-" * 88)
    # Force `full` row first, then the rest sorted by AUROC descending.
    ordered = [r for r in rows if r["subset"] == "full"] + sorted(
        [r for r in rows if r["subset"] != "full"],
        key=lambda r: -r["pooled_auroc"],
    )
    for r in ordered:
        print(
            f"{r['subset']:22s} {r['n_features']:>6d}  "
            f"{r['pooled_auroc']:>7.4f}  "
            f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]  "
            f"{r['diff_vs_full']:>+9.4f}  "
            f"{r['p_vs_full']:>10.2e}"
        )
    print("\nwrote", csv_path)
    print("wrote", json_path)


if __name__ == "__main__":
    main()
