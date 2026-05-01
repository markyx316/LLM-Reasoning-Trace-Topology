"""Phase 3 (T7) — Selective-prediction evaluation.

For each OOF model of interest, compute accuracy as a function of
coverage (what % of items are retained after abstaining on the least
confident). Compare to (a) Phase-2 ULTRA_TEXT_ONLY baseline and
(b) self-consistency N=8 literature numbers, framed per-group.

Deliverables
------------
    reports/month3/phase3_selective.csv
        rows: (variant, group, coverage, accuracy, n_retained)

    reports/month3/phase3_selective_summary.csv
        pivot table: accuracy @ coverage∈{50,70,85,90,95} for each
        (variant, group).

Confidence score used for selection: for binary models,
    conf = max(p, 1 - p)
i.e. distance from 0.5. For calibration-free UQ this is equivalent to
"trust the high-|p - 0.5| items first."

Usage
-----
    PYTHONPATH=. python scripts/phase3/selective_prediction_curves.py \
        --oofs \
          SH_LR=results/month3/superhybrid_SuperHybrid_LR_oof.npz \
          SH_RF=results/month3/superhybrid_SuperHybrid_RF_oof.npz \
          ULTRA_TEXT_ONLY_LR=results/month3/ultra_hybrid/ultrahybrid_ULTRA_TEXT_ONLY__lr_oof.npz \
        --out-dir reports/month3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def selective_curve(
    p: np.ndarray, y: np.ndarray, thresholds: np.ndarray
) -> pd.DataFrame:
    """Compute accuracy@coverage for each threshold in [0..1].

    Coverage := fraction of items with conf >= threshold, where
    conf = max(p, 1 - p). For each coverage, we also report the
    accuracy of a majority-vote classifier (threshold=0.5 on p)
    over the retained items.
    """
    assert p.shape == y.shape
    conf = np.where(p >= 0.5, p, 1.0 - p)  # distance from 0.5 in [0.5, 1]
    order = np.argsort(-conf)  # most confident first
    sorted_p = p[order]
    sorted_y = y[order]
    pred = (sorted_p >= 0.5).astype(int)
    correct = (pred == sorted_y).astype(int)
    # Cumulative accuracy vs coverage
    cum_correct = np.cumsum(correct)
    N = len(p)
    coverage_frac = np.arange(1, N + 1, dtype=float) / N
    cum_acc = cum_correct / np.arange(1, N + 1)
    # Interpolate at requested thresholds
    rows = []
    for t in thresholds:
        k = max(1, int(np.ceil(t * N)))
        rows.append({
            "coverage_req":  float(t),
            "n_retained":    int(k),
            "coverage_act":  float(k / N),
            "accuracy":      float(cum_acc[k - 1]),
            "base_rate":     float(y.mean()),
        })
    return pd.DataFrame(rows)


def _load_oof(path: str):
    z = np.load(path, allow_pickle=True)
    return (
        np.asarray(z["item_ids"]).astype(str),
        np.asarray(z["groups"]).astype(str),
        np.asarray(z["y_true"]).astype(int),
        np.asarray(z["oof_prob"]).astype(float),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--oofs", nargs="+", required=True,
        help="Key=path entries, e.g. SH_LR=results/.../x.npz",
    )
    ap.add_argument(
        "--coverages", nargs="+", type=float,
        default=[1.00, 0.95, 0.90, 0.85, 0.70, 0.50, 0.30],
    )
    ap.add_argument("--out-dir", default="reports/month3")
    ap.add_argument(
        "--per-group", action="store_true", default=True,
        help="Emit curves broken out by group as well as pooled.",
    )
    args = ap.parse_args()

    thresholds = np.array(sorted(args.coverages, reverse=True), dtype=float)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summary_rows = []
    for entry in args.oofs:
        if "=" not in entry:
            print(f"[skip] malformed OOF entry {entry!r}; expected KEY=PATH")
            continue
        name, path = entry.split("=", 1)
        if not os.path.exists(path):
            print(f"[skip] missing OOF: {path}")
            continue
        ids, groups, y, p = _load_oof(path)
        # Pooled
        pooled = selective_curve(p, y, thresholds)
        pooled.insert(0, "variant", name)
        pooled.insert(1, "group", "__pooled__")
        all_rows.append(pooled)

        # Per-group
        if args.per_group:
            for g in sorted(set(groups.tolist())):
                m = groups == g
                if m.sum() < 30:
                    continue
                sub = selective_curve(p[m], y[m], thresholds)
                sub.insert(0, "variant", name)
                sub.insert(1, "group", g)
                all_rows.append(sub)

        # Summary
        for t in thresholds:
            k = max(1, int(np.ceil(t * len(p))))
            conf = np.where(p >= 0.5, p, 1.0 - p)
            order = np.argsort(-conf)
            kept = order[:k]
            acc = float(((p[kept] >= 0.5).astype(int) == y[kept]).mean())
            summary_rows.append({
                "variant": name, "group": "__pooled__",
                "coverage": float(k / len(p)),
                "accuracy": acc,
                "n_retained": k, "n_total": len(p),
                "base_rate": float(y.mean()),
            })
            if args.per_group:
                for g in sorted(set(groups.tolist())):
                    m = groups == g
                    if m.sum() < 30:
                        continue
                    pg = p[m]
                    yg = y[m]
                    kg = max(1, int(np.ceil(t * len(pg))))
                    confg = np.where(pg >= 0.5, pg, 1.0 - pg)
                    og = np.argsort(-confg)
                    kept = og[:kg]
                    summary_rows.append({
                        "variant": name, "group": g,
                        "coverage": float(kg / len(pg)),
                        "accuracy": float(((pg[kept] >= 0.5).astype(int) == yg[kept]).mean()),
                        "n_retained": kg, "n_total": int(m.sum()),
                        "base_rate": float(yg.mean()),
                    })

    full = pd.concat(all_rows, ignore_index=True)
    full_path = out_dir / "phase3_selective.csv"
    full.to_csv(full_path, index=False)
    print(f"wrote {full_path}  ({len(full)} rows)")

    summ = pd.DataFrame(summary_rows)
    summ_path = out_dir / "phase3_selective_summary.csv"
    summ.to_csv(summ_path, index=False)
    print(f"wrote {summ_path}  ({len(summ)} rows)")

    # Pretty pooled summary
    pretty = (
        summ[summ["group"] == "__pooled__"]
        .pivot_table(index=["variant"], columns="coverage", values="accuracy")
        .round(4)
    )
    print("\n=== Accuracy @ coverage (pooled) ===")
    print(pretty.to_string())


if __name__ == "__main__":
    main()
