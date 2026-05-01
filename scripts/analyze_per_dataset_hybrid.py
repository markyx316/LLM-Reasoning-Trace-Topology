#!/usr/bin/env python3
"""
analyze_per_dataset_hybrid.py

Build the per-dataset method comparison table from the per-dataset hybrid
JSONs (output of run_per_dataset_hybrid.sh). For each dataset, reports each
variant's AUROC per classifier, plus deltas vs the text-only and structural-
only baselines so we can immediately see which cells favor structure.

Output:
  - prints a master table to stdout
  - writes results/month2_v2/per_dataset_summary.csv

The master table answers the central question for Approach 1:
  "On which dataset/model combos does adding structural features actually
   beat the RoBERTa text-encoder, and by how much?"

Usage:
    PYTHONPATH=. python scripts/analyze_per_dataset_hybrid.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = [
    "math500_qwen7b", "math500_llama8b",
    "gsm8k_qwen7b", "gsm8k_llama8b",
    "gpqa_diamond_qwen7b", "gpqa_diamond_llama8b",
    "arc_challenge_qwen7b", "arc_challenge_llama8b",
]

VARIANTS = [
    "step_only", "handcrafted+rec", "STRUCTURAL_FULL",
    "deberta_only",   # = roberta_only (the arg name is legacy)
    "deberta+step", "deberta+feats", "FULL_HYBRID",
]


def best_clf_auroc(variant_block: dict) -> tuple[str, float]:
    """Return (best_clf, best_auroc_mean) across LR/RF/XGB."""
    best = (None, -1.0)
    for clf in ("lr", "rf", "xgb"):
        if clf not in variant_block.get("clfs", {}):
            continue
        s = variant_block["clfs"][clf].get("summary", {})
        a = s.get("auroc_mean", -1.0)
        if a > best[1]:
            best = (clf, float(a))
    return best


def all_clf_aurocs(variant_block: dict) -> dict[str, float]:
    out = {}
    for clf in ("lr", "rf", "xgb"):
        if clf not in variant_block.get("clfs", {}):
            continue
        s = variant_block["clfs"][clf].get("summary", {})
        out[clf] = float(s.get("auroc_mean", float("nan")))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", default="results/month2_v2/roberta_per_dataset",
                   help="Directory containing hybrid_<ds>.json files")
    p.add_argument("--out-csv", default="results/month2_v2/per_dataset_summary.csv")
    args = p.parse_args()

    rows = []
    missing = []
    for ds in DATASETS:
        path = os.path.join(args.in_dir, f"hybrid_{ds}.json")
        if not os.path.exists(path):
            missing.append(ds)
            continue
        d = json.load(open(path))
        n = d["variants"][next(iter(d["variants"]))].get("n_samples", -1)
        for v in VARIANTS:
            if v not in d["variants"]:
                continue
            vb = d["variants"][v]
            best_clf, best_a = best_clf_auroc(vb)
            allauc = all_clf_aurocs(vb)
            rows.append({
                "dataset": ds,
                "n": n,
                "variant": v,
                "n_features": vb.get("n_features", -1),
                "auroc_lr":  allauc.get("lr", float("nan")),
                "auroc_rf":  allauc.get("rf", float("nan")),
                "auroc_xgb": allauc.get("xgb", float("nan")),
                "best_clf": best_clf,
                "best_auroc": best_a,
            })

    if missing:
        print(f"WARNING: {len(missing)} dataset(s) missing hybrid output:")
        for m in missing:
            print(f"  - {m}  (expected at {args.in_dir}/hybrid_{m}.json)")
        print()

    if not rows:
        print("No hybrid_<ds>.json files found. Run scripts/run_per_dataset_hybrid.sh first.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}  ({len(df)} rows)")
    print()

    # ---- Master table: best AUROC per (dataset, variant) ----
    print("=" * 110)
    print("MASTER TABLE — best AUROC per (dataset, variant), with best classifier in parens")
    print("=" * 110)
    pivot = df.pivot_table(index="dataset", columns="variant",
                           values="best_auroc", aggfunc="first")
    pivot = pivot.reindex(index=DATASETS, columns=VARIANTS)
    print(pivot.round(4).to_string(na_rep="  .   "))

    # ---- Headline: structural lift over text-only ----
    print()
    print("=" * 110)
    print("HEADLINE — Δ AUROC of best structural-augmented variant over RoBERTa-only")
    print("           (positive = structure adds value on top of text encoder)")
    print("=" * 110)

    deltas = []
    for ds in DATASETS:
        sub = df[df["dataset"] == ds]
        if len(sub) == 0:
            continue
        roberta = sub[sub["variant"] == "deberta_only"]["best_auroc"]
        roberta_a = float(roberta.iloc[0]) if len(roberta) else float("nan")
        # Best of the augmented hybrids
        candidates = sub[sub["variant"].isin(["deberta+step", "deberta+feats", "FULL_HYBRID"])]
        best_aug = candidates["best_auroc"].max() if len(candidates) else float("nan")
        # Structural-only ceiling
        struct = sub[sub["variant"] == "STRUCTURAL_FULL"]["best_auroc"]
        struct_a = float(struct.iloc[0]) if len(struct) else float("nan")
        deltas.append({
            "dataset": ds,
            "RoBERTa": roberta_a,
            "STRUCTURAL_FULL": struct_a,
            "best_hybrid": best_aug,
            "Δ(hybrid - roberta)": best_aug - roberta_a,
            "Δ(hybrid - struct)":  best_aug - struct_a,
        })
    dd = pd.DataFrame(deltas)
    print(dd.to_string(index=False, float_format=lambda x: f"{x:+.4f}"
                       if isinstance(x, float) and not np.isnan(x) else "  .   "))

    # ---- Win counts ----
    print()
    n_struct_wins = int((dd["Δ(hybrid - roberta)"] > 0.005).sum())
    n_struct_ties = int(dd["Δ(hybrid - roberta)"].abs().lt(0.005).sum())
    n_text_wins   = int((dd["Δ(hybrid - roberta)"] < -0.005).sum())
    print(f"Structure-augmented hybrid > RoBERTa (Δ > +0.005): {n_struct_wins} / {len(dd)}")
    print(f"Roughly tied (|Δ| ≤ 0.005):                          {n_struct_ties} / {len(dd)}")
    print(f"RoBERTa > structure-augmented (Δ < -0.005):          {n_text_wins} / {len(dd)}")

    # ---- Detailed per-classifier table for further analysis ----
    print()
    print("=" * 110)
    print("DETAIL — AUROC per (dataset, variant, classifier)")
    print("=" * 110)
    detail = df.pivot_table(
        index=["dataset", "variant"],
        values=["auroc_lr", "auroc_rf", "auroc_xgb"]
    )
    print(detail.round(4).to_string(na_rep="  .   "))


if __name__ == "__main__":
    main()
