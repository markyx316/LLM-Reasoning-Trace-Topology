#!/usr/bin/env python3
"""
compare_transfer_matrices.py

Loads the two cross-dataset transfer summary CSVs (RoBERTa text-encoder and
StepTF structural) and produces side-by-side comparisons for the paper's
"structure transfers, text doesn't" thesis.

Inputs:
  - results/month2_v2/roberta_transfer/roberta_transfer_summary.csv  (Approach 4)
  - results/month2_v2/steptf_transfer/steptf_transfer_summary.csv   (already exists)

Output:
  - prints comparison tables to stdout
  - writes results/month2_v2/transfer_comparison.csv

Usage:
    PYTHONPATH=. python scripts/compare_transfer_matrices.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

DATASETS = [
    "math500_qwen7b", "math500_llama8b",
    "gsm8k_qwen7b", "gsm8k_llama8b",
    "gpqa_diamond_qwen7b", "gpqa_diamond_llama8b",
    "arc_challenge_qwen7b", "arc_challenge_llama8b",
]


def load_matrix(csv_path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f"Missing: {csv_path}  (skip {label})")
        return None
    df = pd.read_csv(csv_path)
    print(f"Loaded {label}: n={len(df)}  unique srcs={df['source'].nunique()}")
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--roberta-csv", default="results/month2_v2/roberta_transfer/roberta_transfer_summary.csv")
    p.add_argument("--steptf-csv",  default="results/month2_v2/steptf_transfer/steptf_transfer_summary.csv")
    p.add_argument("--out-csv",     default="results/month2_v2/transfer_comparison.csv")
    args = p.parse_args()

    rob = load_matrix(args.roberta_csv, "RoBERTa")
    stp = load_matrix(args.steptf_csv,  "StepTF")
    if rob is None or stp is None:
        print("\nMissing one or both transfer summaries — run their generators first.")
        sys.exit(1)

    # Pivots
    rob_pivot = rob.pivot(index="source", columns="target", values="auroc").reindex(
        index=DATASETS, columns=DATASETS)
    stp_pivot = stp.pivot(index="source", columns="target", values="auroc").reindex(
        index=DATASETS, columns=DATASETS)

    print()
    print("=" * 110)
    print("RoBERTa transfer (rows=src, cols=tgt; in-distribution diagonal in bold conceptually)")
    print("=" * 110)
    print(rob_pivot.round(3).to_string(na_rep="  .  "))

    print()
    print("=" * 110)
    print("StepTF transfer (same convention)")
    print("=" * 110)
    print(stp_pivot.round(3).to_string(na_rep="  .  "))

    print()
    print("=" * 110)
    print("DELTA matrix (RoBERTa - StepTF)  — positive = text does better; negative = structure does better")
    print("=" * 110)
    delta = rob_pivot - stp_pivot
    print(delta.round(3).to_string(na_rep="  .  "))

    # Off-diagonal aggregates (the headline number)
    rob_off = rob[rob["source"] != rob["target"]]
    stp_off = stp[stp["source"] != stp["target"]]

    print()
    print("=" * 110)
    print("OFF-DIAGONAL AGGREGATES — the central comparison for 'who transfers?'")
    print("=" * 110)
    print(f"{'metric':<22s}  {'RoBERTa':>10s}  {'StepTF':>10s}  {'Δ (Rob-Stp)':>12s}")
    for stat_name, fn in [("mean",   np.mean),
                           ("median", np.median),
                           ("min",    np.min),
                           ("max",    np.max),
                           ("std",    np.std)]:
        ra = fn(rob_off["auroc"])
        sa = fn(stp_off["auroc"])
        print(f"  off-diag {stat_name:<11s}  {ra:>10.4f}  {sa:>10.4f}  {ra - sa:>+12.4f}")

    print()
    print(f"Items where RoBERTa transfers BETTER (Δ > +0.02): "
          f"{int((delta.values > 0.02).sum())} / {delta.size}")
    print(f"Items where StepTF transfers BETTER  (Δ < -0.02): "
          f"{int((delta.values < -0.02).sum())} / {delta.size}")
    print(f"Roughly tied (|Δ| ≤ 0.02):                        "
          f"{int((np.abs(delta.values) <= 0.02).sum())} / {delta.size}")

    # Per-source breakdown: how much does each text source GENERALIZE vs each struct source?
    print()
    print("=" * 110)
    print("PER-SOURCE GENERALIZATION (mean off-diagonal AUROC for that source)")
    print("How well does each source's trained model transfer to other datasets?")
    print("=" * 110)
    rob_src_mean = rob_off.groupby("source")["auroc"].agg(["mean", "std", "count"])
    stp_src_mean = stp_off.groupby("source")["auroc"].agg(["mean", "std", "count"])
    out = pd.DataFrame({
        "RoBERTa_mean": rob_src_mean["mean"], "RoBERTa_std": rob_src_mean["std"],
        "StepTF_mean":  stp_src_mean["mean"], "StepTF_std":  stp_src_mean["std"],
    })
    out["Δ (Rob-Stp)"] = out["RoBERTa_mean"] - out["StepTF_mean"]
    out = out.reindex(DATASETS)
    print(out.round(3).to_string(na_rep="  .  "))

    # Save merged per-pair comparison
    rob_renamed = rob[["source", "target", "auroc", "diagonal"]].rename(
        columns={"auroc": "auroc_roberta"})
    stp_renamed = stp[["source", "target", "auroc"]].rename(
        columns={"auroc": "auroc_steptf"})
    merged = rob_renamed.merge(stp_renamed, on=["source", "target"], how="outer")
    merged["delta_roberta_minus_steptf"] = merged["auroc_roberta"] - merged["auroc_steptf"]
    merged.to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
