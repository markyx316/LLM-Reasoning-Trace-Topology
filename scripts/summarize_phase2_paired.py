"""Phase 2: collapse every paired_<tag>_by_group.csv in reports/month3/paired/
into a single compact summary table so the paper has a one-shot view of
"who beats whom, where, and how significantly."

The driver script run_phase2_paired_delong.sh emits filenames shaped like:
    paired_p2_<short_variant>_<clf>__vs__<baseline>_by_group.csv
This script unpacks (variant, clf, baseline) from the filename and extracts
the OVERALL slice from each file.

Outputs:
    reports/month3/phase2_paired_matrix.csv
        one row per (variant, clf, baseline) with pooled AUROC_a,
        AUROC_b, diff, CI, p, sig and n_common.

Usage:
    PYTHONPATH=. python scripts/summarize_phase2_paired.py
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import pandas as pd


FILENAME_RE = re.compile(
    r"^paired_p2_(?P<variant>.+?)_(?P<clf>lr|rf|xgb)__vs__(?P<baseline>\w+)_by_group\.csv$"
)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in-dir", default="reports/month3/paired")
    ap.add_argument("--out-csv", default="reports/month3/phase2_paired_matrix.csv")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.in_dir, "paired_p2_*_by_group.csv")))
    print(f"Found {len(files)} paired-DeLong CSVs")
    rows = []
    for f in files:
        base = os.path.basename(f)
        m = FILENAME_RE.match(base)
        if not m:
            continue
        variant = m.group("variant")
        clf     = m.group("clf")
        baseline = m.group("baseline")
        df = pd.read_csv(f)
        overall = df[df["slice_kind"] == "overall"]
        if overall.empty:
            continue
        row = overall.iloc[0]
        rows.append({
            "variant": variant,
            "clf": clf,
            "baseline": baseline,
            "n_common": int(row["n"]),
            "auroc_challenger": float(row["auroc_a"]),
            "auroc_baseline":   float(row["auroc_b"]),
            "diff": float(row["diff"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
            "p_two_sided": float(row["p_two_sided"]),
            "sig": row["sig"],
        })
    out = pd.DataFrame(rows).sort_values(
        ["baseline", "clf", "diff"], ascending=[True, True, False]
    ).reset_index(drop=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}  ({len(out)} rows)")

    # Quick pretty summary: for each baseline, how many (variant, clf) beat it?
    if len(out):
        print("\n=== Who beats whom? (by baseline) ===")
        for blab, grp in out.groupby("baseline"):
            n_total = len(grp)
            n_wins = int((grp["diff"] > 0).sum())
            n_sig_wins = int(((grp["diff"] > 0) & (grp["p_two_sided"] < 0.05)).sum())
            n_sig_losses = int(((grp["diff"] < 0) & (grp["p_two_sided"] < 0.05)).sum())
            print(f"  {blab:8s}: {n_wins}/{n_total} challengers beat; "
                  f"significant wins={n_sig_wins}, significant losses={n_sig_losses}")

        # Top 5 by diff for each baseline
        print("\n=== Top 5 challengers per baseline (by diff) ===")
        for blab, grp in out.groupby("baseline"):
            top = grp.nlargest(5, "diff")[
                ["variant", "clf", "auroc_challenger", "auroc_baseline",
                 "diff", "ci_low", "ci_high", "p_two_sided", "sig"]
            ].copy()
            for c in ("auroc_challenger", "auroc_baseline", "diff", "ci_low", "ci_high"):
                top[c] = top[c].round(4)
            top["p_two_sided"] = top["p_two_sided"].apply(
                lambda v: "nan" if pd.isna(v) else f"{v:.2e}"
            )
            print(f"\n-- vs baseline={blab} --")
            print(top.to_string(index=False))


if __name__ == "__main__":
    main()
