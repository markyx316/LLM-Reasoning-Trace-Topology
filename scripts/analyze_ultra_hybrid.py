"""Phase 2 analysis: distill `results/month3/ultra_hybrid.json` into the
headline / ablation / calibration tables the paper needs.

Reads the nested JSON emitted by `src/modeling/hybrid_route_ab.py` and writes:

  reports/month3/phase2_headline.csv
      one row per (variant, clf) with pooled AUROC, AUPRC, ECE plus 95%
      DeLong CI on AUROC (computed from the per-variant OOF npz when
      available; from the JSON alone otherwise).

  reports/month3/phase2_ablation.csv
      Ablation of ULTRA_HYBRID_ALL.  Row per feature family with its
      leave-one-out delta on pooled AUROC.

  reports/month3/phase2_calibration.csv
      ECE across every variant × classifier, sorted.

Usage:
    PYTHONPATH=. python scripts/analyze_ultra_hybrid.py \
        --json   results/month3/ultra_hybrid.json \
        --oof-dir results/month3/ultra_hybrid \
        --out-dir reports/month3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analysis.delong_ci import delong_auroc_ci  # noqa: E402


def _npz_path_for(oof_dir: Path, variant: str, clf: str) -> Path:
    safe = variant.replace("+", "_plus_").replace("-", "_minus_")
    return oof_dir / f"ultrahybrid_{safe}__{clf}_oof.npz"


def _pooled_ci_from_npz(path: Path, alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (auroc, ci_low, ci_high) from a saved OOF npz via DeLong + logit CI."""
    z = np.load(path, allow_pickle=True)
    y = z["y_true"].astype(int)
    p = z["oof_prob"].astype(float)
    r = delong_auroc_ci(y, p, alpha=alpha, method="logit")
    return float(r.auroc), float(r.ci_low), float(r.ci_high)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", required=True)
    ap.add_argument("--oof-dir", default=None)
    ap.add_argument("--out-dir", default="reports/month3")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    out_dir = _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(Path(args.json).read_text())
    variants: dict = payload["variants"]

    oof_dir = Path(args.oof_dir) if args.oof_dir else None

    # ================================================================
    # 1. Headline table (per variant × classifier)
    # ================================================================
    rows = []
    for vname, vblock in variants.items():
        n_feat = vblock["n_features"]
        n_samples = vblock["n_samples"]
        for clf, cblk in vblock["clfs"].items():
            s = cblk["summary"]
            o = cblk["overall"]
            auc_fold = s.get("auroc_mean", float("nan"))
            auc_pool = o.get("auroc", float("nan"))
            # pooled DeLong CI (requires the OOF npz)
            ci_lo = ci_hi = float("nan")
            if oof_dir is not None:
                npz = _npz_path_for(oof_dir, vname, clf)
                if npz.exists():
                    _, ci_lo, ci_hi = _pooled_ci_from_npz(npz, alpha=args.alpha)
            rows.append({
                "variant": vname, "clf": clf,
                "n_features": n_feat, "n_samples": n_samples,
                "auroc_pooled": auc_pool,
                "auroc_ci_low": ci_lo,
                "auroc_ci_high": ci_hi,
                "auroc_fold_mean": auc_fold,
                "auroc_fold_std":  s.get("auroc_std", float("nan")),
                "auprc_pooled": o.get("auprc", float("nan")),
                "ece_pooled":   o.get("ece",   float("nan")),
                "accuracy_at_80": o.get("accuracy_at_80", float("nan")),
                "accuracy_at_90": o.get("accuracy_at_90", float("nan")),
            })
    head_df = pd.DataFrame(rows).sort_values(
        ["auroc_pooled"], ascending=False).reset_index(drop=True)
    head_path = out_dir / "phase2_headline.csv"
    head_df.to_csv(head_path, index=False)
    print(f"wrote {head_path}  ({len(head_df)} rows)")
    # Pretty print top 10
    pretty = head_df.head(20).copy()
    for c in ("auroc_pooled", "auroc_ci_low", "auroc_ci_high", "auprc_pooled",
              "ece_pooled", "accuracy_at_80"):
        pretty[c] = pretty[c].round(4)
    print("\n=== Top 20 (variant, clf) pooled AUROC ===")
    print(pretty[["variant", "clf", "n_features", "auroc_pooled",
                  "auroc_ci_low", "auroc_ci_high", "ece_pooled"]]
          .to_string(index=False))

    # ================================================================
    # 2. Ablation table (leave-one-out from ULTRA_HYBRID_ALL)
    # ================================================================
    ablation_pairs = [
        ("ULTRA_HYBRID_ALL", "ULTRA_ALL-route_a", "Route A (n-gram/graph/timing/structural-PH)"),
        ("ULTRA_HYBRID_ALL", "ULTRA_ALL-gnn",     "Trace GNNs (structural + hybrid)"),
        ("ULTRA_HYBRID_ALL", "ULTRA_ALL-text",    "Text OOFs (DeBERTa, DeBERTa+Cond, RoBERTa, Step)"),
        ("ULTRA_HYBRID_ALL", "ULTRA_ALL-probe",   "Hidden-state probe (v2)"),
        ("ULTRA_HYBRID_ALL", "ULTRA_ALL-shapelet","Shapelet OOF"),
    ]
    rows = []
    for full_v, minus_v, label in ablation_pairs:
        if full_v not in variants or minus_v not in variants:
            print(f"  [skip ablation] need both {full_v} and {minus_v}")
            continue
        for clf in ("lr", "rf", "xgb"):
            if clf not in variants[full_v]["clfs"] or clf not in variants[minus_v]["clfs"]:
                continue
            auc_full = variants[full_v]["clfs"][clf]["overall"]["auroc"]
            auc_minus = variants[minus_v]["clfs"][clf]["overall"]["auroc"]
            rows.append({
                "removed_family": label,
                "clf": clf,
                "auroc_full":  auc_full,
                "auroc_minus": auc_minus,
                "delta":       auc_full - auc_minus,   # + means family helps
            })
    abl_df = pd.DataFrame(rows)
    abl_path = out_dir / "phase2_ablation.csv"
    abl_df.to_csv(abl_path, index=False)
    print(f"\nwrote {abl_path}  ({len(abl_df)} rows)")
    if len(abl_df):
        pretty = abl_df.copy()
        for c in ("auroc_full", "auroc_minus", "delta"):
            pretty[c] = pretty[c].round(4)
        print("\n=== Leave-one-out ablation ===")
        print(pretty.to_string(index=False))

    # ================================================================
    # 3. Calibration table (ECE) — every variant × classifier, sorted
    # ================================================================
    calib = head_df[["variant", "clf", "auroc_pooled", "ece_pooled"]].copy()
    calib = calib.sort_values("ece_pooled").reset_index(drop=True)
    calib_path = out_dir / "phase2_calibration.csv"
    calib.to_csv(calib_path, index=False)
    print(f"\nwrote {calib_path}")
    pretty = calib.head(10).copy()
    pretty["auroc_pooled"] = pretty["auroc_pooled"].round(4)
    pretty["ece_pooled"] = pretty["ece_pooled"].round(4)
    print("\n=== Top 10 best-calibrated (ECE asc) ===")
    print(pretty.to_string(index=False))


if __name__ == "__main__":
    main()
