#!/usr/bin/env python3
"""
backfill_prr.py - Re-evaluate every existing OOF .npz with the PRR-enabled
metric stack and write augmented results JSONs alongside.

Run once after PRR was added to cv_utils.evaluate(); subsequent training runs
will emit PRR natively.

Usage:
    PYTHONPATH=. python scripts/backfill_prr.py
    PYTHONPATH=. python scripts/backfill_prr.py --dry-run
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.modeling.cv_utils import evaluate

logger = logging.getLogger(__name__)


def backfill_one(oof_path: str, dry_run: bool = False) -> dict:
    z = np.load(oof_path, allow_pickle=True)
    y = np.asarray(z["y_true"]).astype(int)
    p = np.asarray(z["oof_prob"]).astype(float)
    folds = np.asarray(z["oof_fold"]) if "oof_fold" in z.files else None

    overall = evaluate(y, p, name=os.path.basename(oof_path).replace("_oof.npz", ""))

    per_fold = []
    if folds is not None and folds.size and (folds >= 0).any():
        for f in sorted(set(int(x) for x in folds if x >= 0)):
            mask = folds == f
            if mask.sum() < 2 or len(np.unique(y[mask])) < 2:
                continue
            per_fold.append(evaluate(y[mask], p[mask], name=f"fold_{f}"))

    out = {
        "oof_source": oof_path,
        "n_samples": int(len(y)),
        "n_correct": int(y.sum()),
        "base_accuracy": float(y.mean()),
        "overall": overall,
        "per_fold": per_fold,
    }

    json_path = oof_path.replace("_oof.npz", "_metrics_prr.json")
    logger.info(f"  {os.path.basename(oof_path)}  AUROC={overall['auroc']:.4f}  "
                f"PRR={overall['prr']:+.4f}  AURC_method={overall['aurc_method']:.4f}")
    if not dry_run:
        with open(json_path, "w") as f:
            json.dump(out, f, indent=2, default=float)
        logger.info(f"    -> wrote {json_path}")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glob", default="results/**/*_oof.npz",
                        help="Glob pattern for OOF .npz files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print metrics without writing JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    paths = sorted(glob.glob(args.glob, recursive=True))
    if not paths:
        logger.warning(f"No OOF files matched: {args.glob}")
        return

    logger.info(f"Backfilling PRR for {len(paths)} OOF file(s)")
    summary = []
    for p in paths:
        m = backfill_one(p, dry_run=args.dry_run)
        summary.append((p, m["overall"]))

    print()
    print(f"{'File':<60s}  {'AUROC':>7s}  {'PRR':>7s}  {'ECE':>6s}  {'Acc@80':>7s}")
    for p, ov in summary:
        print(f"{os.path.basename(p):<60s}  "
              f"{ov['auroc']:>7.4f}  {ov['prr']:>+7.4f}  "
              f"{ov['ece']:>6.4f}  {ov['accuracy_at_80']:>7.4f}")


if __name__ == "__main__":
    main()
