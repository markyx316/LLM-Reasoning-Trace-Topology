#!/usr/bin/env python3
"""
run_cross_model_transfer.py - Cross-model transfer (qwen <-> llama on the same dataset).

The 6 transfer experiments saved in `results/transfer_*.json` only do
**cross-domain** transfer (math <-> gpqa, math <-> arc, gsm8k <-> math) within
the same model. We never tested whether structural features learned on
R1-Distill-Qwen-7B traces transfer to R1-Distill-Llama-8B traces of the
*same* problems. This script does that — 8 cells (4 datasets x 2 directions).

Tests the implicit "the structural signature of an uncertain reasoning model
is model-family-general" claim. If structural features collapse across model
families, we have a same-problem-different-model leakage story to write up.

Reuses `train_and_evaluate.run_transfer_experiment` and the recurrence-augmented
feature CSVs (28 cols: 23 legacy + 5 recurrence).

Usage:
    PYTHONPATH=. python scripts/run_cross_model_transfer.py
    PYTHONPATH=. python scripts/run_cross_model_transfer.py --classifier xgboost
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.modeling.train_and_evaluate import run_transfer_experiment

logger = logging.getLogger(__name__)


DATASETS = ["math500", "gsm8k", "gpqa_diamond", "arc_challenge"]
MODELS = ["qwen7b", "llama8b"]


# Cols not used as features: identifiers + label.
_NON_FEATURE_COLS = {"item_id", "dataset", "is_correct"}


def load_csv(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    if "is_correct" not in df.columns:
        raise ValueError(f"{path} missing is_correct column")
    feat_cols = [c for c in df.columns if c not in _NON_FEATURE_COLS]
    X = df[feat_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["is_correct"].astype(int).to_numpy()
    return X, y, feat_cols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", default="data/features",
                        help="Directory containing *_features_rec.csv files")
    parser.add_argument("--classifier", default="random_forest",
                        choices=["logistic_regression", "random_forest", "xgboost"])
    parser.add_argument("--out-dir", default="results",
                        help="Directory to write transfer JSONs")
    parser.add_argument("--variant", default="rec",
                        choices=["rec", "base"],
                        help="rec = 23 legacy + 5 recurrence (28 features); "
                             "base = 23 legacy only")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    suffix = "_features_rec.csv" if args.variant == "rec" else "_features.csv"

    rows = []
    for dataset in DATASETS:
        # qwen -> llama and llama -> qwen
        for src_model, tgt_model in [("qwen7b", "llama8b"), ("llama8b", "qwen7b")]:
            src_path = os.path.join(args.features_dir, f"{dataset}_{src_model}{suffix}")
            tgt_path = os.path.join(args.features_dir, f"{dataset}_{tgt_model}{suffix}")
            if not (os.path.exists(src_path) and os.path.exists(tgt_path)):
                logger.warning(f"Skip {dataset} {src_model}->{tgt_model}: missing CSV")
                continue

            X_src, y_src, src_feats = load_csv(src_path)
            X_tgt, y_tgt, tgt_feats = load_csv(tgt_path)

            # Sanity: column alignment
            if src_feats != tgt_feats:
                common = [f for f in src_feats if f in tgt_feats]
                logger.warning(f"  feature mismatch; falling back to {len(common)} common cols")
                src_idx = [src_feats.index(f) for f in common]
                tgt_idx = [tgt_feats.index(f) for f in common]
                X_src = X_src[:, src_idx]
                X_tgt = X_tgt[:, tgt_idx]
                feats = common
            else:
                feats = src_feats

            train_name = f"{dataset}_{src_model}"
            test_name = f"{dataset}_{tgt_model}"
            logger.info(f"Transfer: {train_name} -> {test_name} ({args.classifier}, "
                        f"{len(feats)} features)")
            res = run_transfer_experiment(
                X_train=X_src, y_train=y_src,
                X_test=X_tgt, y_test=y_tgt,
                feature_names=feats,
                train_name=train_name, test_name=test_name,
                classifier_name=args.classifier,
            )
            metrics = res["metrics"]
            logger.info(f"  AUROC={metrics['auroc']:.3f}  AUPRC={metrics['auprc']:.3f}  "
                        f"ECE={metrics['ece']:.3f}  Acc@80={metrics['accuracy_at_80']:.3f}")

            out_path = os.path.join(args.out_dir,
                                    f"transfer_cross_model_{dataset}_{src_model}_to_{tgt_model}_{args.variant}.json")
            os.makedirs(args.out_dir, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump({
                    "transfer": res,
                    "n_features": len(feats),
                    "feature_names": feats,
                    "variant": args.variant,
                }, f, indent=2, default=float)
            logger.info(f"  saved {out_path}")

            rows.append({
                "dataset": dataset,
                "src": src_model,
                "tgt": tgt_model,
                "n_train": len(y_src),
                "n_test": len(y_tgt),
                "src_acc": float(y_src.mean()),
                "tgt_acc": float(y_tgt.mean()),
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "ece": metrics["ece"],
                "acc_at_80": metrics["accuracy_at_80"],
            })

    # Summary table
    print("\n" + "=" * 95)
    print(f"CROSS-MODEL TRANSFER  (clf={args.classifier}, variant={args.variant})")
    print("=" * 95)
    print(f"{'dataset':<20s} {'src->tgt':<22s} {'AUROC':>8s} {'AUPRC':>8s} "
          f"{'ECE':>7s} {'Acc@80':>8s} {'src_acc':>8s} {'tgt_acc':>8s}")
    print("-" * 95)
    for r in rows:
        arrow = f"{r['src']}->{r['tgt']}"
        print(f"{r['dataset']:<20s} {arrow:<22s} "
              f"{r['auroc']:>8.3f} {r['auprc']:>8.3f} "
              f"{r['ece']:>7.3f} {r['acc_at_80']:>8.3f} "
              f"{r['src_acc']:>8.3f} {r['tgt_acc']:>8.3f}")
    print("=" * 95)


if __name__ == "__main__":
    main()
