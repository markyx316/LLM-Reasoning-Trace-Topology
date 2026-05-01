#!/usr/bin/env python3
"""
run_cross_dataset_steptf_transfer.py

8x8 cross-dataset transfer matrix for the Step Transformer.

For each source dataset s (8 combos: math500/gsm8k/gpqa_diamond/arc_challenge
x qwen7b/llama8b):
  1. Train StepTransformer on 100% of s's step embeddings (no CV, no val split)
  2. For each target dataset t (including t == s as a sanity diagonal):
       - Run inference on t's full step-embedding set
       - Compute AUROC, AUPRC, ECE, Acc@80, Acc@90, PRR
  3. Save one JSON per (s, t) + one summary CSV + print an 8x8 AUROC matrix.

Why this experiment:
  The per-dataset StepTF runs (results/month2_v2/step_transformer_<ds>.json)
  test within-dataset CV performance. The cross-dataset matrix tests whether
  the structural signal a small Transformer-over-MiniLM picks up on one
  domain transfers to another. Positive off-diagonal cells support the
  "structure is domain-general" claim; strong diagonals confirm the base
  training is healthy.

Design notes:
  - Trains on 100% of source (no val split) — we already have CV numbers
    on the diagonal from per-dataset runs.
  - Single training run per source (8 total), ~3-5 min each on an
    Ampere+/Blackwell GPU; inference is ~seconds per target.
  - Reuses StepDataset / StepTransformer / collate from
    src.modeling.step_transformer so the architecture matches exactly.
  - Reuses cv_utils.evaluate() so metrics (including PRR) are consistent
    with the rest of the pipeline.

Usage:
    PYTHONPATH=. python scripts/run_cross_dataset_steptf_transfer.py
    # (plus --only to restrict sources, --epochs/--lr to tune)

Outputs:
  results/month2_v2/steptf_transfer/steptf_<src>__to__<tgt>.json  (up to 64 files)
  results/month2_v2/steptf_transfer/steptf_transfer_summary.csv   (long format)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.modeling.step_transformer import (
    EMB_DIM,
    N_TYPES,
    PAD_TYPE,
    StepDataset,
    StepTransformer,
    collate,
)
from src.modeling.cv_utils import evaluate, save_results

logger = logging.getLogger(__name__)

DATASETS = [
    "math500_qwen7b", "math500_llama8b",
    "gsm8k_qwen7b", "gsm8k_llama8b",
    "gpqa_diamond_qwen7b", "gpqa_diamond_llama8b",
    "arc_challenge_qwen7b", "arc_challenge_llama8b",
]


# =============================================================================
# DATA
# =============================================================================

def load_npz(npz_path: str) -> dict:
    """Load a single dataset's step-embedding .npz into the schema
    StepDataset expects (lists of per-item arrays + a labels ndarray)."""
    z = np.load(npz_path, allow_pickle=True)
    return {
        "embeddings": list(z["embeddings"]),
        "step_types": list(z["step_types"]),
        "labels":     z["is_correct"].astype(np.int64),
        "item_ids":   z["item_ids"],
    }


def make_loader(data: dict, batch_size: int, shuffle: bool,
                num_workers: int = 2) -> DataLoader:
    ds = StepDataset(data["embeddings"], data["step_types"], data["labels"])
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate, num_workers=num_workers)


# =============================================================================
# TRAIN / PREDICT
# =============================================================================

def train_on_source(data: dict, device: str, epochs: int, batch_size: int,
                    lr: float, seed: int, log_prefix: str = "") -> torch.nn.Module:
    """Train StepTransformer on all of `data` (no validation split)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    loader = make_loader(data, batch_size=batch_size, shuffle=True)
    model = StepTransformer().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    y = data["labels"]
    n_pos = max(int(y.sum()), 1)
    n_neg = max(len(y) - n_pos, 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        n_seen = 0
        for batch in loader:
            emb = batch["emb"].to(device, non_blocking=True)
            typ = batch["typ"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            yb = batch["y"].to(device, non_blocking=True)

            opt.zero_grad()
            logit = model(emb, typ, mask)
            loss = loss_fn(logit, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_loss += float(loss.item()) * yb.size(0)
            n_seen += yb.size(0)

        avg = ep_loss / max(n_seen, 1)
        logger.info(f"  {log_prefix}ep {ep:02d}  loss={avg:.4f}")

    model.eval()
    return model


@torch.no_grad()
def predict(model: torch.nn.Module, data: dict, device: str,
            batch_size: int = 64) -> np.ndarray:
    """Run inference in the ORIGINAL item order (shuffle=False)."""
    loader = make_loader(data, batch_size=batch_size, shuffle=False)
    probs = []
    for batch in loader:
        emb = batch["emb"].to(device)
        typ = batch["typ"].to(device)
        mask = batch["mask"].to(device)
        logit = model(emb, typ, mask)
        probs.append(torch.sigmoid(logit).float().cpu().numpy())
    return np.concatenate(probs)


# =============================================================================
# DRIVER
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz-dir", default="data/step_embeddings")
    ap.add_argument("--out-dir", default="results/month2_v2/steptf_transfer")
    ap.add_argument("--epochs", type=int, default=15,
                    help="Training epochs on source (per-fold StepTF uses 15)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", nargs="+", default=None,
                    help="Restrict source datasets to these (default: all 8)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip (src, tgt) pairs whose JSON already exists")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    os.makedirs(args.out_dir, exist_ok=True)

    # Preload every dataset once
    data = {}
    for d in DATASETS:
        path = os.path.join(args.npz_dir, f"{d}.npz")
        if not os.path.exists(path):
            logger.warning(f"Missing: {path}")
            continue
        data[d] = load_npz(path)
        logger.info(f"loaded {d}: n={len(data[d]['labels'])}  "
                    f"pos_rate={float(data[d]['labels'].mean()):.3f}")

    sources = args.only or list(data.keys())
    for s in sources:
        if s not in data:
            logger.error(f"--only specified {s} but it's not loaded; skipping")

    summary_rows = []
    for src in sources:
        if src not in data:
            continue
        logger.info("")
        logger.info(f"========== SOURCE: {src} "
                    f"(n={len(data[src]['labels'])}, "
                    f"pos={float(data[src]['labels'].mean()):.3f}) ==========")
        t0 = time.time()
        model = train_on_source(
            data[src], device,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, seed=args.seed,
            log_prefix=f"[{src}] ",
        )
        logger.info(f"  trained in {time.time() - t0:.1f}s")

        for tgt in DATASETS:
            if tgt not in data:
                continue
            out_path = os.path.join(args.out_dir, f"steptf_{src}__to__{tgt}.json")
            if args.skip_existing and os.path.exists(out_path):
                logger.info(f"    {src} -> {tgt}: already done, skip")
                # Still include in summary if we can read it
                try:
                    d = json.load(open(out_path))
                    m = d["metrics"]
                    summary_rows.append({
                        "source": src, "target": tgt,
                        "auroc": m["auroc"], "auprc": m["auprc"],
                        "ece": m["ece"], "prr": m.get("prr", 0.0),
                        "acc_at_80": m.get("accuracy_at_80", 0.0),
                        "n_target": int(len(data[tgt]["labels"])),
                        "target_pos_rate": float(data[tgt]["labels"].mean()),
                    })
                except Exception:
                    pass
                continue

            y = data[tgt]["labels"]
            prob = predict(model, data[tgt], device,
                           batch_size=max(args.batch_size * 2, 32))
            metrics = evaluate(y, prob, name=f"{src}__to__{tgt}")

            logger.info(f"    {src} -> {tgt}:  "
                        f"AUROC={metrics['auroc']:.4f}  "
                        f"AUPRC={metrics['auprc']:.4f}  "
                        f"ECE={metrics['ece']:.4f}  "
                        f"PRR={metrics.get('prr', float('nan')):.4f}")

            out = {
                "source": src, "target": tgt,
                "diagonal": (src == tgt),
                "epochs": args.epochs, "lr": args.lr,
                "batch_size": args.batch_size, "seed": args.seed,
                "n_source_train": int(len(data[src]["labels"])),
                "n_target_eval":  int(len(y)),
                "source_pos_rate": float(data[src]["labels"].mean()),
                "target_pos_rate": float(y.mean()),
                "metrics": metrics,
            }
            save_results(out_path, out)

            summary_rows.append({
                "source": src, "target": tgt,
                "diagonal": (src == tgt),
                "auroc": metrics["auroc"], "auprc": metrics["auprc"],
                "ece": metrics["ece"], "prr": metrics.get("prr", 0.0),
                "acc_at_80": metrics.get("accuracy_at_80", 0.0),
                "n_target": int(len(y)),
                "target_pos_rate": float(y.mean()),
            })

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Summary CSV + AUROC pivot
    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(args.out_dir, "steptf_transfer_summary.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"\nSummary CSV: {csv_path}")

    if len(df):
        print("\n=== StepTF cross-dataset transfer — AUROC matrix (rows=src, cols=tgt) ===")
        try:
            pivot = df.pivot(index="source", columns="target", values="auroc")
            pivot = pivot.reindex(index=DATASETS, columns=DATASETS)
            print(pivot.round(4).to_string(na_rep="  .   "))
        except Exception as e:
            logger.warning(f"Could not pivot summary: {e}")

        print("\n=== Diagonal sanity (same-dataset train+test) vs per-dataset CV ===")
        diag = df[df["source"] == df["target"]][["source", "auroc"]]
        diag.columns = ["dataset", "diagonal_auroc"]
        print(diag.to_string(index=False))

        print("\n=== Off-diagonal summary ===")
        off = df[df["source"] != df["target"]]
        print(f"  n_pairs={len(off)}")
        if len(off):
            print(f"  AUROC mean={off['auroc'].mean():.4f}  "
                  f"median={off['auroc'].median():.4f}  "
                  f"min={off['auroc'].min():.4f}  "
                  f"max={off['auroc'].max():.4f}")


if __name__ == "__main__":
    main()
