#!/usr/bin/env python3
"""
run_cross_dataset_roberta_transfer.py

8x8 cross-dataset transfer matrix for RoBERTa-base (or any AutoModelFor
SequenceClassification HF model). Mirror of run_cross_dataset_steptf_transfer.py
but for the text encoder.

For each source dataset s in the 8 standard combos:
  1. Train RoBERTa on 100% of s's traces (no CV, no validation split)
  2. For each target dataset t (incl. self as sanity):
       - Tokenize, truncate to last 512 tokens
       - Predict softmax probabilities
       - Compute AUROC, AUPRC, ECE, Acc@80, Acc@90, PRR
  3. Save one JSON per (s, t) pair + summary CSV + print 8x8 AUROC matrix.

Why this matters (Approach 4 of the post-disappointment plan):
  Pooled in-domain CV gives RoBERTa AUROC 0.798 — strong but possibly content-
  dependent. The deep question is: does RoBERTa transfer ACROSS datasets and
  ACROSS models? If transfer collapses, it confirms the text encoder is
  picking up domain vocabulary. If it holds, RoBERTa generalises and
  structural features have a harder differentiation story to make.

  Compared to scripts/run_cross_dataset_steptf_transfer.py off-diagonal mean
  of 0.59, this gives us the apples-to-apples text-encoder transfer baseline.

Compute: ~3-4 hours on 1x RTX PRO 6000 Blackwell.
  Smallest source (gpqa, n=198): ~3 min training + 8 min inference = 11 min
  Largest source (gsm8k, n=1319): ~25 min training + 8 min inference = 33 min
  Total (8 sources): ~3 hours

Usage:
    nohup PYTHONPATH=. python scripts/run_cross_dataset_roberta_transfer.py \\
        > logs/roberta_transfer.log 2>&1 &
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
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.modeling.deberta_baseline import (
    TraceTextDataset, collate_pad, load_traces,
)
from src.modeling.cv_utils import evaluate, save_results

logger = logging.getLogger(__name__)

DATASETS = [
    "math500_qwen7b", "math500_llama8b",
    "gsm8k_qwen7b", "gsm8k_llama8b",
    "gpqa_diamond_qwen7b", "gpqa_diamond_llama8b",
    "arc_challenge_qwen7b", "arc_challenge_llama8b",
]


def train_full(texts: list[str], labels: np.ndarray, model_name: str,
               epochs: int, batch_size: int, lr: float, max_len: int,
               device: str, seed: int = 42) -> tuple[torch.nn.Module, AutoTokenizer]:
    """Train on 100% of (texts, labels). No CV, no validation split."""
    torch.manual_seed(seed); np.random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_id = tokenizer.pad_token_id
    ds = TraceTextDataset(texts, labels, tokenizer, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2,
                        collate_fn=lambda b: collate_pad(b, pad_id))

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2,
    ).to(device)

    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=lr)
    n_steps = max(len(loader) * epochs, 1)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * n_steps), n_steps)

    for ep in range(1, epochs + 1):
        model.train()
        loss_sum, n_seen, n_skipped = 0.0, 0, 0
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad()
            out = model(**batch)
            loss = out.loss
            if not torch.isfinite(loss):
                n_skipped += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_sum += float(loss.item()) * batch["labels"].size(0)
            n_seen += batch["labels"].size(0)
        avg = loss_sum / max(n_seen, 1)
        msg = f"  ep {ep:02d}  loss={avg:.4f}"
        if n_skipped > 0:
            msg += f"  (skipped {n_skipped} batches)"
        logger.info(msg)

    model.eval()
    return model, tokenizer


@torch.no_grad()
def predict(model: torch.nn.Module, tokenizer: AutoTokenizer,
            texts: list[str], batch_size: int, max_len: int,
            device: str) -> np.ndarray:
    """Run inference; return P(class==1) for each text in original order."""
    pad_id = tokenizer.pad_token_id
    dummy_labels = np.zeros(len(texts), dtype=np.int64)
    ds = TraceTextDataset(texts, dummy_labels, tokenizer, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2,
                        collate_fn=lambda b: collate_pad(b, pad_id))
    probs = []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        logits = out.logits.float()
        if not torch.isfinite(logits).all():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=40.0, neginf=-40.0)
        p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        probs.append(p)
    return np.concatenate(probs)


def load_dataset(traces_path: str) -> dict:
    """Load all traces from one JSONL file."""
    data = load_traces(traces_path)
    return data  # has texts, labels, ids, groups


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", default="roberta-base",
                   help="HF model id (default: roberta-base)")
    p.add_argument("--traces-dir", default="data/traces")
    p.add_argument("--out-dir", default="results/month2_v2/roberta_transfer")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only", nargs="+", default=None,
                   help="Restrict source datasets")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip (src, tgt) pairs whose JSON already exists")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    os.makedirs(args.out_dir, exist_ok=True)

    # Pre-load all 8 datasets (~5 sec; trace JSONLs are small)
    data = {}
    for ds in DATASETS:
        path = os.path.join(args.traces_dir, f"{ds}_traces.jsonl")
        if not os.path.exists(path):
            logger.warning(f"Missing trace file: {path} (skip)")
            continue
        d = load_dataset(path)
        data[ds] = d
        logger.info(f"loaded {ds}: n={len(d['labels'])}  "
                    f"pos_rate={float(np.asarray(d['labels']).mean()):.3f}")

    sources = args.only or list(data.keys())
    summary = []

    for src in sources:
        if src not in data:
            continue
        logger.info("")
        logger.info(f"========== SOURCE: {src} "
                    f"(n_train={len(data[src]['labels'])}) ==========")

        # Train once on full source
        t0 = time.time()
        try:
            model, tokenizer = train_full(
                data[src]["texts"], np.asarray(data[src]["labels"], dtype=np.int64),
                model_name=args.model,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                max_len=args.max_len, device=device, seed=args.seed,
            )
        except Exception as e:
            logger.error(f"  training failed for source {src}: {e}")
            continue
        logger.info(f"  trained in {time.time() - t0:.1f}s")

        # Infer on every target
        for tgt in DATASETS:
            if tgt not in data:
                continue
            out_path = os.path.join(
                args.out_dir, f"roberta_{src}__to__{tgt}.json")
            if args.skip_existing and os.path.exists(out_path):
                logger.info(f"    {src} -> {tgt}: skip (exists)")
                # Still pull metrics into the summary
                try:
                    d = json.load(open(out_path))
                    m = d["metrics"]
                    summary.append({"source": src, "target": tgt,
                                    "auroc": m["auroc"], "auprc": m["auprc"],
                                    "ece": m["ece"], "prr": m.get("prr", 0.0),
                                    "acc_at_80": m.get("accuracy_at_80", 0.0),
                                    "n_target": int(len(data[tgt]["labels"])),
                                    "diagonal": (src == tgt)})
                except Exception:
                    pass
                continue

            try:
                probs = predict(model, tokenizer,
                                data[tgt]["texts"], args.batch_size * 2,
                                args.max_len, device)
            except Exception as e:
                logger.error(f"    {src} -> {tgt}: inference failed: {e}")
                continue

            y = np.asarray(data[tgt]["labels"], dtype=np.int64)
            metrics = evaluate(y, probs, name=f"{src}__to__{tgt}")
            logger.info(f"    {src} -> {tgt}:  AUROC={metrics['auroc']:.4f}  "
                        f"AUPRC={metrics['auprc']:.4f}  "
                        f"PRR={metrics.get('prr', float('nan')):.4f}")

            out = {
                "model": args.model,
                "source": src, "target": tgt,
                "diagonal": (src == tgt),
                "epochs": args.epochs, "lr": args.lr,
                "batch_size": args.batch_size, "seed": args.seed,
                "n_source_train": int(len(data[src]["labels"])),
                "n_target_eval":  int(len(y)),
                "source_pos_rate": float(np.asarray(data[src]["labels"]).mean()),
                "target_pos_rate": float(y.mean()),
                "metrics": metrics,
            }
            save_results(out_path, out)
            summary.append({"source": src, "target": tgt,
                            "diagonal": (src == tgt),
                            "auroc": metrics["auroc"], "auprc": metrics["auprc"],
                            "ece": metrics["ece"], "prr": metrics.get("prr", 0.0),
                            "acc_at_80": metrics.get("accuracy_at_80", 0.0),
                            "n_target": int(len(y))})

        # Free GPU memory
        del model, tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()

    # Summary CSV + 8x8 pivot
    if not summary:
        logger.warning("No results produced.")
        return
    df = pd.DataFrame(summary)
    csv_path = os.path.join(args.out_dir, "roberta_transfer_summary.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"\nSummary CSV: {csv_path}")

    print("\n=== RoBERTa cross-dataset transfer — AUROC matrix (rows=src, cols=tgt) ===")
    try:
        pivot = df.pivot(index="source", columns="target", values="auroc")
        pivot = pivot.reindex(index=DATASETS, columns=DATASETS)
        print(pivot.round(4).to_string(na_rep="  .   "))
    except Exception as e:
        logger.warning(f"pivot failed: {e}")

    off = df[df["source"] != df["target"]]
    print(f"\nOff-diagonal (n={len(off)}): "
          f"AUROC mean={off['auroc'].mean():.4f}  "
          f"median={off['auroc'].median():.4f}  "
          f"min={off['auroc'].min():.4f}  "
          f"max={off['auroc'].max():.4f}")
    diag = df[df["source"] == df["target"]]
    print(f"Diagonal (n={len(diag)}, train-on-target sanity): "
          f"AUROC mean={diag['auroc'].mean():.4f}")


if __name__ == "__main__":
    main()
