"""
deberta_baseline.py - Fine-tune DeBERTa-v3-base on raw reasoning trace text.

This is the strongest text-encoder baseline we MUST beat to claim that
structure-aware modeling adds value. Reviewers will ask: "what about a plain
text classifier on the trace?"

Strategy:
  - Truncate each trace to LAST 512 tokens (conclusion area is most decisive).
  - Fine-tune microsoft/deberta-v3-base end-to-end with binary CE loss.
  - 5-fold stratified CV pooled across 8 dataset-model files.
  - Same metrics as Step Transformer for direct comparison.

Usage:
    PYTHONPATH=. python src/modeling/deberta_baseline.py \
        --traces-glob "data/traces/*_traces.jsonl" \
        --output results/month2/deberta_pooled.json \
        --epochs 3 --batch-size 8 --lr 2e-5
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
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.modeling.cv_utils import aggregate_folds, evaluate, save_results, stratified_split

logger = logging.getLogger(__name__)


# =============================================================================
# DATA
# =============================================================================

def load_traces(glob_pat: str) -> dict:
    paths = sorted(glob.glob(glob_pat))
    logger.info(f"Found {len(paths)} trace files")
    texts, labels, ids, groups = [], [], [], []
    for p in paths:
        gname = os.path.basename(p).replace("_traces.jsonl", "")
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                it = json.loads(line)
                trace = it.get("reasoning_trace") or it.get("full_response") or ""
                if not trace.strip():
                    continue
                texts.append(trace)
                labels.append(int(it.get("is_correct", False)))
                ids.append(it.get("item_id"))
                groups.append(gname)
    return {
        "texts": texts,
        "labels": np.array(labels, dtype=np.int64),
        "ids": np.array(ids, dtype=object),
        "groups": np.array(groups, dtype=object),
    }


class TraceTextDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray,
                 tokenizer, max_len: int = 512, take_last: bool = True):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer
        self.max_len = max_len
        self.take_last = take_last

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize then optionally take the LAST max_len tokens (decision area)
        enc = self.tok(text, return_tensors="pt", truncation=False,
                       add_special_tokens=False)
        ids = enc["input_ids"][0]
        if self.take_last and len(ids) > self.max_len - 2:
            ids = ids[-(self.max_len - 2):]
        else:
            ids = ids[: self.max_len - 2]
        ids = torch.cat([
            torch.tensor([self.tok.cls_token_id], dtype=torch.long),
            ids,
            torch.tensor([self.tok.sep_token_id], dtype=torch.long),
        ])
        attn = torch.ones_like(ids)
        return {
            "input_ids": ids,
            "attention_mask": attn,
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


def collate_pad(batch: list[dict], pad_id: int) -> dict:
    max_len = max(b["input_ids"].size(0) for b in batch)
    n = len(batch)
    input_ids = torch.full((n, max_len), pad_id, dtype=torch.long)
    attention = torch.zeros(n, max_len, dtype=torch.long)
    labels = torch.zeros(n, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_ids"].size(0)
        input_ids[i, :L] = b["input_ids"]
        attention[i, :L] = 1
        labels[i] = b["labels"]
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


# =============================================================================
# TRAIN ONE FOLD
# =============================================================================

def train_one_fold(model_name: str, train_ds: Dataset, val_ds: Dataset,
                   tokenizer, epochs: int, batch_size: int, lr: float,
                   device: str, log_prefix: str = ""
                   ) -> tuple[np.ndarray, np.ndarray]:

    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2,
        collate_fn=lambda b: collate_pad(b, pad_id),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=2,
        collate_fn=lambda b: collate_pad(b, pad_id),
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2,
    ).to(device)

    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=lr)
    n_steps = len(train_loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * n_steps), n_steps)

    use_amp = (device == "cuda")
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_auroc = -1.0
    best_probs = None
    best_labels = None

    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n_seen = 0
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    out = model(**batch)
                    loss = out.loss
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                out = model(**batch)
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
            loss_sum += loss.item() * batch["labels"].size(0)
            n_seen += batch["labels"].size(0)

        avg_loss = loss_sum / max(n_seen, 1)

        # Eval
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in val_loader:
                lbl = batch.pop("labels")
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.cuda.amp.autocast(dtype=torch.bfloat16) if use_amp else torch.no_grad():
                    out = model(**batch)
                p = torch.softmax(out.logits.float(), dim=-1)[:, 1].cpu().numpy()
                ps.append(p); ys.append(lbl.numpy())
        ys = np.concatenate(ys); ps = np.concatenate(ps)
        m = evaluate(ys, ps, name=f"epoch_{ep}")
        logger.info(f"  {log_prefix} ep {ep:02d}  loss={avg_loss:.4f}  "
                    f"val_AUROC={m['auroc']:.3f}  val_AUPRC={m['auprc']:.3f}  "
                    f"val_ECE={m['ece']:.3f}")
        if m["auroc"] > best_auroc:
            best_auroc = m["auroc"]
            best_probs = ps.copy()
            best_labels = ys.copy()

    # Free
    del model
    if use_amp:
        torch.cuda.empty_cache()
    return best_labels, best_probs


# =============================================================================
# CV ORCHESTRATION
# =============================================================================

def run_cv(traces_glob: str, output: str,
           model_name: str = "microsoft/deberta-v3-base",
           epochs: int = 3, batch_size: int = 8, lr: float = 2e-5,
           max_len: int = 512, n_splits: int = 5, seed: int = 42):
    data = load_traces(traces_glob)
    texts, labels, groups, item_ids = (data["texts"], data["labels"],
                                       data["groups"], data["ids"])
    logger.info(f"Loaded {len(labels)} traces. Pos rate: {labels.mean():.3f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    oof_prob = np.full(len(labels), np.nan, dtype=np.float32)
    oof_fold = np.full(len(labels), -1, dtype=np.int32)

    fold_metrics = []
    all_y, all_p = [], []
    for fold, (tr_idx, te_idx) in enumerate(stratified_split(
            labels, group_id=groups if len(set(groups)) > 1 else None,
            n_splits=n_splits, seed=seed)):
        logger.info(f"\n=== Fold {fold + 1}/{n_splits}  "
                    f"train={len(tr_idx)}  val={len(te_idx)} ===")
        train_ds = TraceTextDataset([texts[i] for i in tr_idx], labels[tr_idx],
                                     tokenizer, max_len=max_len)
        val_ds = TraceTextDataset([texts[i] for i in te_idx], labels[te_idx],
                                   tokenizer, max_len=max_len)

        labels_, probs_ = train_one_fold(
            model_name, train_ds, val_ds, tokenizer,
            epochs=epochs, batch_size=batch_size, lr=lr, device=device,
            log_prefix=f"fold{fold + 1}",
        )
        oof_prob[te_idx] = probs_
        oof_fold[te_idx] = fold
        fm = evaluate(labels_, probs_, name=f"fold_{fold + 1}")
        fold_metrics.append(fm)
        all_y.append(labels_); all_p.append(probs_)
        logger.info(f"  -> Fold {fold + 1} best  AUROC={fm['auroc']:.3f}  "
                    f"ECE={fm['ece']:.3f}")

    all_y = np.concatenate(all_y); all_p = np.concatenate(all_p)
    overall = evaluate(all_y, all_p, name="overall")
    summary = aggregate_folds(fold_metrics)

    oof_path = output.replace(".json", "_oof.npz")
    np.savez_compressed(oof_path,
        item_ids=item_ids, groups=groups,
        y_true=labels, oof_prob=oof_prob, oof_fold=oof_fold,
        seed=np.array([seed]), n_splits=np.array([n_splits]))
    logger.info(f"Saved OOF preds: {oof_path}")

    results = {
        "model": model_name,
        "n_splits": n_splits,
        "epochs": epochs, "batch_size": batch_size, "lr": lr, "max_len": max_len,
        "n_samples": int(len(labels)),
        "base_accuracy": float(labels.mean()),
        "fold_metrics": fold_metrics,
        "summary": summary,
        "overall": overall,
        "oof_path": oof_path,
    }
    save_results(output, results)

    print("\n=== DeBERTa baseline CV summary ===")
    for k in ["auroc_mean", "auprc_mean", "ece_mean",
              "accuracy_at_80_mean", "accuracy_at_90_mean"]:
        print(f"  {k:25s} {summary.get(k, float('nan')):.4f}  "
              f"± {summary.get(k.replace('_mean','_std'), 0):.4f}")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces-glob", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="microsoft/deberta-v3-base")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    run_cv(args.traces_glob, args.output, model_name=args.model,
           epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
           max_len=args.max_len, n_splits=args.n_splits, seed=args.seed)


if __name__ == "__main__":
    main()
