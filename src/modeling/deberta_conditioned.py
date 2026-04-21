"""
deberta_conditioned.py - Problem-Conditioned DeBERTa baseline.

Upgrades the plain DeBERTa baseline by explicitly feeding the PROBLEM text
alongside the trace:

    [CLS] problem_text [SEP] trace_last_tokens [SEP]

Rationale: a trace can be structurally clean yet answer the wrong question.
Plain DeBERTa cannot detect this because it only sees the trace. Giving it
the problem allows attention heads to measure problem-trace consistency.

Token budget (max 512):
  - Reserve up to 128 tokens for the problem (truncated from end if longer)
  - Remaining ~380 tokens for the trace (taken from the TAIL, as in the
    plain DeBERTa baseline — the conclusion area is most decisive)

Usage:
    PYTHONPATH=. python src/modeling/deberta_conditioned.py \
        --traces-glob "data/traces/*_traces.jsonl" \
        --output results/month2/deberta_conditioned_pooled.json \
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
    AutoModelForSequenceClassification, AutoTokenizer,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.modeling.cv_utils import (
    aggregate_folds, evaluate, save_results, stratified_split,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA
# =============================================================================

def load_items(glob_pat: str) -> dict:
    paths = sorted(glob.glob(glob_pat))
    logger.info(f"Found {len(paths)} trace files")
    problems, traces, labels, ids, groups = [], [], [], [], []
    for p in paths:
        gname = os.path.basename(p).replace("_traces.jsonl", "")
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                it = json.loads(line)
                trace = it.get("reasoning_trace") or it.get("full_response") or ""
                prob = it.get("problem") or it.get("prompt") or ""
                if not trace.strip():
                    continue
                traces.append(trace)
                problems.append(prob)
                labels.append(int(it.get("is_correct", False)))
                ids.append(it.get("item_id"))
                groups.append(gname)
    return {"problems": problems, "traces": traces,
            "labels": np.array(labels, dtype=np.int64),
            "ids": np.array(ids, dtype=object),
            "groups": np.array(groups, dtype=object)}


class ConditionedDataset(Dataset):
    """Encode [CLS] problem [SEP] trace_last_tokens [SEP]."""
    def __init__(self, problems: list[str], traces: list[str],
                 labels: np.ndarray, tokenizer,
                 max_len: int = 512, problem_budget: int = 128):
        self.problems = problems
        self.traces = traces
        self.labels = labels
        self.tok = tokenizer
        self.max_len = max_len
        self.problem_budget = problem_budget

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        prob_txt = self.problems[idx] or ""
        trace_txt = self.traces[idx] or ""

        # Tokenize problem (truncate head preserving essentials - take first N tokens)
        p_ids = self.tok(prob_txt, add_special_tokens=False,
                         truncation=True, max_length=self.problem_budget,
                         return_tensors="pt")["input_ids"][0]

        # Tokenize trace, take tail to fit remaining budget
        remaining = self.max_len - 3 - len(p_ids)   # 3 = CLS + 2 SEPs
        remaining = max(remaining, 32)              # floor
        t_ids_full = self.tok(trace_txt, add_special_tokens=False,
                              truncation=False,
                              return_tensors="pt")["input_ids"][0]
        if len(t_ids_full) > remaining:
            t_ids = t_ids_full[-remaining:]
        else:
            t_ids = t_ids_full

        input_ids = torch.cat([
            torch.tensor([self.tok.cls_token_id], dtype=torch.long),
            p_ids,
            torch.tensor([self.tok.sep_token_id], dtype=torch.long),
            t_ids,
            torch.tensor([self.tok.sep_token_id], dtype=torch.long),
        ])
        # token type ids: 0 for problem, 1 for trace
        type_ids = torch.cat([
            torch.zeros(1 + len(p_ids) + 1, dtype=torch.long),
            torch.ones(len(t_ids) + 1, dtype=torch.long),
        ])
        attn = torch.ones_like(input_ids)
        return {"input_ids": input_ids,
                "attention_mask": attn,
                "token_type_ids": type_ids,
                "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long)}


def collate_pad(batch, pad_id):
    max_len = max(b["input_ids"].size(0) for b in batch)
    n = len(batch)
    ids = torch.full((n, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros(n, max_len, dtype=torch.long)
    typ = torch.zeros(n, max_len, dtype=torch.long)
    lbl = torch.zeros(n, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_ids"].size(0)
        ids[i, :L] = b["input_ids"]
        attn[i, :L] = 1
        typ[i, :L] = b["token_type_ids"]
        lbl[i] = b["labels"]
    return {"input_ids": ids, "attention_mask": attn,
            "token_type_ids": typ, "labels": lbl}


# =============================================================================
# TRAIN ONE FOLD
# =============================================================================

def train_one_fold(model_name, train_ds, val_ds, tokenizer,
                   epochs, batch_size, lr, device, log_prefix=""):
    pad_id = tokenizer.pad_token_id
    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                           num_workers=2, collate_fn=lambda b: collate_pad(b, pad_id))
    vl_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                           num_workers=2, collate_fn=lambda b: collate_pad(b, pad_id))

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)

    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=lr)
    n_steps = len(tr_loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * n_steps), n_steps)

    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_auroc = -1.0
    best_p = None; best_y = None

    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0; n_seen = 0
        for batch in tr_loader:
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

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in vl_loader:
                lbl = batch.pop("labels")
                batch = {k: v.to(device) for k, v in batch.items()}
                if use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        out = model(**batch)
                else:
                    out = model(**batch)
                p = torch.softmax(out.logits.float(), dim=-1)[:, 1].cpu().numpy()
                ps.append(p); ys.append(lbl.numpy())
        ys = np.concatenate(ys); ps = np.concatenate(ps)
        m = evaluate(ys, ps, name=f"epoch_{ep}")
        logger.info(f"  {log_prefix} ep {ep:02d}  loss={avg_loss:.4f}  "
                    f"val_AUROC={m['auroc']:.3f}  ECE={m['ece']:.3f}")
        if m["auroc"] > best_auroc:
            best_auroc = m["auroc"]
            best_p = ps.copy(); best_y = ys.copy()

    del model
    if use_amp:
        torch.cuda.empty_cache()
    return best_y, best_p


# =============================================================================
# DRIVER
# =============================================================================

def run_cv(traces_glob, output, model_name="microsoft/deberta-v3-base",
           epochs=3, batch_size=8, lr=2e-5,
           max_len=512, problem_budget=128,
           n_splits=5, seed=42):
    data = load_items(traces_glob)
    probs, traces, labels = data["problems"], data["traces"], data["labels"]
    groups, ids = data["groups"], data["ids"]
    logger.info(f"Loaded {len(labels)} items. Pos rate: {labels.mean():.3f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        train_ds = ConditionedDataset(
            [probs[i] for i in tr_idx], [traces[i] for i in tr_idx],
            labels[tr_idx], tokenizer,
            max_len=max_len, problem_budget=problem_budget,
        )
        val_ds = ConditionedDataset(
            [probs[i] for i in te_idx], [traces[i] for i in te_idx],
            labels[te_idx], tokenizer,
            max_len=max_len, problem_budget=problem_budget,
        )
        y_, p_ = train_one_fold(model_name, train_ds, val_ds, tokenizer,
                                epochs, batch_size, lr, device,
                                log_prefix=f"fold{fold + 1}")
        oof_prob[te_idx] = p_
        oof_fold[te_idx] = fold
        fm = evaluate(y_, p_, name=f"fold_{fold + 1}")
        fold_metrics.append(fm)
        all_y.append(y_); all_p.append(p_)
        logger.info(f"  -> Fold {fold + 1} AUROC={fm['auroc']:.3f}  ECE={fm['ece']:.3f}")

    all_y = np.concatenate(all_y); all_p = np.concatenate(all_p)
    overall = evaluate(all_y, all_p, name="overall")
    summary = aggregate_folds(fold_metrics)

    oof_path = output.replace(".json", "_oof.npz")
    np.savez_compressed(oof_path,
        item_ids=ids, groups=groups,
        y_true=labels, oof_prob=oof_prob, oof_fold=oof_fold,
        seed=np.array([seed]), n_splits=np.array([n_splits]))
    logger.info(f"Saved OOF preds: {oof_path}")

    results = {
        "model": model_name,
        "conditioned": True,
        "n_splits": n_splits,
        "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "max_len": max_len, "problem_budget": problem_budget,
        "n_samples": int(len(labels)),
        "base_accuracy": float(labels.mean()),
        "fold_metrics": fold_metrics,
        "summary": summary,
        "overall": overall,
        "oof_path": oof_path,
    }
    save_results(output, results)
    print("\n=== DeBERTa-CONDITIONED CV summary ===")
    for k in ["auroc_mean","auprc_mean","ece_mean","accuracy_at_80_mean","accuracy_at_90_mean"]:
        print(f"  {k:25s} {summary.get(k,0):.4f}  ± {summary.get(k.replace('_mean','_std'),0):.4f}")
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
    p.add_argument("--problem-budget", type=int, default=128)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    run_cv(args.traces_glob, args.output, model_name=args.model,
           epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
           max_len=args.max_len, problem_budget=args.problem_budget,
           n_splits=args.n_splits, seed=args.seed)


if __name__ == "__main__":
    main()
