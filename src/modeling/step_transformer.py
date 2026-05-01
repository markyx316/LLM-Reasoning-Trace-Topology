"""
step_transformer.py - Step-Sequence Transformer for trace correctness prediction.

Architecture:
  [CLS] step_emb_1 step_emb_2 ... step_emb_N
       (each step_emb = projection of frozen MiniLM 384-d
                       + step-type embedding
                       + sinusoidal position)
   ↓ Transformer encoder (small: 4 layers, hidden 256, 4 heads)
   ↓ take CLS hidden
   ↓ MLP (256 -> 64 -> 1)
   ↓ sigmoid -> P(is_correct)

Trained on POOLED 8 datasets (~5900 items) with stratified 5-fold CV.
Input is the precomputed .npz files from build_step_embeddings.py.

Usage:
    PYTHONPATH=. python src/modeling/step_transformer.py \
        --npz-glob "data/step_embeddings/*.npz" \
        --output results/month2/step_transformer_pooled.json \
        --epochs 15 --batch-size 32

    # Per-dataset evaluation (no pooling):
    PYTHONPATH=. python src/modeling/step_transformer.py \
        --npz data/step_embeddings/math500_qwen7b.npz \
        --output results/month2/step_transformer_math500_qwen7b.json
"""

from __future__ import annotations

import argparse
import glob
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.modeling.cv_utils import (
    aggregate_folds, evaluate, load_pooled_npz, save_results, stratified_split,
)

logger = logging.getLogger(__name__)

# Must match scripts/build_step_embeddings.py
PAD_TYPE = 0
N_TYPES = 7       # 6 BehaviorType (F, V, X, R, H, C) + PAD
EMB_DIM = 384     # MiniLM


# =============================================================================
# DATASET
# =============================================================================

class StepDataset(Dataset):
    """In-memory dataset of (step_embeddings, step_types, label)."""

    def __init__(self, embeddings: list, step_types: list, labels: np.ndarray,
                 max_len: int = 256):
        self.embs = embeddings
        self.types = step_types
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emb = self.embs[idx]
        typ = self.types[idx]
        if len(emb) > self.max_len:
            emb = emb[:self.max_len]
            typ = typ[:self.max_len]
        return {
            "emb": torch.from_numpy(np.ascontiguousarray(emb)).float(),
            "typ": torch.from_numpy(np.ascontiguousarray(typ)).long(),
            "y":   torch.tensor(int(self.labels[idx]), dtype=torch.float32),
        }


def collate(batch: list[dict]) -> dict:
    """Pad sequences to batch max length; build attention mask."""
    n = len(batch)
    max_len = max(b["emb"].shape[0] for b in batch)
    if max_len == 0:
        max_len = 1  # avoid empty
    d = batch[0]["emb"].shape[1] if batch[0]["emb"].numel() > 0 else EMB_DIM

    emb = torch.zeros(n, max_len, d, dtype=torch.float32)
    typ = torch.full((n, max_len), PAD_TYPE, dtype=torch.long)
    mask = torch.zeros(n, max_len, dtype=torch.bool)
    y = torch.zeros(n, dtype=torch.float32)

    for i, b in enumerate(batch):
        L = b["emb"].shape[0]
        if L > 0:
            emb[i, :L] = b["emb"]
            typ[i, :L] = b["typ"]
            mask[i, :L] = True
        y[i] = b["y"]

    return {"emb": emb, "typ": typ, "mask": mask, "y": y}


# =============================================================================
# MODEL
# =============================================================================

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class StepTransformer(nn.Module):
    def __init__(self, emb_dim: int = EMB_DIM, hidden: int = 256,
                 n_layers: int = 4, n_heads: int = 4, n_types: int = N_TYPES,
                 dropout: float = 0.2, max_len: int = 1024):
        super().__init__()
        self.proj = nn.Linear(emb_dim, hidden)
        self.type_emb = nn.Embedding(n_types, hidden, padding_idx=PAD_TYPE)
        self.pos = SinusoidalPositionalEmbedding(hidden, max_len=max_len)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.cls, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads,
            dim_feedforward=hidden * 4, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, emb: torch.Tensor, typ: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # emb: (B, L, emb_dim), typ: (B, L), mask: (B, L) bool, True=valid
        b = emb.size(0)
        x = self.proj(emb) + self.type_emb(typ)         # (B, L, H)
        # Prepend CLS
        cls = self.cls.expand(b, -1, -1)                # (B, 1, H)
        x = torch.cat([cls, x], dim=1)                  # (B, L+1, H)
        cls_mask = torch.ones(b, 1, dtype=torch.bool, device=mask.device)
        m = torch.cat([cls_mask, mask], dim=1)          # (B, L+1)

        x = self.pos(x)
        # nn.TransformerEncoder expects key_padding_mask = True for positions to MASK OUT
        key_padding = ~m
        x = self.encoder(x, src_key_padding_mask=key_padding)

        cls_out = self.norm(x[:, 0])                    # (B, H)
        logit = self.head(cls_out).squeeze(-1)          # (B,)
        return logit


# =============================================================================
# TRAIN / EVAL ONE FOLD
# =============================================================================

def train_one_fold(
    train_ds: StepDataset, val_ds: StepDataset,
    epochs: int, batch_size: int, lr: float, device: str,
    pos_weight: float = 1.0,
    log_prefix: str = "",
    emb_dim: int = EMB_DIM,
) -> tuple[np.ndarray, np.ndarray]:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate, num_workers=2)

    model = StepTransformer(emb_dim=emb_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    # ----- LEAKAGE-SAFE OOF PROTOCOL -----
    # The OOF probability we return becomes an input to downstream stackers
    # (hybrid_route_ab, tune_hybrid). Selecting the "best" epoch by
    # val_AUROC on the held-out fold is test-set model selection — it
    # inflates the reported OOF AUROC and makes the probs not reusable as
    # leakage-free stacker inputs. Fix: return LAST epoch's predictions,
    # period. Per-epoch val metrics are still logged for monitoring only.
    last_probs = None
    last_labels = None

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for batch in train_loader:
            emb = batch["emb"].to(device, non_blocking=True)
            typ = batch["typ"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            logit = model(emb, typ, mask)
            loss = bce(logit, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * y.size(0)
        ep_loss /= len(train_ds)

        # Validate
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in val_loader:
                emb = batch["emb"].to(device); typ = batch["typ"].to(device)
                mask = batch["mask"].to(device)
                logit = model(emb, typ, mask)
                ps.append(torch.sigmoid(logit).cpu().numpy())
                ys.append(batch["y"].numpy())
        ys = np.concatenate(ys); ps = np.concatenate(ps)
        m = evaluate(ys, ps, name=f"epoch_{ep}")
        logger.info(f"  {log_prefix} ep {ep:02d}  loss={ep_loss:.4f}  "
                    f"val_AUROC={m['auroc']:.3f}  val_AUPRC={m['auprc']:.3f}  "
                    f"val_ECE={m['ece']:.3f}  [logged for monitoring only]")
        # Always overwrite — we return the LAST epoch's predictions
        last_probs = ps.copy()
        last_labels = ys.copy()

    return last_labels, last_probs


# =============================================================================
# CV ORCHESTRATION
# =============================================================================

def run_cv(npz_paths: list[str], output_path: str,
           epochs: int = 15, batch_size: int = 32, lr: float = 3e-4,
           n_splits: int = 5, seed: int = 42):
    data = load_pooled_npz(npz_paths)
    embs, types, y, groups = (data["embeddings"], data["step_types"],
                              data["labels"], data["groups"])
    item_ids = data["item_ids"]
    logger.info(f"Loaded {len(y)} items from {len(npz_paths)} files. "
                f"Pos rate: {y.mean():.3f}")

    # Auto-detect embedding dimension from the first non-empty item so the
    # script works with both MiniLM (384-d) and bge-base (768-d) .npz files.
    emb_dim = EMB_DIM
    for e in embs:
        if e is not None and len(e) > 0:
            emb_dim = int(e.shape[1])
            break
    logger.info(f"Auto-detected embedding dim: {emb_dim}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    n_pos = max(int(y.sum()), 1)
    n_neg = max(int(len(y) - n_pos), 1)
    pos_weight = n_neg / n_pos
    logger.info(f"pos_weight = {pos_weight:.3f}")

    # Out-of-fold (OOF) predictions for stacking
    oof_prob = np.full(len(y), np.nan, dtype=np.float32)
    oof_fold = np.full(len(y), -1, dtype=np.int32)

    fold_metrics = []
    all_y, all_p = [], []
    for fold, (tr_idx, te_idx) in enumerate(stratified_split(
            y, group_id=groups if len(set(groups)) > 1 else None,
            n_splits=n_splits, seed=seed)):
        logger.info(f"\n=== Fold {fold + 1}/{n_splits}  "
                    f"train={len(tr_idx)}  val={len(te_idx)} ===")
        tr_emb = [embs[i] for i in tr_idx]
        tr_typ = [types[i] for i in tr_idx]
        tr_y = y[tr_idx]
        te_emb = [embs[i] for i in te_idx]
        te_typ = [types[i] for i in te_idx]
        te_y = y[te_idx]

        train_ds = StepDataset(tr_emb, tr_typ, tr_y)
        val_ds = StepDataset(te_emb, te_typ, te_y)

        labels, probs = train_one_fold(
            train_ds, val_ds, epochs=epochs, batch_size=batch_size,
            lr=lr, device=device, pos_weight=pos_weight,
            log_prefix=f"fold{fold + 1}", emb_dim=emb_dim,
        )
        oof_prob[te_idx] = probs
        oof_fold[te_idx] = fold
        fm = evaluate(labels, probs, name=f"fold_{fold + 1}")
        fold_metrics.append(fm)
        all_y.append(labels); all_p.append(probs)
        logger.info(f"  -> Fold {fold + 1} best  AUROC={fm['auroc']:.3f}  "
                    f"AUPRC={fm['auprc']:.3f}  ECE={fm['ece']:.3f}")

    all_y = np.concatenate(all_y); all_p = np.concatenate(all_p)
    overall = evaluate(all_y, all_p, name="overall")
    summary = aggregate_folds(fold_metrics)

    # Save OOF predictions alongside results JSON
    oof_path = output_path.replace(".json", "_oof.npz")
    np.savez_compressed(oof_path,
        item_ids=item_ids, groups=groups,
        y_true=y, oof_prob=oof_prob, oof_fold=oof_fold,
        seed=np.array([seed]), n_splits=np.array([n_splits]))
    logger.info(f"Saved OOF preds: {oof_path}")

    results = {
        "model": "StepTransformer",
        "n_splits": n_splits,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "n_samples": int(len(y)),
        "base_accuracy": float(y.mean()),
        "fold_metrics": fold_metrics,
        "summary": summary,
        "overall": overall,
        "oof_path": oof_path,
    }
    save_results(output_path, results)

    print("\n=== Step Transformer CV summary ===")
    for k in ["auroc_mean", "auprc_mean", "ece_mean",
              "accuracy_at_80_mean", "accuracy_at_90_mean"]:
        print(f"  {k:25s} {summary.get(k, float('nan')):.4f}  "
              f"± {summary.get(k.replace('_mean','_std'), 0):.4f}")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", nargs="+", default=None,
                   help="Explicit list of .npz files")
    p.add_argument("--npz-glob", default=None,
                   help="Glob pattern to find .npz files")
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    if args.npz_glob:
        paths = sorted(glob.glob(args.npz_glob))
    elif args.npz:
        paths = args.npz
    else:
        p.error("--npz or --npz-glob")
    logger.info(f"Loading from {len(paths)} npz files: {paths}")

    run_cv(paths, args.output,
           epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
           n_splits=args.n_splits, seed=args.seed)


if __name__ == "__main__":
    main()
