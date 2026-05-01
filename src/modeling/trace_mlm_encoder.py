"""
trace_mlm_encoder.py — Masked-step-reconstruction pretraining for reasoning traces.

Trains a small Transformer encoder on per-step MiniLM (or bge) embeddings
using a continuous-MLM objective (mask 15% of steps with a learnable [MASK]
vector, reconstruct them), then fine-tunes for correctness prediction.

Hypothesis (Approach 6b): StepTF's random initialization forces the encoder
to learn step-transition patterns from scratch. A trace-specific pretraining
step should give it a better starting point — the encoder learns "what makes
step N plausibly follow step N-1" without labels. When fine-tuned on
correctness prediction, this should yield a better classifier than random-
init StepTF.

Architecture: same as StepTransformer (4-layer Transformer, 256-d hidden,
4 heads) but with two heads:
  - reconstruct_head: hidden -> emb_dim (MSE loss, used during pretraining)
  - classifier_head:  hidden -> 1 (BCE loss, used during fine-tuning)

Training protocol:
  Stage 1 — PRETRAIN:
    Input: sequence of step embeddings (all traces, no labels used)
    - Randomly mask 15% of step positions with learnable [MASK] vector
    - Model reconstructs the masked embeddings; MSE loss on masked positions only
    - 30 epochs, ~10 min on 1x RTX PRO 6000

  Stage 2 — FINE-TUNE (per CV fold):
    Load pretrained weights
    - Encoder: lr=1e-5 (slow fine-tune)
    - Classifier: lr=3e-4 (fresh head)
    - 15 epochs, ~5 min per fold x 5 folds = 25 min

Caveat: Pretraining uses all 6378 traces — same pool fine-tuning sees. This
is the standard "unsupervised pretraining on unlabeled data + supervised
fine-tuning" protocol common in ML, but it means pretraining did see
(unlabeled) versions of future test items. Document in paper.

Usage:
    # Pretrain once on pooled embeddings:
    PYTHONPATH=. python src/modeling/trace_mlm_encoder.py pretrain \\
        --npz-glob "data/step_embeddings/*.npz" \\
        --output-ckpt results/month2_v2/trace_mlm_pretrained.pt \\
        --epochs 30 --batch-size 32 --lr 3e-4

    # Fine-tune 5-fold CV:
    PYTHONPATH=. python src/modeling/trace_mlm_encoder.py finetune \\
        --npz-glob "data/step_embeddings/*.npz" \\
        --checkpoint results/month2_v2/trace_mlm_pretrained.pt \\
        --output results/month2_v2/trace_mlm_pooled.json \\
        --epochs 15 --batch-size 32 --lr-encoder 1e-5 --lr-head 3e-4

    # Baseline (no pretraining, random init):
    PYTHONPATH=. python src/modeling/trace_mlm_encoder.py finetune \\
        --npz-glob "data/step_embeddings/*.npz" \\
        --output results/month2_v2/trace_mlm_no_pretrain.json \\
        --epochs 15 --batch-size 32 --lr-encoder 1e-5 --lr-head 3e-4
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.modeling.cv_utils import (
    aggregate_folds, evaluate, save_results, stratified_split,
)
from src.modeling.step_transformer import (
    SinusoidalPositionalEmbedding, EMB_DIM,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATASET (embedding-only; step_types and labels used in different stages)
# =============================================================================

class TraceEmbeddingDataset(Dataset):
    """Dataset returning just the step-embedding sequence per item.
    Used for pretraining (labels ignored) and fine-tuning (labels used)."""

    def __init__(self, embeddings: list[np.ndarray], labels: np.ndarray,
                 max_len: int = 256):
        self.embs = embeddings
        self.labels = labels
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        emb = self.embs[idx]
        if len(emb) > self.max_len:
            emb = emb[:self.max_len]
        return {
            "emb": torch.from_numpy(np.ascontiguousarray(emb)).float(),
            "y":   torch.tensor(int(self.labels[idx]), dtype=torch.float32),
        }


def collate(batch: list[dict]) -> dict:
    """Pad to batch max length; build mask (True=valid)."""
    n = len(batch)
    max_len = max(b["emb"].shape[0] for b in batch)
    if max_len == 0:
        max_len = 1
    d = batch[0]["emb"].shape[1] if batch[0]["emb"].numel() > 0 else EMB_DIM

    emb = torch.zeros(n, max_len, d, dtype=torch.float32)
    mask = torch.zeros(n, max_len, dtype=torch.bool)
    y = torch.zeros(n, dtype=torch.float32)

    for i, b in enumerate(batch):
        L = b["emb"].shape[0]
        if L > 0:
            emb[i, :L] = b["emb"]
            mask[i, :L] = True
        y[i] = b["y"]

    return {"emb": emb, "mask": mask, "y": y}


# =============================================================================
# MODEL
# =============================================================================

class TraceMLMEncoder(nn.Module):
    """Transformer encoder with two heads:
      - reconstruct_head: for MLM pretraining (hidden -> emb_dim)
      - classifier:       for fine-tuning     (hidden -> 1)

    During pretraining: select 15% of step positions, replace with [MASK]
    vector, encode the sequence, and reconstruct the original embeddings
    at the masked positions via MSE loss.

    During fine-tuning: no masking; use the CLS position's hidden state for
    binary classification."""

    def __init__(self, emb_dim: int = EMB_DIM, hidden: int = 256,
                 n_layers: int = 4, n_heads: int = 4, dropout: float = 0.2,
                 max_len: int = 1024):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden = hidden

        # Project raw step embedding to hidden dim
        self.proj = nn.Linear(emb_dim, hidden)

        # Learnable [CLS] and [MASK] vectors in HIDDEN space (not emb_dim)
        # We compare reconstructions in emb_dim space via self.reconstruct_head
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.pos = SinusoidalPositionalEmbedding(hidden, max_len=max_len)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads,
            dim_feedforward=hidden * 4, dropout=dropout,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(hidden)
        self.reconstruct_head = nn.Linear(hidden, emb_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward_encoder(self, emb: torch.Tensor, mask: torch.Tensor,
                        mlm_mask: torch.Tensor = None) -> torch.Tensor:
        """Run the encoder. If mlm_mask is provided (shape (B, L) bool, True
        where the position is masked), those positions receive the [MASK]
        vector instead of their projected embedding.

        Returns: (B, L+1, hidden) — the first position is [CLS].
        """
        B, L, _ = emb.shape
        # Project to hidden dim
        x = self.proj(emb)  # (B, L, H)
        # Replace masked positions with the learnable [MASK] vector
        if mlm_mask is not None:
            mask_vec = self.mask_token.expand(B, L, -1)
            x = torch.where(mlm_mask.unsqueeze(-1), mask_vec, x)
        # Prepend CLS
        cls = self.cls.expand(B, -1, -1)  # (B, 1, H)
        x = torch.cat([cls, x], dim=1)     # (B, L+1, H)
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)  # (B, L+1)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=~full_mask)
        return x

    def forward_reconstruct(self, emb: torch.Tensor, mask: torch.Tensor,
                            mlm_mask: torch.Tensor) -> torch.Tensor:
        """For pretraining. Returns predicted embeddings at all (non-CLS)
        positions; caller computes MSE loss only at mlm_mask positions."""
        x = self.forward_encoder(emb, mask, mlm_mask=mlm_mask)  # (B, L+1, H)
        # Drop the CLS position; reconstruct only at step positions
        x = self.norm(x[:, 1:])      # (B, L, H)
        return self.reconstruct_head(x)  # (B, L, emb_dim)

    def forward_classify(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """For fine-tuning / inference. Returns (B,) logits."""
        x = self.forward_encoder(emb, mask, mlm_mask=None)  # (B, L+1, H)
        cls_out = self.norm(x[:, 0])   # (B, H)
        return self.classifier(cls_out).squeeze(-1)  # (B,)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_pooled(npz_paths: list[str]) -> dict:
    all_emb, all_y, all_id, all_group = [], [], [], []
    for p in npz_paths:
        z = np.load(p, allow_pickle=True)
        embs = list(z["embeddings"])
        labels = z["is_correct"].astype(np.int64)
        ids = z["item_ids"]
        gname = os.path.basename(p).replace(".npz", "")
        all_emb.extend(embs)
        all_y.extend(list(labels))
        all_id.extend(list(ids))
        all_group.extend([gname] * len(labels))
    return {
        "embeddings": all_emb,
        "labels":     np.array(all_y, dtype=np.int64),
        "item_ids":   np.array(all_id, dtype=object),
        "groups":     np.array(all_group, dtype=object),
    }


def auto_detect_emb_dim(embeddings: list) -> int:
    for e in embeddings:
        if e is not None and len(e) > 0:
            return int(e.shape[1])
    return EMB_DIM


# =============================================================================
# PRETRAINING
# =============================================================================

def pretrain(args):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    npz_paths = sorted(glob.glob(args.npz_glob))
    logger.info(f"Loading {len(npz_paths)} .npz files...")
    data = load_pooled(npz_paths)
    embs = data["embeddings"]
    labels = data["labels"]
    emb_dim = auto_detect_emb_dim(embs)
    logger.info(f"  n_traces={len(embs)}  emb_dim={emb_dim}")

    ds = TraceEmbeddingDataset(embs, labels)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=2, pin_memory=True)

    model = TraceMLMEncoder(emb_dim=emb_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {n_params:,} parameters")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n_steps = len(loader) * args.epochs
    # Cosine schedule
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    mask_rate = args.mask_rate
    logger.info(f"Starting pretraining: {args.epochs} epochs, mask_rate={mask_rate}")

    for ep in range(1, args.epochs + 1):
        model.train()
        tot_loss, n_masked_seen = 0.0, 0
        for batch in loader:
            emb = batch["emb"].to(device, non_blocking=True)   # (B, L, D)
            mask = batch["mask"].to(device, non_blocking=True)  # (B, L)

            # Sample mlm_mask: 15% of VALID positions
            B, L, _ = emb.shape
            rand = torch.rand(B, L, device=device)
            mlm_mask = (rand < mask_rate) & mask  # only mask valid positions
            # Guarantee at least 1 masked position per sequence (sample longest valid position)
            # If a sample has no masked position, loss is 0 and we skip gradient on it
            n_masked = int(mlm_mask.sum().item())
            if n_masked == 0:
                continue

            recon = model.forward_reconstruct(emb, mask, mlm_mask)  # (B, L, D)
            # MSE loss at masked positions only
            per_pos_loss = F.mse_loss(recon, emb, reduction="none").mean(dim=-1)  # (B, L)
            loss = (per_pos_loss * mlm_mask.float()).sum() / max(n_masked, 1)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot_loss += float(loss.item()) * n_masked
            n_masked_seen += n_masked

        avg = tot_loss / max(n_masked_seen, 1)
        logger.info(f"  ep {ep:02d}/{args.epochs}  loss={avg:.6f}  "
                    f"lr={sched.get_last_lr()[0]:.2e}")

    os.makedirs(os.path.dirname(args.output_ckpt) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "emb_dim": emb_dim,
                "hidden": 256, "n_layers": 4, "n_heads": 4,
                "mask_rate": mask_rate,
                "pretrain_epochs": args.epochs,
                "n_traces": len(embs)},
               args.output_ckpt)
    logger.info(f"Saved pretrained ckpt to {args.output_ckpt}")


# =============================================================================
# FINE-TUNING
# =============================================================================

def finetune_one_fold(
    model: TraceMLMEncoder,
    train_ds: TraceEmbeddingDataset, val_ds: TraceEmbeddingDataset,
    epochs: int, batch_size: int,
    lr_encoder: float, lr_head: float,
    device: str, pos_weight: float, log_prefix: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate, num_workers=2)

    # Two param groups: encoder body gets small LR, classifier head gets normal LR
    head_params = list(model.classifier.parameters())
    head_ids = set(id(p) for p in head_params)
    encoder_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([
        {"params": encoder_params, "lr": lr_encoder},
        {"params": head_params,    "lr": lr_head},
    ], weight_decay=1e-4)

    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    # ----- LEAKAGE-SAFE OOF PROTOCOL -----
    # Return LAST epoch's predictions, not best-by-val-AUROC (which is
    # test-set epoch selection). Val metrics logged for monitoring only.
    last_probs, last_labels = None, None
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss, n_seen = 0.0, 0
        for batch in train_loader:
            emb = batch["emb"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            logit = model.forward_classify(emb, mask)
            loss = bce(logit, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss.item()) * y.size(0)
            n_seen += y.size(0)
        ep_loss /= max(n_seen, 1)

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in val_loader:
                emb = batch["emb"].to(device)
                mask = batch["mask"].to(device)
                logit = model.forward_classify(emb, mask)
                ps.append(torch.sigmoid(logit).cpu().numpy())
                ys.append(batch["y"].numpy())
        ys = np.concatenate(ys); ps = np.concatenate(ps)
        m = evaluate(ys, ps, name=f"epoch_{ep}")
        if ep == 1 or ep == epochs or ep % 5 == 0:
            logger.info(f"  {log_prefix}ep {ep:02d}  loss={ep_loss:.4f}  "
                        f"val_AUROC={m['auroc']:.3f}  [monitor only]")
        # Always overwrite — last-epoch protocol
        last_probs = ps.copy(); last_labels = ys.copy()

    return last_labels, last_probs


def finetune(args):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    npz_paths = sorted(glob.glob(args.npz_glob))
    data = load_pooled(npz_paths)
    embs = data["embeddings"]
    labels = data["labels"]
    groups = data["groups"]
    item_ids = data["item_ids"]
    emb_dim = auto_detect_emb_dim(embs)
    logger.info(f"Loaded {len(embs)} traces; emb_dim={emb_dim}; "
                f"pos_rate={labels.mean():.3f}")

    oof_prob = np.full(len(labels), np.nan, dtype=np.float32)
    oof_fold = np.full(len(labels), -1, dtype=np.int32)
    fold_metrics = []
    all_y, all_p = [], []

    for fold, (tr, te) in enumerate(stratified_split(
            labels, group_id=groups if len(set(groups)) > 1 else None,
            n_splits=args.n_splits, seed=args.seed)):
        logger.info("")
        logger.info(f"=== Fold {fold + 1}/{args.n_splits}  train={len(tr)}  val={len(te)} ===")
        tr_emb = [embs[i] for i in tr]; tr_y = labels[tr]
        te_emb = [embs[i] for i in te]; te_y = labels[te]
        train_ds = TraceEmbeddingDataset(tr_emb, tr_y)
        val_ds = TraceEmbeddingDataset(te_emb, te_y)

        n_pos = max(int(tr_y.sum()), 1)
        n_neg = max(len(tr_y) - n_pos, 1)
        pos_weight = n_neg / n_pos

        # Fresh model for each fold (so fine-tuning is truly per-fold)
        model = TraceMLMEncoder(emb_dim=emb_dim).to(device)
        if args.checkpoint:
            ckpt = torch.load(args.checkpoint, map_location=device)
            # Load only encoder + [MASK]/[CLS]/proj/pos; classifier stays fresh
            sd = ckpt["state_dict"]
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if missing:
                logger.info(f"  ckpt loaded; missing keys (fresh-init'd head): "
                            f"{[k for k in missing if 'classifier' in k]}")
            logger.info(f"  LOADED pretrained ckpt: {args.checkpoint}")
        else:
            logger.info("  NO pretrained ckpt supplied — fine-tuning from random init")

        labels_, probs_ = finetune_one_fold(
            model, train_ds, val_ds,
            epochs=args.epochs, batch_size=args.batch_size,
            lr_encoder=args.lr_encoder, lr_head=args.lr_head,
            device=device, pos_weight=pos_weight,
            log_prefix=f"fold{fold + 1} ",
        )
        oof_prob[te] = probs_
        oof_fold[te] = fold
        fm = evaluate(labels_, probs_, name=f"fold_{fold + 1}")
        fold_metrics.append(fm)
        all_y.append(labels_); all_p.append(probs_)
        logger.info(f"  -> Fold {fold + 1} best  AUROC={fm['auroc']:.3f}  ECE={fm['ece']:.3f}")

        del model
        if device == "cuda": torch.cuda.empty_cache()

    all_y = np.concatenate(all_y); all_p = np.concatenate(all_p)
    overall = evaluate(all_y, all_p, name="overall")
    summary = aggregate_folds(fold_metrics)

    oof_path = args.output.replace(".json", "_oof.npz")
    os.makedirs(os.path.dirname(oof_path) or ".", exist_ok=True)
    np.savez_compressed(oof_path,
        item_ids=item_ids, groups=groups,
        y_true=labels, oof_prob=oof_prob, oof_fold=oof_fold,
        seed=np.array([args.seed]), n_splits=np.array([args.n_splits]))
    logger.info(f"Saved OOF: {oof_path}")

    results = {
        "model": "TraceMLMEncoder",
        "pretrained": bool(args.checkpoint),
        "checkpoint": args.checkpoint,
        "n_splits": args.n_splits,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr_encoder": args.lr_encoder,
        "lr_head": args.lr_head,
        "n_samples": int(len(labels)),
        "emb_dim": emb_dim,
        "base_accuracy": float(labels.mean()),
        "fold_metrics": fold_metrics,
        "summary": summary,
        "overall": overall,
        "oof_path": oof_path,
    }
    save_results(args.output, results)

    print("\n=== TraceMLMEncoder fine-tune summary ===")
    for k in ["auroc_mean", "auprc_mean", "ece_mean",
              "accuracy_at_80_mean", "accuracy_at_90_mean", "prr_mean"]:
        print(f"  {k:25s} {summary.get(k, float('nan')):.4f}  "
              f"± {summary.get(k.replace('_mean', '_std'), 0):.4f}")


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- pretrain ----
    pp = sub.add_parser("pretrain", help="MLM pretrain on step embeddings")
    pp.add_argument("--npz-glob", required=True)
    pp.add_argument("--output-ckpt", required=True)
    pp.add_argument("--epochs", type=int, default=30)
    pp.add_argument("--batch-size", type=int, default=32)
    pp.add_argument("--lr", type=float, default=3e-4)
    pp.add_argument("--mask-rate", type=float, default=0.15)
    pp.add_argument("--seed", type=int, default=42)

    # ---- finetune ----
    pf = sub.add_parser("finetune", help="Fine-tune for correctness prediction")
    pf.add_argument("--npz-glob", required=True)
    pf.add_argument("--output", required=True)
    pf.add_argument("--checkpoint", default=None,
                    help="Path to pretrained ckpt (.pt). Omit to fine-tune from random init.")
    pf.add_argument("--epochs", type=int, default=15)
    pf.add_argument("--batch-size", type=int, default=32)
    pf.add_argument("--lr-encoder", type=float, default=1e-5)
    pf.add_argument("--lr-head", type=float, default=3e-4)
    pf.add_argument("--n-splits", type=int, default=5)
    pf.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    if args.cmd == "pretrain":
        pretrain(args)
    elif args.cmd == "finetune":
        finetune(args)


if __name__ == "__main__":
    main()
