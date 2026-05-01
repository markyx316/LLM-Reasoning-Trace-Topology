"""
behavior_seq_lm.py — Content-free sequence model over cognitive-behavior tokens.

Trains a tiny BiGRU on the ordered F/V/X/R/H/C ordinal sequence (vocab size 7
including PAD=0) and predicts is_correct. Reads the behavior tokens from the
same .npz files StepTransformer uses (data/step_embeddings/*.npz contains
step_types — the ordinal behavior sequence per trace, from the 6-class
rule_based_parser).

Why this exists (Approach 2 of the post-result pivot):
  Handcrafted-25 features count behaviors (prop_forward, prop_verify, etc.)
  but lose all ordering information. StepTF sees semantic embeddings plus
  behavior type, but the type-embedding channel was empirically shown to
  carry ~zero signal on top of MiniLM. Nobody has explicitly modeled the
  BEHAVIOR SEQUENCE as a sequence. This module fills that gap:
  a ~85K-parameter model on a 6-token vocabulary, isolating the pure
  structural-ordering signal from all content.

Hypotheses:
  H1 (strong): Order-aware model beats counts (handcrafted_only ≈ 0.66 pooled).
               Would indicate meaningful sequential patterns exist.
  H2 (weak):   Order-aware model ≈ counts. "It's all in the counts" —
               publishable null result with large implications.
  H3 (stretch): Order-aware model transfers better cross-dataset because it
                captures model-agnostic reasoning flow.

Output format: same as StepTransformer — pooled+per-dataset JSONs plus
OOF .npz files (item_ids, groups, y_true, oof_prob, oof_fold) that plug
straight into hybrid.py as a new signal.

Usage:
    # Pooled 5-fold CV across all 8 datasets
    PYTHONPATH=. python src/modeling/behavior_seq_lm.py \\
        --npz-glob "data/step_embeddings/*.npz" \\
        --output results/month2_v2/behavior_seq_lm_pooled.json \\
        --epochs 30 --batch-size 32 --lr 3e-4

    # Per-dataset (for the per-dataset analysis)
    PYTHONPATH=. python src/modeling/behavior_seq_lm.py \\
        --npz data/step_embeddings/math500_qwen7b.npz \\
        --output results/month2_v2/behavior_seq_lm_math500_qwen7b.json
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
    aggregate_folds, evaluate, save_results, stratified_split,
)

logger = logging.getLogger(__name__)


# --- Vocab ---
PAD_TYPE = 0
VOCAB_SIZE = 7   # PAD + 6 behaviors (F, V, X, R, H, C)
MAX_LEN = 256    # truncate very long traces here; matches StepTF default


# =============================================================================
# DATASET
# =============================================================================

class BehaviorSeqDataset(Dataset):
    """Per-item dataset: (behavior_token_sequence, binary_label)."""

    def __init__(self, step_types_list: list[np.ndarray], labels: np.ndarray,
                 max_len: int = MAX_LEN):
        self.types = step_types_list
        self.labels = labels
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        typ = self.types[idx]
        if len(typ) > self.max_len:
            typ = typ[:self.max_len]
        return {
            "typ": torch.from_numpy(np.ascontiguousarray(typ)).long(),
            "y":   torch.tensor(int(self.labels[idx]), dtype=torch.float32),
        }


def collate(batch: list[dict]) -> dict:
    """Pad sequences to the batch max, build attention mask."""
    n = len(batch)
    max_len = max(b["typ"].shape[0] for b in batch)
    if max_len == 0:
        max_len = 1
    typ = torch.full((n, max_len), PAD_TYPE, dtype=torch.long)
    mask = torch.zeros(n, max_len, dtype=torch.bool)
    y = torch.zeros(n, dtype=torch.float32)
    for i, b in enumerate(batch):
        L = b["typ"].shape[0]
        if L > 0:
            typ[i, :L] = b["typ"]
            mask[i, :L] = True
        y[i] = b["y"]
    return {"typ": typ, "mask": mask, "y": y}


# =============================================================================
# MODEL
# =============================================================================

class BehaviorSeqLM(nn.Module):
    """Content-free BiGRU over F/V/X/R/H/C ordinal behavior sequences.

    Parameter count @ defaults:
       embed (7 × 32)               = 224
       BiGRU (2-layer, 32 → 64 × 2) ≈ 75k
       head (128 → 64 → 1)          ≈ 8.4k
       --------------------------------
       total                         ≈ 84k params

    In contrast StepTransformer has ~3.4M params and consumes 384-d MiniLM
    embeddings; this is ~40× smaller and operates on a 6-token vocabulary.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, embed_dim: int = 32,
                 hidden: int = 64, n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TYPE)
        self.rnn = nn.GRU(
            embed_dim, hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Linear(2 * hidden, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, typ: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        typ: (B, L) long tensor of behavior ordinals (0=PAD, 1..6 = F V X R H C)
        mask: (B, L) bool tensor, True at valid positions

        Returns: (B,) logits
        """
        x = self.embed(typ)                     # (B, L, D)
        x, _ = self.rnn(x)                      # (B, L, 2H)
        # Masked mean pool
        m = mask.unsqueeze(-1).float()
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        logit = self.head(pooled).squeeze(-1)   # (B,)
        return logit


# =============================================================================
# DATA LOADING
# =============================================================================

def load_pooled_behavior_sequences(npz_paths: list[str]) -> dict:
    """Load step_types arrays + labels + item_ids from multiple .npz files and
    concatenate. Returns a dict with embeddings-free data (we only need types)."""
    all_types, all_y, all_id, all_group = [], [], [], []
    for p in npz_paths:
        z = np.load(p, allow_pickle=True)
        step_types = list(z["step_types"])
        labels     = z["is_correct"].astype(np.int64)
        item_ids   = z["item_ids"]
        gname = os.path.basename(p).replace(".npz", "")
        all_types.extend(step_types)
        all_y.extend(list(labels))
        all_id.extend(list(item_ids))
        all_group.extend([gname] * len(labels))
    return {
        "step_types": all_types,
        "labels":     np.array(all_y, dtype=np.int64),
        "item_ids":   np.array(all_id, dtype=object),
        "groups":     np.array(all_group, dtype=object),
    }


# =============================================================================
# TRAIN ONE FOLD
# =============================================================================

def train_one_fold(
    train_ds: BehaviorSeqDataset, val_ds: BehaviorSeqDataset,
    epochs: int, batch_size: int, lr: float, device: str,
    hidden: int, n_layers: int, dropout: float,
    pos_weight: float = 1.0, log_prefix: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                            collate_fn=collate, num_workers=2)

    model = BehaviorSeqLM(hidden=hidden, n_layers=n_layers, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    # ----- LEAKAGE-SAFE OOF PROTOCOL -----
    # Previously tracked best val_AUROC across epochs and returned those
    # predictions — that's test-set epoch selection. Now we return the LAST
    # epoch's predictions and only log val metrics for monitoring.
    last_probs, last_labels = None, None

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss, n_seen = 0.0, 0
        for batch in train_loader:
            typ = batch["typ"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            logit = model(typ, mask)
            loss = bce(logit, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * y.size(0)
            n_seen += y.size(0)
        ep_loss /= max(n_seen, 1)

        # Validate
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in val_loader:
                typ = batch["typ"].to(device)
                mask = batch["mask"].to(device)
                logit = model(typ, mask)
                ps.append(torch.sigmoid(logit).cpu().numpy())
                ys.append(batch["y"].numpy())
        ys = np.concatenate(ys); ps = np.concatenate(ps)
        m = evaluate(ys, ps, name=f"epoch_{ep}")
        if ep == 1 or ep == epochs or ep % 5 == 0:
            logger.info(f"  {log_prefix}ep {ep:02d}  loss={ep_loss:.4f}  "
                        f"val_AUROC={m['auroc']:.3f}  val_ECE={m['ece']:.3f}  "
                        f"[logged for monitoring only]")
        # Always overwrite — last-epoch protocol
        last_probs = ps.copy()
        last_labels = ys.copy()

    return last_labels, last_probs


# =============================================================================
# CV ORCHESTRATION
# =============================================================================

def run_cv(npz_paths: list[str], output: str,
           epochs: int, batch_size: int, lr: float,
           hidden: int, n_layers: int, dropout: float,
           n_splits: int, seed: int) -> dict:
    data = load_pooled_behavior_sequences(npz_paths)
    step_types = data["step_types"]
    labels     = data["labels"]
    groups     = data["groups"]
    item_ids   = data["item_ids"]

    logger.info(f"Loaded {len(labels)} traces from {len(npz_paths)} files. "
                f"Pos rate: {labels.mean():.3f}")
    logger.info(f"Groups: {sorted(set(groups))}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # OOF buffers
    oof_prob = np.full(len(labels), np.nan, dtype=np.float32)
    oof_fold = np.full(len(labels), -1, dtype=np.int32)
    fold_metrics = []
    all_y, all_p = [], []

    splitter = stratified_split(
        labels, group_id=groups if len(set(groups)) > 1 else None,
        n_splits=n_splits, seed=seed,
    )
    for fold, (tr_idx, te_idx) in enumerate(splitter):
        logger.info("")
        logger.info(f"=== Fold {fold + 1}/{n_splits}  train={len(tr_idx)}  val={len(te_idx)} ===")
        tr_types = [step_types[i] for i in tr_idx]
        te_types = [step_types[i] for i in te_idx]

        train_ds = BehaviorSeqDataset(tr_types, labels[tr_idx])
        val_ds   = BehaviorSeqDataset(te_types, labels[te_idx])

        y_tr = labels[tr_idx]
        n_pos = max(int(y_tr.sum()), 1)
        n_neg = max(len(y_tr) - n_pos, 1)
        pos_weight = n_neg / n_pos

        labels_, probs_ = train_one_fold(
            train_ds, val_ds,
            epochs=epochs, batch_size=batch_size, lr=lr, device=device,
            hidden=hidden, n_layers=n_layers, dropout=dropout,
            pos_weight=pos_weight, log_prefix=f"fold{fold + 1} ",
        )

        oof_prob[te_idx] = probs_
        oof_fold[te_idx] = fold
        fm = evaluate(labels_, probs_, name=f"fold_{fold + 1}")
        fold_metrics.append(fm)
        all_y.append(labels_); all_p.append(probs_)
        logger.info(f"  -> Fold {fold + 1} best  AUROC={fm['auroc']:.3f}  ECE={fm['ece']:.3f}")

    all_y = np.concatenate(all_y); all_p = np.concatenate(all_p)
    overall = evaluate(all_y, all_p, name="overall")
    summary = aggregate_folds(fold_metrics)

    # Save OOF .npz — same schema as step_transformer + deberta
    oof_path = output.replace(".json", "_oof.npz")
    os.makedirs(os.path.dirname(oof_path) or ".", exist_ok=True)
    np.savez_compressed(
        oof_path,
        item_ids=item_ids, groups=groups,
        y_true=labels, oof_prob=oof_prob, oof_fold=oof_fold,
        seed=np.array([seed]), n_splits=np.array([n_splits]),
    )
    logger.info(f"Saved OOF preds: {oof_path}")

    results = {
        "model": "BehaviorSeqLM",
        "n_splits": n_splits,
        "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "hidden": hidden, "n_layers": n_layers, "dropout": dropout,
        "n_samples": int(len(labels)),
        "base_accuracy": float(labels.mean()),
        "fold_metrics": fold_metrics,
        "summary": summary,
        "overall": overall,
        "oof_path": oof_path,
        "n_npz_files": len(npz_paths),
        "pooled": len(npz_paths) > 1,
    }
    save_results(output, results)

    print("\n=== BehaviorSeqLM CV summary ===")
    for k in ["auroc_mean", "auprc_mean", "ece_mean",
              "accuracy_at_80_mean", "accuracy_at_90_mean", "prr_mean"]:
        print(f"  {k:25s} {summary.get(k, float('nan')):.4f}  "
              f"± {summary.get(k.replace('_mean', '_std'), 0):.4f}")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--npz", nargs="+", default=None,
                   help="Explicit .npz paths")
    g.add_argument("--npz-glob", default=None,
                   help='Glob, e.g. "data/step_embeddings/*.npz"')
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    if args.npz_glob:
        npz_paths = sorted(glob.glob(args.npz_glob))
    else:
        npz_paths = list(args.npz)

    if not npz_paths:
        logger.error("No .npz files matched.")
        sys.exit(1)

    logger.info(f"Input .npz files ({len(npz_paths)}):")
    for p_ in npz_paths:
        logger.info(f"  - {p_}")

    run_cv(npz_paths, args.output,
           epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
           hidden=args.hidden, n_layers=args.n_layers, dropout=args.dropout,
           n_splits=args.n_splits, seed=args.seed)


if __name__ == "__main__":
    main()
