"""
trace_gnn.py - Graph Neural Network over trace-episode graphs (Route B1).

Strategy:
  Each reasoning trace is a directed graph whose nodes are cognitive episodes
  (labeled by behavior type in {F,V,B,R,S,H,C}) and whose edges capture:
    (a) temporal chain:    ep_i -> ep_{i+1}  (weight 1.0)
    (b) behavior recurrence: ep_i -> ep_j for j > i if behavior[i] == behavior[j]
         and j - i >= MIN_GAP, weight 1 / (j - i)  -- content-free recurrence
  We train a 2-layer GIN (Xu et al. 2019) with attention pooling + mean/max
  concat readout, emit OOF probabilities via 5-fold stratified CV, and dump
  a .npz in the same schema as DeBERTa / StepTF OOF files so hybrid.py can
  consume it.

Two variants (selected by --variant):
  * structural  :  node features = [7-d one-hot behavior, normalized position,
                   z-scored token count, log1p confidence] = 10-d.
                   Content-free. Writes trace_gnn_structural_oof.npz.
  * hybrid      :  same + optional MiniLM 384-d episode-text embedding
                   concatenated (requires graph .npz built with --with-content).
                   Writes trace_gnn_hybrid_oof.npz.

This module is PyG-free: we implement a minimal padded-batch GIN on dense
adjacency matrices so it runs on any torch install (including the torch311
miniconda env). Graphs are small (<= 256 nodes) so dense batching is fine.

Inputs:
  data/graphs/{dataset}_{model}_graph.npz   (one per dataset, from
  scripts/build_trace_graphs.py).
  Keys per file:
    item_ids     (N,)     str
    is_correct   (N,)     int8
    node_feats   (N,)     object; each entry (L_i, d_node) float32
    edge_indices (N,)     object; each (2, E_i) int32, PyG-style
    edge_weights (N,)     object; each (E_i,) float32

Output:
  results/route_ab/trace_gnn_{variant}_pooled.json   -- metrics + per-fold
  results/route_ab/trace_gnn_{variant}_oof.npz       -- OOF probs for hybrid

Usage:
  PYTHONPATH=. python src/modeling/trace_gnn.py \\
      --graph-glob "data/graphs/*_graph.npz" \\
      --variant structural \\
      --output results/route_ab/trace_gnn_structural_pooled.json \\
      --epochs 30 --batch-size 32
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.modeling.cv_utils import (
    aggregate_folds, evaluate, save_results, stratified_split,
)

logger = logging.getLogger(__name__)


# =============================================================================
# GRAPH POOLING / LOADING
# =============================================================================

def load_pooled_graphs(npz_paths: list[str]) -> dict:
    """Load multiple graph .npz files (one per dataset) and concat.

    Mirrors cv_utils.load_pooled_npz but for graph artifacts.
    Returns dict with lists of per-item arrays so the model can iterate
    without ever materializing a batch-level tensor until collate time.
    """
    all_nf, all_ei, all_ew, all_y, all_id, all_group = [], [], [], [], [], []
    for p in npz_paths:
        z = np.load(p, allow_pickle=True)
        nf = z["node_feats"]      # object array
        ei = z["edge_indices"]
        ew = z["edge_weights"]
        y = z["is_correct"]
        ids = z["item_ids"]
        for i in range(len(y)):
            all_nf.append(np.asarray(nf[i], dtype=np.float32))
            all_ei.append(np.asarray(ei[i], dtype=np.int64))
            all_ew.append(np.asarray(ew[i], dtype=np.float32))
        all_y.extend(list(y))
        all_id.extend(list(ids))
        gname = os.path.basename(p).replace("_graph.npz", "").replace(".npz", "")
        all_group.extend([gname] * len(y))
    return {
        "node_feats": all_nf,
        "edge_indices": all_ei,
        "edge_weights": all_ew,
        "labels": np.array(all_y, dtype=np.int64),
        "item_ids": np.array(all_id, dtype=object),
        "groups": np.array(all_group, dtype=object),
    }


# =============================================================================
# DATASET + COLLATION (pads graphs to batch max nodes, builds dense A)
# =============================================================================

class GraphDataset(Dataset):
    def __init__(self, node_feats, edge_indices, edge_weights, labels,
                 max_nodes: int = 256):
        self.nf = node_feats
        self.ei = edge_indices
        self.ew = edge_weights
        self.y = labels
        self.max_nodes = max_nodes

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        nf = self.nf[idx]
        ei = self.ei[idx]
        ew = self.ew[idx]
        if nf.shape[0] > self.max_nodes:
            nf = nf[: self.max_nodes]
            keep = (ei[0] < self.max_nodes) & (ei[1] < self.max_nodes)
            ei = ei[:, keep]
            ew = ew[keep]
        return {
            "x": torch.from_numpy(np.ascontiguousarray(nf)).float(),
            "edge_index": torch.from_numpy(np.ascontiguousarray(ei)).long(),
            "edge_weight": torch.from_numpy(np.ascontiguousarray(ew)).float(),
            "y": torch.tensor(int(self.y[idx]), dtype=torch.float32),
        }


def collate_graphs(batch: list[dict]) -> dict:
    """Pad to batch max nodes; build (B, N, N) dense adjacency.

    The feature dim is inferred from the first *non-empty* graph (and verified
    against every other non-empty graph in the batch). Using batch[0] blindly
    is unsafe: zero-node graphs built in the hybrid mode sometimes have an
    inconsistent stored shape (legacy bug in build_trace_graphs.py that kept
    empty traces as (0, 10) rather than (0, 10+d_content)). Picking from the
    first non-empty graph keeps this function correct even when the stored
    data has that legacy defect.
    """
    n = len(batch)
    # d: feat-dim of the first non-empty graph in the batch; fall back to
    # batch[0] only if every graph in the batch is empty (very rare).
    d = next(
        (int(b["x"].shape[1]) for b in batch if int(b["x"].shape[0]) > 0),
        int(batch[0]["x"].shape[1]),
    )
    max_nodes = max(1, max(int(b["x"].shape[0]) for b in batch))

    x = torch.zeros(n, max_nodes, d, dtype=torch.float32)
    A = torch.zeros(n, max_nodes, max_nodes, dtype=torch.float32)
    mask = torch.zeros(n, max_nodes, dtype=torch.bool)
    y = torch.zeros(n, dtype=torch.float32)

    for i, b in enumerate(batch):
        L = int(b["x"].shape[0])
        if L > 0:
            if int(b["x"].shape[1]) != d:
                # Shouldn't happen for a well-built dataset; loud error beats a
                # silent miscopy. If the builder has drift, catch it here.
                raise RuntimeError(
                    f"Inconsistent node-feat dim in batch: graph {i} has "
                    f"d={int(b['x'].shape[1])}, batch d={d}. "
                    f"Rebuild the .npz files with the fixed builder."
                )
            x[i, :L] = b["x"]
            mask[i, :L] = True
            ei = b["edge_index"]
            ew = b["edge_weight"]
            if ei.numel() > 0:
                src, dst = ei[0], ei[1]
                # Symmetrize: add both directions so GIN propagates both ways.
                A[i, src, dst] += ew
                A[i, dst, src] += ew
        y[i] = b["y"]

    return {"x": x, "A": A, "mask": mask, "y": y}


# =============================================================================
# MODEL
# =============================================================================

class GINLayer(nn.Module):
    """Padded-batch GIN layer:  h' = MLP((1+eps) * h + sum_j A_ij * h_j)."""

    def __init__(self, d_in: int, d_out: int, dropout: float = 0.2):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.LayerNorm(d_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_out, d_out),
        )
        self.norm = nn.LayerNorm(d_out)

    def forward(self, H: torch.Tensor, A: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        # H: (B, N, d_in), A: (B, N, N), mask: (B, N) bool
        AH = torch.bmm(A, H)                           # aggregated neighbors
        Hnew = self.mlp((1.0 + self.eps) * H + AH)
        Hnew = self.norm(Hnew)
        Hnew = Hnew * mask.unsqueeze(-1).float()
        return Hnew


class AttentionPool(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1)
        )

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        s = self.score(H).squeeze(-1)                  # (B, N)
        s = s.masked_fill(~mask, float("-inf"))
        # Guard: if a graph has zero valid nodes (shouldn't happen), avoid NaN.
        any_valid = mask.any(dim=1, keepdim=True)
        s = torch.where(any_valid, s, torch.zeros_like(s))
        a = torch.softmax(s, dim=1)                    # (B, N)
        pooled = (H * a.unsqueeze(-1)).sum(dim=1)       # (B, d)
        return pooled


class TraceGIN(nn.Module):
    """2-layer GIN + (attention, mean, max) pooling concat + 2-layer MLP head."""

    def __init__(self, d_node: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(d_node, hidden)
        self.gin1 = GINLayer(hidden, hidden, dropout=dropout)
        self.gin2 = GINLayer(hidden, hidden, dropout=dropout)
        self.att = AttentionPool(hidden)
        self.head = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def _pool(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        att = self.att(H, mask)                         # (B, hidden)
        mask_f = mask.unsqueeze(-1).float()
        denom = mask_f.sum(dim=1).clamp(min=1.0)
        mean = (H * mask_f).sum(dim=1) / denom
        mx = H.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        # If no valid nodes, -inf -> 0 (still the padding mask guard).
        mx = torch.where(torch.isinf(mx), torch.zeros_like(mx), mx)
        return torch.cat([att, mean, mx], dim=-1)       # (B, 3*hidden)

    def forward(self, x: torch.Tensor, A: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        H0 = self.proj(x) * mask.unsqueeze(-1).float()
        H1 = self.gin1(H0, A, mask)
        H2 = self.gin2(H1, A, mask)
        pooled = self._pool(H2, mask)
        logit = self.head(pooled).squeeze(-1)
        return logit


# =============================================================================
# TRAIN / EVAL ONE FOLD
# =============================================================================

def train_one_fold(train_ds: GraphDataset, val_ds: GraphDataset,
                   d_node: int,
                   epochs: int = 30, batch_size: int = 32, lr: float = 3e-4,
                   patience: int = 5, weight_decay: float = 1e-5,
                   hidden: int = 128, dropout: float = 0.2,
                   pos_weight: float = 1.0, device: str = "cpu",
                   log_prefix: str = "") -> tuple[np.ndarray, np.ndarray]:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_graphs, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_graphs, num_workers=0)

    model = TraceGIN(d_node=d_node, hidden=hidden, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight],
                                                       device=device))

    # ----- LEAKAGE-SAFE OOF PROTOCOL -----
    # Previous version tracked best_auroc on val + early-stopped on it. That
    # is two distinct leaks: (a) epoch selection on test labels, and (b)
    # epoch-count selection on test labels. We now train a fixed number of
    # epochs and return the LAST epoch's predictions. Val metrics are still
    # logged for monitoring only, never consulted for picks.
    last_probs = None
    last_labels = None

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        n_seen = 0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            A = batch["A"].to(device, non_blocking=True)
            mk = batch["mask"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            logit = model(x, A, mk)
            loss = bce(logit, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item() * y.size(0)
            n_seen += y.size(0)
        ep_loss /= max(n_seen, 1)

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device); A = batch["A"].to(device)
                mk = batch["mask"].to(device)
                logit = model(x, A, mk)
                ps.append(torch.sigmoid(logit).cpu().numpy())
                ys.append(batch["y"].numpy())
        ys = np.concatenate(ys); ps = np.concatenate(ps)
        m = evaluate(ys, ps, name=f"epoch_{ep}")
        logger.info(f"  {log_prefix} ep {ep:02d}  loss={ep_loss:.4f}  "
                    f"val_AUROC={m['auroc']:.3f}  val_AUPRC={m['auprc']:.3f}  "
                    f"val_ECE={m['ece']:.3f}  [logged for monitoring only]")
        # Always overwrite — last-epoch protocol
        last_probs = ps.copy()
        last_labels = ys.copy()
        # NOTE: early stopping removed. Previously this block early-stopped
        # when val_AUROC failed to improve for `patience` epochs — another
        # form of test-set-driven decision. `patience` kwarg is now ignored.

    return last_labels, last_probs


# =============================================================================
# CV ORCHESTRATION
# =============================================================================

def run_cv(npz_paths: list[str], output_path: str,
           variant: str = "structural",
           epochs: int = 30, batch_size: int = 32, lr: float = 3e-4,
           hidden: int = 128, dropout: float = 0.2, patience: int = 5,
           n_splits: int = 5, seed: int = 42, device: Optional[str] = None):
    data = load_pooled_graphs(npz_paths)
    labels = data["labels"]
    item_ids = data["item_ids"]
    groups = data["groups"]
    node_feats = data["node_feats"]
    edge_indices = data["edge_indices"]
    edge_weights = data["edge_weights"]
    n = len(labels)
    logger.info(f"Loaded {n} graphs from {len(npz_paths)} files. "
                f"Pos rate: {labels.mean():.3f}")

    # Auto-detect node feature dim from first non-empty graph.
    d_node = 10
    for nf in node_feats:
        if nf.size > 0:
            d_node = int(nf.shape[1])
            break
    logger.info(f"Auto-detected node feature dim: {d_node}  variant={variant}")

    sizes = [nf.shape[0] for nf in node_feats]
    logger.info(f"Node count stats: "
                f"min={min(sizes)}  mean={np.mean(sizes):.1f}  "
                f"max={max(sizes)}  median={int(np.median(sizes))}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    n_pos = max(int(labels.sum()), 1)
    n_neg = max(int(n - n_pos), 1)
    pos_weight = n_neg / n_pos
    logger.info(f"pos_weight = {pos_weight:.3f}")

    oof_prob = np.full(n, np.nan, dtype=np.float32)
    oof_fold = np.full(n, -1, dtype=np.int32)

    fold_metrics = []
    all_y, all_p = [], []

    for fold, (tr_idx, te_idx) in enumerate(stratified_split(
            labels, group_id=groups if len(set(groups)) > 1 else None,
            n_splits=n_splits, seed=seed)):
        logger.info(f"\n=== Fold {fold + 1}/{n_splits}  "
                    f"train={len(tr_idx)}  val={len(te_idx)} ===")
        tr_ds = GraphDataset(
            [node_feats[i] for i in tr_idx],
            [edge_indices[i] for i in tr_idx],
            [edge_weights[i] for i in tr_idx],
            labels[tr_idx],
        )
        te_ds = GraphDataset(
            [node_feats[i] for i in te_idx],
            [edge_indices[i] for i in te_idx],
            [edge_weights[i] for i in te_idx],
            labels[te_idx],
        )

        ys, ps = train_one_fold(
            tr_ds, te_ds, d_node=d_node,
            epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
            hidden=hidden, dropout=dropout, pos_weight=pos_weight,
            device=device, log_prefix=f"fold{fold + 1}",
        )
        oof_prob[te_idx] = ps
        oof_fold[te_idx] = fold
        fm = evaluate(ys, ps, name=f"fold_{fold + 1}")
        fold_metrics.append(fm)
        all_y.append(ys); all_p.append(ps)
        logger.info(f"  -> Fold {fold + 1} best  AUROC={fm['auroc']:.3f}  "
                    f"AUPRC={fm['auprc']:.3f}  ECE={fm['ece']:.3f}")

    all_y = np.concatenate(all_y); all_p = np.concatenate(all_p)
    overall = evaluate(all_y, all_p, name="overall")
    summary = aggregate_folds(fold_metrics)

    oof_path = output_path.replace(".json", "_oof.npz")
    os.makedirs(os.path.dirname(oof_path) or ".", exist_ok=True)
    np.savez_compressed(oof_path,
        item_ids=item_ids, groups=groups,
        y_true=labels, oof_prob=oof_prob, oof_fold=oof_fold,
        seed=np.array([seed]), n_splits=np.array([n_splits]))
    logger.info(f"Saved OOF preds: {oof_path}")

    results = {
        "model": f"TraceGIN_{variant}",
        "variant": variant,
        "n_splits": n_splits,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "hidden": hidden,
        "dropout": dropout,
        "patience": patience,
        "n_samples": int(n),
        "d_node": int(d_node),
        "base_accuracy": float(labels.mean()),
        "fold_metrics": fold_metrics,
        "summary": summary,
        "overall": overall,
        "oof_path": oof_path,
    }
    save_results(output_path, results)

    print("\n=== TraceGIN CV summary ===")
    for k in ["auroc_mean", "auprc_mean", "ece_mean",
              "accuracy_at_80_mean", "accuracy_at_90_mean"]:
        print(f"  {k:25s} {summary.get(k, float('nan')):.4f}  "
              f"± {summary.get(k.replace('_mean', '_std'), 0):.4f}")
    return results


# =============================================================================
# DRIVER
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph-glob", default=None,
                    help="Glob for per-dataset graph .npz files")
    ap.add_argument("--graph", nargs="+", default=None,
                    help="Explicit list of .npz files")
    ap.add_argument("--variant", choices=["structural", "hybrid"],
                    default="structural",
                    help="Tag for the run; does not change training, the "
                         "distinction is baked in at graph-build time via "
                         "scripts/build_trace_graphs.py --with-content")
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None,
                    help="cpu|cuda  (default auto)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    if args.graph_glob:
        paths = sorted(glob.glob(args.graph_glob))
    elif args.graph:
        paths = args.graph
    else:
        ap.error("--graph or --graph-glob")
    if not paths:
        logger.error(f"No graph .npz files matched  ({args.graph_glob})")
        sys.exit(1)
    logger.info(f"Loading from {len(paths)} npz files: {paths}")

    run_cv(paths, args.output, variant=args.variant,
           epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
           hidden=args.hidden, dropout=args.dropout, patience=args.patience,
           n_splits=args.n_splits, seed=args.seed, device=args.device)


# =============================================================================
# SELF-TEST
# =============================================================================

def _synth_graph(n_nodes: int, d_node: int, label: int, rng: np.random.Generator
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a toy graph whose topology correlates with label.

    Correct (label=1): chain-only, no recurrence edges.
    Incorrect (label=0): chain + many recurrence cycles (hubs).
    """
    nf = rng.standard_normal((n_nodes, d_node)).astype(np.float32)
    nf[:, 0] = float(label)  # leak the label into node feat so AUROC > 0.5
    ei = []
    ew = []
    for i in range(n_nodes - 1):
        ei.append([i, i + 1]); ew.append(1.0)
    if label == 0:
        for i in range(0, n_nodes - 5, 2):
            ei.append([i, min(i + 5, n_nodes - 1)])
            ew.append(0.2)
    ei = np.asarray(ei, dtype=np.int64).T if ei else np.zeros((2, 0),
                                                              dtype=np.int64)
    ew = np.asarray(ew, dtype=np.float32) if len(ew) else np.zeros((0,),
                                                                   dtype=np.float32)
    return nf, ei, ew


def _test_collate_mixed_dim_batch():
    """Regression test: if batch[0] is an empty graph with the legacy buggy
    shape (0, 10) but other graphs in the batch have d=394, collate_graphs
    must still correctly pick d=394 from a non-empty graph (and complain
    loudly if two non-empty graphs disagree)."""
    # Empty graph first (stale shape 0x10), then two valid 394-d graphs.
    batch = [
        {
            "x": torch.zeros(0, 10, dtype=torch.float32),
            "edge_index": torch.zeros(2, 0, dtype=torch.long),
            "edge_weight": torch.zeros(0, dtype=torch.float32),
            "y": torch.tensor(0.0),
        },
        {
            "x": torch.randn(5, 394, dtype=torch.float32),
            "edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
            "edge_weight": torch.ones(4, dtype=torch.float32),
            "y": torch.tensor(1.0),
        },
        {
            "x": torch.randn(3, 394, dtype=torch.float32),
            "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            "edge_weight": torch.ones(2, dtype=torch.float32),
            "y": torch.tensor(0.0),
        },
    ]
    out = collate_graphs(batch)
    assert out["x"].shape == (3, 5, 394), f"expected (3, 5, 394), got {out['x'].shape}"
    # The empty graph should remain all zeros; mask should be False for it.
    assert out["mask"][0].sum().item() == 0, "empty graph should have empty mask"
    assert out["mask"][1].sum().item() == 5
    assert out["mask"][2].sum().item() == 3

    # Disagreement among non-empty graphs MUST raise (caller needs to rebuild).
    bad = [
        {
            "x": torch.randn(5, 394, dtype=torch.float32),
            "edge_index": torch.zeros(2, 0, dtype=torch.long),
            "edge_weight": torch.zeros(0, dtype=torch.float32),
            "y": torch.tensor(1.0),
        },
        {
            "x": torch.randn(3, 10, dtype=torch.float32),
            "edge_index": torch.zeros(2, 0, dtype=torch.long),
            "edge_weight": torch.zeros(0, dtype=torch.float32),
            "y": torch.tensor(0.0),
        },
    ]
    try:
        collate_graphs(bad)
        raise AssertionError("expected RuntimeError on dim-mismatched batch")
    except RuntimeError as e:
        assert "Inconsistent node-feat dim" in str(e)
    print("  collate_graphs: mixed-dim batch ok, dim-mismatch raises  (regression guarded)")


def _run_self_test():
    print("Running trace_gnn self-test...")
    _test_collate_mixed_dim_batch()
    import tempfile
    rng = np.random.default_rng(0)

    n = 60
    d_node = 10
    nfs, eis, ews, ys, ids = [], [], [], [], []
    for i in range(n):
        label = i % 2
        L = int(rng.integers(20, 60))
        nf, ei, ew = _synth_graph(L, d_node, label, rng)
        nfs.append(nf); eis.append(ei); ews.append(ew)
        ys.append(label); ids.append(f"syn_{i:03d}")

    tmp = tempfile.mkdtemp()
    npz_path = os.path.join(tmp, "syn_graph.npz")

    def _as_object(seq):
        arr = np.empty(len(seq), dtype=object)
        for i, v in enumerate(seq):
            arr[i] = v
        return arr

    np.savez_compressed(
        npz_path,
        item_ids=np.array(ids, dtype=object),
        is_correct=np.array(ys, dtype=np.int8),
        node_feats=_as_object(nfs),
        edge_indices=_as_object(eis),
        edge_weights=_as_object(ews),
    )

    out_json = os.path.join(tmp, "syn_gnn.json")
    res = run_cv([npz_path], out_json, variant="structural",
                 epochs=5, batch_size=8, n_splits=3, device="cpu")
    auroc = res["summary"].get("auroc_mean", 0.0)
    print(f"  Synthetic CV AUROC: {auroc:.4f}")
    assert auroc > 0.7, f"synthetic GNN AUROC too low: {auroc:.4f}"

    # OOF npz schema check
    oof = np.load(res["oof_path"], allow_pickle=True)
    assert set(oof.files) >= {"item_ids", "y_true", "oof_prob", "groups"}
    assert len(oof["item_ids"]) == n
    assert np.isnan(oof["oof_prob"]).sum() == 0, "OOF probs should be filled"
    print("All trace_gnn tests passed.")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _run_self_test()
    else:
        main()
