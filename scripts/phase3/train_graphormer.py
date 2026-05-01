"""Phase 3 (T5) — Graphormer over enriched trace DAGs.

Graphormer (Ying et al. 2021, "Do Transformers Really Perform Bad for
Graph Representation?") adds three graph-aware biases to the standard
self-attention head:

    1. Centrality encoding — per-node learned embedding of its
       (in_degree, out_degree). We use clipped bins in [0, D_MAX=16].
    2. Spatial encoding    — shortest-path-distance bias added to the
       attention logits between every pair of nodes. Unreachable pairs
       are given a large negative bias. We clip SPD at SPD_MAX=8.
    3. Edge encoding       — per-edge-type learned scalar bias summed along
       the shortest path between two nodes. For Phase 3 we have 3 edge
       types: 0 temporal, 1 recurrence, 2 revision.

Our variant keeps all three ideas but simplifies:
    - Dense attention (graphs are small, L ≤ 256 nodes).
    - Edge encoding summed over hops using only the edge type at each hop
      (we precompute the shortest path between all pairs via Floyd-Warshall
      on the unweighted graph, then record the max edge type along the
      path for a simple "edge-type-on-path" feature).
    - [CLS] readout token concatenated to the node set.

Inputs: data/graphs_v3/*.npz produced by scripts/phase3/build_trace_dags.py
        (must have keys item_ids, is_correct, node_feats, edge_indices,
         edge_weights, edge_types).

Outputs
-------
    results/month3/graphormer_v3.json
    results/month3/graphormer_v3_oof.npz

Usage
-----
    PYTHONPATH=. python scripts/phase3/train_graphormer.py \
        --npz-glob 'data/graphs_v3/*_graph_v3.npz' \
        --output   results/month3/graphormer_v3.json
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.modeling.cv_utils import stratified_split, evaluate, save_results  # type: ignore


# Hyper-parameters
SPD_MAX = 8        # clip shortest-path distance at this hop count
DEG_MAX = 16       # clip node degree at this value
N_EDGE_TYPES = 3   # temporal / recurrence / revision


# =========================================================================
# DATA
# =========================================================================

def load_pooled_graphs(npz_paths: list[str]) -> dict:
    item_ids, labels, groups = [], [], []
    nfs, eis, ews, ets = [], [], [], []
    for p in npz_paths:
        z = np.load(p, allow_pickle=True)
        base = os.path.basename(p).replace("_graph_v3.npz", "").replace(".npz", "")
        n = len(z["item_ids"])
        item_ids.extend([str(x) for x in z["item_ids"]])
        labels.extend(list(z["is_correct"].astype(int)))
        groups.extend([base] * n)
        nfs.extend(list(z["node_feats"]))
        eis.extend(list(z["edge_indices"]))
        ews.extend(list(z["edge_weights"]))
        ets.extend(list(z["edge_types"]))
    return {
        "item_ids": np.asarray(item_ids, dtype=object),
        "labels":   np.asarray(labels, dtype=np.int64),
        "groups":   np.asarray(groups, dtype=object),
        "nfs":      nfs, "eis": eis, "ews": ews, "ets": ets,
    }


def _spd_and_edge_path_type(ei: np.ndarray, et: np.ndarray, L: int):
    """Return (spd[L,L] int16, path_edge_type[L,L] int8).

    Uses Floyd-Warshall on the *undirected* view for SPD, and records the
    maximum edge type encountered along the shortest path.
    """
    INF = 32767
    spd = np.full((L, L), INF, dtype=np.int16)
    pet = np.zeros((L, L), dtype=np.int8)  # default 0 (temporal)
    for i in range(L):
        spd[i, i] = 0
    # Build adjacency with direct edges
    for k in range(ei.shape[1]):
        a, b = int(ei[0, k]), int(ei[1, k])
        t = int(et[k]) if et is not None else 0
        if spd[a, b] > 1:
            spd[a, b] = 1; pet[a, b] = t
            spd[b, a] = 1; pet[b, a] = t
    # Floyd-Warshall
    for k in range(L):
        spdk = spd[k].astype(np.int32)
        for i in range(L):
            d_ik = spd[i, k]
            if d_ik >= INF:
                continue
            new = d_ik + spdk
            mask = new < spd[i].astype(np.int32)
            if mask.any():
                spd[i, mask] = new[mask].astype(np.int16)
                # Edge-type-on-path: pass through the max of (pet[i,k], pet[k,j])
                pet[i, mask] = np.maximum(pet[i, k], pet[k, mask])
    spd = np.clip(spd, 0, SPD_MAX)
    return spd, pet


class GraphDataset(Dataset):
    def __init__(self, nfs, eis, ews, ets, labels, max_nodes: int = 256):
        self.nfs, self.eis, self.ews, self.ets = nfs, eis, ews, ets
        self.labels = labels
        self.max_nodes = max_nodes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        nf = np.asarray(self.nfs[idx], dtype=np.float32)
        ei = np.asarray(self.eis[idx], dtype=np.int64)
        ew = np.asarray(self.ews[idx], dtype=np.float32)
        et = np.asarray(self.ets[idx], dtype=np.int8)
        L = nf.shape[0]
        if L > self.max_nodes:
            # Truncate: keep first max_nodes
            nf = nf[: self.max_nodes]
            keep = (ei[0] < self.max_nodes) & (ei[1] < self.max_nodes)
            ei = ei[:, keep]; ew = ew[keep]; et = et[keep]
            L = self.max_nodes
        if L == 0:
            L = 1
            nf = np.zeros((1, nf.shape[1] if nf.size else 10), dtype=np.float32)
            ei = np.zeros((2, 0), dtype=np.int64)
            ew = np.zeros((0,), dtype=np.float32)
            et = np.zeros((0,), dtype=np.int8)
        in_deg = np.zeros(L, dtype=np.int16)
        out_deg = np.zeros(L, dtype=np.int16)
        for k in range(ei.shape[1]):
            a = int(ei[0, k]); b = int(ei[1, k])
            if 0 <= a < L: out_deg[a] += 1
            if 0 <= b < L: in_deg[b] += 1
        in_deg = np.clip(in_deg, 0, DEG_MAX - 1)
        out_deg = np.clip(out_deg, 0, DEG_MAX - 1)
        spd, pet = _spd_and_edge_path_type(ei, et, L)
        return {
            "nf":       torch.from_numpy(nf).float(),
            "spd":      torch.from_numpy(spd.astype(np.int64)),
            "pet":      torch.from_numpy(pet.astype(np.int64)),
            "in_deg":   torch.from_numpy(in_deg.astype(np.int64)),
            "out_deg":  torch.from_numpy(out_deg.astype(np.int64)),
            "y":        torch.tensor(int(self.labels[idx]), dtype=torch.float32),
        }


def collate_graphs(batch: list[dict]) -> dict:
    """Pad to max L in batch (+1 for [CLS]). Attention bias shape (B, L+1, L+1)."""
    n = len(batch)
    Lmax = max(b["nf"].shape[0] for b in batch)
    D_in = batch[0]["nf"].shape[1]

    nf = torch.zeros(n, Lmax, D_in, dtype=torch.float32)
    mask = torch.zeros(n, Lmax, dtype=torch.bool)
    spd = torch.full((n, Lmax + 1, Lmax + 1), SPD_MAX, dtype=torch.long)
    pet = torch.zeros((n, Lmax + 1, Lmax + 1), dtype=torch.long)
    in_deg = torch.zeros(n, Lmax, dtype=torch.long)
    out_deg = torch.zeros(n, Lmax, dtype=torch.long)
    y = torch.zeros(n, dtype=torch.float32)

    for i, b in enumerate(batch):
        L = b["nf"].shape[0]
        nf[i, :L] = b["nf"]
        mask[i, :L] = True
        # +1 for CLS: leave row/col 0 as "distance 0" (self) for CLS to any
        # node — i.e., CLS is connected to everyone with SPD 0.
        spd[i, 1:L+1, 1:L+1] = b["spd"]
        pet[i, 1:L+1, 1:L+1] = b["pet"]
        spd[i, 0, 1:L+1] = 1  # CLS to each real node: distance 1
        spd[i, 1:L+1, 0] = 1
        spd[i, 0, 0] = 0
        in_deg[i, :L] = b["in_deg"]
        out_deg[i, :L] = b["out_deg"]
        y[i] = b["y"]
    return {
        "nf": nf, "mask": mask, "spd": spd, "pet": pet,
        "in_deg": in_deg, "out_deg": out_deg, "y": y,
    }


# =========================================================================
# GRAPHORMER MODEL
# =========================================================================

class GraphormerLayer(nn.Module):
    def __init__(self, d: int, n_heads: int, dropout: float = 0.2):
        super().__init__()
        assert d % n_heads == 0
        self.d = d
        self.n_heads = n_heads
        self.h = d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ffn = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(4 * d, d),
        )
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_bias, key_padding):
        # x: (B, N, d); attn_bias: (B, 1, N, N); key_padding: (B, N) True=valid
        B, N, d = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.n_heads, self.h).permute(
            2, 0, 3, 1, 4
        )  # (3, B, H, N, h)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.h)
        scores = scores + attn_bias
        # Mask out padded KEY positions (columns)
        if key_padding is not None:
            m = key_padding.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            scores = scores.masked_fill(~m, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.drop(attn)
        o = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, d)
        o = self.out(o)
        x = x + self.drop(o)
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class Graphormer(nn.Module):
    def __init__(self, in_dim: int, d: int = 192, n_heads: int = 6,
                 n_layers: int = 4, dropout: float = 0.2,
                 spd_max: int = SPD_MAX, deg_max: int = DEG_MAX,
                 n_edge_types: int = N_EDGE_TYPES):
        super().__init__()
        self.proj = nn.Linear(in_dim, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.emb_in_deg = nn.Embedding(deg_max, d)
        self.emb_out_deg = nn.Embedding(deg_max, d)
        # Per-head scalar biases indexed by SPD and edge-type-on-path
        self.spd_bias = nn.Embedding(spd_max + 1, n_heads)
        self.pet_bias = nn.Embedding(n_edge_types, n_heads)
        self.layers = nn.ModuleList([
            GraphormerLayer(d, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d, 1),
        )

    def forward(self, nf, mask, spd, pet, in_deg, out_deg):
        B, L, _ = nf.shape
        x = self.proj(nf)                                  # (B, L, d)
        cent = self.emb_in_deg(in_deg) + self.emb_out_deg(out_deg)
        x = x + cent
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                     # (B, L+1, d)
        # Attention bias: (B, H, L+1, L+1)
        spd_b = self.spd_bias(spd).permute(0, 3, 1, 2)      # (B, H, L+1, L+1)
        pet_b = self.pet_bias(pet).permute(0, 3, 1, 2)
        attn_bias = spd_b + pet_b                          # (B, H, L+1, L+1)
        # Key-padding mask (CLS always valid)
        cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
        kp = torch.cat([cls_mask, mask], dim=1)            # (B, L+1)
        for layer in self.layers:
            x = layer(x, attn_bias, kp)
        x = self.norm(x)
        return self.head(x[:, 0]).squeeze(-1)              # (B,)


# =========================================================================
# TRAINING
# =========================================================================

def run_cv(npz_paths: list[str], output_path: str,
           epochs: int = 25, batch_size: int = 16, lr: float = 2e-4,
           n_splits: int = 5, seed: int = 42,
           d: int = 192, n_heads: int = 6, n_layers: int = 4):
    logger = logging.getLogger("graphormer")
    data = load_pooled_graphs(npz_paths)
    y = data["labels"]
    groups = data["groups"]
    item_ids = data["item_ids"]

    in_dim = data["nfs"][0].shape[1] if len(data["nfs"]) and data["nfs"][0].size else 10
    logger.info("Loaded %d graphs. pos=%.3f  in_dim=%d",
                len(y), y.mean(), in_dim)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_pos = max(int(y.sum()), 1); n_neg = max(int(len(y) - n_pos), 1)
    pos_weight = n_neg / n_pos

    oof_prob = np.full(len(y), np.nan, dtype=np.float32)
    oof_fold = np.full(len(y), -1, dtype=np.int32)
    fold_metrics = []

    for fold, (tr, te) in enumerate(stratified_split(
            y, group_id=groups if len(set(groups)) > 1 else None,
            n_splits=n_splits, seed=seed)):
        logger.info("=== Fold %d/%d  train=%d val=%d ===", fold + 1, n_splits, len(tr), len(te))
        tr_ds = GraphDataset([data["nfs"][i] for i in tr],
                             [data["eis"][i] for i in tr],
                             [data["ews"][i] for i in tr],
                             [data["ets"][i] for i in tr],
                             y[tr])
        va_ds = GraphDataset([data["nfs"][i] for i in te],
                             [data["eis"][i] for i in te],
                             [data["ews"][i] for i in te],
                             [data["ets"][i] for i in te],
                             y[te])
        tl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_graphs, num_workers=2)
        vl = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_graphs, num_workers=2)

        model = Graphormer(in_dim=in_dim, d=d, n_heads=n_heads, n_layers=n_layers).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

        best_auroc = -1.0
        best_probs = None
        best_labels = None

        for ep in range(1, epochs + 1):
            model.train()
            total = 0.0
            for batch in tl:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                logit = model(batch["nf"], batch["mask"], batch["spd"],
                              batch["pet"], batch["in_deg"], batch["out_deg"])
                loss = bce(logit, batch["y"])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += float(loss.item()) * batch["y"].size(0)
            # Validate
            model.eval()
            ys, ps = [], []
            with torch.no_grad():
                for batch in vl:
                    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                    logit = model(batch["nf"], batch["mask"], batch["spd"],
                                  batch["pet"], batch["in_deg"], batch["out_deg"])
                    p = torch.sigmoid(logit).cpu().numpy()
                    ys.append(batch["y"].cpu().numpy()); ps.append(p)
            yv = np.concatenate(ys); pv = np.concatenate(ps)
            fm = evaluate(yv, pv, name=f"fold{fold+1}_ep{ep}")
            if fm["auroc"] > best_auroc:
                best_auroc = fm["auroc"]; best_probs = pv; best_labels = yv
            logger.info("  ep %02d  loss=%.4f  AUROC=%.4f  best=%.4f",
                        ep, total / max(len(tr), 1), fm["auroc"], best_auroc)

        oof_prob[te] = best_probs
        oof_fold[te] = fold
        fold_metrics.append({"fold": fold + 1, "auroc": best_auroc})

    # Save results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    overall = evaluate(y, np.where(np.isnan(oof_prob), 0.5, oof_prob), name="pooled")
    results = {
        "n_samples": len(y),
        "fold_metrics": fold_metrics,
        "pooled": overall,
    }
    save_results(output_path, results)
    oof_path = output_path.replace(".json", "_oof.npz")
    np.savez_compressed(oof_path,
        item_ids=item_ids, groups=groups,
        y_true=y, oof_prob=oof_prob.astype(np.float32), oof_fold=oof_fold,
        seed=np.array([seed]), n_splits=np.array([n_splits]))
    logger.info("Saved OOF: %s", oof_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz-glob", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--n-layers", type=int, default=4)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    paths = sorted(glob.glob(args.npz_glob))
    if not paths:
        raise SystemExit(f"No graph files match {args.npz_glob}")

    run_cv(
        paths, args.output,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        n_splits=args.n_splits, seed=args.seed,
        d=args.d, n_heads=args.n_heads, n_layers=args.n_layers,
    )


if __name__ == "__main__":
    main()
