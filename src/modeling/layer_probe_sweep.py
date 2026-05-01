"""
layer_probe_sweep.py - Train MLP probes across (layer, position) and produce
a heatmap of correctness AUROC.

Reads .npz produced by scripts/extract_layer_atlas.py. For each (layer,
position) cell, trains an MLP probe in 5-fold CV (within each hidden_dim
group) and reports pooled AUROC.

Output:
    - JSON with full per-cell metrics
    - Heatmap data ready to plot

Usage:
    PYTHONPATH=. python src/modeling/layer_probe_sweep.py \
        --npz-glob "data/hidden_atlas/*.npz" \
        --output   results/month3/layer_atlas.json
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.modeling.cv_utils import evaluate, save_results, stratified_split

logger = logging.getLogger(__name__)


# =============================================================================
# Probe (same as hidden_state_probe)
# =============================================================================

class MLPProbe(nn.Module):
    def __init__(self, in_dim, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_fold(X_tr, y_tr, X_te, device, epochs=20, batch_size=128,
                   lr=1e-3, wd=1e-3, dropout=0.3,
                   pos_weight=1.0):
    model = MLPProbe(X_tr.shape[1], dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    tr_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr).float(),
                                         torch.from_numpy(y_tr).float()),
                           batch_size=batch_size, shuffle=True)
    Xte_t = torch.from_numpy(X_te).float().to(device)

    best_auroc = -1; best_p = None
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            l = bce(model(xb), yb); l.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xte_t)).cpu().numpy()
        # Track best by pseudo-AUROC (no labels passed here, so use early-stop later)
        best_p = p
    return best_p


# =============================================================================
# CORE SWEEP
# =============================================================================

def cv_sweep_one_cell(X, y, group, n_splits=5, seed=42, device="cuda"):
    n_pos = max(int(y.sum()), 1); n_neg = max(len(y) - n_pos, 1)
    pos_w = n_neg / n_pos
    oof = np.full(len(y), np.nan, dtype=np.float32)
    for tr, te in stratified_split(
            y, group_id=group if len(set(group)) > 1 else None,
            n_splits=n_splits, seed=seed):
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr]); Xte = scaler.transform(X[te])
        p = train_mlp_fold(Xtr, y[tr], Xte, device=device,
                           epochs=15, batch_size=128, lr=1e-3,
                           pos_weight=pos_w)
        oof[te] = p
    return oof


def load_pooled_atlas(paths):
    """Group by hidden_dim. Each group: dict with hidden (N, L, P, H), y, group, ids, layer_indices."""
    dim_groups: dict[int, list[str]] = {}
    for p in paths:
        z = np.load(p, allow_pickle=True)
        d = int(z["hidden"].shape[-1])
        dim_groups.setdefault(d, []).append(p)

    out = {}
    for d, plist in dim_groups.items():
        h_list = []; y_list = []; g_list = []; id_list = []
        layer_idx = None; pos_names = None; n_total_layers = None
        for p in plist:
            z = np.load(p, allow_pickle=True)
            h_list.append(z["hidden"].astype(np.float32))
            y_list.append(z["y_true"].astype(int))
            g_list.append(z["groups"].astype(str))
            id_list.append(z["item_ids"].astype(str))
            layer_idx = z["layer_indices"].astype(int) if layer_idx is None else layer_idx
            pos_names = z["position_names"].astype(str) if pos_names is None else pos_names
            n_total_layers = int(z["n_total_layers"][0]) if n_total_layers is None else n_total_layers
        out[d] = {
            "hidden":     np.concatenate(h_list),     # (N, L, P, H)
            "y_true":     np.concatenate(y_list),
            "groups":     np.concatenate(g_list),
            "item_ids":   np.concatenate(id_list),
            "layer_indices": layer_idx,
            "position_names": list(pos_names),
            "n_total_layers": n_total_layers,
        }
        logger.info(f"  dim={d}: hidden shape={out[d]['hidden'].shape}, "
                    f"layers={layer_idx.tolist()}, "
                    f"positions={list(pos_names)}, "
                    f"n_total_layers={n_total_layers}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz-glob", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    paths = sorted(glob.glob(args.npz_glob))
    logger.info(f"Loading {len(paths)} npz files")
    atlas = load_pooled_atlas(paths)

    results = {"hidden_dims": list(atlas.keys()), "cells": {}}

    # Each (dim, layer, position) cell: train probe, save OOF, compute AUROC
    pooled_cells: dict[tuple[int, str], dict] = {}

    for dim, data in atlas.items():
        h = data["hidden"]
        y = data["y_true"]; group = data["groups"]
        layer_idx = data["layer_indices"]
        pos_names = data["position_names"]
        n_total = data["n_total_layers"]
        logger.info(f"\n========== hidden_dim={dim}  n={len(y)} ==========")

        for li, layer_i in enumerate(layer_idx):
            for pi, pos_name in enumerate(pos_names):
                X = h[:, li, pi, :]   # (N, H)
                logger.info(f"  -- layer={layer_i} ({100*layer_i/(n_total-1):.0f}%) "
                            f"pos={pos_name} ...")
                oof = cv_sweep_one_cell(X, y, group, n_splits=args.n_splits,
                                        seed=args.seed, device=device)
                m = evaluate(y, oof, name=f"L{layer_i}_{pos_name}_dim{dim}")
                key = (dim, int(layer_i), pos_name)
                pooled_cells[key] = {
                    "auroc": m["auroc"],
                    "ece":   m["ece"],
                    "auprc": m["auprc"],
                    "n":     int(len(y)),
                    "layer_frac": float(layer_i / (n_total - 1)),
                }
                logger.info(f"     AUROC={m['auroc']:.4f}  ECE={m['ece']:.4f}")

    # Aggregate per-dim into JSON-able structure
    out = {}
    for (dim, li, pn), v in pooled_cells.items():
        out.setdefault(str(dim), {}).setdefault(pn, []).append({
            "layer_index": li, "layer_frac": v["layer_frac"],
            "auroc": v["auroc"], "ece": v["ece"], "auprc": v["auprc"],
            "n": v["n"],
        })
    results["heatmap"] = out

    # Flat (dim/layer/position) -> cell projection for downstream lookup.
    # Prior versions left this empty; kept in sync with heatmap so that
    # every cell is addressable by a single key.
    results["cells"] = {
        f"dim{dim}/L{li}/{pn}": {
            "dim": dim,
            "layer_index": li,
            "layer_frac": v["layer_frac"],
            "position": pn,
            "auroc": v["auroc"],
            "ece": v["ece"],
            "auprc": v["auprc"],
            "n": v["n"],
        }
        for (dim, li, pn), v in pooled_cells.items()
    }

    save_results(args.output, results)

    # Pretty: print for each (dim, position) the AUROC vs layer trace
    print("\n=== Layer-Atlas Sweep Summary ===")
    for dim, pos_dict in out.items():
        print(f"\nHidden_dim = {dim}")
        for pos, layers in pos_dict.items():
            layers_sorted = sorted(layers, key=lambda x: x["layer_index"])
            cells = " ".join(f"L{c['layer_index']}={c['auroc']:.3f}"
                             for c in layers_sorted)
            print(f"  {pos:14s} -> {cells}")


if __name__ == "__main__":
    main()
