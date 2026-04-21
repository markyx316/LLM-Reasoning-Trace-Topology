"""
multi_layer_probe.py - Multi-layer hidden-state concatenation probes.

The layer atlas (src/modeling/layer_probe_sweep.py) shows that per-layer
AUROC plateaus across the ~60-85% depth range. This suggests DIFFERENT
layers might encode correctness from different angles. This script tests
that hypothesis by concatenating hidden states from multiple layers before
the probe.

We test a ladder of layer subsets (using ANSWER_MARKER position by default,
since atlas shows it is near-optimal in every layer):

    single_best:      [L20 / L23]                      (baseline)
    pair_adjacent:    [L16, L20]                       (two late)
    spread_3:         [L4, L12, L20]                   (early+mid+late)
    late_3:           [L16, L20, L24]
    spread_5:         [L4, L8, L12, L16, L20]
    all_8_layers:     all atlas layers concatenated

For each variant, probes are trained per hidden_dim group (Qwen / Llama)
and OOF predictions are pooled across both groups for a global AUROC.

Usage:
    PYTHONPATH=. python src/modeling/multi_layer_probe.py \
        --npz-glob "data/hidden_atlas/*.npz" \
        --output   results/month3/multi_layer_probe.json
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
# Probe MLP (same as layer_probe_sweep)
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


def train_mlp_fold(X_tr, y_tr, X_te, y_te, device,
                   epochs=25, batch_size=128, lr=1e-3,
                   wd=1e-3, dropout=0.3, pos_weight=1.0,
                   return_best=True):
    """Returns OOF probs (best-epoch by val AUROC)."""
    model = MLPProbe(X_tr.shape[1], hidden=256, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    tr_ds = TensorDataset(torch.from_numpy(X_tr).float(),
                          torch.from_numpy(y_tr).float())
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    Xte_t = torch.from_numpy(X_te).float().to(device)

    best_auroc = -1.0
    best_p = None
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            bce(model(xb), yb).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xte_t)).cpu().numpy()
        if return_best:
            m = evaluate(y_te, p, name=f"ep{ep}")
            if m["auroc"] > best_auroc:
                best_auroc = m["auroc"]; best_p = p.copy()
        else:
            best_p = p
    return best_p


# =============================================================================
# Per-dim-group CV with layer selection
# =============================================================================

def cv_one_variant(X_big, y, group, device, n_splits=5, seed=42):
    """X_big: (N, d). Returns OOF probs aligned to X_big."""
    n_pos = max(int(y.sum()), 1); n_neg = max(len(y) - n_pos, 1)
    pw = n_neg / n_pos
    oof = np.full(len(y), np.nan, dtype=np.float32)
    for tr, te in stratified_split(
            y, group_id=group if len(set(group)) > 1 else None,
            n_splits=n_splits, seed=seed):
        scaler = StandardScaler().fit(X_big[tr])
        Xtr = scaler.transform(X_big[tr])
        Xte = scaler.transform(X_big[te])
        p = train_mlp_fold(Xtr, y[tr], Xte, y[te], device=device,
                           epochs=25, pos_weight=pw)
        oof[te] = p
    return oof


# =============================================================================
# Data loading
# =============================================================================

def load_atlas_grouped(npz_paths):
    """Return {hidden_dim: dict(hidden (N,L,P,H), y, group, item_ids,
                                  layer_indices, position_names, n_total_layers)}."""
    groups: dict[int, dict] = {}
    for p in npz_paths:
        z = np.load(p, allow_pickle=True)
        d = int(z["hidden"].shape[-1])
        if d not in groups:
            groups[d] = {"hidden": [], "y": [], "group": [], "ids": [],
                         "layer_indices": None,
                         "position_names": None,
                         "n_total_layers": None}
        groups[d]["hidden"].append(z["hidden"].astype(np.float32))
        groups[d]["y"].append(z["y_true"].astype(int))
        groups[d]["group"].append(z["groups"].astype(str))
        groups[d]["ids"].append(z["item_ids"].astype(str))
        if groups[d]["layer_indices"] is None:
            groups[d]["layer_indices"] = z["layer_indices"].astype(int)
            groups[d]["position_names"] = z["position_names"].astype(str).tolist()
            groups[d]["n_total_layers"] = int(z["n_total_layers"][0])
    for d in groups:
        groups[d]["hidden"] = np.concatenate(groups[d]["hidden"])
        groups[d]["y"] = np.concatenate(groups[d]["y"])
        groups[d]["group"] = np.concatenate(groups[d]["group"])
        groups[d]["ids"] = np.concatenate(groups[d]["ids"])
    return groups


def pick_layer_subset(all_layer_idx, n_total_layers, kind: str):
    """Return a list of indices INTO all_layer_idx array matching the subset."""
    # all_layer_idx are model-relative indices already. We need array indices.
    # Find elements closest to desired fractions.
    def nearest(frac):
        target = frac * (n_total_layers - 1)
        return int(np.argmin(np.abs(all_layer_idx - target)))

    if kind == "single_best":        # ~71%
        return [nearest(0.71)]
    if kind == "single_last":
        return [len(all_layer_idx) - 1]
    if kind == "pair_adjacent":      # ~57% + ~71%
        return sorted({nearest(0.57), nearest(0.71)})
    if kind == "late_3":             # 57, 71, 86
        return sorted({nearest(0.57), nearest(0.71), nearest(0.86)})
    if kind == "spread_3":           # 14, 43, 71
        return sorted({nearest(0.14), nearest(0.43), nearest(0.71)})
    if kind == "spread_5":           # 14, 29, 43, 57, 71
        return sorted({nearest(0.14), nearest(0.29), nearest(0.43),
                       nearest(0.57), nearest(0.71)})
    if kind == "spread_6":           # 14, 29, 43, 57, 71, 86
        return sorted({nearest(0.14), nearest(0.29), nearest(0.43),
                       nearest(0.57), nearest(0.71), nearest(0.86)})
    if kind == "all":
        return list(range(len(all_layer_idx)))
    raise ValueError(kind)


def pick_positions(position_names, kind="answer"):
    if kind == "answer":
        return [position_names.index("answer_marker")]
    if kind == "ans_last":
        return [position_names.index("answer_marker"),
                position_names.index("last_token")]
    if kind == "ans_last_think":
        return [position_names.index("answer_marker"),
                position_names.index("last_token"),
                position_names.index("think_close")]
    if kind == "all_pos":
        return list(range(len(position_names)))
    raise ValueError(kind)


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz-glob", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    paths = sorted(glob.glob(args.npz_glob))
    groups = load_atlas_grouped(paths)
    logger.info(f"Loaded {len(paths)} files into {len(groups)} hidden_dim groups")

    # Variants: (layer_subset, position_subset)
    layer_kinds = ["single_best", "single_last",
                   "pair_adjacent", "late_3", "spread_3",
                   "spread_5", "spread_6", "all"]
    position_kinds = ["answer", "ans_last", "ans_last_think", "all_pos"]

    results = {"variants": {}}

    # Accumulator for pooled OOF across dim-groups
    pooled: dict[str, dict] = {}

    for lk in layer_kinds:
        for pk in position_kinds:
            vname = f"L={lk}|P={pk}"
            pooled[vname] = {"y": [], "p": [], "ids": [], "grp": []}

    for dim, data in groups.items():
        H = data["hidden"]          # (N, L, P, D)
        y = data["y"]
        grp = data["group"]
        ids = data["ids"]
        layer_idx = data["layer_indices"]
        pos_names = data["position_names"]
        n_total = data["n_total_layers"]

        logger.info(f"\n========== hidden_dim={dim}  n={len(y)} ==========")

        for lk in layer_kinds:
            L_sel = pick_layer_subset(layer_idx, n_total, lk)
            for pk in position_kinds:
                P_sel = pick_positions(pos_names, pk)
                X = H[:, L_sel][:, :, P_sel, :]
                X = X.reshape(len(y), -1)   # (N, len(L) * len(P) * D)
                vname = f"L={lk}|P={pk}"
                logger.info(f"  -- {vname:40s}  d={X.shape[1]}")
                oof = cv_one_variant(X, y, grp, device=device,
                                     n_splits=args.n_splits, seed=args.seed)
                pooled[vname]["y"].extend(y.tolist())
                pooled[vname]["p"].extend(oof.tolist())
                pooled[vname]["ids"].extend(ids.tolist())
                pooled[vname]["grp"].extend(grp.tolist())

    # Save pooled metrics + per-variant OOFs
    out_dir = os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)

    for vname, buf in pooled.items():
        y = np.asarray(buf["y"], dtype=int)
        p = np.asarray(buf["p"], dtype=float)
        ids = np.asarray(buf["ids"], dtype=object)
        grp = np.asarray(buf["grp"], dtype=object)
        m = evaluate(y, p, name=vname)
        results["variants"][vname] = {
            "n_items": int(len(y)),
            "auroc": m["auroc"],
            "auprc": m["auprc"],
            "ece":   m["ece"],
            "accuracy_at_80": m.get("accuracy_at_80", None),
            "accuracy_at_90": m.get("accuracy_at_90", None),
        }
        # Save OOF for best variants
        safe = vname.replace("|", "_").replace("=", "_")
        oof_path = args.output.replace(".json", f"_{safe}_oof.npz")
        np.savez_compressed(oof_path,
            item_ids=ids, groups=grp, y_true=y, oof_prob=p.astype(np.float32),
            seed=np.array([args.seed]),
            n_splits=np.array([args.n_splits]))

    save_results(args.output, results)

    # Pretty table
    print("\n" + "=" * 80)
    print("Multi-Layer Probe Results (pooled across hidden_dim groups)")
    print("=" * 80)
    print(f"{'variant':38s}  {'AUROC':>6s}  {'AUPRC':>6s}  {'ECE':>6s}  {'Acc@80':>6s}")
    print("-" * 80)
    # Sort by AUROC
    sorted_v = sorted(results["variants"].items(),
                      key=lambda kv: -kv[1]["auroc"])
    for vname, m in sorted_v:
        print(f"{vname:38s}  "
              f"{m['auroc']:.4f}  "
              f"{m['auprc']:.3f}   "
              f"{m['ece']:.3f}   "
              f"{m['accuracy_at_80'] or 0:.3f}")


if __name__ == "__main__":
    main()
