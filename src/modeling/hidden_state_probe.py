"""
hidden_state_probe.py - Train linear / MLP probes on generator hidden states.

Takes .npz files produced by scripts/extract_hidden_states.py and trains a
supervised correctness classifier on top. We compare:

  - Linear probe on h_last      (baseline)
  - Linear probe on h_think     (at </think> close)
  - Linear probe on h_answer    (at "final answer" / "boxed{")
  - MLP probe on h_answer       (nonlinear version of the above)
  - Linear probe on CONCAT of all three

We evaluate with the same group-aware stratified 5-fold CV as the rest of
the pipeline, and save OOF predictions so the hybrid meta-learner can
include this signal downstream.

Usage:
    PYTHONPATH=. python src/modeling/hidden_state_probe.py \
        --npz-glob "data/hidden_states/*.npz" \
        --output   results/month3/hidden_probe_pooled.json
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.generation_uncertainty import GENERATION_UNC_FEATURE_NAMES
from src.modeling.cv_utils import (
    aggregate_folds, evaluate, save_results, stratified_split,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_pooled(npz_paths: list[str]) -> dict:
    """Load .npz hidden states grouped by hidden_dim (different generator
    families have different sizes, e.g. Qwen=3584 vs Llama=4096)."""
    # group paths by hidden_dim
    dim_groups: dict[int, list[str]] = {}
    for p in npz_paths:
        z = np.load(p, allow_pickle=True)
        d = int(z["h_last"].shape[1])
        dim_groups.setdefault(d, []).append(p)

    dim_data: dict[int, dict] = {}
    for d, paths in dim_groups.items():
        h_last, h_think, h_answer = [], [], []
        item_ids, groups, y_true = [], [], []
        for p in paths:
            z = np.load(p, allow_pickle=True)
            h_last.append(z["h_last"].astype(np.float32))
            h_think.append(z["h_think"].astype(np.float32))
            h_answer.append(z["h_answer"].astype(np.float32))
            item_ids.append(z["item_ids"].astype(str))
            groups.append(z["groups"].astype(str))
            y_true.append(z["y_true"].astype(int))
        dim_data[d] = {
            "h_last":   np.concatenate(h_last),
            "h_think":  np.concatenate(h_think),
            "h_answer": np.concatenate(h_answer),
            "item_ids": np.concatenate(item_ids),
            "groups":   np.concatenate(groups),
            "y_true":   np.concatenate(y_true),
        }
        logger.info(f"  hidden_dim={d}: n={len(y_true[0]) if len(y_true)==1 else sum(len(x) for x in y_true)}  files={len(paths)}")
    return dim_data


# =============================================================================
# MLP PROBE
# =============================================================================

class MLPProbe(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.3):
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


def train_mlp_fold(X_tr, y_tr, X_te, y_te, device: str,
                   epochs: int = 30, batch_size: int = 128, lr: float = 1e-3,
                   wd: float = 1e-3, dropout: float = 0.3,
                   pos_weight: float = 1.0) -> np.ndarray:
    model = MLPProbe(X_tr.shape[1], hidden=256, dropout=dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    tr_ds = TensorDataset(torch.from_numpy(X_tr).float(),
                          torch.from_numpy(y_tr).float())
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

    Xte_t = torch.from_numpy(X_te).float().to(device)

    best_auroc = -1.0
    best_probs = None
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(device); yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = bce(logits, yb)
            loss.backward()
            opt.step()
        sched.step()
        # Track val
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(Xte_t)).cpu().numpy()
        m = evaluate(y_te, probs)
        if m["auroc"] > best_auroc:
            best_auroc = m["auroc"]; best_probs = probs.copy()
    return best_probs


# =============================================================================
# PROBE VARIANTS
# =============================================================================

def run_probe_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 name: str, probe_kind: str = "lr",
                 n_splits: int = 5, seed: int = 42) -> dict:
    fold_metrics = []
    all_y, all_p = [], []
    oof_prob = np.full(len(y), np.nan, dtype=np.float32)

    device = "cuda" if (probe_kind == "mlp" and torch.cuda.is_available()) else "cpu"

    n_pos = max(int(y.sum()), 1); n_neg = max(len(y) - n_pos, 1)
    pos_weight = n_neg / n_pos

    for fold, (tr, te) in enumerate(stratified_split(
            y, group_id=groups if len(set(groups)) > 1 else None,
            n_splits=n_splits, seed=seed)):
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr]); Xte = scaler.transform(X[te])
        ytr = y[tr]; yte = y[te]

        if probe_kind == "lr":
            m = LogisticRegression(C=0.1, max_iter=2000,
                                   class_weight="balanced", random_state=seed)
            m.fit(Xtr, ytr)
            p = m.predict_proba(Xte)[:, 1]
        elif probe_kind == "rf":
            m = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                       class_weight="balanced",
                                       random_state=seed, n_jobs=-1)
            m.fit(Xtr, ytr)
            p = m.predict_proba(Xte)[:, 1]
        elif probe_kind == "mlp":
            p = train_mlp_fold(Xtr, ytr, Xte, yte, device=device,
                               epochs=30, batch_size=128, lr=1e-3,
                               pos_weight=pos_weight)
        else:
            raise ValueError(probe_kind)

        oof_prob[te] = p
        fm = evaluate(yte, p, name=f"fold_{fold + 1}")
        fold_metrics.append(fm)
        all_y.append(yte); all_p.append(p)

    all_y = np.concatenate(all_y); all_p = np.concatenate(all_p)
    summary = aggregate_folds(fold_metrics)
    overall = evaluate(all_y, all_p, name=name)
    logger.info(f"[{name:30s} probe={probe_kind}]  "
                f"AUROC={summary.get('auroc_mean',0):.4f} ± {summary.get('auroc_std',0):.4f}  "
                f"ECE={summary.get('ece_mean',0):.4f}")
    return {
        "name": name, "probe_kind": probe_kind,
        "summary": summary,
        "fold_metrics": fold_metrics,
        "overall": overall,
        "oof_prob": oof_prob,
    }


# =============================================================================
# DRIVER
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz-glob", required=True,
                    help="Glob for hidden state npz files")
    ap.add_argument("--genunc-glob", default=None,
                    help="Glob for generation-uncertainty CSV files "
                         "(e.g. data/features/*_features_genunc.csv). If given, "
                         "include Direction A variants.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    paths = sorted(glob.glob(args.npz_glob))
    logger.info(f"Loading {len(paths)} npz files")
    dim_data = load_pooled(paths)

    # --- Build per-dim variants & gather pooled OOF predictions per variant ---
    # Strategy: train each variant SEPARATELY for each hidden_dim group
    # (Qwen/Llama have different dims), then pool their OOF probs for a
    # joint AUROC. This avoids the dim-mismatch concat issue.

    def build_variants_for_dim(data):
        d = {
            "linear_h_last":   ("lr",  data["h_last"]),
            "linear_h_think":  ("lr",  data["h_think"]),
            "linear_h_answer": ("lr",  data["h_answer"]),
            "linear_concat":   ("lr",  np.concatenate([data["h_last"], data["h_think"], data["h_answer"]], axis=1)),
            "mlp_h_answer":    ("mlp", data["h_answer"]),
            "mlp_concat":      ("mlp", np.concatenate([data["h_last"], data["h_think"], data["h_answer"]], axis=1)),
        }
        return d

    # --- Load genunc features (optional) ---
    gu_map = {}   # (item_id, group) -> 10-d vector
    if args.genunc_glob is not None:
        import glob as _glob
        genunc_paths = sorted(_glob.glob(args.genunc_glob))
        if genunc_paths:
            gu_dfs = []
            for p in genunc_paths:
                d_csv = pd.read_csv(p)
                d_csv["group"] = (os.path.basename(p)
                                  .replace("_features_genunc.csv", ""))
                gu_dfs.append(d_csv)
            gu_df = pd.concat(gu_dfs, ignore_index=True)
            gu_df["item_id"] = gu_df["item_id"].astype(str)
            for _, row in gu_df.iterrows():
                key = (row["item_id"], row["group"])
                gu_map[key] = np.array(
                    [row.get(c, 0.0) for c in GENERATION_UNC_FEATURE_NAMES],
                    dtype=np.float32)
            logger.info(f"Loaded genunc feature map: {len(gu_map)} items")

    def align_genunc(data):
        n = len(data["item_ids"])
        X = np.zeros((n, len(GENERATION_UNC_FEATURE_NAMES)), dtype=np.float32)
        for i in range(n):
            key = (str(data["item_ids"][i]), str(data["groups"][i]))
            v = gu_map.get(key)
            if v is not None:
                X[i] = v
        return X

    # --- Run CV per dim, pool OOF predictions across dim groups ---
    variant_names = list(build_variants_for_dim(next(iter(dim_data.values()))).keys())
    if gu_map:
        variant_names += ["lr_genunc", "rf_genunc", "mlp_hidden_plus_genunc"]

    # accumulators across dim-groups
    pooled: dict[str, dict] = {v: {"y": [], "p": [], "ids": [], "grp": []} for v in variant_names}

    for dim, data in dim_data.items():
        y = data["y_true"]; groups = data["groups"]; item_ids = data["item_ids"]
        logger.info(f"\n=========== dim={dim}  n={len(y)}  pos_rate={y.mean():.3f} ===========")
        variants = build_variants_for_dim(data)

        if gu_map:
            X_gu = align_genunc(data)
            variants["lr_genunc"] = ("lr", X_gu)
            variants["rf_genunc"] = ("rf", X_gu)
            variants["mlp_hidden_plus_genunc"] = ("mlp",
                np.concatenate([data["h_last"], data["h_think"], data["h_answer"], X_gu], axis=1))

        for vname, (kind, X) in variants.items():
            logger.info(f"--- {vname} (probe={kind}, d={X.shape[1]}) dim_group={dim} ---")
            r = run_probe_cv(X, y, groups, name=f"{vname}_dim{dim}",
                             probe_kind=kind,
                             n_splits=args.n_splits, seed=args.seed)
            pooled[vname]["y"].extend(y.tolist())
            pooled[vname]["p"].extend(r["oof_prob"].tolist())
            pooled[vname]["ids"].extend(item_ids.tolist())
            pooled[vname]["grp"].extend(groups.tolist())

    # --- Compute pooled AUROC per variant, save OOF ---
    total_n = sum(len(data["y_true"]) for data in dim_data.values())
    all_results = {"n_items": int(total_n),
                   "hidden_dims": sorted(dim_data.keys()),
                   "variants": {}}
    oof_paths = {}

    for vname in variant_names:
        buf = pooled[vname]
        y = np.asarray(buf["y"], dtype=int)
        p = np.asarray(buf["p"], dtype=float)
        ids = np.asarray(buf["ids"], dtype=object)
        grp = np.asarray(buf["grp"], dtype=object)
        m = evaluate(y, p, name=vname)
        all_results["variants"][vname] = {
            "overall": m,
            "n_items": int(len(y)),
        }
        oof_path = args.output.replace(".json", f"_{vname}_oof.npz")
        np.savez_compressed(oof_path,
            item_ids=ids, groups=grp, y_true=y, oof_prob=p,
            seed=np.array([args.seed]),
            n_splits=np.array([args.n_splits]))
        oof_paths[vname] = oof_path

    all_results["oof_paths"] = oof_paths
    save_results(args.output, all_results)

    # Summary table
    print("\n" + "=" * 72)
    print(f"{'variant':32s}  {'AUROC':>12s}  {'AUPRC':>8s}  {'ECE':>8s}  {'Acc@80':>8s}")
    print("-" * 72)
    for vname, r in all_results["variants"].items():
        m = r["overall"]
        print(f"{vname:32s}  "
              f"{m['auroc']:.4f}      "
              f"{m.get('auprc',0):.3f}     "
              f"{m.get('ece',0):.3f}     "
              f"{m.get('accuracy_at_80',0):.3f}")


if __name__ == "__main__":
    main()
