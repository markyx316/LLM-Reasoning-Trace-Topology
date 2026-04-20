"""
cv_utils.py - Shared cross-validation, evaluation, and pooling helpers
for the Month-2 learned models (Step Transformer, DeBERTa).

Keeps eval consistent with train_and_evaluate.py (same metrics) so we can
compare Month-1 vs Month-2 numbers apples-to-apples.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


# =============================================================================
# METRICS (mirror src/modeling/train_and_evaluate.py)
# =============================================================================

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        n = mask.sum()
        if n == 0:
            continue
        ece += (n / len(y_true)) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def selective_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    sorted_idx = np.argsort(y_prob)[::-1]
    ys = y_true[sorted_idx]
    n = len(y_true)
    return {
        "accuracy_at_80": float(ys[:max(int(0.8 * n), 1)].mean()),
        "accuracy_at_90": float(ys[:max(int(0.9 * n), 1)].mean()),
    }


def evaluate(y_true, y_prob, name: str = "") -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(np.unique(y_true)) < 2:
        return {"method": name, "auroc": 0.5, "auprc": float(y_true.mean()),
                "ece": 0.0, "accuracy": float(y_true.mean()), "f1": 0.0,
                "n_samples": len(y_true)}
    auroc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    ece = compute_ece(y_true, y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    sel = selective_metrics(y_true, y_prob)
    return {"method": name, "auroc": auroc, "auprc": auprc, "ece": ece,
            "accuracy": acc, "f1": f1,
            "accuracy_at_80": sel["accuracy_at_80"],
            "accuracy_at_90": sel["accuracy_at_90"],
            "n_samples": len(y_true), "n_correct": int(y_true.sum()),
            "base_accuracy": float(y_true.mean())}


# =============================================================================
# DATA POOLING
# =============================================================================

def stratified_split(y: np.ndarray, group_id: Optional[np.ndarray] = None,
                     n_splits: int = 5, seed: int = 42):
    """
    Yield (train_idx, test_idx) pairs.
    If group_id given, stratify on (group_id, y) joint key so each fold
    has balanced datasets AND labels.
    """
    if group_id is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in skf.split(np.zeros_like(y), y):
            yield tr, te
    else:
        # Build joint stratification key
        key = np.array([f"{g}_{c}" for g, c in zip(group_id, y)])
        # Map to integer codes
        _, codes = np.unique(key, return_inverse=True)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in skf.split(np.zeros_like(codes), codes):
            yield tr, te


# =============================================================================
# RESULT AGGREGATION
# =============================================================================

def aggregate_folds(fold_metrics: list[dict],
                    keys: tuple = ("auroc", "auprc", "ece",
                                   "accuracy_at_80", "accuracy_at_90")) -> dict:
    out = {}
    for k in keys:
        vals = [fm[k] for fm in fold_metrics if k in fm]
        if vals:
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_std"] = float(np.std(vals))
    return out


def save_results(path: str, results: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info(f"Saved: {path}")


# =============================================================================
# DATASET POOLING (load multiple .npz into a single training set with group_id)
# =============================================================================

def load_pooled_npz(npz_paths: list[str]):
    """Load multiple .npz step-embedding files and concatenate. Returns dict."""
    all_emb, all_typ, all_y, all_id, all_group = [], [], [], [], []
    for p in npz_paths:
        z = np.load(p, allow_pickle=True)
        emb = z["embeddings"]; typ = z["step_types"]; y = z["is_correct"]; ids = z["item_ids"]
        all_emb.extend(list(emb)); all_typ.extend(list(typ))
        all_y.extend(list(y)); all_id.extend(list(ids))
        gname = os.path.basename(p).replace(".npz", "")
        all_group.extend([gname] * len(y))
    return {
        "embeddings": all_emb,           # list of (n_steps, d)
        "step_types": all_typ,           # list of (n_steps,)
        "labels": np.array(all_y, dtype=np.int64),
        "item_ids": np.array(all_id, dtype=object),
        "groups": np.array(all_group, dtype=object),
    }
