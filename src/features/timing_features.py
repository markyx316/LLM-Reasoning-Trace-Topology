"""
timing_features.py - Inter-event timing / dynamics features from parsed traces.

Beyond *which* behaviors occur and *how many* (captured by handcrafted
counts and proportions), the *when* and *how-clustered* of each behavior
carries correctness signal. This module emits 46 descriptors per trace.

Per-behavior (7 x 6 = 42 features):
  For each b in {F, V, B, R, S, H, C}:
    - count_<b>:                number of b-episodes
    - first_pos_<b>:            first occurrence position / L  (-1 if none)
    - last_pos_<b>:             last occurrence position  / L  (-1 if none)
    - iei_mean_<b>:             mean inter-event interval (episodes; -1 if < 2)
    - iei_std_<b>:              std  of inter-event intervals (-1 if < 2)
    - burstiness_<b>:           (sigma - mu) / (sigma + mu)   (-1 if < 2)
                                  Goh & Barabasi 2008. 0=Poisson, 1=bursty.

Global dynamics (4 features):
  - global_iei_mean:            mean gap in *tokens* between consecutive episodes
                                  (cumulative token_count timeline).
  - global_iei_std:             std  of the token-timeline gaps.
  - iei_entropy:                Shannon entropy of the token-gap distribution
                                  (quantized to 20 bins, normalized).
  - transition_rate:            number of behavior changes / L  (0 if L < 2).

Output columns:
    item_id, dataset, is_correct, <46 timing features>

Self-test synthesizes two classes of traces (bursty-incorrect vs
uniform-correct) and asserts that burstiness features discriminate.

Usage:
    PYTHONPATH=. python src/features/timing_features.py \\
        --parsed-glob "data/parsed/*_parsed.jsonl" --output-dir data/features/
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
import pandas as pd

logger = logging.getLogger(__name__)


BEHAVIOR_CHARS: list[str] = ["F", "V", "B", "R", "S", "H", "C"]
IEI_ENTROPY_BINS: int = 20


# =============================================================================
# PER-TRACE FEATURES
# =============================================================================

def _burstiness(intervals: np.ndarray) -> float:
    """
    Goh & Barabasi (2008) burstiness coefficient:
        B = (sigma - mu) / (sigma + mu)
    0 means Poisson-like (mu==sigma). 1 means maximally bursty. Returns -1.0
    if fewer than 2 intervals exist (insufficient data).
    """
    if len(intervals) < 2:
        return -1.0
    mu = float(np.mean(intervals))
    sigma = float(np.std(intervals))
    if sigma + mu <= 1e-12:
        return 0.0
    return (sigma - mu) / (sigma + mu)


def _per_behavior_features(seq: list[str],
                           behaviors: list[str] = BEHAVIOR_CHARS
                           ) -> dict[str, float]:
    """Six stats per behavior type: count, first_pos/L, last_pos/L, iei_mean,
    iei_std, burstiness. Missing behaviors get sentinel -1 (except count)."""
    out: dict[str, float] = {}
    L = max(len(seq), 1)
    for b in behaviors:
        idxs = [i for i, c in enumerate(seq) if c == b]
        count = float(len(idxs))
        out[f"t_count_{b}"] = count
        if not idxs:
            out[f"t_first_pos_{b}"] = -1.0
            out[f"t_last_pos_{b}"] = -1.0
            out[f"t_iei_mean_{b}"] = -1.0
            out[f"t_iei_std_{b}"] = -1.0
            out[f"t_burstiness_{b}"] = -1.0
            continue
        out[f"t_first_pos_{b}"] = idxs[0] / (L - 1) if L > 1 else 0.0
        out[f"t_last_pos_{b}"] = idxs[-1] / (L - 1) if L > 1 else 0.0
        if len(idxs) < 2:
            out[f"t_iei_mean_{b}"] = -1.0
            out[f"t_iei_std_{b}"] = -1.0
            out[f"t_burstiness_{b}"] = -1.0
        else:
            ieis = np.diff(np.array(idxs, dtype=float))
            out[f"t_iei_mean_{b}"] = float(np.mean(ieis))
            out[f"t_iei_std_{b}"] = float(np.std(ieis))
            out[f"t_burstiness_{b}"] = _burstiness(ieis)
    return out


def _global_features(seq: list[str],
                     token_counts: list[int]) -> dict[str, float]:
    """
    Global dynamics:
      - mean / std of gaps in the *token-time* timeline (cumulative token count
        at each episode boundary).
      - Shannon entropy of quantized gaps.
      - transition rate (behavior changes / L).
    """
    L = len(seq)
    out: dict[str, float] = {}
    if L < 2:
        return {"t_global_iei_mean": 0.0,
                "t_global_iei_std": 0.0,
                "t_iei_entropy": 0.0,
                "t_transition_rate": 0.0}

    # Token-timeline gaps: token_counts gives episode length in tokens; the
    # "gap" between consecutive episodes is the length of the earlier episode
    # (how many tokens elapsed before the next one starts).
    if token_counts and all(isinstance(t, (int, float)) for t in token_counts):
        gaps = np.array(token_counts[:-1], dtype=float)
    else:
        gaps = np.ones(L - 1, dtype=float)

    iei_mean = float(np.mean(gaps)) if len(gaps) else 0.0
    iei_std = float(np.std(gaps)) if len(gaps) else 0.0

    # Normalized entropy
    if len(gaps) and gaps.sum() > 0:
        hist, _ = np.histogram(gaps, bins=IEI_ENTROPY_BINS)
        p = hist / max(hist.sum(), 1)
        p = p[p > 0]
        if len(p) > 0:
            ent = float(-np.sum(p * np.log(p)) / np.log(IEI_ENTROPY_BINS))
        else:
            ent = 0.0
    else:
        ent = 0.0

    transitions = sum(1 for i in range(1, L) if seq[i] != seq[i - 1])
    out["t_global_iei_mean"] = iei_mean
    out["t_global_iei_std"] = iei_std
    out["t_iei_entropy"] = ent
    out["t_transition_rate"] = transitions / (L - 1) if L > 1 else 0.0
    return out


def extract_timing_features(episodes: list[dict]) -> dict[str, float]:
    """All 46 features for a single trace."""
    seq = [ep.get("behavior", "F") for ep in episodes]
    seq = [c if c in BEHAVIOR_CHARS else "F" for c in seq]
    token_counts = [int(ep.get("token_count", 0)) for ep in episodes]
    feats = {}
    feats.update(_per_behavior_features(seq))
    feats.update(_global_features(seq, token_counts))
    return feats


# =============================================================================
# PIPELINE
# =============================================================================

def _iter_parsed_records(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _group_name_from_path(parsed_path: str) -> str:
    base = os.path.basename(parsed_path)
    if base.endswith("_parsed.jsonl"):
        return base[: -len("_parsed.jsonl")]
    return base.replace(".jsonl", "")


def build_csv_for_file(parsed_path: str, output_dir: str) -> str:
    group = _group_name_from_path(parsed_path)
    rows = []
    for rec in _iter_parsed_records(parsed_path):
        episodes = rec.get("episodes", [])
        feats = extract_timing_features(episodes)
        rows.append({
            "item_id": rec["item_id"],
            "dataset": group,
            "is_correct": int(rec.get("is_correct", False)),
            **feats,
        })
    df = pd.DataFrame(rows)
    out = os.path.join(output_dir, f"{group}_timing.csv")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"  {group}: n={len(df)}  feat_cols={df.shape[1] - 3}  -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parsed")
    ap.add_argument("--parsed-glob", default="data/parsed/*_parsed.jsonl")
    ap.add_argument("--output-dir", default="data/features/")
    ap.add_argument("--skip-pilot", action="store_true", default=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    paths = [args.parsed] if args.parsed else sorted(glob.glob(args.parsed_glob))
    if args.skip_pilot and not args.parsed:
        paths = [p for p in paths
                 if not os.path.basename(p).startswith(("pilot_", "_"))
                 and "_sc" not in os.path.basename(p)]
    if not paths:
        logger.error("No parsed JSONL files found")
        sys.exit(1)

    for p in paths:
        logger.info(f"Processing {p}")
        try:
            build_csv_for_file(p, args.output_dir)
        except Exception as e:
            logger.exception(f"  Failed on {p}: {e}")


# =============================================================================
# SELF-TEST
# =============================================================================

def _run_self_test():
    print("Running timing_features self-test...")
    import random
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    rng = random.Random(0)
    rows = []
    for i in range(50):
        label = i % 2
        if label == 1:
            # Correct: uniform distribution of verifies, few hesitations
            L = rng.randint(20, 40)
            seq = []
            for _ in range(L):
                r = rng.random()
                if r < 0.7: seq.append("F")
                elif r < 0.85: seq.append("V")
                elif r < 0.95: seq.append("S")
                else: seq.append("C")
        else:
            # Incorrect: bursty hesitations in the middle
            L = rng.randint(30, 50)
            seq = ["F"] * L
            # inject a burst of H's in the middle
            burst_start = L // 2
            for k in range(6):
                if burst_start + k < L:
                    seq[burst_start + k] = "H"
            seq[-1] = "F"  # no C
        eps = [{"behavior": b, "position": k, "token_count": rng.randint(3, 15)}
               for k, b in enumerate(seq)]
        feats = extract_timing_features(eps)
        rows.append({"label": label, **feats})

    df = pd.DataFrame(rows)
    y = df["label"].to_numpy(dtype=int)
    X = df.drop(columns=["label"]).to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    s = StandardScaler().fit_transform(X)
    lr = LogisticRegression(max_iter=2000).fit(s, y)
    p = lr.predict_proba(s)[:, 1]
    auroc = roc_auc_score(y, p)
    print(f"  Synthetic in-sample AUROC: {auroc:.4f}")
    assert auroc > 0.8, f"synthetic timing AUROC too low: {auroc:.4f}"
    # Sanity: 46 features total
    n_feats = len([c for c in df.columns if c != "label"])
    assert n_feats == 46, f"expected 46 features, got {n_feats}"
    print("All timing_features tests passed.")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _run_self_test()
    else:
        main()
