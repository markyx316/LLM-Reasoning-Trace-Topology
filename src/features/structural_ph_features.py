"""
structural_ph_features.py - Content-free persistent-homology features.

Replaces the MiniLM 384-d point cloud used by topology_features_v2.py with
a 13-d CONTENT-FREE per-episode coordinate, then runs the same persistence
pipeline (ripser + persistence images). The motivation: v2's features mix
content signal (MiniLM captures lexical / semantic context) with structural
signal; this module isolates the structural contribution.

Per-episode 13-d coordinate:
    [0..6]  7-d one-hot behavior (F, V, B, R, S, H, C)
    [7]     normalized position = i / max(L - 1, 1)
    [8]     token_count z-score within the trace
    [9]     running behavior-diversity H(counts_so_far) / log(7)
    [10]    time-since-last-B (normalized by L, -1 if no prior B)
    [11]    time-since-last-V (normalized by L, -1 if no prior V)
    [12]    time-since-last-R (normalized by L, -1 if no prior R)

Features emitted (same schema as topology_features_v2.py, 36 feats):
    h0_total_per_step, h0_n_per_step, h1_total_per_step, h1_n_per_step
    h{0,1}_pi_{0..3}_{0..3}   (4x4 persistence-image grids)

CSV output:  data/features/{dataset}_{model}_structural_ph.csv
    columns: item_id, dataset, is_correct, <36 features>

Why CONTENT-FREE?
  The text-encoder falsifier asks: "does structural modeling add value
  over content?" The MiniLM-based PH can't answer that in isolation —
  it carries content. This module does.

Usage:
    PYTHONPATH=. python src/features/structural_ph_features.py \\
        --parsed-glob "data/parsed/*_parsed.jsonl" --output-dir data/features/

Dependencies: ripser, numpy, pandas.
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
import pandas as pd

logger = logging.getLogger(__name__)

# Reuse the persistence-image helpers from topology_features_v2
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.features.topology_features_v2 import (  # noqa: E402
    PI_RES, _normalize_diagram, _persistence_image, feature_names as v2_feature_names,
)


BEHAVIOR_CHARS: list[str] = ["F", "V", "B", "R", "S", "H", "C"]
BEHAVIOR_TO_IDX: dict[str, int] = {b: i for i, b in enumerate(BEHAVIOR_CHARS)}
COORD_DIM: int = 13


# =============================================================================
# 13-D CONTENT-FREE COORDS
# =============================================================================

def build_structural_coords(episodes: list[dict]) -> np.ndarray:
    """
    Return (L, 13) float32 coordinate matrix. Empty -> (0, 13).

    Feature layout (see module docstring for rationale):
        cols 0..6  : one-hot behavior (F, V, B, R, S, H, C)
        col 7      : normalized position  i / max(L-1, 1)
        col 8      : token_count z-score within the trace
        col 9      : running diversity entropy / log(7)
        col 10     : time-since-last-B, normalized (-1 if none yet)
        col 11     : time-since-last-V, normalized (-1 if none yet)
        col 12     : time-since-last-R, normalized (-1 if none yet)
    """
    L = len(episodes)
    if L == 0:
        return np.zeros((0, COORD_DIM), dtype=np.float32)

    beh = [ep.get("behavior", "F") for ep in episodes]
    beh = [b if b in BEHAVIOR_TO_IDX else "F" for b in beh]
    tok = np.array([float(ep.get("token_count", 0.0)) for ep in episodes],
                   dtype=np.float32)

    # Token z-score (stable for zero-var case)
    mu, sd = float(tok.mean()), float(tok.std())
    if sd < 1e-6:
        z = np.zeros(L, dtype=np.float32)
    else:
        z = ((tok - mu) / sd).astype(np.float32)

    # Position / running diversity / time-since-last trackers
    counts = np.zeros(7, dtype=np.int64)
    last_B = last_V = last_R = -1
    coords = np.zeros((L, COORD_DIM), dtype=np.float32)
    denom_L = max(L - 1, 1)
    for i, b in enumerate(beh):
        idx = BEHAVIOR_TO_IDX[b]
        coords[i, idx] = 1.0
        coords[i, 7] = i / denom_L
        coords[i, 8] = z[i]

        counts[idx] += 1
        n_seen = counts.sum()
        if n_seen > 0:
            p = counts / n_seen
            p = p[p > 0]
            ent = float(-np.sum(p * np.log(p)))
        else:
            ent = 0.0
        coords[i, 9] = ent / math.log(7) if ent > 0 else 0.0

        coords[i, 10] = (i - last_B) / L if last_B >= 0 else -1.0
        coords[i, 11] = (i - last_V) / L if last_V >= 0 else -1.0
        coords[i, 12] = (i - last_R) / L if last_R >= 0 else -1.0

        if b == "B":
            last_B = i
        if b == "V":
            last_V = i
        if b == "R":
            last_R = i
    return coords


# =============================================================================
# PH COMPUTATION
# =============================================================================

def _zero_features() -> dict[str, float]:
    return {name: 0.0 for name in v2_feature_names()}


def compute_structural_ph(coords: np.ndarray, maxdim: int = 1
                          ) -> dict[str, float]:
    """
    Run ripser on the 13-d content-free coordinates and return the same
    36-feature dictionary topology_features_v2 produces on MiniLM coords.
    """
    if coords is None or len(coords) < 3:
        return _zero_features()

    try:
        from ripser import ripser
    except ImportError:
        logger.error("ripser not installed: pip install ripser")
        raise

    try:
        result = ripser(coords.astype(np.float32), maxdim=maxdim)
    except Exception as e:
        logger.warning(f"ripser failed on shape {coords.shape}: {e}")
        return _zero_features()

    dgms = result["dgms"]
    h0 = dgms[0] if len(dgms) > 0 else np.zeros((0, 2))
    h1 = dgms[1] if len(dgms) > 1 else np.zeros((0, 2))

    h0_finite = h0[np.isfinite(h0[:, 1])]
    h1_finite = h1[np.isfinite(h1[:, 1])]
    h0_lens = (h0_finite[:, 1] - h0_finite[:, 0]) if len(h0_finite) else np.array([])
    h1_lens = (h1_finite[:, 1] - h1_finite[:, 0]) if len(h1_finite) else np.array([])
    h0_lens = h0_lens[h0_lens > 0]
    h1_lens = h1_lens[h1_lens > 0]

    n_steps = len(coords)
    feats: dict[str, float] = {
        "h0_total_per_step": float(h0_lens.sum()) / max(n_steps, 1),
        "h0_n_per_step":     float(len(h0_lens)) / max(n_steps, 1),
        "h1_total_per_step": float(h1_lens.sum()) / max(n_steps, 1),
        "h1_n_per_step":     float(len(h1_lens)) / max(n_steps, 1),
    }

    # Scale normalization: use max finite death over all bars
    if len(h0_finite) or len(h1_finite):
        deaths = np.concatenate([
            h0_finite[:, 1] if len(h0_finite) else np.array([]),
            h1_finite[:, 1] if len(h1_finite) else np.array([]),
        ])
        deaths = deaths[np.isfinite(deaths) & (deaths > 0)]
        scale = float(deaths.max()) if len(deaths) else 1.0
    else:
        scale = 1.0

    h0_norm = _normalize_diagram(h0, scale)
    h1_norm = _normalize_diagram(h1, scale)
    h0_img = _persistence_image(h0_norm)
    h1_img = _persistence_image(h1_norm)
    for r in range(PI_RES):
        for c in range(PI_RES):
            idx = r * PI_RES + c
            feats[f"h0_pi_{r}_{c}"] = float(h0_img[idx])
            feats[f"h1_pi_{r}_{c}"] = float(h1_img[idx])
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


def build_csv_for_file(parsed_path: str, output_dir: str,
                       max_steps: int = 256) -> str:
    """Read one parsed JSONL -> structural-PH CSV."""
    group = _group_name_from_path(parsed_path)
    rows = []
    for rec in _iter_parsed_records(parsed_path):
        episodes = rec.get("episodes", []) or []
        if len(episodes) > max_steps:
            episodes = episodes[:max_steps]
        coords = build_structural_coords(episodes)
        feats = compute_structural_ph(coords)
        rows.append({
            "item_id": rec["item_id"],
            "dataset": group,
            "is_correct": int(rec.get("is_correct", False)),
            **feats,
        })
    df = pd.DataFrame(rows)
    out = os.path.join(output_dir, f"{group}_structural_ph.csv")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info(f"  {group}: n={len(df)}  feat_cols={df.shape[1] - 3}  -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parsed")
    ap.add_argument("--parsed-glob", default="data/parsed/*_parsed.jsonl")
    ap.add_argument("--output-dir", default="data/features/")
    ap.add_argument("--max-steps", type=int, default=256,
                    help="Cap episodes per trace to bound ripser cost (256 is "
                         "well above typical trace length ~50)")
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
            build_csv_for_file(p, args.output_dir, max_steps=args.max_steps)
        except Exception as e:
            logger.exception(f"  Failed on {p}: {e}")


# =============================================================================
# SELF-TEST
# =============================================================================

def _run_self_test():
    print("Running structural_ph_features self-test...")

    # Coord builder: empty
    c = build_structural_coords([])
    assert c.shape == (0, 13)

    # Coord builder: simple case
    eps = [{"behavior": "F", "position": i, "token_count": 5} for i in range(5)]
    c = build_structural_coords(eps)
    assert c.shape == (5, 13), f"expected (5,13), got {c.shape}"
    # First row: F one-hot, pos 0, time-since -1, -1, -1
    assert abs(c[0, 0] - 1.0) < 1e-6 and abs(c[0, 7]) < 1e-6
    assert c[0, 10] == -1.0 and c[0, 11] == -1.0 and c[0, 12] == -1.0

    # PH on synthetic
    try:
        import ripser  # noqa: F401
    except ImportError:
        print("  ripser not installed; skipping PH check.")
        return

    import random
    rng = random.Random(0)
    # 20-episode trace
    seq = []
    for _ in range(20):
        r = rng.random()
        if r < 0.6: seq.append("F")
        elif r < 0.7: seq.append("V")
        elif r < 0.8: seq.append("B")
        else: seq.append("H")
    seq[-1] = "C"
    eps = [{"behavior": b, "position": i, "token_count": rng.randint(3, 15)}
           for i, b in enumerate(seq)]
    coords = build_structural_coords(eps)
    feats = compute_structural_ph(coords)
    n_expected = len(v2_feature_names())
    assert len(feats) == n_expected, f"expected {n_expected} feats, got {len(feats)}"
    # h0 should have some persistence
    assert feats["h0_n_per_step"] > 0, "h0 should have bars on 20 points"

    print("All structural_ph_features tests passed.")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _run_self_test()
    else:
        main()
