"""
topology_features.py - Persistent homology features on per-step embeddings.

For each reasoning trace, treats the per-step MiniLM (all-MiniLM-L6-v2, 384-d)
sentence embeddings as a point cloud in embedding space and extracts summary
features of its Vietoris–Rips persistent homology up to dimension 1.

This is the text-surface analogue of Minegishi et al. (NeurIPS 2025,
"Topology of Reasoning", which uses hidden-state features) and the direct
text-only competitor to Tan et al. (arXiv:2510.20665, "The Shape of Reasoning",
which applied text-PH on AIME only).

Features per trace (7 total):
  h0_total_persistence    - sum of finite H0 bar lengths (cluster-gap signal)
  h0_max_persistence      - longest finite H0 bar (dominant cluster gap)
  h0_n_bars               - number of finite H0 bars (# of distinct clusters)
  h0_persistence_entropy  - Shannon entropy of normalised H0 bar lengths
  h1_total_persistence    - sum of H1 bar lengths (# and strength of loops)
  h1_max_persistence      - longest H1 bar (most persistent cycle)
  h1_n_bars               - number of H1 bars (# of loops detected)

Hypothesis, downstream of the recurrence-feature findings:
  Correct traces walk a linear-ish trajectory in semantic space — few loops
  (low h1_*) and compact clusters (low h0_*). Incorrect traces rumination
  revisits earlier steps, creating persistent cycles (high h1_*) and
  multiple disjoint clusters (high h0_*).

Distance metric:
  MiniLM embeddings are L2-normalised at encode time (see
  scripts/build_step_embeddings.py), so Euclidean distance is a monotone
  transform of cosine distance: ||x-y||^2 = 2(1 - x·y). We pass the point
  cloud directly to ripser and use Euclidean (ripser default).

Usage:
    # Single dataset:
    PYTHONPATH=. python src/features/topology_features.py \\
        --npz data/step_embeddings/math500_qwen7b.npz

    # All 8 datasets at once:
    PYTHONPATH=. python src/features/topology_features.py --all

Output schema:
    data/features/v2/<dataset>_features_ph.csv
    columns: item_id, is_correct, dataset, + 7 h0_* / h1_* features

Dependencies:
    pip install ripser numpy pandas tqdm
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


FEATURE_NAMES = [
    "h0_total_persistence",
    "h0_max_persistence",
    "h0_n_bars",
    "h0_persistence_entropy",
    "h1_total_persistence",
    "h1_max_persistence",
    "h1_n_bars",
]

DATASETS = [
    "math500_qwen7b", "math500_llama8b",
    "gsm8k_qwen7b", "gsm8k_llama8b",
    "gpqa_diamond_qwen7b", "gpqa_diamond_llama8b",
    "arc_challenge_qwen7b", "arc_challenge_llama8b",
]


# =============================================================================
# PH computation
# =============================================================================

def _zero_features() -> dict[str, float]:
    return {f: 0.0 for f in FEATURE_NAMES}


def compute_ph_features(emb: np.ndarray, maxdim: int = 1,
                        n_perm: int | None = None) -> dict[str, float]:
    """
    Persistent-homology features for a single trace's step embeddings.

    Parameters
    ----------
    emb : np.ndarray of shape (n_steps, emb_dim)
        Per-step embeddings. Must have n_steps >= 3 for a meaningful PH;
        zero features are returned for shorter traces.
    maxdim : int
        Maximum homology dimension to compute. 1 covers H0 + H1.
    n_perm : int or None
        If set, subsample to this many landmark points to speed up ripser on
        long traces (>200 steps). None = use all points.

    Returns
    -------
    dict mapping each name in FEATURE_NAMES to a float.
    """
    if emb is None or len(emb) < 3:
        return _zero_features()

    # Lazy import so the file can be imported even without ripser installed
    from ripser import ripser

    emb = np.asarray(emb, dtype=np.float32)
    kwargs = {"maxdim": maxdim}
    if n_perm is not None and n_perm < len(emb):
        kwargs["n_perm"] = n_perm

    try:
        result = ripser(emb, **kwargs)
    except Exception as e:  # pragma: no cover - diagnostic
        logger.warning(f"ripser failed on shape {emb.shape}: {e}")
        return _zero_features()

    dgms = result["dgms"]  # list of (n_bars, 2) arrays per dimension

    # H0: drop the single infinite bar (always present, represents the overall
    # connected component that never dies); keep the finite bars.
    h0 = dgms[0] if len(dgms) > 0 else np.zeros((0, 2))
    h0_finite = h0[np.isfinite(h0[:, 1])]
    h0_lengths = (h0_finite[:, 1] - h0_finite[:, 0])
    h0_lengths = h0_lengths[h0_lengths > 0]

    # H1: all finite by construction at the max filtration
    h1 = dgms[1] if len(dgms) > 1 else np.zeros((0, 2))
    h1_finite = h1[np.isfinite(h1[:, 1])]
    h1_lengths = (h1_finite[:, 1] - h1_finite[:, 0])
    h1_lengths = h1_lengths[h1_lengths > 0]

    # H0 aggregates
    if len(h0_lengths) > 0:
        h0_total = float(h0_lengths.sum())
        h0_max = float(h0_lengths.max())
        h0_n = int(len(h0_lengths))
        if h0_total > 0 and len(h0_lengths) > 1:
            p = h0_lengths / h0_total
            h0_ent = float(-(p * np.log(p + 1e-12)).sum())
        else:
            h0_ent = 0.0
    else:
        h0_total = h0_max = h0_ent = 0.0
        h0_n = 0

    # H1 aggregates
    if len(h1_lengths) > 0:
        h1_total = float(h1_lengths.sum())
        h1_max = float(h1_lengths.max())
        h1_n = int(len(h1_lengths))
    else:
        h1_total = h1_max = 0.0
        h1_n = 0

    return {
        "h0_total_persistence": h0_total,
        "h0_max_persistence": h0_max,
        "h0_n_bars": float(h0_n),
        "h0_persistence_entropy": h0_ent,
        "h1_total_persistence": h1_total,
        "h1_max_persistence": h1_max,
        "h1_n_bars": float(h1_n),
    }


# =============================================================================
# Batch driver
# =============================================================================

def extract_from_npz(npz_path: str, max_steps: int = 256,
                     n_perm: int | None = None) -> pd.DataFrame:
    """Compute PH features for every trace in a step-embedding .npz file."""
    z = np.load(npz_path, allow_pickle=True)
    item_ids = z["item_ids"]
    labels = z["is_correct"].astype(int)
    embeddings = z["embeddings"]

    dataset = os.path.basename(npz_path).replace(".npz", "")
    rows = []
    for i, emb in enumerate(tqdm(embeddings, desc=f"PH {dataset}", total=len(embeddings))):
        if emb is None or len(emb) == 0:
            feats = _zero_features()
        else:
            if len(emb) > max_steps:
                emb = emb[:max_steps]
            feats = compute_ph_features(emb, n_perm=n_perm)
        rows.append({
            "item_id": str(item_ids[i]),
            "is_correct": int(labels[i]),
            "dataset": dataset,
            **feats,
        })
    return pd.DataFrame(rows)


def _summary_stats(df: pd.DataFrame) -> str:
    """Quick diagnostic: mean of each feature by is_correct class."""
    parts = []
    for f in FEATURE_NAMES:
        c = df.loc[df.is_correct == 1, f].mean()
        w = df.loc[df.is_correct == 0, f].mean()
        diff = c - w
        parts.append(f"  {f:<26s}  correct={c:.3f}  incorrect={w:.3f}  Δ={diff:+.3f}")
    return "\n".join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--npz", nargs="+", help="Input .npz step-embedding file(s)")
    g.add_argument("--all", action="store_true",
                   help="Process all 8 standard datasets under --npz-dir")
    p.add_argument("--out-dir", default="data/features/v2",
                   help="Output dir for *_features_ph.csv files")
    p.add_argument("--out", default=None,
                   help="Explicit output CSV path (only valid with single --npz)")
    p.add_argument("--max-steps", type=int, default=256,
                   help="Truncate traces beyond this many steps")
    p.add_argument("--n-perm", type=int, default=None,
                   help="Landmark subsampling for traces longer than n_perm "
                        "(default: no subsampling)")
    p.add_argument("--npz-dir", default="data/step_embeddings",
                   help="Step-embedding .npz directory (for --all mode)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Confirm ripser is installed before any work begins
    try:
        import ripser  # noqa: F401
    except ImportError:
        logger.error("ripser is not installed in this env. "
                     "Run: pip install ripser")
        sys.exit(2)

    os.makedirs(args.out_dir, exist_ok=True)

    # Resolve (input, output) pairs
    if args.all:
        pairs = [(os.path.join(args.npz_dir, f"{d}.npz"),
                  os.path.join(args.out_dir, f"{d}_features_ph.csv"))
                 for d in DATASETS]
    else:
        if args.out and len(args.npz) > 1:
            p.error("--out is only valid with a single --npz path")
        if args.out:
            pairs = [(args.npz[0], args.out)]
        else:
            pairs = [(n,
                      os.path.join(args.out_dir,
                                   os.path.basename(n).replace(".npz", "_features_ph.csv")))
                     for n in args.npz]

    for npz_path, out_path in pairs:
        if not os.path.exists(npz_path):
            logger.warning(f"Missing input: {npz_path} (skip)")
            continue
        if os.path.exists(out_path):
            logger.info(f"Already exists: {out_path} (skip; delete to rerun)")
            continue
        logger.info(f"{npz_path} -> {out_path}")
        df = extract_from_npz(npz_path, max_steps=args.max_steps, n_perm=args.n_perm)
        df.to_csv(out_path, index=False)
        logger.info(f"  wrote {len(df)} rows; feature means by correctness:")
        logger.info("\n" + _summary_stats(df))


if __name__ == "__main__":
    main()
