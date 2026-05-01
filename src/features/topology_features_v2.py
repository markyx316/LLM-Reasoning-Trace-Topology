"""
topology_features_v2.py — Length-normalized persistent-homology features.

Improves on topology_features.py based on the v1 diagnostic finding that
4 of 7 v1 features had r ≈ 0.76 with trace length (i.e. they were length
in disguise).

This module produces TWO families of features per trace, written to a single
CSV per dataset (data/features/v2/*_features_phimg.csv):

  (A) 4 LENGTH-NORMALIZED scalar features (sanity-check baseline)
        h0_total_per_step, h0_n_per_step,
        h1_total_per_step, h1_n_per_step

  (B) 32 PERSISTENCE-IMAGE features (Adams et al. JMLR 2017)
        H0: 4x4 grid = 16 features → h0_pi_<row>_<col>
        H1: 4x4 grid = 16 features → h1_pi_<row>_<col>
      Each diagram is first NORMALIZED by max-pairwise-distance so all
      images live in a fixed [0,1] × [0,1] box, decoupling shape from
      absolute scale (the v1 length-confound).

Total: 36 features per trace. Stored alongside item_id, dataset, is_correct.

Hypothesis: if normalization removes the length confound, persistence images
will capture genuine geometric structure that the recurrence-5 and
handcrafted-25 don't. If they STILL add zero lift over STRUCTURAL_FULL, we
have strong evidence that PH on text-step embeddings is fundamentally
redundant with semantic-recurrence features at this point-cloud size
(~50 steps in 384-d).

Dependencies:
    pip install ripser persim numpy pandas tqdm

Usage:
    PYTHONPATH=. python src/features/topology_features_v2.py --all
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


PI_RES = 4   # 4x4 persistence-image grid per homology dimension
PI_SIGMA = 0.05  # Gaussian bandwidth for persistence image kernel

DATASETS = [
    "math500_qwen7b", "math500_llama8b",
    "gsm8k_qwen7b", "gsm8k_llama8b",
    "gpqa_diamond_qwen7b", "gpqa_diamond_llama8b",
    "arc_challenge_qwen7b", "arc_challenge_llama8b",
]


def _norm_scalar_names() -> list[str]:
    return [
        "h0_total_per_step", "h0_n_per_step",
        "h1_total_per_step", "h1_n_per_step",
    ]


def _pi_feature_names() -> list[str]:
    cols = []
    for h in (0, 1):
        for r in range(PI_RES):
            for c in range(PI_RES):
                cols.append(f"h{h}_pi_{r}_{c}")
    return cols


def feature_names() -> list[str]:
    return _norm_scalar_names() + _pi_feature_names()


def _zero_features() -> dict[str, float]:
    return {f: 0.0 for f in feature_names()}


def _normalize_diagram(dgm: np.ndarray, scale: float) -> np.ndarray:
    """Divide birth/death by `scale` (typically max pairwise distance) so
    the diagram lives in [0,1] × [0,1]. Drops infinite bars."""
    if dgm is None or len(dgm) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    finite = dgm[np.isfinite(dgm[:, 1])]
    if len(finite) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if scale <= 0:
        return finite.astype(np.float32)
    return (finite / scale).astype(np.float32)


def _persistence_image(dgm_norm: np.ndarray, res: int = PI_RES,
                       sigma: float = PI_SIGMA) -> np.ndarray:
    """Compute a persistence image for a single normalized diagram.

    Lightweight in-house implementation following Adams et al. 2017:
      1. Convert (birth, death) -> (birth, persistence) where persistence = d - b
      2. Place an isotropic Gaussian at each point with a linear weight
         w(b, p) = p   (more persistent features get more weight)
      3. Discretize over the unit square into a res×res grid
      4. Return the flattened res*res vector

    Avoids needing the `persim` package (which has older numpy dependency
    issues in some envs); only requires numpy.
    """
    img = np.zeros((res, res), dtype=np.float32)
    if len(dgm_norm) == 0:
        return img.ravel()

    # (birth, persistence) coordinates, both in [0, 1] after normalization
    b = dgm_norm[:, 0].clip(0, 1)
    p = (dgm_norm[:, 1] - dgm_norm[:, 0]).clip(0, 1)

    # Grid centers
    centers = (np.arange(res) + 0.5) / res   # 0.125, 0.375, 0.625, 0.875 for res=4
    bx, py = np.meshgrid(centers, centers, indexing="ij")
    # Gaussian width — 2*sigma^2 in the exponent
    inv2s2 = 1.0 / (2.0 * sigma * sigma)

    # Vectorized: for each (b_i, p_i, w_i), add w_i * exp(-((bx-b_i)^2 + (py-p_i)^2)/2sigma^2)
    for bi, pi in zip(b, p):
        d2 = (bx - bi) ** 2 + (py - pi) ** 2
        img += pi * np.exp(-d2 * inv2s2)
    return img.ravel().astype(np.float32)


def compute_phimg_features(emb: np.ndarray, maxdim: int = 1) -> dict[str, float]:
    """Per-trace feature dict — 4 length-normalized scalars + 32 PI features."""
    if emb is None or len(emb) < 3:
        return _zero_features()

    from ripser import ripser

    emb = np.asarray(emb, dtype=np.float32)
    n_steps = len(emb)

    try:
        result = ripser(emb, maxdim=maxdim)
    except Exception as e:  # pragma: no cover
        logger.warning(f"ripser failed on shape {emb.shape}: {e}")
        return _zero_features()

    dgms = result["dgms"]
    h0 = dgms[0] if len(dgms) > 0 else np.zeros((0, 2))
    h1 = dgms[1] if len(dgms) > 1 else np.zeros((0, 2))

    # Length-normalized scalar baselines
    h0_finite = h0[np.isfinite(h0[:, 1])]
    h1_finite = h1[np.isfinite(h1[:, 1])]
    h0_lens = (h0_finite[:, 1] - h0_finite[:, 0]) if len(h0_finite) else np.array([])
    h1_lens = (h1_finite[:, 1] - h1_finite[:, 0]) if len(h1_finite) else np.array([])
    h0_lens = h0_lens[h0_lens > 0]
    h1_lens = h1_lens[h1_lens > 0]

    feats = {
        "h0_total_per_step": float(h0_lens.sum()) / max(n_steps, 1),
        "h0_n_per_step":     float(len(h0_lens)) / max(n_steps, 1),
        "h1_total_per_step": float(h1_lens.sum()) / max(n_steps, 1),
        "h1_n_per_step":     float(len(h1_lens)) / max(n_steps, 1),
    }

    # Normalize each diagram by max pairwise distance (decouples geometry
    # from absolute scale, mitigating the v1 length confound)
    if len(h0_finite) or len(h1_finite):
        all_d = np.concatenate(
            [h0_finite[:, 1] if len(h0_finite) else np.array([]),
             h1_finite[:, 1] if len(h1_finite) else np.array([])]
        )
        all_d = all_d[np.isfinite(all_d) & (all_d > 0)]
        scale = float(all_d.max()) if len(all_d) else 1.0
    else:
        scale = 1.0

    h0_norm = _normalize_diagram(h0, scale)
    h1_norm = _normalize_diagram(h1, scale)

    h0_img = _persistence_image(h0_norm)   # length res*res
    h1_img = _persistence_image(h1_norm)
    for r in range(PI_RES):
        for c in range(PI_RES):
            idx = r * PI_RES + c
            feats[f"h0_pi_{r}_{c}"] = float(h0_img[idx])
            feats[f"h1_pi_{r}_{c}"] = float(h1_img[idx])

    return feats


def extract_from_npz(npz_path: str, max_steps: int = 256) -> pd.DataFrame:
    z = np.load(npz_path, allow_pickle=True)
    item_ids = z["item_ids"]
    labels = z["is_correct"].astype(int)
    embeddings = z["embeddings"]

    dataset = os.path.basename(npz_path).replace(".npz", "")
    rows = []
    for i, emb in enumerate(tqdm(embeddings, desc=f"PHimg {dataset}", total=len(embeddings))):
        if emb is None or len(emb) == 0:
            feats = _zero_features()
        else:
            if len(emb) > max_steps:
                emb = emb[:max_steps]
            feats = compute_phimg_features(emb)
        rows.append({
            "item_id": str(item_ids[i]),
            "is_correct": int(labels[i]),
            "dataset": dataset,
            **feats,
        })
    return pd.DataFrame(rows)


def _summary(df: pd.DataFrame) -> str:
    """Print mean correct vs mean incorrect for the 4 normalized scalars
    and a few representative image cells, plus their effect-size proxy."""
    parts = ["Length-normalized scalars:"]
    for f in _norm_scalar_names():
        c = df.loc[df.is_correct == 1, f].mean()
        w = df.loc[df.is_correct == 0, f].mean()
        s = df[f].std()
        parts.append(f"  {f:<26s}  c={c:.4f}  w={w:.4f}  Δ/std={(c - w) / (s + 1e-9):+.3f}")
    parts.append("Persistence-image cells (selected):")
    for f in ("h0_pi_0_0", "h0_pi_3_3", "h1_pi_0_0", "h1_pi_3_3"):
        c = df.loc[df.is_correct == 1, f].mean()
        w = df.loc[df.is_correct == 0, f].mean()
        s = df[f].std()
        parts.append(f"  {f:<26s}  c={c:.4f}  w={w:.4f}  Δ/std={(c - w) / (s + 1e-9):+.3f}")
    return "\n".join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--npz", nargs="+", help="Input .npz step-embedding file(s)")
    g.add_argument("--all", action="store_true",
                   help="Process all 8 standard datasets under --npz-dir")
    p.add_argument("--out-dir", default="data/features/v2")
    p.add_argument("--out", default=None,
                   help="Explicit output path (only with single --npz)")
    p.add_argument("--max-steps", type=int, default=256)
    p.add_argument("--npz-dir", default="data/step_embeddings")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        import ripser  # noqa: F401
    except ImportError:
        logger.error("ripser not installed: pip install --user ripser")
        sys.exit(2)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.all:
        pairs = [(os.path.join(args.npz_dir, f"{d}.npz"),
                  os.path.join(args.out_dir, f"{d}_features_phimg.csv"))
                 for d in DATASETS]
    else:
        if args.out and len(args.npz) > 1:
            p.error("--out only valid with a single --npz path")
        if args.out:
            pairs = [(args.npz[0], args.out)]
        else:
            pairs = [(n, os.path.join(args.out_dir,
                                      os.path.basename(n).replace(".npz", "_features_phimg.csv")))
                     for n in args.npz]

    for npz_path, out_path in pairs:
        if not os.path.exists(npz_path):
            logger.warning(f"Missing input: {npz_path} (skip)")
            continue
        if os.path.exists(out_path):
            logger.info(f"Already exists: {out_path} (skip; delete to rerun)")
            continue
        logger.info(f"{npz_path} -> {out_path}")
        df = extract_from_npz(npz_path, max_steps=args.max_steps)
        df.to_csv(out_path, index=False)
        logger.info(f"  wrote {len(df)} rows; {len(df.columns) - 3} feature cols")
        logger.info("\n" + _summary(df))


if __name__ == "__main__":
    main()
