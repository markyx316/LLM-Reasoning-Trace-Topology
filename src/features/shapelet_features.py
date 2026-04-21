"""
shapelet_features.py - Precompute shapelet-candidate distance matrices per dataset.

A **shapelet** (Ye & Keogh, 2009) is a short ordered subsequence that
maximally separates classes. For categorical behavior sequences like ours,
the natural distance between a shapelet `s` of length k and a trace `t`
of length L is:

    d(t, s) = min over i in [0, L-k]  of  (# mismatches in t[i:i+k] vs s) / k

i.e. the best-aligning-window Hamming distance, normalized by k so all
distances live in [0, 1].

This module computes **every** such distance:

    input:   data/parsed/{dataset}_{model}_parsed.jsonl
    output:  data/features/{dataset}_{model}_shapelet_distmat.npz
             keys:
                item_ids:   (N,) object/string
                is_correct: (N,) int8
                candidates: (M,) object/string   (unique k-subsequences)
                cand_lens:  (M,) int8            (k = len(candidates[j]))
                distances:  (N, M) float32       (d(t_i, c_j))
                dataset:    str                  (stamped for audit)

The .npz is **not** fold-aware — it's a raw precomputation. The leakage-
safe mining/selection/OOF logic lives in src/modeling/shapelet_eval.py,
which consumes this .npz inside each CV fold's training split.

Complexity budget:
    M unique candidates across k in {3..8} -> empirically ~2-6k / dataset.
    N items ~500-1300. Distance matrix size: N*M * 4B <~ 30 MB per dataset.
    Wall time: a few minutes per dataset on CPU (numpy sliding-window).

Usage:
    PYTHONPATH=. python src/features/shapelet_features.py \\
        --parsed-glob "data/parsed/*_parsed.jsonl" --output-dir data/features/

    # Single dataset
    PYTHONPATH=. python src/features/shapelet_features.py \\
        --parsed data/parsed/math500_qwen7b_parsed.jsonl \\
        --output-dir data/features/
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

logger = logging.getLogger(__name__)


BEHAVIOR_CHARS: list[str] = ["F", "V", "B", "R", "S", "H", "C"]
CHAR_TO_CODE: dict[str, int] = {c: i for i, c in enumerate(BEHAVIOR_CHARS)}

# Default shapelet length range
DEFAULT_K_MIN: int = 3
DEFAULT_K_MAX: int = 8

# Cap on unique candidates per length to bound memory.
# After extracting every k-window from every trace we uniquify; if the
# unique set exceeds this, we keep the top-K by *occurrence count* (common
# first, under the assumption that a shapelet appearing rarely in the
# corpus will never discriminate).
MAX_CANDIDATES_PER_K: int = 2000


# =============================================================================
# LOAD + ENCODE
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


def _encode_sequence(seq: str) -> np.ndarray:
    """Map 7-char alphabet to int8 codes. Unknown chars become 0 (FORWARD)."""
    return np.array([CHAR_TO_CODE.get(c, 0) for c in seq], dtype=np.int8)


def load_encoded_traces(parsed_path: str
                        ) -> tuple[list[np.ndarray], list[str], list[int]]:
    """Return (encoded_traces, item_ids, labels)."""
    encoded: list[np.ndarray] = []
    ids: list[str] = []
    labels: list[int] = []
    for rec in _iter_parsed_records(parsed_path):
        seq = rec.get("behavior_sequence", "")
        if isinstance(seq, list):
            seq = "".join(seq)
        seq = "".join(c for c in seq if c in CHAR_TO_CODE)
        if not seq:
            continue
        encoded.append(_encode_sequence(seq))
        ids.append(str(rec["item_id"]))
        labels.append(int(rec.get("is_correct", False)))
    return encoded, ids, labels


# =============================================================================
# CANDIDATE GENERATION
# =============================================================================

def build_candidate_pool(encoded_traces: list[np.ndarray],
                         k_min: int = DEFAULT_K_MIN,
                         k_max: int = DEFAULT_K_MAX,
                         max_per_k: int = MAX_CANDIDATES_PER_K
                         ) -> tuple[list[str], list[int]]:
    """
    Build the set of unique k-subsequences across all traces for each k in
    [k_min, k_max]. If per-k uniques exceed max_per_k, keep the top by
    corpus occurrence count.

    Returns (candidates, lengths) where candidates[i] is a string and
    lengths[i] is its length.
    """
    from collections import Counter

    all_candidates: list[str] = []
    all_lengths: list[int] = []
    for k in range(k_min, k_max + 1):
        counter: Counter = Counter()
        for trace in encoded_traces:
            if len(trace) < k:
                continue
            for i in range(len(trace) - k + 1):
                # Reconstruct as string key
                s = "".join(BEHAVIOR_CHARS[c] for c in trace[i:i + k])
                counter[s] += 1
        # Select top-max_per_k by corpus frequency
        top = counter.most_common(max_per_k)
        for s, _ in top:
            all_candidates.append(s)
            all_lengths.append(k)
        logger.info(f"  k={k}: {len(counter):5d} unique -> kept {len(top):5d}")
    return all_candidates, all_lengths


# =============================================================================
# DISTANCE MATRIX
# =============================================================================

def compute_distance_matrix(encoded_traces: list[np.ndarray],
                            candidates: list[str],
                            cand_lens: list[int]) -> np.ndarray:
    """
    Return (N, M) float32 matrix of min-Hamming distances.

    Implementation: for each length k in cand_lens, build a stacked
    (N, max_windows_k, k) window tensor per trace (masked to actual windows),
    then vectorize the distance per shapelet of that length.

    On CPU this hits numpy's SIMD path; HPC-friendly, no GPU required.
    """
    N = len(encoded_traces)
    M = len(candidates)
    dist = np.ones((N, M), dtype=np.float32)  # default 1.0 = "no match possible"

    # Group candidates by length for batch processing
    by_len: dict[int, list[int]] = {}
    for j, k in enumerate(cand_lens):
        by_len.setdefault(k, []).append(j)

    # Encode all candidates into int8 arrays grouped by k
    enc_cands: dict[int, np.ndarray] = {}
    for k, idxs in by_len.items():
        buf = np.zeros((len(idxs), k), dtype=np.int8)
        for row, j in enumerate(idxs):
            s = candidates[j]
            for col, c in enumerate(s):
                buf[row, col] = CHAR_TO_CODE.get(c, 0)
        enc_cands[k] = buf  # shape (m_k, k)

    # For each trace, for each k, compute min-hamming for all candidates of that length
    for i, trace in enumerate(encoded_traces):
        L = len(trace)
        for k, idxs in by_len.items():
            if L < k:
                # Distance stays at 1.0 (impossible to match)
                continue
            # Build (n_windows, k) matrix of windows
            windows = np.lib.stride_tricks.sliding_window_view(trace, window_shape=k)
            # shape: (L - k + 1, k)
            cand_mat = enc_cands[k]  # (m_k, k)
            # Broadcast: (m_k, 1, k)  vs (1, n_windows, k)  ->  (m_k, n_windows, k)
            # This can blow memory on long traces; chunk over candidates.
            chunk = 256
            mk = cand_mat.shape[0]
            for start in range(0, mk, chunk):
                stop = min(start + chunk, mk)
                c_chunk = cand_mat[start:stop][:, None, :]  # (chunk, 1, k)
                w_chunk = windows[None, :, :]                # (1, nw, k)
                mism = (c_chunk != w_chunk).sum(axis=2)      # (chunk, nw)
                min_mism = mism.min(axis=1).astype(np.float32)  # (chunk,)
                for local, j in enumerate(idxs[start:stop]):
                    dist[i, j] = float(min_mism[local]) / k
    return dist


# =============================================================================
# PIPELINE
# =============================================================================

def build_distmat_for_file(parsed_path: str, output_dir: str,
                           k_min: int = DEFAULT_K_MIN,
                           k_max: int = DEFAULT_K_MAX,
                           max_per_k: int = MAX_CANDIDATES_PER_K
                           ) -> str:
    group = _group_name_from_path(parsed_path)
    logger.info(f"Loading {parsed_path}...")
    traces, ids, labels = load_encoded_traces(parsed_path)
    logger.info(f"  {group}: {len(traces)} traces")
    logger.info(f"Building candidate pool (k={k_min}..{k_max})...")
    cands, lens = build_candidate_pool(traces, k_min=k_min, k_max=k_max,
                                       max_per_k=max_per_k)
    logger.info(f"  total candidates: {len(cands)}")
    logger.info(f"Computing distance matrix (N={len(traces)} x M={len(cands)})...")
    dist = compute_distance_matrix(traces, cands, lens)
    out = os.path.join(output_dir, f"{group}_shapelet_distmat.npz")
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        out,
        item_ids=np.array(ids, dtype=object),
        is_correct=np.array(labels, dtype=np.int8),
        candidates=np.array(cands, dtype=object),
        cand_lens=np.array(lens, dtype=np.int8),
        distances=dist,
        dataset=group,
    )
    logger.info(f"  wrote {out}  size={dist.nbytes / 1e6:.1f} MB")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parsed")
    ap.add_argument("--parsed-glob", default="data/parsed/*_parsed.jsonl")
    ap.add_argument("--output-dir", default="data/features/")
    ap.add_argument("--k-min", type=int, default=DEFAULT_K_MIN)
    ap.add_argument("--k-max", type=int, default=DEFAULT_K_MAX)
    ap.add_argument("--max-per-k", type=int, default=MAX_CANDIDATES_PER_K)
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
        try:
            build_distmat_for_file(
                p, args.output_dir,
                k_min=args.k_min, k_max=args.k_max,
                max_per_k=args.max_per_k,
            )
        except Exception as e:
            logger.exception(f"  Failed on {p}: {e}")


# =============================================================================
# SELF-TEST
# =============================================================================

def _run_self_test():
    print("Running shapelet_features self-test...")
    import random
    rng = random.Random(0)

    # Two classes: correct traces contain "BVF" somewhere, incorrect contain "HHH"
    fakes = []
    for i in range(20):
        label = i % 2
        L = rng.randint(15, 25)
        body = "".join(rng.choices(BEHAVIOR_CHARS, k=L))
        body += "BVF" if label == 1 else "HHH"
        fakes.append((body, f"synth_{i:04d}", label))

    encoded = [_encode_sequence(s) for s, _, _ in fakes]
    ids = [x[1] for x in fakes]
    labels = [x[2] for x in fakes]

    cands, lens = build_candidate_pool(encoded, k_min=3, k_max=4, max_per_k=200)
    assert len(cands) > 0 and len(cands) == len(lens)
    dist = compute_distance_matrix(encoded, cands, lens)
    assert dist.shape == (20, len(cands)), f"expected (20, {len(cands)}), got {dist.shape}"
    assert dist.dtype == np.float32
    assert (dist >= 0.0).all() and (dist <= 1.0 + 1e-6).all()

    # BVF should be a candidate and have distance 0 on label=1 traces
    if "BVF" in cands:
        j = cands.index("BVF")
        correct_dists = [dist[i, j] for i in range(20) if labels[i] == 1]
        incorrect_dists = [dist[i, j] for i in range(20) if labels[i] == 0]
        # Correct traces definitely contain BVF -> distance == 0
        for d in correct_dists:
            assert d == 0.0, f"correct trace should have BVF distance 0, got {d}"
        assert np.mean(incorrect_dists) > np.mean(correct_dists), \
            "BVF should be closer in correct traces"

    # Write distmat round-trip
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "test_parsed.jsonl")
        with open(out, "w") as f:
            for body, i_id, label in fakes:
                f.write(json.dumps({
                    "item_id": i_id,
                    "is_correct": bool(label),
                    "behavior_sequence": body,
                }) + "\n")
        z_out = build_distmat_for_file(out, tmp, k_min=3, k_max=4, max_per_k=200)
        z = np.load(z_out, allow_pickle=True)
        assert set(z.keys()) >= {"item_ids", "is_correct", "candidates",
                                  "cand_lens", "distances"}
        assert z["distances"].shape == (20, len(z["candidates"]))

    print("All shapelet_features tests passed.")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _run_self_test()
    else:
        main()
