#!/usr/bin/env python3
"""
build_trace_graphs.py - Dump per-dataset PyG-style graph .npz files used by
src/modeling/trace_gnn.py.

For each trace in data/parsed/{group}_parsed.jsonl we build:
  * Node features (float32, shape (L, d_node))
      Structural (d_node = 10):
        [0..6] 7-d one-hot behavior  (F V B R S H C)
        [7]    normalized position   (i / max(L-1, 1))
        [8]    token-count z-score   (per-trace standardization)
        [9]    log1p(confidence)     (stored per-episode in parsed JSONL)
      With --with-content   (d_node = 10 + 384 = 394):
        above + MiniLM embedding of episode text.
  * Edge index (int64, shape (2, E)) and edge weights (float32, shape (E,)):
        Temporal chain:     (i -> i+1), weight 1.0
        Behavior recurrence: (i -> j) for j > i if behavior[i]==behavior[j]
          AND j - i >= MIN_GAP, weight 1 / (j - i).
  * Per-item label (is_correct) and item_id.

Output NPZ schema:
  item_ids     (N,)     str
  is_correct   (N,)     int8
  node_feats   (N,)     object of (L_i, d_node) float32
  edge_indices (N,)     object of (2, E_i) int64
  edge_weights (N,)     object of (E_i,)    float32

Usage:
  # Structural graphs (CPU, fast):
  PYTHONPATH=. python scripts/build_trace_graphs.py \\
      --parsed-glob "data/parsed/*_parsed.jsonl" \\
      --output-dir data/graphs/

  # Hybrid graphs (MiniLM per-episode text, needs sentence-transformers + GPU recommended):
  PYTHONPATH=. python scripts/build_trace_graphs.py \\
      --parsed-glob "data/parsed/*_parsed.jsonl" \\
      --output-dir data/graphs/hybrid/ \\
      --with-content --device cuda
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

BEHAVIOR_CHARS = ["F", "V", "B", "R", "S", "H", "C"]
BEHAVIOR_INDEX = {c: i for i, c in enumerate(BEHAVIOR_CHARS)}
MIN_RECURRENCE_GAP = 3


# =============================================================================
# PER-TRACE GRAPH CONSTRUCTION
# =============================================================================

def _structural_node_feats(episodes: list[dict]) -> np.ndarray:
    """10-d content-free node features."""
    L = len(episodes)
    if L == 0:
        return np.zeros((0, 10), dtype=np.float32)
    behavior = [ep.get("behavior", "F") for ep in episodes]
    behavior = [b if b in BEHAVIOR_INDEX else "F" for b in behavior]
    token_counts = np.array(
        [float(ep.get("token_count", 0)) for ep in episodes], dtype=np.float32
    )
    confidence = np.array(
        [float(ep.get("confidence", 0.5)) for ep in episodes], dtype=np.float32
    )
    # Token-count z-score per trace
    mu = float(token_counts.mean()) if L else 0.0
    sd = float(token_counts.std()) if L else 0.0
    if sd < 1e-6:
        z = np.zeros_like(token_counts)
    else:
        z = (token_counts - mu) / sd
    feats = np.zeros((L, 10), dtype=np.float32)
    for i, b in enumerate(behavior):
        feats[i, BEHAVIOR_INDEX[b]] = 1.0
    feats[:, 7] = np.arange(L, dtype=np.float32) / max(L - 1, 1)
    feats[:, 8] = z
    feats[:, 9] = np.log1p(np.clip(confidence, 0.0, None))
    return feats


def _trace_edges(behavior_seq: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Temporal chain + behavior-recurrence edges."""
    L = len(behavior_seq)
    if L == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    srcs, dsts, ws = [], [], []
    for i in range(L - 1):
        srcs.append(i); dsts.append(i + 1); ws.append(1.0)
    # Behavior-recurrence: same-behavior pairs with gap >= MIN_GAP
    # Indexing by behavior type to avoid O(L^2) scans on long traces.
    pos_by_b: dict[str, list[int]] = {c: [] for c in BEHAVIOR_CHARS}
    for i, b in enumerate(behavior_seq):
        if b in pos_by_b:
            pos_by_b[b].append(i)
    for positions in pos_by_b.values():
        if len(positions) < 2:
            continue
        for a_idx in range(len(positions)):
            for b_idx in range(a_idx + 1, len(positions)):
                gap = positions[b_idx] - positions[a_idx]
                if gap >= MIN_RECURRENCE_GAP:
                    srcs.append(positions[a_idx])
                    dsts.append(positions[b_idx])
                    ws.append(1.0 / gap)
    if not srcs:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    ei = np.stack([np.asarray(srcs, dtype=np.int64),
                   np.asarray(dsts, dtype=np.int64)], axis=0)
    ew = np.asarray(ws, dtype=np.float32)
    return ei, ew


# =============================================================================
# PIPELINE PER FILE
# =============================================================================

def _iter_parsed(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _group_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith("_parsed.jsonl"):
        return base[: -len("_parsed.jsonl")]
    return base.replace(".jsonl", "")


def build_npz_for_file(parsed_path: str, output_dir: str,
                       with_content: bool = False,
                       embedder=None,
                       max_nodes: int = 256) -> str:
    group = _group_name_from_path(parsed_path)
    out_path = os.path.join(output_dir, f"{group}_graph.npz")

    item_ids, labels = [], []
    nfs, eis, ews = [], [], []
    all_texts, all_offsets = [], []  # for batch MiniLM encoding

    for rec in _iter_parsed(parsed_path):
        eps = rec.get("episodes", [])
        if len(eps) > max_nodes:
            eps = eps[:max_nodes]
        behavior_seq = [
            ep.get("behavior", "F") if ep.get("behavior", "F") in BEHAVIOR_INDEX
            else "F"
            for ep in eps
        ]
        nf = _structural_node_feats(eps)
        ei, ew = _trace_edges(behavior_seq)

        start = len(all_texts)
        if with_content:
            all_texts.extend([ep.get("text", "") or "" for ep in eps])
        end = len(all_texts)
        all_offsets.append((start, end))

        nfs.append(nf)
        eis.append(ei)
        ews.append(ew)
        item_ids.append(str(rec.get("item_id")))
        labels.append(int(rec.get("is_correct", False)))

    # Optional: MiniLM content embedding concatenated to structural features
    if with_content:
        if embedder is None:
            raise ValueError("with_content=True requires a loaded embedder")
        if all_texts:
            logger.info(f"  MiniLM encoding {len(all_texts)} episode texts...")
            embs = embedder.encode(
                all_texts, batch_size=256, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=True,
            ).astype(np.float32)
            d_content = int(embs.shape[1])
        else:
            d_content = 384
            embs = np.zeros((0, d_content), dtype=np.float32)
        for i, (s, e) in enumerate(all_offsets):
            structural = nfs[i]
            if e > s:
                content = embs[s:e]
            else:
                content = np.zeros((0, d_content), dtype=np.float32)
            if structural.shape[0] != content.shape[0]:
                # Truncate to the shorter of the two (defensive).
                L = min(structural.shape[0], content.shape[0])
                structural = structural[:L]; content = content[:L]
            # ALWAYS concatenate so empty traces get shape (0, 10+d_content)
            # instead of staying as (0, 10). Mixing dims corrupts collate_graphs
            # which uses batch[0]['x'].shape[1] to infer the padded tensor dim.
            nfs[i] = np.concatenate([structural, content], axis=1)
        # Post-condition: every trace has the same node-feature dim.
        dims = {int(nf.shape[1]) for nf in nfs}
        if len(dims) > 1:
            raise RuntimeError(
                f"Inconsistent node-feat dims after hybrid concat: {dims}. "
                f"This is a bug — all traces in a file must share d_node."
            )

    os.makedirs(output_dir, exist_ok=True)
    # np.array(list_of_ndarrays, dtype=object) raises when every inner array
    # happens to share a leading dim (e.g. all edge_indices are (2, *)).
    # Build an empty object array and fill per-slot to force raggedness.
    def _as_object(seq):
        arr = np.empty(len(seq), dtype=object)
        for i, v in enumerate(seq):
            arr[i] = v
        return arr

    np.savez_compressed(
        out_path,
        item_ids=np.array(item_ids, dtype=object),
        is_correct=np.array(labels, dtype=np.int8),
        node_feats=_as_object(nfs),
        edge_indices=_as_object(eis),
        edge_weights=_as_object(ews),
    )
    d_node = nfs[0].shape[1] if nfs and nfs[0].size else 10
    avg_nodes = float(np.mean([nf.shape[0] for nf in nfs])) if nfs else 0.0
    avg_edges = float(np.mean([ei.shape[1] for ei in eis])) if eis else 0.0
    logger.info(f"  {group}: n={len(item_ids)}  d_node={d_node}  "
                f"avg_nodes={avg_nodes:.1f}  avg_edges={avg_edges:.1f}  "
                f"-> {out_path}")
    return out_path


# =============================================================================
# DRIVER
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parsed")
    ap.add_argument("--parsed-glob", default="data/parsed/*_parsed.jsonl")
    ap.add_argument("--output-dir", default="data/graphs/")
    ap.add_argument("--with-content", action="store_true",
                    help="Concat MiniLM episode-text embedding to node features "
                         "(enables the GNN hybrid variant).")
    ap.add_argument("--embedder-name",
                    default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-nodes", type=int, default=256)
    ap.add_argument("--skip-pilot", action="store_true", default=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    embedder = None
    if args.with_content:
        from src.features.recurrence_features import load_embedder
        logger.info(f"Loading embedder: {args.embedder_name}")
        embedder = load_embedder(args.embedder_name)
        if args.device:
            try:
                embedder.to(args.device)
            except Exception:
                pass

    paths = [args.parsed] if args.parsed else sorted(glob.glob(args.parsed_glob))
    if args.skip_pilot and not args.parsed:
        paths = [p for p in paths
                 if not os.path.basename(p).startswith(("pilot_", "_"))
                 and "_sc" not in os.path.basename(p)]
    if not paths:
        logger.error("No parsed JSONL files found")
        sys.exit(1)

    logger.info(f"Building graphs for {len(paths)} files  "
                f"with_content={args.with_content}")
    for p in paths:
        logger.info(f"Processing {p}")
        try:
            build_npz_for_file(p, args.output_dir,
                               with_content=args.with_content,
                               embedder=embedder,
                               max_nodes=args.max_nodes)
        except Exception as e:
            logger.exception(f"  Failed on {p}: {e}")


# =============================================================================
# SELF-TEST
# =============================================================================

class _MockEmbedder:
    """Minimal sentence_transformers stand-in for self-test (avoids network / GPU)."""
    def __init__(self, dim: int = 384):
        self.dim = int(dim)
    def encode(self, texts, batch_size=256, convert_to_numpy=True,
               normalize_embeddings=True, show_progress_bar=False):
        rng = np.random.default_rng(abs(hash(tuple(texts))) % (2**32))
        out = rng.standard_normal((len(texts), self.dim)).astype(np.float32)
        if normalize_embeddings and len(texts):
            n = np.linalg.norm(out, axis=1, keepdims=True)
            n[n == 0] = 1.0
            out = out / n
        return out


def _run_self_test():
    print("Running build_trace_graphs self-test...")
    import tempfile
    # Create a synthetic parsed JSONL. First record is deliberately empty to
    # exercise the zero-episode path that previously produced mis-shaped
    # hybrid node_feats (see the RuntimeError hit during HPC hybrid training).
    rows = []
    rng = np.random.default_rng(0)
    rows.append({
        "item_id": "syn_empty",
        "dataset": "synthetic",
        "is_correct": 0,
        "episodes": [],
    })
    for i in range(12):
        L = int(rng.integers(15, 30))
        behaviors = list("FVBRSHC")
        seq = rng.choice(behaviors, size=L).tolist()
        rec = {
            "item_id": f"syn_{i:03d}",
            "dataset": "synthetic",
            "is_correct": int(i % 2),
            "episodes": [
                {"text": f"step {k} of item {i}",
                 "behavior": seq[k],
                 "position": k,
                 "token_count": int(rng.integers(3, 20)),
                 "confidence": float(rng.random())}
                for k in range(L)
            ],
        }
        rows.append(rec)

    tmp = tempfile.mkdtemp()
    p_in = os.path.join(tmp, "synthetic_parsed.jsonl")
    with open(p_in, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # --- Structural variant ---
    p_out = build_npz_for_file(p_in, tmp, with_content=False)
    z = np.load(p_out, allow_pickle=True)
    assert len(z["item_ids"]) == 13
    assert z["is_correct"].shape == (13,)
    dims = {int(np.asarray(nf).shape[1]) for nf in z["node_feats"]}
    assert dims == {10}, f"structural should be uniformly d=10, got {dims}"
    # First record is the empty trace; assert it still has d=10.
    nf_empty = np.asarray(z["node_feats"][0])
    assert nf_empty.shape == (0, 10), f"empty trace shape {nf_empty.shape} != (0, 10)"
    # Some normal trace should have edges.
    ei1 = np.asarray(z["edge_indices"][1])
    assert ei1.shape[0] == 2 and ei1.shape[1] >= 1, "expected non-empty edge_index for a normal trace"
    print(f"  structural: n=13  d_node=10  empty_shape=(0,10)  ok")

    # --- Hybrid variant (REGRESSION: empty traces must get d=10+384=394) ---
    embedder = _MockEmbedder(dim=384)
    tmp2 = tempfile.mkdtemp()
    p_out2 = build_npz_for_file(p_in, tmp2, with_content=True, embedder=embedder)
    z2 = np.load(p_out2, allow_pickle=True)
    dims2 = {int(np.asarray(nf).shape[1]) for nf in z2["node_feats"]}
    assert dims2 == {394}, f"hybrid should be uniformly d=394, got {dims2}"
    # Empty trace must NOT fall back to structural shape (0, 10) — that's the bug.
    nf_empty2 = np.asarray(z2["node_feats"][0])
    assert nf_empty2.shape == (0, 394), (
        f"empty trace in hybrid mode has shape {nf_empty2.shape}, must be (0, 394) "
        f"— this is the regression that crashed collate_graphs on HPC."
    )
    # Non-empty trace should have d=394 too.
    nf_full2 = np.asarray(z2["node_feats"][1])
    assert nf_full2.shape[1] == 394, f"hybrid non-empty trace d={nf_full2.shape[1]} != 394"
    print(f"  hybrid:     n=13  d_node=394  empty_shape=(0,394)  ok  (regression guarded)")

    print("All build_trace_graphs tests passed.")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _run_self_test()
    else:
        main()
