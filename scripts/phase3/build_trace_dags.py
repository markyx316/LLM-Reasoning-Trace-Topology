"""Phase 3 (T5) — Build enriched trace DAGs with REVISION-REFERENCE edges.

Extends the Route-B trace graphs (scripts/build_trace_graphs.py) by adding
a third edge type:

    (c) revision reference: for each episode at position i with behavior
        in {X=Revise, R=Restart}, add a directed edge (i -> j) where j is
        the most recent preceding episode with behavior in {F=Forward,
        V=Verify}. Weight = 1 / (i - j). Intuition: a "revise" step
        *references* the forward/verify chain that it is revising. This
        gives the graph model an explicit pointer that the bag-of-edges
        behavior-recurrence term cannot convey.

The output npz schema is the SAME as build_trace_graphs.py:

    item_ids     (N,)     str
    is_correct   (N,)     int8
    node_feats   (N,)     object of (L_i, d_node) float32
    edge_indices (N,)     object of (2, E_i) int64
    edge_weights (N,)     object of (E_i,)    float32
    edge_types   (N,)     object of (E_i,)    int8       ← NEW
                                0 = temporal, 1 = recurrence, 2 = revision

so downstream models that only need the adjacency work without change, and
the Graphormer trainer can optionally use `edge_types` as a typed attention
bias input.

Usage
-----
    PYTHONPATH=. python scripts/phase3/build_trace_dags.py \
        --parsed-glob 'data/parsed/*_parsed.jsonl' \
        --output-dir data/graphs_v3/

    # With content features (MiniLM embedding per episode):
    PYTHONPATH=. python scripts/phase3/build_trace_dags.py \
        --parsed-glob 'data/parsed/*_parsed.jsonl' \
        --output-dir data/graphs_v3_hybrid/ \
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

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# Matches the legacy 7-char vocabulary in build_trace_graphs.py
BEHAVIOR_CHARS = ["F", "V", "B", "R", "S", "H", "C"]
BEHAVIOR_INDEX = {c: i for i, c in enumerate(BEHAVIOR_CHARS)}

# rule_based_parser emits 6-char {F, V, X, R, H, C}. We treat X (revise) as
# its own anchor for revision edges, and also honor the older "B" (backtrack)
# from the legacy parser as equivalent.
REVISION_BEHAVIORS = {"X", "R", "B", "S"}
FORWARD_OR_VERIFY = {"F", "V"}

MIN_RECURRENCE_GAP = 3


# =========================================================================
# NODE FEATURES (same shape as legacy; expands 6→7 one-hot)
# =========================================================================

def structural_node_feats(episodes: list[dict]) -> np.ndarray:
    L = len(episodes)
    if L == 0:
        return np.zeros((0, 10), dtype=np.float32)
    behavior = [ep.get("behavior", "F") for ep in episodes]
    # Map any non-legacy char to "F" so the 7-d one-hot is well-defined.
    behavior = [b if b in BEHAVIOR_INDEX else "F" for b in behavior]
    token_counts = np.array(
        [float(ep.get("token_count", 0)) for ep in episodes], dtype=np.float32
    )
    confidence = np.array(
        [float(ep.get("confidence", 0.5)) for ep in episodes], dtype=np.float32
    )
    mu = float(token_counts.mean()) if L else 0.0
    sd = float(token_counts.std()) + 1e-6
    z_tc = (token_counts - mu) / sd
    feats = np.zeros((L, 10), dtype=np.float32)
    for i, b in enumerate(behavior):
        feats[i, BEHAVIOR_INDEX[b]] = 1.0
        feats[i, 7] = i / max(L - 1, 1)
        feats[i, 8] = float(z_tc[i])
        feats[i, 9] = float(np.log1p(confidence[i]))
    return feats


# =========================================================================
# EDGES
# =========================================================================

def build_edges(behavior_seq: list[str]):
    """Return (edge_index (2,E), edge_weight (E,), edge_type (E,)).

    edge_type codes:
        0 = temporal chain
        1 = behavior recurrence
        2 = revision reference
    """
    L = len(behavior_seq)
    srcs, dsts, ws, ts = [], [], [], []

    # (0) temporal chain
    for i in range(L - 1):
        srcs.append(i); dsts.append(i + 1); ws.append(1.0); ts.append(0)

    # (1) behavior recurrence (unchanged from legacy)
    pos_by_b: dict[str, list[int]] = {}
    for i, b in enumerate(behavior_seq):
        pos_by_b.setdefault(b, []).append(i)
    for positions in pos_by_b.values():
        if len(positions) < 2:
            continue
        for a in range(len(positions)):
            for c in range(a + 1, len(positions)):
                gap = positions[c] - positions[a]
                if gap >= MIN_RECURRENCE_GAP:
                    srcs.append(positions[a])
                    dsts.append(positions[c])
                    ws.append(1.0 / gap)
                    ts.append(1)

    # (2) revision reference: each revise/restart episode i points back to
    # the nearest preceding F/V episode.
    for i, b in enumerate(behavior_seq):
        if b not in REVISION_BEHAVIORS:
            continue
        # Walk backwards looking for F or V
        for k in range(i - 1, -1, -1):
            if behavior_seq[k] in FORWARD_OR_VERIFY:
                srcs.append(i); dsts.append(k)
                ws.append(1.0 / max(i - k, 1))
                ts.append(2)
                break
        # If none found, the edge is skipped (graph still valid).

    if not srcs:
        return (np.zeros((2, 0), dtype=np.int64),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int8))
    ei = np.stack([np.asarray(srcs, dtype=np.int64),
                   np.asarray(dsts, dtype=np.int64)], axis=0)
    ew = np.asarray(ws, dtype=np.float32)
    et = np.asarray(ts, dtype=np.int8)
    return ei, ew, et


# =========================================================================
# PIPELINE
# =========================================================================

def _iter_parsed(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _group_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith("_parsed.jsonl"):
        return base[: -len("_parsed.jsonl")]
    return base.replace(".jsonl", "")


def build_for_file(parsed_path: str, output_dir: str,
                   with_content: bool = False,
                   embedder=None,
                   max_nodes: int = 256) -> str:
    log = logging.getLogger("dags")
    group = _group_name_from_path(parsed_path)
    out_path = os.path.join(output_dir, f"{group}_graph_v3.npz")
    if os.path.exists(out_path):
        log.info("exists, skip: %s", out_path)
        return out_path

    item_ids, labels = [], []
    nfs, eis, ews, ets = [], [], [], []
    all_texts, all_offsets = [], []

    for rec in _iter_parsed(parsed_path):
        eps = rec.get("episodes", [])
        if len(eps) > max_nodes:
            eps = eps[:max_nodes]
        behavior_seq = [
            ep.get("behavior", "F") if ep.get("behavior", "F") in BEHAVIOR_INDEX
            else ("F" if ep.get("behavior") != "X" else "B")  # map X→B for legacy 7-char
            for ep in eps
        ]
        # NOTE: for REVISE (X from rule_based_parser), build_edges needs to
        # recognize it. We keep both the legacy one-hot (7-char) AND the
        # revision behaviors in its own detection logic. So here we preserve
        # the ORIGINAL chars for edge building and only coerce for one-hot.
        edge_behavior = [ep.get("behavior", "F") for ep in eps]
        nf = structural_node_feats(eps)
        ei, ew, et = build_edges(edge_behavior)

        start = len(all_texts)
        if with_content:
            all_texts.extend([(ep.get("text") or "")[:1000] for ep in eps])
            all_offsets.append((start, start + len(eps)))
        else:
            all_offsets.append((start, start))  # empty slice

        nfs.append(nf); eis.append(ei); ews.append(ew); ets.append(et)
        item_ids.append(rec.get("item_id"))
        labels.append(int(bool(rec.get("is_correct", 0))))

    # Optionally append MiniLM embedding of each episode's text
    if with_content and embedder is not None and all_texts:
        log.info("Encoding %d step texts with embedder", len(all_texts))
        embs = embedder.encode(
            all_texts, batch_size=256, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=True,
        ).astype(np.float32)
        # Re-slice and concat to node feats
        new_nfs = []
        for nf, (lo, hi) in zip(nfs, all_offsets):
            if hi > lo:
                e = embs[lo:hi]
                if e.shape[0] != nf.shape[0]:
                    # defensive: fall back to zeros
                    e = np.zeros((nf.shape[0], embs.shape[1]), dtype=np.float32)
                new_nfs.append(np.concatenate([nf, e], axis=1))
            else:
                new_nfs.append(nf)
        nfs = new_nfs

    os.makedirs(output_dir, exist_ok=True)
    # np.array(list_of_ndarrays, dtype=object) raises when every inner array
    # happens to share a leading dim (e.g. all edge_indices are (2, *) and all
    # node_feats are (L_i, d) where d is constant across items). In newer
    # numpy (>=1.24) the constructor tries to stack into a rectangular array
    # and fails to broadcast the ragged suffix axis. Build an empty object
    # array and fill per-slot to force raggedness.
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
        edge_types=_as_object(ets),
    )
    log.info("Wrote %s  n=%d  avg_L=%.1f  avg_E=%.1f",
             out_path, len(item_ids),
             float(np.mean([nf.shape[0] for nf in nfs])) if nfs else 0.0,
             float(np.mean([ei.shape[1] for ei in eis])) if eis else 0.0)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parsed-glob", default="data/parsed/*_parsed.jsonl")
    ap.add_argument("--output-dir", default="data/graphs_v3")
    ap.add_argument("--with-content", action="store_true")
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-nodes", type=int, default=256)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger("dags_main")

    embedder = None
    if args.with_content:
        try:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer(args.embedder, device=args.device)
            log.info("Loaded embedder: %s on %s", args.embedder, args.device)
        except Exception as e:
            log.warning("Embedder load failed (%s); continuing structural-only", e)
            args.with_content = False

    files = sorted(glob.glob(args.parsed_glob))
    log.info("Processing %d parsed files", len(files))
    for fp in files:
        # Skip pilots and sc_ variants
        base = os.path.basename(fp).lower()
        if any(x in base for x in ("pilot", "dryrun", "_sc", ".bak")):
            log.info("skip (excluded): %s", fp)
            continue
        try:
            build_for_file(
                fp, args.output_dir,
                with_content=args.with_content,
                embedder=embedder,
                max_nodes=args.max_nodes,
            )
        except Exception as e:
            log.exception("Failed on %s: %s", fp, e)


if __name__ == "__main__":
    main()
