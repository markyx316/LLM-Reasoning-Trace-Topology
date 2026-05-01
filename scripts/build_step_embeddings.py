#!/usr/bin/env python3
"""
build_step_embeddings.py - Pre-compute step embeddings + types for every trace.

Reads trace JSONL files, parses each trace into cognitive episodes, encodes
every step text with MiniLM, and saves a single .npz per dataset:

    {item_id}: <stored as parallel arrays since npz is keyed>
    item_ids:  array of item_id strings
    is_correct: array of int8 labels
    embeddings: object array of (n_steps_i, 384) float32 matrices
    step_types: object array of (n_steps_i,) int8 (BehaviorType ordinal)

This is consumed by step_transformer.py for fast iteration during training
(no re-encoding per fold). Done once on GPU (~15 min for all 8 datasets).

Usage:
    python scripts/build_step_embeddings.py --all
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.recurrence_features import load_embedder
from src.parsing.rule_based_parser import BehaviorType, parse_trace

logger = logging.getLogger(__name__)

DATASETS = [
    "math500_qwen7b", "math500_llama8b",
    "gsm8k_qwen7b", "gsm8k_llama8b",
    "gpqa_diamond_qwen7b", "gpqa_diamond_llama8b",
    "arc_challenge_qwen7b", "arc_challenge_llama8b",
]

# Stable ordinal mapping: 0=PAD, 1..6 = behavior types (F, V, X, R, H, C).
# Source enum is rule_based_parser.BehaviorType (the 6-class taxonomy actually
# emitted by parse_trace). Importing from src.parsing.taxonomy here would
# silently map every step to PAD because that legacy enum has different members
# (BACKTRACK/SUBGOAL instead of REVISE).
BEHAVIOR_VOCAB = {bt: i + 1 for i, bt in enumerate(BehaviorType)}
PAD_TYPE = 0
N_TYPES = len(BehaviorType) + 1  # 7 with PAD (6 behaviors + PAD)


def process_dataset(
    traces_path: str,
    out_path: str,
    model,
    max_steps: int = 256,
    batch_size: int = 256,
):
    """Encode all step texts for one dataset, save to .npz."""
    logger.info(f"Loading traces: {traces_path}")
    items = []
    with open(traces_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    logger.info(f"  {len(items)} items")

    # Parse and collect all step texts (flat) with offsets so we can batch-encode
    all_step_texts = []
    item_offsets = []      # (start, end) into all_step_texts
    item_types = []        # list of np.int8 arrays (per item)
    item_ids = []
    labels = []

    for it in tqdm(items, desc=f"Parsing {os.path.basename(traces_path)}"):
        trace = it.get("reasoning_trace") or it.get("full_response") or ""
        try:
            episodes = parse_trace(trace) if trace else []
        except Exception:
            episodes = []
        # Truncate
        episodes = episodes[:max_steps]
        texts = [getattr(ep, "text", "").strip() for ep in episodes]
        types = np.array(
            [BEHAVIOR_VOCAB.get(getattr(ep, "behavior", None), PAD_TYPE)
             for ep in episodes],
            dtype=np.int8,
        )

        start = len(all_step_texts)
        all_step_texts.extend(texts)
        end = len(all_step_texts)

        item_offsets.append((start, end))
        item_types.append(types)
        item_ids.append(it.get("item_id"))
        labels.append(int(it.get("is_correct", False)))

    logger.info(f"  total steps to encode: {len(all_step_texts)}")

    # Batch-encode all steps in one GPU pass
    if all_step_texts:
        embs = model.encode(
            all_step_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)
    else:
        embs = np.zeros((0, 384), dtype=np.float32)

    # Slice back into per-item embedding arrays
    item_embs = []
    for (start, end) in item_offsets:
        if end > start:
            item_embs.append(embs[start:end])
        else:
            item_embs.append(np.zeros((0, embs.shape[1] if embs.size else 384),
                                      dtype=np.float32))

    # Save
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        item_ids=np.array(item_ids, dtype=object),
        is_correct=np.array(labels, dtype=np.int8),
        embeddings=np.array(item_embs, dtype=object),
        step_types=np.array(item_types, dtype=object),
    )
    logger.info(f"Saved: {out_path}  items={len(items)}  "
                f"avg_steps={np.mean([len(t) for t in item_types]):.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-dir", default="data/step_embeddings")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dataset", default=None,
                        help="Single dataset name, e.g. math500_qwen7b")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info(f"Loading embedder: {args.model}")
    model = load_embedder(args.model)

    targets = DATASETS if args.all else (
        [args.dataset] if args.dataset else parser.error("--all or --dataset"))

    for name in targets:
        traces = os.path.join(args.data_root, "traces", f"{name}_traces.jsonl")
        out = os.path.join(args.out_dir, f"{name}.npz")
        if not os.path.exists(traces):
            logger.warning(f"Missing: {traces}, skip")
            continue
        if os.path.exists(out):
            logger.info(f"Already exists, skip: {out}")
            continue
        process_dataset(traces, out, model,
                        max_steps=args.max_steps,
                        batch_size=args.batch_size)


if __name__ == "__main__":
    main()
