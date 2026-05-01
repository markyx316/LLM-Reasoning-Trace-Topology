#!/usr/bin/env python3
"""
build_step_embeddings_bge.py

Same as build_step_embeddings.py but uses BAAI/bge-base-en-v1.5 (768-d)
instead of sentence-transformers/all-MiniLM-L6-v2 (384-d). Writes to a
separate directory (data/step_embeddings_bge/) so both generations coexist.

Why: Approach 6a of the post-disappointment plan. MiniLM-L6 is small (22M
params, 384-d) and its embedding space was showing signs of being too coarse
for PH (ripser warning about column/row mismatch, low dynamic range on unit
sphere). bge-base-en-v1.5 is 5x larger (110M params, 768-d) and consistently
tops MTEB for English sentence similarity. If StepTF improves with 768-d
embeddings, we have evidence that step-representation quality was bottlenecked.

Downstream:
  - Feed these .npz files to src/modeling/step_transformer.py (with EMB_DIM=768
    patched, or via --emb-dim flag if supported) and compare AUROC.
  - Also usable by behavior_seq_lm.py (which only reads step_types, not
    embeddings — bge vs MiniLM produces identical behavior sequences since
    parsing is upstream of embedding).

Usage:
    python scripts/build_step_embeddings_bge.py --all --batch-size 128

Compute: ~20 min on 1x RTX PRO 6000. bge-base-en-v1.5 is 5x larger than
MiniLM so batch size drops to 128 (from 256) to fit memory comfortably.
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

from src.parsing.rule_based_parser import BehaviorType, parse_trace

logger = logging.getLogger(__name__)

DATASETS = [
    "math500_qwen7b", "math500_llama8b",
    "gsm8k_qwen7b", "gsm8k_llama8b",
    "gpqa_diamond_qwen7b", "gpqa_diamond_llama8b",
    "arc_challenge_qwen7b", "arc_challenge_llama8b",
]

# Stable ordinal mapping — same as build_step_embeddings.py (post-fix)
BEHAVIOR_VOCAB = {bt: i + 1 for i, bt in enumerate(BehaviorType)}
PAD_TYPE = 0
N_TYPES = len(BehaviorType) + 1  # 7 (PAD + 6 behaviors)


def load_embedder(model_name: str, device: str = None):
    """Load a SentenceTransformer embedding model; auto-CUDA if available."""
    from sentence_transformers import SentenceTransformer
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    logger.info(f"Loading embedder: {model_name} on {device}")
    return SentenceTransformer(model_name, device=device)


def process_dataset(
    traces_path: str, out_path: str, model,
    max_steps: int = 256, batch_size: int = 128,
):
    logger.info(f"Loading traces: {traces_path}")
    items = []
    with open(traces_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    logger.info(f"  {len(items)} items")

    all_step_texts = []
    item_offsets = []
    item_types = []
    item_ids = []
    labels = []

    for it in tqdm(items, desc=f"Parsing {os.path.basename(traces_path)}"):
        trace = it.get("reasoning_trace") or it.get("full_response") or ""
        try:
            episodes = parse_trace(trace) if trace else []
        except Exception:
            episodes = []
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

    if all_step_texts:
        embs = model.encode(
            all_step_texts, batch_size=batch_size,
            convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)
    else:
        embs = np.zeros((0, model.get_sentence_embedding_dimension()), dtype=np.float32)

    # Slice back per item
    item_embs = []
    emb_dim = embs.shape[1] if embs.size else model.get_sentence_embedding_dimension()
    for (start, end) in item_offsets:
        if end > start:
            item_embs.append(embs[start:end])
        else:
            item_embs.append(np.zeros((0, emb_dim), dtype=np.float32))

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        item_ids=np.array(item_ids, dtype=object),
        is_correct=np.array(labels, dtype=np.int8),
        embeddings=np.array(item_embs, dtype=object),
        step_types=np.array(item_types, dtype=object),
    )
    logger.info(f"Saved: {out_path}  items={len(items)}  "
                f"emb_dim={emb_dim}  avg_steps={np.mean([len(t) for t in item_types]):.1f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir",  default="data/step_embeddings_bge")
    p.add_argument("--model",    default="BAAI/bge-base-en-v1.5")
    p.add_argument("--max-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=128,
                   help="Smaller than MiniLM default (256) because bge is ~5x larger")
    p.add_argument("--all", action="store_true")
    p.add_argument("--dataset", default=None,
                   help="Single dataset name, e.g. math500_qwen7b")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    model = load_embedder(args.model)
    emb_dim = model.get_sentence_embedding_dimension()
    logger.info(f"Embedding dimension: {emb_dim}")

    if args.all:
        targets = DATASETS
    elif args.dataset:
        targets = [args.dataset]
    else:
        p.error("--all or --dataset required")

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
