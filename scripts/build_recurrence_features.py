#!/usr/bin/env python3
"""
build_recurrence_features.py - Batch compute 5 recurrence features for all traces.

Reads a trace JSONL, parses each trace into episodes (for revision detection
and step text), encodes step texts once per trace, computes 5 recurrence
features, and MERGES them into the existing per-dataset feature CSV so the
downstream training pipeline can pick them up without modification.

Usage:
    # Single dataset:
    PYTHONPATH=. python scripts/build_recurrence_features.py \
        --traces data/traces/math500_qwen7b_traces.jsonl \
        --in-features data/features/math500_qwen7b_features.csv \
        --out-features data/features/math500_qwen7b_features_rec.csv

    # All 8 dataset-model combos:
    PYTHONPATH=. python scripts/build_recurrence_features.py --all
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.recurrence_features import (
    RECURRENCE_FEATURE_NAMES,
    extract_recurrence_features,
    load_embedder,
)

logger = logging.getLogger(__name__)


DATASETS = [
    "math500_qwen7b",
    "math500_llama8b",
    "gsm8k_qwen7b",
    "gsm8k_llama8b",
    "gpqa_diamond_qwen7b",
    "gpqa_diamond_llama8b",
    "arc_challenge_qwen7b",
    "arc_challenge_llama8b",
]


def process_trace_file(
    traces_path: str,
    in_features_path: str,
    out_features_path: str,
    model,
    threshold: float = 0.70,
    limit: int = None,
) -> pd.DataFrame:
    """Compute recurrence features for every trace and merge into feature CSV."""
    # Lazy-import parser (heavy spacy-less module is fine but keep it here)
    from src.parsing.rule_based_parser import parse_trace

    logger.info(f"Loading traces: {traces_path}")
    traces = []
    with open(traces_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            traces.append(json.loads(line))
            if limit and len(traces) >= limit:
                break
    logger.info(f"  {len(traces)} traces loaded")

    # Compute recurrence features per trace
    records = []
    for tr in tqdm(traces, desc=f"Embedding {os.path.basename(traces_path)}"):
        trace_text = tr.get("reasoning_trace") or tr.get("full_response") or ""
        item_id = tr.get("item_id")

        if not trace_text.strip():
            feats = {k: 0.0 for k in RECURRENCE_FEATURE_NAMES}
        else:
            try:
                episodes = parse_trace(trace_text)
            except Exception as e:
                logger.warning(f"Parse failed for {item_id}: {e}")
                episodes = None
            try:
                feats = extract_recurrence_features(
                    trace_text=trace_text,
                    model=model,
                    episodes=episodes,
                    threshold=threshold,
                )
            except Exception as e:
                logger.warning(f"Feature extract failed for {item_id}: {e}")
                feats = {k: 0.0 for k in RECURRENCE_FEATURE_NAMES}

        rec = {"item_id": item_id}
        rec.update(feats)
        records.append(rec)

    rec_df = pd.DataFrame(records)

    # Merge with existing feature CSV on item_id
    logger.info(f"Loading existing features: {in_features_path}")
    feat_df = pd.read_csv(in_features_path)

    before = len(feat_df)
    merged = feat_df.merge(rec_df, on="item_id", how="left")
    assert len(merged) == before, "Row count changed after merge"

    # Fill any missing (shouldn't happen, but safe)
    for col in RECURRENCE_FEATURE_NAMES:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    os.makedirs(os.path.dirname(out_features_path) or ".", exist_ok=True)
    merged.to_csv(out_features_path, index=False)
    logger.info(f"Wrote: {out_features_path}  shape={merged.shape}")

    # Quick sanity: correlation with correctness
    if "is_correct" in merged.columns:
        print(f"\n  Correlation with is_correct ({os.path.basename(traces_path)}):")
        for col in RECURRENCE_FEATURE_NAMES:
            if col in merged.columns:
                corr = merged[col].corr(merged["is_correct"].astype(float))
                print(f"    {col:30s} r = {corr:+.3f}")

    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", default=None,
                        help="Single trace JSONL path")
    parser.add_argument("--in-features", default=None,
                        help="Existing feature CSV")
    parser.add_argument("--out-features", default=None,
                        help="Output merged feature CSV")
    parser.add_argument("--all", action="store_true",
                        help="Process all 8 dataset-model combinations")
    parser.add_argument("--data-root", default="data",
                        help="Root of data directory")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--threshold", type=float, default=0.70,
                        help="Cosine similarity threshold for recurrence")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of traces per file (for debugging)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info(f"Loading embedder: {args.model}")
    model = load_embedder(args.model)

    if args.all:
        for name in DATASETS:
            traces = os.path.join(args.data_root, "traces", f"{name}_traces.jsonl")
            in_csv = os.path.join(args.data_root, "features", f"{name}_features.csv")
            out_csv = os.path.join(args.data_root, "features", f"{name}_features_rec.csv")
            if not os.path.exists(traces):
                logger.warning(f"Missing: {traces}, skip")
                continue
            if not os.path.exists(in_csv):
                logger.warning(f"Missing: {in_csv}, skip")
                continue
            process_trace_file(traces, in_csv, out_csv, model,
                               threshold=args.threshold, limit=args.limit)
    else:
        if not (args.traces and args.in_features and args.out_features):
            parser.error("Either --all or all of --traces/--in-features/--out-features")
        process_trace_file(args.traces, args.in_features, args.out_features,
                           model, threshold=args.threshold, limit=args.limit)


if __name__ == "__main__":
    main()
