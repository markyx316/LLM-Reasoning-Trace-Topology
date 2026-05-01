#!/usr/bin/env python3
"""
build_problem_alignment.py - Compute 4 problem-trace alignment features per item.

Reuses precomputed step embeddings from data/step_embeddings/*.npz to avoid
re-encoding steps. Only needs to additionally encode the problems themselves.

Merges into existing feature CSVs alongside recurrence features.

Usage:
    # One dataset:
    python scripts/build_problem_alignment.py \
        --traces data/traces/math500_qwen7b_traces.jsonl \
        --step-embeddings data/step_embeddings/math500_qwen7b.npz \
        --in-features data/features/math500_qwen7b_features_rec.csv \
        --out-features data/features/math500_qwen7b_features_align.csv

    # All 8 at once:
    python scripts/build_problem_alignment.py --all
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.problem_alignment import (
    ALIGNMENT_FEATURE_NAMES, encode_problems, extract_alignment_features,
)
from src.features.recurrence_features import load_embedder

logger = logging.getLogger(__name__)


DATASETS = [
    "math500_qwen7b", "math500_llama8b",
    "gsm8k_qwen7b", "gsm8k_llama8b",
    "gpqa_diamond_qwen7b", "gpqa_diamond_llama8b",
    "arc_challenge_qwen7b", "arc_challenge_llama8b",
]


def process(traces_path: str, step_npz: str,
            in_features: str, out_features: str, model):
    # Load traces (for problem text + trace text + item_id)
    logger.info(f"Loading traces: {traces_path}")
    items = []
    with open(traces_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    logger.info(f"  {len(items)} traces")

    # Load precomputed step embeddings
    logger.info(f"Loading step embeddings: {step_npz}")
    z = np.load(step_npz, allow_pickle=True)
    npz_ids = z["item_ids"].astype(str)
    npz_embs = z["embeddings"]     # object array
    id_to_emb = {iid: emb for iid, emb in zip(npz_ids, npz_embs)}

    # Encode problems in batch
    problems = [it.get("problem") or it.get("prompt") or "" for it in items]
    item_ids = [it.get("item_id") for it in items]
    traces = [it.get("reasoning_trace") or it.get("full_response") or "" for it in items]
    logger.info(f"Encoding {len(problems)} problems")
    problem_embs = encode_problems(problems, model)

    # Compute features
    records = []
    for iid, p_text, p_emb, t_text in tqdm(
            list(zip(item_ids, problems, problem_embs, traces)),
            desc=f"Aligning {os.path.basename(traces_path)}"):
        step_emb = id_to_emb.get(str(iid), None)
        if step_emb is None or len(step_emb) == 0:
            feats = {k: 0.0 for k in ALIGNMENT_FEATURE_NAMES}
        else:
            try:
                feats = extract_alignment_features(
                    problem_text=p_text,
                    step_embeddings=np.asarray(step_emb, dtype=np.float32),
                    trace_text=t_text,
                    model=model,
                    problem_emb=p_emb,
                )
            except Exception as e:
                logger.warning(f"alignment failed for {iid}: {e}")
                feats = {k: 0.0 for k in ALIGNMENT_FEATURE_NAMES}
        rec = {"item_id": iid}
        rec.update(feats)
        records.append(rec)

    align_df = pd.DataFrame(records)

    # Merge with existing feature CSV
    logger.info(f"Merging into: {in_features}")
    feat_df = pd.read_csv(in_features)
    feat_df["item_id"] = feat_df["item_id"].astype(str)
    align_df["item_id"] = align_df["item_id"].astype(str)
    merged = feat_df.merge(align_df, on="item_id", how="left")

    for c in ALIGNMENT_FEATURE_NAMES:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0.0)

    os.makedirs(os.path.dirname(out_features) or ".", exist_ok=True)
    merged.to_csv(out_features, index=False)
    logger.info(f"Wrote: {out_features}  shape={merged.shape}")

    if "is_correct" in merged.columns:
        print(f"\n  Correlation with is_correct ({os.path.basename(traces_path)}):")
        for c in ALIGNMENT_FEATURE_NAMES:
            if c in merged.columns:
                r = merged[c].corr(merged["is_correct"].astype(float))
                print(f"    {c:30s} r = {r:+.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces")
    ap.add_argument("--step-embeddings")
    ap.add_argument("--in-features")
    ap.add_argument("--out-features")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info(f"Loading embedder: {args.model}")
    model = load_embedder(args.model)

    if args.all:
        for name in DATASETS:
            traces = os.path.join(args.data_root, "traces", f"{name}_traces.jsonl")
            npz = os.path.join(args.data_root, "step_embeddings", f"{name}.npz")
            incsv = os.path.join(args.data_root, "features", f"{name}_features_rec.csv")
            outcsv = os.path.join(args.data_root, "features", f"{name}_features_align.csv")
            for path in (traces, npz, incsv):
                if not os.path.exists(path):
                    logger.warning(f"Missing {path}, skip dataset {name}")
                    break
            else:
                process(traces, npz, incsv, outcsv, model)
    else:
        if not (args.traces and args.step_embeddings and args.in_features and args.out_features):
            ap.error("--all or all four of --traces/--step-embeddings/--in-features/--out-features")
        process(args.traces, args.step_embeddings,
                args.in_features, args.out_features, model)


if __name__ == "__main__":
    main()
