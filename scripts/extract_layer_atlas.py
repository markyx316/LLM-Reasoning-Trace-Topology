#!/usr/bin/env python3
"""
extract_layer_atlas.py - Extract hidden states across LAYERS x POSITIONS.

For S1 (Layer-wise Probe Atlas): instead of taking only the last hidden
layer at the answer position, we extract a grid of (layer, position) pairs.
This lets us train a probe at every cell and find where correctness signal
is most concentrated.

We sample LAYERS uniformly across the model depth (8 layers per model:
indices 0%, 14%, 28%, 42%, 57%, 71%, 85%, 100% of depth) and four
POSITIONS: start, think_close, answer_marker, last_token.

Storage per item: 8 layers x 4 positions x hidden_dim x fp16 = ~260 KB
Total across 8 datasets x 6378 items: ~1.6 GB.

Usage:
    python scripts/extract_layer_atlas.py --all
    python scripts/extract_layer_atlas.py --traces ... --model ... --output ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


DATASET_MODEL_MAP = {
    "math500_qwen7b":       "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "math500_llama8b":      "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "gsm8k_qwen7b":         "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "gsm8k_llama8b":        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "gpqa_diamond_qwen7b":  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "gpqa_diamond_llama8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "arc_challenge_qwen7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "arc_challenge_llama8b":"deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
}

POSITION_NAMES = ["start", "think_close", "answer_marker", "last_token"]
ANSWER_MARKER_PAT = re.compile(
    r"(?:final\s+answer|answer\s+is|\\boxed\{)", re.IGNORECASE)


# =============================================================================
# Anchor finding
# =============================================================================

def find_anchor_indices(text, tokenizer, input_ids):
    L = input_ids.shape[-1]
    out = {"last_token": L - 1, "start": 0}

    close_think = text.rfind("</think>")
    if close_think != -1:
        prefix = text[:close_think + len("</think>")]
        tok_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        out["think_close"] = min(len(tok_ids) - 1, L - 1)
    else:
        out["think_close"] = L - 1

    matches = list(ANSWER_MARKER_PAT.finditer(text))
    if matches:
        end = matches[-1].end()
        prefix = text[:end]
        tok_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        out["answer_marker"] = min(len(tok_ids) - 1, L - 1)
    else:
        out["answer_marker"] = L - 1
    return out


def pick_layer_indices(n_total: int, n_keep: int = 8) -> list[int]:
    """Uniformly sample n_keep layer indices from [0..n_total-1]."""
    if n_total <= n_keep:
        return list(range(n_total))
    fracs = np.linspace(0.0, 1.0, n_keep)
    return sorted(set(int(round(f * (n_total - 1))) for f in fracs))


# =============================================================================
# EXTRACTION
# =============================================================================

def extract_one(text, tokenizer, model, layer_indices, device, max_len=4096):
    """Returns (n_layers, n_positions, hidden_dim) tensor on CPU as fp16."""
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_len, add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    am = enc.get("attention_mask")
    am = am.to(device) if am is not None else None

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out = model(input_ids=input_ids, attention_mask=am,
                    output_hidden_states=True, use_cache=False)
    # tuple of (n_layers+1) of (1, L, H)
    L = input_ids.shape[-1]
    anchors = find_anchor_indices(text, tokenizer, input_ids)
    pos_idx = [anchors[name] for name in POSITION_NAMES]

    arr = np.zeros((len(layer_indices), len(POSITION_NAMES),
                    out.hidden_states[0].shape[-1]), dtype=np.float16)
    for li, layer_idx in enumerate(layer_indices):
        h = out.hidden_states[layer_idx][0]   # (L, H)
        for pi, p in enumerate(pos_idx):
            p = min(max(int(p), 0), L - 1)
            arr[li, pi] = h[p].to(torch.float16).cpu().numpy()
    return arr


def process(traces_path, model_name, output, max_len=4096, device="cuda",
            n_layers_keep: int = 8):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    items = []
    with open(traces_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    logger.info(f"Loaded {len(items)} traces from {traces_path}")

    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map={"": device},
    )
    model.eval()

    n_total_layers = len(model.model.layers) + 1   # +1 for embedding
    layer_idx = pick_layer_indices(n_total_layers, n_keep=n_layers_keep)
    logger.info(f"  total layers (with emb)={n_total_layers}, "
                f"sampling indices={layer_idx}")
    hidden_dim = model.config.hidden_size

    n = len(items)
    arr = np.zeros((n, len(layer_idx), len(POSITION_NAMES), hidden_dim),
                   dtype=np.float16)
    item_ids = np.empty(n, dtype=object)
    group = os.path.basename(traces_path).replace("_traces.jsonl", "")
    groups = np.full(n, group, dtype=object)
    y_true = np.zeros(n, dtype=np.int8)

    for i, it in enumerate(tqdm(items, desc=group)):
        trace = it.get("reasoning_trace") or it.get("full_response") or ""
        item_ids[i] = it.get("item_id")
        y_true[i] = int(it.get("is_correct", False))
        if not trace.strip():
            continue
        try:
            arr[i] = extract_one(trace, tokenizer, model, layer_idx, device,
                                 max_len=max_len)
        except Exception as e:
            logger.warning(f"failed for {item_ids[i]}: {e}")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    np.savez_compressed(
        output,
        item_ids=item_ids, groups=groups, y_true=y_true,
        hidden=arr,           # (N, n_layers_keep, n_positions, hidden_dim)
        layer_indices=np.array(layer_idx, dtype=np.int32),
        position_names=np.array(POSITION_NAMES, dtype=object),
        model_name=np.array([model_name]),
        n_total_layers=np.array([n_total_layers]),
    )
    logger.info(f"Saved {output}, shape={arr.shape}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces"); ap.add_argument("--model")
    ap.add_argument("--output")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="data/hidden_atlas")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--n-layers", type=int, default=8)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.all:
        for ds, mpath in DATASET_MODEL_MAP.items():
            traces = f"{args.data_root}/traces/{ds}_traces.jsonl"
            out = f"{args.out_dir}/{ds}.npz"
            if not os.path.exists(traces):
                logger.warning(f"missing {traces}, skip"); continue
            if os.path.exists(out):
                logger.info(f"exists, skip: {out}"); continue
            process(traces, mpath, out, max_len=args.max_len, device=device,
                    n_layers_keep=args.n_layers)
    else:
        if not (args.traces and args.model and args.output):
            ap.error("--all OR all of --traces/--model/--output")
        process(args.traces, args.model, args.output,
                max_len=args.max_len, device=device,
                n_layers_keep=args.n_layers)


if __name__ == "__main__":
    main()
