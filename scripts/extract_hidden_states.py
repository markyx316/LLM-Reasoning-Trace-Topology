#!/usr/bin/env python3
"""
extract_hidden_states.py - Extract hidden states from the generator LLM for each trace.

For each trace, we run a single TEACHER-FORCING forward pass through the
generator (Qwen-7B or Llama-8B) on the trace tokens, then save the hidden
state at a chosen "anchor" position per layer.

Why this is a useful UQ signal:
  The generator's own internal state at the moment of producing its answer
  carries compressed "decision" information that plain text cannot encode.
  A linear probe on this state has been shown to detect correctness of
  short-form answers (Kadavath et al. 2022). We adapt the idea to long,
  free-form reasoning traces — which to our knowledge has not been studied.

Positions we extract (all from the LAST hidden layer of the model):
  - last_token:       The final token of the full trace+answer text.
  - think_close:      Position right after "</think>" if present in the
                      R1-style trace, else falls back to last_token.
  - answer_marker:    Position after the last occurrence of phrases like
                      "answer is", "final answer", "boxed{" (case insensitive),
                      else falls back to last_token.

Output: a compressed .npz per trace file:
    item_ids       (N,)     object (str)
    groups         (N,)     object (str)
    y_true         (N,)     int8
    h_last         (N, d)   float16  -- hidden state at last token
    h_think        (N, d)   float16  -- hidden state at </think>
    h_answer       (N, d)   float16  -- hidden state at answer marker

Usage:
    python scripts/extract_hidden_states.py \
        --traces data/traces/math500_qwen7b_traces.jsonl \
        --model  deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --output data/hidden_states/math500_qwen7b.npz
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
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.generation_uncertainty import (
    GENERATION_UNC_FEATURE_NAMES, summarize_trajectory,
)

logger = logging.getLogger(__name__)

# Dataset -> (generator HF path). We assume these match what was used to
# produce the traces. If you regenerate traces with a different model,
# update this map.
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


# =============================================================================
# ANCHOR POSITION FINDING
# =============================================================================

ANSWER_MARKER_PAT = re.compile(
    r"(?:final\s+answer|answer\s+is|\\boxed\{)",
    re.IGNORECASE,
)


def find_anchor_positions(text: str, tokenizer, input_ids) -> dict[str, int]:
    """Return dict of {anchor_name: token_index (0-based into input_ids[0])}.
    All positions fall back to last_token if the target phrase is not found."""
    L = input_ids.shape[-1]
    out = {"last_token": L - 1}

    # </think> — common in DeepSeek-R1 style traces
    close_think = text.rfind("</think>")
    if close_think != -1:
        # Tokens up to and including this character
        prefix = text[:close_think + len("</think>")]
        tok_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        out["think_close"] = min(len(tok_ids) - 1, L - 1)
    else:
        out["think_close"] = L - 1

    # answer marker — last match
    matches = list(ANSWER_MARKER_PAT.finditer(text))
    if matches:
        end = matches[-1].end()
        prefix = text[:end]
        tok_ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        out["answer_marker"] = min(len(tok_ids) - 1, L - 1)
    else:
        out["answer_marker"] = L - 1

    return out


# =============================================================================
# EXTRACTION
# =============================================================================

def extract_one(text: str, tokenizer, model, device: str,
                max_len: int = 4096) -> dict:
    """Forward pass on the given text. Returns both:
       - vecs: hidden states at 3 anchor positions (Direction B signal)
       - gen_unc: 10 summary features of per-token logprob/entropy (Direction A)
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len,
                    add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask").to(device) if enc.get("attention_mask") is not None else None

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True, use_cache=False)

    # --- Direction B: Hidden state at 3 positions ---
    last_layer = out.hidden_states[-1][0]       # (L, H)
    L = last_layer.size(0)
    anchors = find_anchor_positions(text, tokenizer, input_ids)
    vecs = {}
    for name, pos in anchors.items():
        pos = min(max(int(pos), 0), L - 1)
        vecs[name] = last_layer[pos].to(torch.float16).cpu().numpy()

    # --- Direction A: per-token logprob + entropy trajectory ---
    # logits: (1, L, V). Predict token_{t+1} from positions 0..L-2.
    logits = out.logits[0]                       # (L, V), fp16 or bf16
    # Upcast to fp32 just for stable softmax/log
    logits_f = logits[:-1].to(torch.float32)     # (L-1, V)
    targets = input_ids[0, 1:]                   # (L-1,)

    # log_softmax
    log_probs = torch.log_softmax(logits_f, dim=-1)         # (L-1, V)
    token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (L-1,)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)              # (L-1,) in nats

    token_lp_np = token_log_probs.detach().cpu().numpy()
    entropy_np  = entropy.detach().cpu().numpy()

    # Summary features
    gen_unc = summarize_trajectory(token_lp_np, entropy_np)

    return {"vecs": vecs, "gen_unc": gen_unc}


# =============================================================================
# BATCH DRIVER
# =============================================================================

def process(traces_path: str, model_name: str, output: str,
            max_len: int = 4096, device: str = "cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading traces: {traces_path}")
    items = []
    with open(traces_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    logger.info(f"  {len(items)} items")

    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()
    hidden_dim = model.config.hidden_size
    logger.info(f"  hidden_dim = {hidden_dim}")

    group = os.path.basename(traces_path).replace("_traces.jsonl", "")

    # Pre-allocate
    n = len(items)
    h_last   = np.zeros((n, hidden_dim), dtype=np.float16)
    h_think  = np.zeros((n, hidden_dim), dtype=np.float16)
    h_answer = np.zeros((n, hidden_dim), dtype=np.float16)
    item_ids = np.empty(n, dtype=object)
    groups = np.full(n, group, dtype=object)
    y_true = np.zeros(n, dtype=np.int8)

    # Direction A: collect summary features per trace (for CSV)
    gen_unc_records = []

    for i, it in enumerate(tqdm(items, desc=f"Forward {group}")):
        trace = it.get("reasoning_trace") or it.get("full_response") or ""
        iid = it.get("item_id")
        item_ids[i] = iid
        y_true[i] = int(it.get("is_correct", False))
        if not trace.strip():
            gen_unc_records.append({"item_id": iid,
                                    **{k: 0.0 for k in GENERATION_UNC_FEATURE_NAMES}})
            continue
        try:
            result = extract_one(trace, tokenizer, model, device, max_len=max_len)
        except Exception as e:
            logger.warning(f"extract failed for {iid}: {e}")
            gen_unc_records.append({"item_id": iid,
                                    **{k: 0.0 for k in GENERATION_UNC_FEATURE_NAMES}})
            continue
        h_last[i]   = result["vecs"]["last_token"]
        h_think[i]  = result["vecs"]["think_close"]
        h_answer[i] = result["vecs"]["answer_marker"]
        rec = {"item_id": iid}
        rec.update(result["gen_unc"])
        gen_unc_records.append(rec)

    # Save hidden states npz (Direction B)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    np.savez_compressed(
        output,
        item_ids=item_ids, groups=groups, y_true=y_true,
        h_last=h_last, h_think=h_think, h_answer=h_answer,
        model_name=np.array([model_name]),
    )
    logger.info(f"Saved hidden states: {output}  items={n}  hidden_dim={hidden_dim}")

    # Save generation uncertainty CSV (Direction A) next to existing feature CSVs
    features_dir = os.path.join(os.path.dirname(os.path.dirname(output)), "features")
    os.makedirs(features_dir, exist_ok=True)
    gen_unc_df = pd.DataFrame(gen_unc_records)
    gen_unc_csv = os.path.join(features_dir, f"{group}_features_genunc.csv")
    gen_unc_df.to_csv(gen_unc_csv, index=False)
    logger.info(f"Saved generation-uncertainty features: {gen_unc_csv}")

    # Quick correlation check
    corrs = {}
    for c in GENERATION_UNC_FEATURE_NAMES:
        r = np.corrcoef(gen_unc_df[c].to_numpy(), y_true)[0, 1]
        corrs[c] = float(r) if not np.isnan(r) else 0.0
    logger.info(f"  gen_unc correlations with is_correct ({group}):")
    for c, r in corrs.items():
        logger.info(f"    {c:25s} r = {r:+.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces")
    ap.add_argument("--model")
    ap.add_argument("--output")
    ap.add_argument("--all", action="store_true",
                    help="Process all 8 dataset-model files")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="data/hidden_states")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    if args.all:
        # Group by model to avoid repeated model loading
        by_model: dict[str, list[str]] = {}
        for name, mpath in DATASET_MODEL_MAP.items():
            by_model.setdefault(mpath, []).append(name)
        for mpath, ds_list in by_model.items():
            for ds in ds_list:
                traces = os.path.join(args.data_root, "traces", f"{ds}_traces.jsonl")
                out = os.path.join(args.out_dir, f"{ds}.npz")
                if not os.path.exists(traces):
                    logger.warning(f"missing: {traces}, skip")
                    continue
                if os.path.exists(out):
                    logger.info(f"exists, skip: {out}")
                    continue
                process(traces, mpath, out, max_len=args.max_len, device=device)
    else:
        if not (args.traces and args.model and args.output):
            ap.error("--all or all of --traces/--model/--output")
        process(args.traces, args.model, args.output,
                max_len=args.max_len, device=device)


if __name__ == "__main__":
    main()
