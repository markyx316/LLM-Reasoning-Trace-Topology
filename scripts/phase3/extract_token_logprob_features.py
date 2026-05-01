"""Phase 3 (T2) — Richer token-level logprob features for every trace.

The existing `scripts/extract_hidden_states.py` already teacher-forces the
generator over every trace and saves 10 trace-level generation-uncertainty
summary features (`_features_genunc.csv`). Those are decent. They're also
already used by Phase 2's super-hybrid.

This script adds *novel* features that the 10-feature summary misses:

  A. Positional segments
     quartile NLL + entropy (4 segments × 2 stats = 8 features)
     — captures "does the model get more / less confident late?"

  B. Tail / answer-moment
     tail_50_mean_nll      mean NLL of the last 50 tokens
     tail_50_max_nll       biggest token-level NLL among last 50
     boxed_mean_nll        NLL of tokens inside `\\boxed{...}` if present
     boxed_max_nll
     answer_marker_nll     mean NLL of tokens immediately after the most
                           recent "answer is / final answer" marker
     — Captures "how confident is the model *at the answer*?"

  C. Sliding-window spikes
     max_window_50_nll     worst 50-token rolling-mean NLL
     max_window_50_pos     position of that worst window (normalized)
     nll_spike_count       # of single-token NLL > 8 nats
     nll_spike_frac_tail   fraction of those spikes in the last 20%
     — Captures "did the model have a localized meltdown, and where?"

  D. Entropy collapse / peak
     entropy_peak_pos      (same as max_entropy_pos but on a smoothed curve)
     entropy_collapse      max drop in entropy over a 20-token window
                           (abrupt transition from confusion to certainty)

Output
------
  - data/features/{group}_features_steplp.csv   (24 new features per trace)
  - data/token_logprobs/{group}_raw.npz         (padded raw arrays for
    downstream models; contains `token_lp`, `entropy`, `mask`, `item_ids`)

Usage (single dataset)
----------------------
    python scripts/phase3/extract_token_logprob_features.py \
        --traces data/traces/math500_qwen7b_traces.jsonl \
        --model  deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --output-features data/features/math500_qwen7b_features_steplp.csv \
        --output-raw      data/token_logprobs/math500_qwen7b_raw.npz

Usage (all 8)
-------------
    python scripts/phase3/extract_token_logprob_features.py --all
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

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# Keep this dataset->model map in lockstep with extract_hidden_states.py
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


ANSWER_MARKER_PAT = re.compile(
    r"(?:final\s+answer|answer\s+is|\\boxed\{)",
    re.IGNORECASE,
)
BOXED_START_PAT = re.compile(r"\\boxed\{")


# =============================================================================
# FEATURE COMPUTATION
# =============================================================================

NEW_FEATURES = [
    # Quartile (8)
    "q1_mean_nll", "q2_mean_nll", "q3_mean_nll", "q4_mean_nll",
    "q1_mean_ent", "q2_mean_ent", "q3_mean_ent", "q4_mean_ent",
    # Tail / answer (5)
    "tail_50_mean_nll", "tail_50_max_nll",
    "boxed_mean_nll", "boxed_max_nll",
    "answer_marker_nll",
    # Sliding-window (4)
    "max_window_50_nll", "max_window_50_pos",
    "nll_spike_count", "nll_spike_frac_tail",
    # Entropy collapse (2)
    "entropy_peak_pos_smooth", "entropy_collapse",
    # Normalization info (1)
    "n_tokens",
]


def _mean_over_range(arr: np.ndarray, lo: int, hi: int) -> float:
    """Mean of arr[lo:hi]; NaN if empty."""
    if hi <= lo or lo < 0 or hi > len(arr):
        return float("nan")
    sub = arr[lo:hi]
    if len(sub) == 0:
        return float("nan")
    return float(np.mean(sub))


def _max_over_range(arr: np.ndarray, lo: int, hi: int) -> float:
    if hi <= lo or lo < 0 or hi > len(arr):
        return float("nan")
    sub = arr[lo:hi]
    if len(sub) == 0:
        return float("nan")
    return float(np.max(sub))


def compute_features(
    token_lp: np.ndarray,   # (L-1,) log p(token_i | ctx)  — NATS, NEGATIVE when surprising
    entropy:  np.ndarray,   # (L-1,) entropy in nats
    text: str,
    tokenizer,
    input_ids,              # (1, L)
) -> dict:
    """Compute the 20 novel features from per-token logprob + entropy arrays."""
    nll = -token_lp  # positive NLL; bigger = more surprising
    L = len(nll)
    if L == 0:
        return {k: float("nan") for k in NEW_FEATURES} | {"n_tokens": 0}

    # --- (A) Quartiles -----------------------------------------------------
    q = L // 4
    q_nll = [
        _mean_over_range(nll, 0, q),
        _mean_over_range(nll, q, 2 * q),
        _mean_over_range(nll, 2 * q, 3 * q),
        _mean_over_range(nll, 3 * q, L),
    ]
    q_ent = [
        _mean_over_range(entropy, 0, q),
        _mean_over_range(entropy, q, 2 * q),
        _mean_over_range(entropy, 2 * q, 3 * q),
        _mean_over_range(entropy, 3 * q, L),
    ]

    # --- (B) Tail / answer -------------------------------------------------
    tail_n = min(50, L)
    tail_mean = _mean_over_range(nll, L - tail_n, L)
    tail_max = _max_over_range(nll, L - tail_n, L)

    # \boxed{...}
    boxed_mean = float("nan")
    boxed_max = float("nan")
    m_box = list(BOXED_START_PAT.finditer(text))
    if m_box:
        # Find the character range of the last \boxed{...} (matching braces)
        start_char = m_box[-1].end()  # right after `\boxed{`
        depth = 1
        end_char = start_char
        while end_char < len(text) and depth > 0:
            c = text[end_char]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            end_char += 1
        # Map char indices to token indices via length-prefix tokenization
        try:
            tok_prefix = tokenizer(
                text[:start_char], return_tensors="pt", add_special_tokens=False,
            )["input_ids"][0]
            tok_full = tokenizer(
                text[:end_char], return_tensors="pt", add_special_tokens=False,
            )["input_ids"][0]
            tk_lo, tk_hi = len(tok_prefix), len(tok_full)
            # Clamp to valid range within (L-1,) logprob array
            tk_lo = max(0, min(tk_lo - 1, L - 1))  # -1 because token_lp is shifted
            tk_hi = max(0, min(tk_hi - 1, L))
            if tk_hi > tk_lo:
                boxed_mean = _mean_over_range(nll, tk_lo, tk_hi)
                boxed_max = _max_over_range(nll, tk_lo, tk_hi)
        except Exception:
            pass

    # answer-marker: mean NLL of the 20 tokens immediately after the LAST
    # "final answer"/"answer is"/"\boxed{" match
    answer_marker_nll = float("nan")
    matches = list(ANSWER_MARKER_PAT.finditer(text))
    if matches:
        end_char = matches[-1].end()
        try:
            tok_prefix = tokenizer(
                text[:end_char], return_tensors="pt", add_special_tokens=False,
            )["input_ids"][0]
            lo = max(0, len(tok_prefix) - 1)
            hi = min(lo + 20, L)
            if hi > lo:
                answer_marker_nll = _mean_over_range(nll, lo, hi)
        except Exception:
            pass

    # --- (C) Sliding-window spikes ----------------------------------------
    max_w_nll = float("nan")
    max_w_pos = float("nan")
    if L >= 50:
        # Rolling mean over window 50
        cs = np.cumsum(np.insert(nll, 0, 0.0))  # (L+1,)
        w = 50
        roll = (cs[w:] - cs[:-w]) / w           # (L-w+1,)
        idx = int(np.argmax(roll))
        max_w_nll = float(roll[idx])
        max_w_pos = float(idx / max(L - w, 1))
    else:
        max_w_nll = float(np.mean(nll))
        max_w_pos = 0.0

    # Spike detection: NLL > 8 nats (≈ p < 3e-4 under Gaussian-ish ctx)
    spikes = nll > 8.0
    n_spikes = int(spikes.sum())
    if n_spikes > 0:
        tail_start = int(0.8 * L)
        n_tail_spikes = int(spikes[tail_start:].sum())
        frac_tail = float(n_tail_spikes / max(n_spikes, 1))
    else:
        frac_tail = 0.0

    # --- (D) Entropy collapse ---------------------------------------------
    # Smooth entropy with a 20-token mean, then find the biggest drop over
    # any 20-token stride.
    if L >= 20:
        w = 20
        cs_e = np.cumsum(np.insert(entropy, 0, 0.0))
        smooth = (cs_e[w:] - cs_e[:-w]) / w   # (L-w+1,)
        peak_pos = float(np.argmax(smooth) / max(L - w, 1))
        # Collapse = max(smooth[i]) - min(smooth[j]) for j > i (entropy falls)
        # O(L) by maintaining running max to the left:
        running_max = np.maximum.accumulate(smooth)
        drop = running_max - smooth
        collapse = float(drop.max())
    else:
        peak_pos = float("nan")
        collapse = float("nan")

    feats = {
        "q1_mean_nll": q_nll[0], "q2_mean_nll": q_nll[1],
        "q3_mean_nll": q_nll[2], "q4_mean_nll": q_nll[3],
        "q1_mean_ent": q_ent[0], "q2_mean_ent": q_ent[1],
        "q3_mean_ent": q_ent[2], "q4_mean_ent": q_ent[3],
        "tail_50_mean_nll": tail_mean,
        "tail_50_max_nll":  tail_max,
        "boxed_mean_nll":   boxed_mean,
        "boxed_max_nll":    boxed_max,
        "answer_marker_nll": answer_marker_nll,
        "max_window_50_nll": max_w_nll,
        "max_window_50_pos": max_w_pos,
        "nll_spike_count":   float(n_spikes),
        "nll_spike_frac_tail": frac_tail,
        "entropy_peak_pos_smooth": peak_pos,
        "entropy_collapse":  collapse,
        "n_tokens":          int(L),
    }
    return feats


# =============================================================================
# EXTRACT + SAVE
# =============================================================================

def extract_arrays(text: str, tokenizer, model, device: str,
                   max_len: int = 4096) -> tuple[np.ndarray, np.ndarray, object]:
    """Single teacher-forcing pass; returns (token_lp, entropy, input_ids)."""
    import torch
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len,
                    add_special_tokens=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = (enc.get("attention_mask") or None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=False, use_cache=False)

    logits = out.logits[0][:-1].to(torch.float32)     # (L-1, V)
    targets = input_ids[0, 1:]                        # (L-1,)
    log_probs = torch.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).cpu().numpy()
    probs = log_probs.exp()
    entropy = (-(probs * log_probs).sum(dim=-1)).cpu().numpy()
    return token_lp, entropy, input_ids


def process(traces_path: str, model_name: str,
            out_feats: str, out_raw: str, max_len: int = 4096):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    logger = logging.getLogger("steplp")

    with open(traces_path) as f:
        items = [json.loads(ln) for ln in f if ln.strip()]
    logger.info("Loaded %d traces from %s", len(items), traces_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map={"": device},
    )
    model.eval()

    group = os.path.basename(traces_path).replace("_traces.jsonl", "")

    # For raw npz, we save padded sequences clipped at max_len.
    raw_lp = np.zeros((len(items), max_len), dtype=np.float16)
    raw_ent = np.zeros((len(items), max_len), dtype=np.float16)
    raw_mask = np.zeros((len(items), max_len), dtype=np.bool_)

    rows = []
    for i, it in enumerate(tqdm(items, desc=f"TeacherForce {group}")):
        iid = it.get("item_id")
        text = it.get("reasoning_trace") or it.get("full_response") or ""
        rec = {"item_id": iid}
        if not text.strip():
            rec.update({k: float("nan") for k in NEW_FEATURES})
            rec["n_tokens"] = 0
            rows.append(rec)
            continue
        try:
            lp, ent, input_ids = extract_arrays(
                text, tokenizer, model, device, max_len=max_len,
            )
            feats = compute_features(lp, ent, text, tokenizer, input_ids)
            rec.update(feats)
            L = len(lp)
            raw_lp[i, :L] = lp.astype(np.float16)
            raw_ent[i, :L] = ent.astype(np.float16)
            raw_mask[i, :L] = True
        except Exception as e:
            logger.warning("extract failed for %s: %s", iid, e)
            rec.update({k: float("nan") for k in NEW_FEATURES})
            rec["n_tokens"] = 0
        rows.append(rec)

    df = pd.DataFrame(rows)
    # Reorder: item_id first, then feature columns in NEW_FEATURES order
    cols = ["item_id"] + NEW_FEATURES
    df = df[cols]
    os.makedirs(os.path.dirname(out_feats) or ".", exist_ok=True)
    df.to_csv(out_feats, index=False)
    logger.info("Wrote features %s  (n=%d, %d cols)", out_feats, len(df), len(cols))

    item_ids = np.array([it.get("item_id") for it in items], dtype=object)
    y_true = np.array([int(bool(it.get("is_correct", 0))) for it in items], dtype=np.int8)
    groups = np.array([group] * len(items), dtype=object)
    os.makedirs(os.path.dirname(out_raw) or ".", exist_ok=True)
    np.savez_compressed(
        out_raw,
        item_ids=item_ids, groups=groups, y_true=y_true,
        token_lp=raw_lp, entropy=raw_ent, mask=raw_mask,
        model_name=np.array([model_name]),
    )
    logger.info("Wrote raw arrays %s  shape=(%d, %d)",
                out_raw, len(items), max_len)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--traces")
    ap.add_argument("--model")
    ap.add_argument("--output-features")
    ap.add_argument("--output-raw")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--data-root", default="data")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger("steplp_main")

    if args.all:
        for name, mpath in DATASET_MODEL_MAP.items():
            traces = os.path.join(args.data_root, "traces", f"{name}_traces.jsonl")
            out_feats = os.path.join("data/features", f"{name}_features_steplp.csv")
            out_raw = os.path.join("data/token_logprobs", f"{name}_raw.npz")
            if not os.path.exists(traces):
                log.warning("missing: %s, skip", traces)
                continue
            if os.path.exists(out_feats) and os.path.exists(out_raw):
                log.info("exists, skip: %s", name)
                continue
            process(traces, mpath, out_feats, out_raw, max_len=args.max_len)
    else:
        if not (args.traces and args.model and args.output_features and args.output_raw):
            ap.error("need --all OR --traces/--model/--output-features/--output-raw")
        process(args.traces, args.model, args.output_features, args.output_raw,
                max_len=args.max_len)


if __name__ == "__main__":
    main()
