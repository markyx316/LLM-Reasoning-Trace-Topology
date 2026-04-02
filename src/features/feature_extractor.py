"""
feature_extractor.py - Feature extraction for the 6-class rule-based parser.

Extracts 25 features from a parsed reasoning trace in three groups:

  Group 1 — Length & Proportion   (8 features)
    total_tokens, total_episodes,
    prop_forward, prop_verify, prop_revise, prop_restart,
    prop_hesitate, prop_conclude

  Group 2 — Structural / Topological  (10 features)
    revise_count, verify_count, restart_count,
    vf_ratio, revise_position_mean, first_conclude_pos,
    v_clustering, max_forward_run,
    transition_entropy, cycle_count

  Group 3 — Content-Free Meta         (7 features)
    wait_density, maybe_density, verify_density, actually_density,
    negation_density, question_mark_rate, repetition_rate_4gram

All features are deterministic, content-free (Group 3 uses only
surface-form tokens, no domain knowledge), and require only a single
generation of the trace.

Usage
-----
    # From pre-parsed episodes
    from src.parsing.rule_based_parser import parse_trace
    from src.features.feature_extractor import extract_features, to_array

    episodes = parse_trace(trace_text)
    feat_dict = extract_features(trace_text, episodes)
    feat_vec  = to_array(feat_dict)           # shape (25,)

    # Direct from raw trace text (parses internally)
    from src.features.feature_extractor import extract_from_text
    feat_dict = extract_from_text(trace_text)

    # Batch from JSONL trace file
    from src.features.feature_extractor import extract_from_jsonl
    extract_from_jsonl("data/traces/math500_qwen7b_traces.jsonl",
                       "data/features/math500_qwen7b_rbp_features.csv")
"""

import csv
import json
import logging
import math
import os
import re
from collections import Counter
from typing import Optional

import numpy as np

from src.parsing.rule_based_parser import (
    BehaviorType, Episode, parse_trace, sequence_to_string,
)

logger = logging.getLogger(__name__)

# Canonical behavior order — used for proportion / count features
_BEHAVIORS = [
    BehaviorType.FORWARD,
    BehaviorType.VERIFY,
    BehaviorType.REVISE,
    BehaviorType.RESTART,
    BehaviorType.HESITATE,
    BehaviorType.CONCLUDE,
]
_B_IDX = {b: i for i, b in enumerate(_BEHAVIORS)}


# =============================================================================
# FEATURE NAME MANIFEST
# =============================================================================

def get_feature_names() -> list[str]:
    """Return the ordered list of all 25 feature names."""
    return [
        # Group 1: Length & Proportion (8)
        "total_tokens", "total_episodes",
        "prop_forward", "prop_verify", "prop_revise",
        "prop_restart", "prop_hesitate", "prop_conclude",
        # Group 2: Structural / Topological (10)
        "revise_count", "verify_count", "restart_count",
        "vf_ratio", "revise_position_mean", "first_conclude_pos",
        "v_clustering", "max_forward_run",
        "transition_entropy", "cycle_count",
        # Group 3: Content-Free Meta (7)
        "wait_density", "maybe_density", "verify_density",
        "actually_density", "negation_density",
        "question_mark_rate", "repetition_rate_4gram",
    ]


def get_feature_groups() -> dict[str, list[str]]:
    """Return feature names organized by group."""
    return {
        "group1_length": [
            "total_tokens", "total_episodes",
            "prop_forward", "prop_verify", "prop_revise",
            "prop_restart", "prop_hesitate", "prop_conclude",
        ],
        "group2_structural": [
            "revise_count", "verify_count", "restart_count",
            "vf_ratio", "revise_position_mean", "first_conclude_pos",
            "v_clustering", "max_forward_run",
            "transition_entropy", "cycle_count",
        ],
        "group3_meta": [
            "wait_density", "maybe_density", "verify_density",
            "actually_density", "negation_density",
            "question_mark_rate", "repetition_rate_4gram",
        ],
    }


# =============================================================================
# GROUP 1: LENGTH & PROPORTION
# =============================================================================

def _extract_length(trace_text: str, episodes: list[Episode]) -> dict[str, float]:
    """
    8 features: raw counts + per-behavior proportions.

    total_tokens    — whitespace-split word count of the full trace
    total_episodes  — number of classified sentences
    prop_*          — fraction of episodes for each behavior type
    """
    total_tokens   = len(trace_text.split()) if trace_text else 0
    total_episodes = len(episodes)
    denom          = max(total_episodes, 1)

    counts = Counter(ep.behavior for ep in episodes)

    return {
        "total_tokens":   float(total_tokens),
        "total_episodes": float(total_episodes),
        "prop_forward":   counts.get(BehaviorType.FORWARD,   0) / denom,
        "prop_verify":    counts.get(BehaviorType.VERIFY,    0) / denom,
        "prop_revise":    counts.get(BehaviorType.REVISE,    0) / denom,
        "prop_restart":   counts.get(BehaviorType.RESTART,   0) / denom,
        "prop_hesitate":  counts.get(BehaviorType.HESITATE,  0) / denom,
        "prop_conclude":  counts.get(BehaviorType.CONCLUDE,  0) / denom,
    }


# =============================================================================
# GROUP 2: STRUCTURAL / TOPOLOGICAL
# =============================================================================

def _extract_structural(episodes: list[Episode]) -> dict[str, float]:
    """
    10 features capturing the shape of the reasoning trajectory.

    revise_count         — total number of Revise (X) episodes
    verify_count         — total number of Verify (V) episodes
    restart_count        — total number of Restart (R) episodes

    vf_ratio             — verify_count / max(forward_count, 1)
                           High ratio → lots of self-checking relative to new content

    revise_position_mean — mean normalized position of Revise episodes [0, 1]
                           0 = very early, 1 = very late in trace
                           Late revisions → the model realized errors only near the end

    first_conclude_pos   — normalized position of the first Conclude episode [0, 1]
                           1.0 when no Conclude found (trace never reaches an answer)

    v_clustering         — coefficient of variation of inter-Verify gaps
                           = std(gaps) / mean(gaps)
                           High = verifications are bunched; low = evenly spaced

    max_forward_run      — longest consecutive streak of Forward episodes
                           Very long runs without revision → potentially over-confident

    transition_entropy   — Shannon entropy of the behavior bigram distribution
                           H = 0 → always the same transition (highly stereotyped)
                           H > 0 → diverse, complex reasoning trajectory

    cycle_count          — number of semantic revisitations:
                           episode pairs ≥ 3 apart with Jaccard word-overlap ≥ 0.30
                           and a Revise / Verify / Restart episode between them
    """
    _zero = {
        "revise_count": 0.0, "verify_count": 0.0, "restart_count": 0.0,
        "vf_ratio": 0.0, "revise_position_mean": 0.0,
        "first_conclude_pos": 1.0, "v_clustering": 0.0,
        "max_forward_run": 0.0, "transition_entropy": 0.0, "cycle_count": 0.0,
    }
    if not episodes:
        return _zero

    behaviors = [ep.behavior for ep in episodes]
    n = len(behaviors)

    # --- Raw counts ---
    revise_count  = sum(1 for b in behaviors if b == BehaviorType.REVISE)
    verify_count  = sum(1 for b in behaviors if b == BehaviorType.VERIFY)
    restart_count = sum(1 for b in behaviors if b == BehaviorType.RESTART)
    forward_count = sum(1 for b in behaviors if b == BehaviorType.FORWARD)

    # --- Verify-to-Forward ratio ---
    vf_ratio = verify_count / max(forward_count, 1)

    # --- Revise position (mean normalized) ---
    revise_positions = [
        i / max(n - 1, 1)
        for i, b in enumerate(behaviors)
        if b == BehaviorType.REVISE
    ]
    revise_position_mean = float(np.mean(revise_positions)) if revise_positions else 0.0

    # --- First Conclude position ---
    conclude_indices = [i for i, b in enumerate(behaviors) if b == BehaviorType.CONCLUDE]
    first_conclude_pos = (
        conclude_indices[0] / max(n - 1, 1) if conclude_indices else 1.0
    )

    # --- Verification clustering (CV of inter-Verify gaps) ---
    v_positions = [i for i, b in enumerate(behaviors) if b == BehaviorType.VERIFY]
    if len(v_positions) >= 2:
        gaps = np.diff(v_positions).astype(float)
        mean_gap = np.mean(gaps)
        v_clustering = float(np.std(gaps) / mean_gap) if mean_gap > 0 else 0.0
    else:
        v_clustering = 0.0

    # --- Longest consecutive Forward run ---
    max_forward_run = current_run = 0
    for b in behaviors:
        if b == BehaviorType.FORWARD:
            current_run += 1
            max_forward_run = max(max_forward_run, current_run)
        else:
            current_run = 0

    # --- Behavior bigram transition entropy ---
    transition_entropy = _compute_transition_entropy(behaviors)

    # --- Semantic cycle count (word-overlap heuristic) ---
    cycle_count = _count_cycles(episodes)

    return {
        "revise_count":         float(revise_count),
        "verify_count":         float(verify_count),
        "restart_count":        float(restart_count),
        "vf_ratio":             vf_ratio,
        "revise_position_mean": revise_position_mean,
        "first_conclude_pos":   first_conclude_pos,
        "v_clustering":         v_clustering,
        "max_forward_run":      float(max_forward_run),
        "transition_entropy":   transition_entropy,
        "cycle_count":          float(cycle_count),
    }


def _compute_transition_entropy(behaviors: list[BehaviorType]) -> float:
    """
    Shannon entropy of the behavior bigram distribution.

    We build the row-normalized transition matrix over all 6 behavior types,
    compute per-row entropy, and return the (weighted) mean.
    """
    if len(behaviors) < 2:
        return 0.0

    n_types = len(_BEHAVIORS)
    trans = np.zeros((n_types, n_types), dtype=float)
    for i in range(len(behaviors) - 1):
        trans[_B_IDX[behaviors[i]], _B_IDX[behaviors[i + 1]]] += 1

    row_entropies, row_weights = [], []
    for row in trans:
        s = row.sum()
        if s == 0:
            continue
        p = row / s
        p = p[p > 0]
        row_entropies.append(-np.sum(p * np.log2(p)))
        row_weights.append(s)

    if not row_entropies:
        return 0.0
    weights = np.array(row_weights)
    return float(np.average(row_entropies, weights=weights))


_STOP_WORDS = frozenset({
    "the", "and", "for", "that", "this", "with", "are", "was",
    "has", "have", "not", "from", "but", "can", "will", "get",
    "let", "now", "its", "our", "also", "into", "than", "then",
    "them", "they", "their", "there", "which", "when", "what",
})


def _count_cycles(
    episodes: list[Episode],
    min_gap: int = 3,
    overlap_threshold: float = 0.30,
) -> int:
    """
    Count semantic revisitations via word-overlap heuristic.

    A cycle is detected when:
      1. Episodes i and j (j ≥ i + min_gap) share Jaccard word-overlap ≥ threshold
      2. A Revise, Verify, or Restart episode occurs between them
         (indicating the model revisited for a reason)

    Each source episode i is counted at most once.
    """
    if len(episodes) < min_gap + 1:
        return 0

    # Pre-compute content word sets
    word_sets = []
    for ep in episodes:
        words = frozenset(
            w for w in re.findall(r'\b\w{3,}\b', ep.text.lower())
            if w not in _STOP_WORDS
        )
        word_sets.append(words)

    _correction_types = {BehaviorType.REVISE, BehaviorType.VERIFY, BehaviorType.RESTART}
    cycle_count = 0

    for i in range(len(episodes)):
        if not word_sets[i]:
            continue
        for j in range(i + min_gap, len(episodes)):
            if not word_sets[j]:
                continue
            inter = len(word_sets[i] & word_sets[j])
            union = len(word_sets[i] | word_sets[j])
            if union == 0:
                continue
            if inter / union >= overlap_threshold:
                between = [ep.behavior for ep in episodes[i + 1:j]]
                if any(b in _correction_types for b in between):
                    cycle_count += 1
                    break  # Count episode i at most once

    return cycle_count


# =============================================================================
# GROUP 3: CONTENT-FREE META
# =============================================================================

# Compiled regex patterns (shared with baseline_b_lexical.py)
_WAIT_WORDS = frozenset({"wait", "hmm", "hmmm", "hmmmm", "uh", "um", "well"})

_RE_MAYBE    = re.compile(r'\b(maybe|perhaps|possibly|might|could be|probably)\b', re.I)
_RE_VERIFY   = re.compile(
    r'\b(verify|verified|verif(?:ying|ication)|check(?:ed|ing)?|'
    r'let me (?:verify|check|confirm|re-?check)|double.?check(?:ed|ing)?|'
    r'confirm(?:ed|ing)?)\b', re.I,
)
_RE_ACTUALLY = re.compile(
    r'\b(actually|but actually|wait actually|in fact|on second thought)\b', re.I,
)
_RE_NEGATION = [
    re.compile(p, re.I) for p in [
        r'\bno\b', r'\bnot\b', r'\bwrong\b', r'\bincorrect\b',
        r'\berror\b', r'\bmistake\b',
        r"\bcan't\b", r"\bcannot\b", r"\bdon't\b", r"\bdoesn't\b",
        r"\bisn't\b", r"\bwon't\b", r"\bwouldn't\b", r"\bshouldn't\b",
    ]
]


def _extract_meta(trace_text: str) -> dict[str, float]:
    """
    7 surface-level text statistics, all normalized per token.

    wait_density          — "wait"/"hmm"/"uh"/"um" family per token
    maybe_density         — hedging words ("maybe", "perhaps", …) per token
    verify_density        — verification words ("verify", "check", …) per token
    actually_density      — revision markers ("actually", "in fact", …) per token
    negation_density      — negation words ("no", "not", "wrong", …) per token
    question_mark_rate    — "?" per token
    repetition_rate_4gram — fraction of 4-grams appearing more than once
    """
    _zero = {k: 0.0 for k in [
        "wait_density", "maybe_density", "verify_density", "actually_density",
        "negation_density", "question_mark_rate", "repetition_rate_4gram",
    ]}
    if not trace_text or not trace_text.strip():
        return _zero

    tokens    = trace_text.lower().split()
    total     = max(len(tokens), 1)

    # wait
    wait_count = sum(1 for t in tokens if t.strip(".,!?;:'\"") in _WAIT_WORDS)

    # regex counts
    maybe_count    = len(_RE_MAYBE.findall(trace_text))
    verify_count   = len(_RE_VERIFY.findall(trace_text))
    actually_count = len(_RE_ACTUALLY.findall(trace_text))
    negation_count = sum(len(p.findall(trace_text)) for p in _RE_NEGATION)
    qmark_count    = trace_text.count("?")

    # 4-gram repetition
    if len(tokens) >= 4:
        four_grams = [tuple(tokens[i:i + 4]) for i in range(len(tokens) - 3)]
        unique     = len(set(four_grams))
        rep_rate   = 1.0 - unique / max(len(four_grams), 1)
    else:
        rep_rate = 0.0

    return {
        "wait_density":          wait_count    / total,
        "maybe_density":         maybe_count   / total,
        "verify_density":        verify_count  / total,
        "actually_density":      actually_count / total,
        "negation_density":      negation_count / total,
        "question_mark_rate":    qmark_count   / total,
        "repetition_rate_4gram": rep_rate,
    }


# =============================================================================
# UNIFIED EXTRACTION API
# =============================================================================

def extract_features(
    trace_text: str,
    episodes: list[Episode],
) -> dict[str, float]:
    """
    Extract all 25 features from a parsed trace.

    Args:
        trace_text: Raw reasoning trace string (contents of <think> block).
        episodes:   List of Episode objects from rule_based_parser.parse_trace().

    Returns:
        Ordered dict with all 25 feature names → float values.
        Key order matches get_feature_names().
    """
    feats: dict[str, float] = {}
    feats.update(_extract_length(trace_text, episodes))
    feats.update(_extract_structural(episodes))
    feats.update(_extract_meta(trace_text))
    # Return in canonical order
    names = get_feature_names()
    return {k: feats[k] for k in names}


def to_array(feat_dict: dict[str, float]) -> np.ndarray:
    """
    Convert a feature dict to a 1-D numpy array in canonical order.

    Shape: (25,)  dtype: float64
    """
    names = get_feature_names()
    return np.array([feat_dict[k] for k in names], dtype=np.float64)


def extract_from_text(trace_text: str) -> dict[str, float]:
    """
    Convenience wrapper: parse trace and extract features in one call.

    Equivalent to:
        episodes = parse_trace(trace_text)
        return extract_features(trace_text, episodes)
    """
    episodes = parse_trace(trace_text)
    return extract_features(trace_text, episodes)


# =============================================================================
# BATCH EXTRACTION — from raw JSONL trace file
# =============================================================================

def extract_from_jsonl(
    traces_path: str,
    output_csv: str,
    trace_field: str = "reasoning_trace",
) -> list[dict]:
    """
    Parse and extract features for every record in a raw JSONL trace file.

    Parses each trace with rule_based_parser.parse_trace(), extracts features,
    and writes a CSV with columns:

        item_id | dataset | is_correct | <25 feature columns>

    Args:
        traces_path: Path to JSONL file produced by generate_traces.py
        output_csv:  Destination CSV path
        trace_field: Field name containing the reasoning trace text

    Returns:
        List of row dicts (same content as the CSV).
    """
    feature_names = get_feature_names()
    meta_fields   = ["item_id", "dataset", "is_correct"]
    all_fields    = meta_fields + feature_names

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    rows = []

    with open(traces_path) as fin:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            trace   = record.get(trace_field, "") or ""
            episodes = parse_trace(trace)
            feats   = extract_features(trace, episodes)

            row = {
                "item_id":    record.get("item_id",   f"item_{line_num}"),
                "dataset":    record.get("dataset",   "unknown"),
                "is_correct": int(bool(record.get("is_correct", False))),
            }
            row.update(feats)
            rows.append(row)

            if line_num % 200 == 0:
                logger.info(f"  Extracted {line_num} traces…")

    with open(output_csv, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        f"Done: {len(rows)} traces → {output_csv}  "
        f"({len(feature_names)} features each)"
    )
    return rows


def extract_from_parsed_jsonl(
    parsed_path: str,
    output_csv: str,
) -> list[dict]:
    """
    Extract features from a pre-parsed JSONL file (output of
    rule_based_parser.parse_trace_file).

    Reconstructs Episode objects from stored dicts — avoids re-parsing.

    Args:
        parsed_path: Path to JSONL file with 'episodes' field already populated.
        output_csv:  Destination CSV path.
    """
    feature_names = get_feature_names()
    meta_fields   = ["item_id", "dataset", "is_correct"]
    all_fields    = meta_fields + feature_names

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    rows = []

    with open(parsed_path) as fin:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            record   = json.loads(line)
            episodes = [Episode.from_dict(ep) for ep in record.get("episodes", [])]
            trace    = record.get("reasoning_trace", "") or ""
            feats    = extract_features(trace, episodes)

            row = {
                "item_id":    record.get("item_id",   f"item_{line_num}"),
                "dataset":    record.get("dataset",   "unknown"),
                "is_correct": int(bool(record.get("is_correct", False))),
            }
            row.update(feats)
            rows.append(row)

            if line_num % 200 == 0:
                logger.info(f"  Extracted {line_num} traces…")

    with open(output_csv, "w", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Done: {len(rows)} traces → {output_csv}")
    return rows


# =============================================================================
# SELF-TESTS
# =============================================================================

def _run_tests() -> bool:
    print("Running feature_extractor self-tests…")

    passed = failed = 0

    def chk(name: str, cond: bool, msg: str = ""):
        nonlocal passed, failed
        if cond:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL [{name}]: {msg}")

    # ------------------------------------------------------------------ helpers
    def make_ep(text: str, btype: BehaviorType, pos: int) -> Episode:
        return Episode(text, btype, pos, len(text.split()), 1.0)

    F, V, X, R, H, C = (
        BehaviorType.FORWARD, BehaviorType.VERIFY, BehaviorType.REVISE,
        BehaviorType.RESTART, BehaviorType.HESITATE, BehaviorType.CONCLUDE,
    )

    # ------------------------------------------------------------------ Test 1: feature count
    names = get_feature_names()
    chk("feature_count", len(names) == 25, f"expected 25, got {len(names)}")

    groups = get_feature_groups()
    all_in_groups = sum(len(v) for v in groups.values())
    chk("group_total", all_in_groups == 25, f"expected 25, got {all_in_groups}")
    chk("groups_cover_all",
        set(names) == set(n for ns in groups.values() for n in ns),
        "group names don't match feature names")

    # ------------------------------------------------------------------ Test 2: empty trace
    feats_empty = extract_features("", [])
    chk("empty_total_tokens",   feats_empty["total_tokens"]   == 0.0)
    chk("empty_total_episodes", feats_empty["total_episodes"] == 0.0)
    chk("empty_revise_count",   feats_empty["revise_count"]   == 0.0)
    chk("empty_first_conclude", feats_empty["first_conclude_pos"] == 1.0,
        f"expected 1.0, got {feats_empty['first_conclude_pos']}")
    chk("empty_all_present",    set(feats_empty.keys()) == set(names),
        "missing keys in empty output")

    # ------------------------------------------------------------------ Test 3: simple forward trace
    eps3 = [make_ep("Compute x squared.", F, 0),
            make_ep("We get 25.",         F, 1),
            make_ep("The answer is 5.",   C, 2)]
    feats3 = extract_features("Compute x squared. We get 25. The answer is 5.", eps3)
    chk("t3_total_episodes",  feats3["total_episodes"]  == 3.0)
    chk("t3_prop_forward",    abs(feats3["prop_forward"] - 2/3) < 0.01,
        f"got {feats3['prop_forward']:.3f}")
    chk("t3_prop_conclude",   abs(feats3["prop_conclude"] - 1/3) < 0.01,
        f"got {feats3['prop_conclude']:.3f}")
    chk("t3_revise_count",    feats3["revise_count"]    == 0.0)
    chk("t3_max_forward_run", feats3["max_forward_run"] == 2.0,
        f"got {feats3['max_forward_run']}")
    chk("t3_first_conclude",  abs(feats3["first_conclude_pos"] - 2/2) < 0.01,
        f"got {feats3['first_conclude_pos']}")

    # ------------------------------------------------------------------ Test 4: revise + verify
    eps4 = [
        make_ep("Set up equation.",    F, 0),
        make_ep("x = 3.",              F, 1),
        make_ep("Wait, that's wrong.", X, 2),
        make_ep("x = 5.",              F, 3),
        make_ep("Let me verify.",      V, 4),
        make_ep("Answer is 5.",        C, 5),
    ]
    trace4 = "Set up equation. x = 3. Wait, that's wrong. x = 5. Let me verify. Answer is 5."
    feats4 = extract_features(trace4, eps4)
    chk("t4_revise_count",         feats4["revise_count"]  == 1.0)
    chk("t4_verify_count",         feats4["verify_count"]  == 1.0)
    chk("t4_revise_pos",           abs(feats4["revise_position_mean"] - 2/5) < 0.01,
        f"got {feats4['revise_position_mean']:.3f}")
    chk("t4_first_conclude_pos",   feats4["first_conclude_pos"] == 1.0,
        f"got {feats4['first_conclude_pos']}")
    chk("t4_vf_ratio",             feats4["vf_ratio"] == 1/3,
        f"got {feats4['vf_ratio']:.3f}")

    # ------------------------------------------------------------------ Test 5: all behavior types
    eps5 = [make_ep("Forward step.", F, 0),
            make_ep("Hmm.",          H, 1),
            make_ep("Try again.",    R, 2),
            make_ep("Check.",        V, 3),
            make_ep("Oh wait wrong.",X, 4),
            make_ep("Done.",         C, 5)]
    feats5 = extract_features("Forward step. Hmm. Try again. Check. Oh wait wrong. Done.", eps5)
    for behavior, key in [(F,"prop_forward"),(H,"prop_hesitate"),(R,"prop_restart"),
                           (V,"prop_verify"),(X,"prop_revise"),(C,"prop_conclude")]:
        chk(f"t5_{key}", abs(feats5[key] - 1/6) < 0.02, f"got {feats5[key]:.3f}")

    # ------------------------------------------------------------------ Test 6: transition entropy
    # Uniform transitions (all-Forward) → entropy = 0
    eps_all_f = [make_ep("step.", F, i) for i in range(5)]
    feats_f = extract_features("step " * 5, eps_all_f)
    chk("t6_entropy_uniform", feats_f["transition_entropy"] == 0.0,
        f"got {feats_f['transition_entropy']}")

    # Mixed successors: F→V, F→X, F→F → Forward has 3 different successors → entropy > 0
    eps_mixed = [make_ep("a.", F, 0), make_ep("b.", V, 1),
                 make_ep("c.", F, 2), make_ep("d.", X, 3),
                 make_ep("e.", F, 4), make_ep("f.", F, 5)]
    feats_mixed = extract_features("a b c d e f", eps_mixed)
    chk("t6_entropy_mixed", feats_mixed["transition_entropy"] > 0,
        f"got {feats_mixed['transition_entropy']}")

    # ------------------------------------------------------------------ Test 7: meta features
    trace_meta = "Wait, hmm. Is this right? No, that's wrong. Actually, maybe not."
    eps_meta   = [make_ep(trace_meta, F, 0)]
    feats_meta = extract_features(trace_meta, eps_meta)
    chk("t7_wait_density",      feats_meta["wait_density"]     > 0)
    chk("t7_negation_density",  feats_meta["negation_density"] > 0)
    chk("t7_qmark_rate",        feats_meta["question_mark_rate"] > 0)
    chk("t7_maybe_density",     feats_meta["maybe_density"]    > 0)
    chk("t7_actually_density",  feats_meta["actually_density"] > 0)

    # ------------------------------------------------------------------ Test 8: to_array shape
    arr = to_array(feats_meta)
    chk("t8_array_shape",  arr.shape == (25,), f"got {arr.shape}")
    chk("t8_array_dtype",  arr.dtype == np.float64)
    chk("t8_no_nan",       not np.any(np.isnan(arr)), "contains NaN")

    # ------------------------------------------------------------------ Test 9: extract_from_text
    trace9 = "I need to solve x^2 = 4. So x = ±2. Let me verify: 2^2 = 4. Therefore the answer is \\boxed{2}."
    feats9 = extract_from_text(trace9)
    chk("t9_returns_dict",     isinstance(feats9, dict))
    chk("t9_all_features",     set(feats9.keys()) == set(names))
    chk("t9_nonzero_tokens",   feats9["total_tokens"] > 0)

    # ------------------------------------------------------------------ Test 10: v_clustering
    # Single verify → clustering = 0
    eps_1v = [make_ep("x.", F, 0), make_ep("check.", V, 1), make_ep("done.", C, 2)]
    f_1v   = extract_features("x check done", eps_1v)
    chk("t10_clustering_single_v", f_1v["v_clustering"] == 0.0)

    # Two verifies with different gaps → clustering > 0 only if gaps differ
    eps_2v_equal = [make_ep("a.", F, 0), make_ep("b.", V, 1),
                    make_ep("c.", F, 2), make_ep("d.", V, 3)]
    f_2v = extract_features("a b c d", eps_2v_equal)
    chk("t10_clustering_equal_gap", f_2v["v_clustering"] == 0.0,
        f"got {f_2v['v_clustering']}")

    # ------------------------------------------------------------------ summary
    print(f"\n  {passed} passed, {failed} failed out of {passed + failed} tests")
    if failed == 0:
        print("  All tests passed.")
    else:
        print("  WARNING: some tests failed.")
    return failed == 0


# =============================================================================
# CLI / DEMO
# =============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ok = _run_tests()

    if ok:
        print("\n" + "=" * 65)
        print("Demo: feature vector for a sample trace")
        print("=" * 65)
        sample = (
            "First, I need to find the roots of x^2 - 5x + 6 = 0. "
            "Using the quadratic formula: x = (5 ± √(25-24)) / 2 = (5 ± 1) / 2. "
            "Wait, that's wrong. I made an arithmetic error. "
            "25 - 4·6 = 25 - 24 = 1, so √1 = 1. "
            "x = (5 + 1)/2 = 3 or x = (5 - 1)/2 = 2. "
            "Let me verify: (x-3)(x-2) = x^2 - 5x + 6. Correct. "
            "Therefore, the answer is \\boxed{x = 2 \\text{ or } x = 3}."
        )
        print(f"\nTrace:\n  {sample}\n")
        feats = extract_from_text(sample)
        arr   = to_array(feats)

        groups = get_feature_groups()
        for gname, gfeats in groups.items():
            print(f"  {gname}:")
            for fname in gfeats:
                print(f"    {fname:28s} = {feats[fname]:.4f}")
        print(f"\n  Feature vector shape: {arr.shape}")
        print(f"  Vector: {np.round(arr, 4)}")

    sys.exit(0 if ok else 1)
