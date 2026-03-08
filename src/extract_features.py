"""
extract_features.py — Extract 20 structural features from parsed reasoning traces.

Feature groups (from proposal Section 3.5):
  Group 1: Length and proportion features (6)
  Group 2: Structural / topological features (10)
  Group 3: Content-free meta features (4)

Usage:
  python src/extract_features.py --input data/traces/math500.jsonl \
                                  --output data/features/math500.csv
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from parse_trace import parse_trace, ParsedTrace

ROOT = Path(__file__).parent.parent


# ─── Group 1: Length and proportion features ──────────────────────────────────

def feat_length_proportion(parsed: ParsedTrace, trace_text: str) -> dict:
    behaviors = parsed.behaviors
    n = max(len(behaviors), 1)
    tokens = len(trace_text.split())  # word-level proxy

    counts = Counter(behaviors)
    return {
        "g1_token_count": tokens,
        "g1_episode_count": n,
        "g1_prop_F": counts.get("F", 0) / n,
        "g1_prop_V": counts.get("V", 0) / n,
        "g1_prop_B": counts.get("B", 0) / n,
        "g1_prop_R": counts.get("R", 0) / n,
        "g1_prop_S": counts.get("S", 0) / n,
        "g1_prop_H": counts.get("H", 0) / n,
    }


# ─── Group 2: Structural / topological features ───────────────────────────────

def feat_structural(parsed: ParsedTrace) -> dict:
    behaviors = parsed.behaviors
    n = max(len(behaviors), 1)
    counts = Counter(behaviors)

    # Basic counts
    b_count = counts.get("B", 0)
    v_count = counts.get("V", 0)
    r_count = counts.get("R", 0)
    f_count = max(counts.get("F", 0), 1)

    # Verification-to-Forward ratio
    v_f_ratio = v_count / f_count

    # Backtrack position (normalized mean position of B episodes)
    b_positions = [i / n for i, b in enumerate(behaviors) if b == "B"]
    backtrack_pos_mean = float(np.mean(b_positions)) if b_positions else 0.0

    # First conclusion position (normalized)
    c_positions = [i for i, b in enumerate(behaviors) if b == "C"]
    first_c_pos = (c_positions[0] / n) if c_positions else 1.0

    # Verification clustering coefficient: std of gaps between consecutive V episodes
    v_positions = [i for i, b in enumerate(behaviors) if b == "V"]
    if len(v_positions) >= 2:
        gaps = [v_positions[k + 1] - v_positions[k] for k in range(len(v_positions) - 1)]
        v_cluster_coeff = float(np.std(gaps)) / n
    else:
        v_cluster_coeff = 0.0

    # Longest forward run (max consecutive F episodes)
    longest_f_run = _longest_run(behaviors, "F")

    # Behavior transition entropy (Shannon entropy of bigram transitions)
    trans_entropy = _transition_entropy(behaviors)

    # Cycle count (text-level): how many times does a behavior sequence re-enter V or B
    # after a forward stretch? Count (non-V/B) -> (V or B) transitions
    cycle_count = sum(
        1 for i in range(1, len(behaviors))
        if behaviors[i] in ("V", "B") and behaviors[i - 1] == "F"
    )

    return {
        "g2_backtrack_count": b_count,
        "g2_verification_count": v_count,
        "g2_restart_count": r_count,
        "g2_vf_ratio": v_f_ratio,
        "g2_backtrack_pos_mean": backtrack_pos_mean,
        "g2_first_conclusion_pos": first_c_pos,
        "g2_verification_cluster_coeff": v_cluster_coeff,
        "g2_longest_forward_run": longest_f_run,
        "g2_transition_entropy": trans_entropy,
        "g2_cycle_count": cycle_count,
    }


def _longest_run(seq: list[str], symbol: str) -> int:
    max_run = 0
    cur = 0
    for s in seq:
        if s == symbol:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


def _transition_entropy(seq: list[str]) -> float:
    if len(seq) < 2:
        return 0.0
    bigrams = [(seq[i], seq[i + 1]) for i in range(len(seq) - 1)]
    counts = Counter(bigrams)
    total = sum(counts.values())
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return entropy


# ─── Group 3: Content-free meta features ─────────────────────────────────────

def feat_meta(trace_text: str) -> dict:
    tokens = trace_text.lower().split()
    n_tokens = max(len(tokens), 1)

    # "Wait"-family token ratio
    wait_tokens = sum(1 for t in tokens if re.match(r"wait[,.]?", t))
    wait_ratio = wait_tokens / n_tokens

    # Number of question marks (self-questioning)
    question_marks = trace_text.count("?")

    # Explicit negations
    negation_count = len(re.findall(
        r"\b(no[,.]|wrong|incorrect|that'?s not right|mistaken|error)\b",
        trace_text, re.IGNORECASE
    ))

    # 4-gram repetition rate (fraction of 4-grams appearing more than once)
    fourgrams = [" ".join(tokens[i:i + 4]) for i in range(len(tokens) - 3)]
    if fourgrams:
        fg_counts = Counter(fourgrams)
        repeated = sum(1 for c in fg_counts.values() if c > 1)
        repetition_rate = repeated / len(fg_counts)
    else:
        repetition_rate = 0.0

    return {
        "g3_wait_ratio": wait_ratio,
        "g3_question_mark_count": question_marks,
        "g3_negation_count": negation_count,
        "g3_fourgram_repetition_rate": repetition_rate,
    }


# ─── Master extractor ─────────────────────────────────────────────────────────

def extract_features(record: dict) -> dict:
    """
    Extract all 20 features from a single trace record.

    Args:
        record: dict with at least {"id", "trace", "correct", "dataset"}

    Returns:
        Feature dict with all ~20 features + metadata.
    """
    trace_text = record.get("trace", "")
    parsed = parse_trace(trace_text)

    feats = {}
    feats["id"] = record["id"]
    feats["dataset"] = record.get("dataset", "")
    feats["correct"] = int(record.get("correct", False))
    feats["token_count_raw"] = record.get("tokens", len(trace_text.split()))

    feats.update(feat_length_proportion(parsed, trace_text))
    feats.update(feat_structural(parsed))
    feats.update(feat_meta(trace_text))

    return feats


FEATURE_COLS = [
    # Group 1
    "g1_token_count", "g1_episode_count",
    "g1_prop_F", "g1_prop_V", "g1_prop_B", "g1_prop_R", "g1_prop_S", "g1_prop_H",
    # Group 2
    "g2_backtrack_count", "g2_verification_count", "g2_restart_count",
    "g2_vf_ratio", "g2_backtrack_pos_mean", "g2_first_conclusion_pos",
    "g2_verification_cluster_coeff", "g2_longest_forward_run",
    "g2_transition_entropy", "g2_cycle_count",
    # Group 3
    "g3_wait_ratio", "g3_question_mark_count", "g3_negation_count",
    "g3_fourgram_repetition_rate",
]


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to .jsonl trace file")
    parser.add_argument("--output", default=None, help="Output CSV path (default: data/features/<name>.csv)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        feat_dir = ROOT / "data" / "features"
        feat_dir.mkdir(parents=True, exist_ok=True)
        output_path = feat_dir / (input_path.stem + ".csv")

    records = []
    with open(input_path) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Extracting features from {len(records)} traces...")
    rows = [extract_features(r) for r in tqdm(records)]
    df = pd.DataFrame(rows)

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows × {len(df.columns)} columns to {output_path}")

    # Quick summary
    print(f"\nClass balance: {df['correct'].mean():.1%} correct")
    print("\nFeature means by label:")
    print(df.groupby("correct")[FEATURE_COLS[:6]].mean().to_string())


if __name__ == "__main__":
    main()
