"""
feature_pipeline.py - Unified feature extraction from parsed reasoning traces.

Extracts three groups of features from cognitive episode sequences:
  Group 1 (Length & Proportion):  9 features  - How long and what mix of behaviors
  Group 2 (Structural):         10 features  - Topology of the reasoning process
  Group 3 (Content-Free Meta):   4 features  - Surface-level text statistics

Total: 23 features per trace.

Usage:
    from src.features.feature_pipeline import extract_all_features

    episodes = parse_trace(trace_text)
    features = extract_all_features(trace_text, episodes)
"""

import re
import math
import logging
from collections import Counter
from typing import Optional

import numpy as np

from src.parsing.taxonomy import BehaviorType, CognitiveEpisode

logger = logging.getLogger(__name__)

# All behavior types in canonical order
ALL_BEHAVIORS = list(BehaviorType)
BEHAVIOR_VALUES = [b.value for b in ALL_BEHAVIORS]


# =============================================================================
# GROUP 1: LENGTH & PROPORTION FEATURES
# =============================================================================

def extract_length_features(
    trace_text: str,
    episodes: list[CognitiveEpisode],
) -> dict[str, float]:
    """
    Extract length and proportion features.

    Features:
      - total_tokens:      Approximate token count (whitespace split)
      - total_episodes:    Number of cognitive episodes
      - prop_forward:      Proportion of FORWARD episodes
      - prop_verification: Proportion of VERIFICATION episodes
      - prop_backtrack:    Proportion of BACKTRACK episodes
      - prop_restart:      Proportion of RESTART episodes
      - prop_hesitation:   Proportion of HESITATION episodes
      - prop_subgoal:      Proportion of SUBGOAL episodes
      - prop_conclusion:   Proportion of CONCLUSION episodes
    """
    total_tokens = len(trace_text.split()) if trace_text else 0
    total_episodes = len(episodes)

    # Behavior counts
    counts = Counter(ep.behavior for ep in episodes)
    denom = max(total_episodes, 1)

    features = {
        "total_tokens": float(total_tokens),
        "total_episodes": float(total_episodes),
        "prop_forward": counts.get(BehaviorType.FORWARD, 0) / denom,
        "prop_verification": counts.get(BehaviorType.VERIFICATION, 0) / denom,
        "prop_backtrack": counts.get(BehaviorType.BACKTRACK, 0) / denom,
        "prop_restart": counts.get(BehaviorType.RESTART, 0) / denom,
        "prop_hesitation": counts.get(BehaviorType.HESITATION, 0) / denom,
        "prop_subgoal": counts.get(BehaviorType.SUBGOAL, 0) / denom,
        "prop_conclusion": counts.get(BehaviorType.CONCLUSION, 0) / denom,
    }

    return features


# =============================================================================
# GROUP 2: STRUCTURAL / TOPOLOGICAL FEATURES
# =============================================================================

def extract_structural_features(
    episodes: list[CognitiveEpisode],
    use_embeddings: bool = False,
) -> dict[str, float]:
    """
    Extract structural and topological features.

    Features:
      - backtrack_count:      Number of BACKTRACK episodes
      - verification_count:   Number of VERIFICATION episodes
      - restart_count:        Number of RESTART episodes
      - vf_ratio:             Verification-to-Forward ratio
      - bt_position_mean:     Mean normalized position of backtracks (0=start, 1=end)
      - first_conclusion_pos: Normalized position of first CONCLUSION episode
      - v_clustering:         Verification clustering coefficient
      - max_forward_run:      Longest consecutive FORWARD streak
      - transition_entropy:   Shannon entropy of behavior transition bigrams
      - cycle_count:          Number of semantic-level revisitations

    Args:
        episodes: Parsed cognitive episodes.
        use_embeddings: If True, use sentence embeddings for cycle detection.
                       If False, use text overlap heuristic (faster).
    """
    if not episodes:
        return {k: 0.0 for k in [
            "backtrack_count", "verification_count", "restart_count",
            "vf_ratio", "bt_position_mean", "first_conclusion_pos",
            "v_clustering", "max_forward_run", "transition_entropy",
            "cycle_count",
        ]}

    behaviors = [ep.behavior for ep in episodes]
    n = len(behaviors)

    # --- Counts ---
    backtrack_count = sum(1 for b in behaviors if b == BehaviorType.BACKTRACK)
    verification_count = sum(1 for b in behaviors if b == BehaviorType.VERIFICATION)
    restart_count = sum(1 for b in behaviors if b == BehaviorType.RESTART)

    # --- Verification-to-Forward ratio ---
    forward_count = sum(1 for b in behaviors if b == BehaviorType.FORWARD)
    vf_ratio = verification_count / max(forward_count, 1)

    # --- Backtrack position (mean normalized) ---
    # Early backtracks (near 0) = exploratory; late backtracks (near 1) = corrective
    bt_positions = [
        i / max(n - 1, 1)
        for i, b in enumerate(behaviors)
        if b == BehaviorType.BACKTRACK
    ]
    bt_position_mean = float(np.mean(bt_positions)) if bt_positions else 0.0

    # --- First conclusion position ---
    # How early does the model first attempt an answer?
    conclusion_indices = [
        i for i, b in enumerate(behaviors) if b == BehaviorType.CONCLUSION
    ]
    if conclusion_indices:
        first_conclusion_pos = conclusion_indices[0] / max(n - 1, 1)
    else:
        first_conclusion_pos = 1.0  # No conclusion → treat as "very late"

    # --- Verification clustering coefficient ---
    # High clustering = verifications are bunched together
    # Low clustering = verifications are evenly distributed
    v_positions = [i for i, b in enumerate(behaviors) if b == BehaviorType.VERIFICATION]
    if len(v_positions) >= 2:
        gaps = np.diff(v_positions).astype(float)
        mean_gap = np.mean(gaps)
        if mean_gap > 0:
            v_clustering = float(np.std(gaps) / mean_gap)
        else:
            v_clustering = 0.0
    else:
        v_clustering = 0.0

    # --- Longest forward run ---
    max_forward_run = 0
    current_run = 0
    for b in behaviors:
        if b == BehaviorType.FORWARD:
            current_run += 1
            max_forward_run = max(max_forward_run, current_run)
        else:
            current_run = 0

    # --- Behavior transition entropy ---
    # Captures how chaotic vs. structured the reasoning is.
    # High entropy = unpredictable transitions (F→B→V→H→F...)
    # Low entropy = structured patterns (F→F→F→V→F→F→C)
    transition_entropy = _compute_transition_entropy(behaviors)

    # --- Cycle count (semantic-level revisitation) ---
    if use_embeddings:
        cycle_count = _count_cycles_embedding(episodes)
    else:
        cycle_count = _count_cycles_heuristic(episodes)

    return {
        "backtrack_count": float(backtrack_count),
        "verification_count": float(verification_count),
        "restart_count": float(restart_count),
        "vf_ratio": vf_ratio,
        "bt_position_mean": bt_position_mean,
        "first_conclusion_pos": first_conclusion_pos,
        "v_clustering": v_clustering,
        "max_forward_run": float(max_forward_run),
        "transition_entropy": transition_entropy,
        "cycle_count": float(cycle_count),
    }


def _compute_transition_entropy(behaviors: list[BehaviorType]) -> float:
    """
    Compute Shannon entropy of the behavior bigram transition matrix.

    For each behavior type that appears, we compute the entropy of its
    outgoing transition distribution, then average across behavior types.
    """
    if len(behaviors) < 2:
        return 0.0

    n_types = len(ALL_BEHAVIORS)
    type_to_idx = {b: i for i, b in enumerate(ALL_BEHAVIORS)}

    # Build transition count matrix
    trans_matrix = np.zeros((n_types, n_types))
    for i in range(len(behaviors) - 1):
        from_idx = type_to_idx[behaviors[i]]
        to_idx = type_to_idx[behaviors[i + 1]]
        trans_matrix[from_idx, to_idx] += 1

    # Compute entropy for each row (each "from" behavior)
    row_entropies = []
    for row in trans_matrix:
        row_sum = row.sum()
        if row_sum > 0:
            probs = row / row_sum
            # Filter out zero entries to avoid log(0)
            probs = probs[probs > 0]
            row_entropy = -np.sum(probs * np.log2(probs))
            row_entropies.append(row_entropy)

    return float(np.mean(row_entropies)) if row_entropies else 0.0


def _count_cycles_heuristic(
    episodes: list[CognitiveEpisode],
    min_gap: int = 3,
    overlap_threshold: float = 0.3,
) -> int:
    """
    Count semantic-level cycles using word overlap heuristic.

    A "cycle" is detected when:
      1. Two episodes ≥ min_gap apart share significant word overlap
      2. There's a non-forward behavior between them (suggesting the
         model went back to reconsider something)

    This is a fast alternative to embedding-based cycle detection.
    """
    if len(episodes) < min_gap + 1:
        return 0

    cycle_count = 0

    # Extract word sets for each episode (normalized)
    word_sets = []
    for ep in episodes:
        words = set(re.findall(r'\b\w{3,}\b', ep.text.lower()))
        # Remove very common words
        stop_words = {
            'the', 'and', 'for', 'that', 'this', 'with', 'are', 'was',
            'has', 'have', 'not', 'from', 'but', 'can', 'will', 'get',
            'let', 'now', 'its', 'our', 'also',
        }
        words -= stop_words
        word_sets.append(words)

    # Check for revisitations
    for i in range(len(episodes)):
        if not word_sets[i]:
            continue
        for j in range(i + min_gap, len(episodes)):
            if not word_sets[j]:
                continue

            # Compute Jaccard similarity
            intersection = len(word_sets[i] & word_sets[j])
            union = len(word_sets[i] | word_sets[j])
            if union == 0:
                continue
            similarity = intersection / union

            if similarity >= overlap_threshold:
                # Check for non-forward behavior between i and j
                between = [ep.behavior for ep in episodes[i+1:j]]
                has_correction = any(
                    b in (BehaviorType.BACKTRACK, BehaviorType.RESTART,
                          BehaviorType.VERIFICATION)
                    for b in between
                )
                if has_correction:
                    cycle_count += 1
                    break  # Count each source episode only once

    return cycle_count


def _count_cycles_embedding(
    episodes: list[CognitiveEpisode],
    min_gap: int = 3,
    similarity_threshold: float = 0.85,
) -> int:
    """
    Count semantic-level cycles using sentence embeddings.

    More accurate than the heuristic method but requires the
    sentence-transformers library and is slower.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers not installed, falling back to heuristic")
        return _count_cycles_heuristic(episodes, min_gap)

    if len(episodes) < min_gap + 1:
        return 0

    # Encode all episode texts
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [ep.text for ep in episodes]
    embeddings = model.encode(texts, show_progress_bar=False)

    cycle_count = 0
    for i in range(len(episodes)):
        for j in range(i + min_gap, len(episodes)):
            # Cosine similarity
            sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-8
            )

            if sim >= similarity_threshold:
                between = [ep.behavior for ep in episodes[i+1:j]]
                has_correction = any(
                    b in (BehaviorType.BACKTRACK, BehaviorType.RESTART,
                          BehaviorType.VERIFICATION)
                    for b in between
                )
                if has_correction:
                    cycle_count += 1
                    break

    return cycle_count


# =============================================================================
# GROUP 3: CONTENT-FREE META FEATURES
# =============================================================================

def extract_meta_features(trace_text: str) -> dict[str, float]:
    """
    Extract content-free meta features from the raw trace text.

    These features don't require parsing — they operate on raw text only.
    This makes them very cheap to compute and parser-independent.

    Features:
      - wait_ratio:         Fraction of tokens that are "wait" family words
      - question_mark_count: Number of question marks (self-questioning)
      - negation_count:     Number of explicit negations
      - repetition_rate_4gram: Fraction of 4-grams that appear more than once
    """
    if not trace_text or not trace_text.strip():
        return {
            "wait_ratio": 0.0,
            "question_mark_count": 0.0,
            "negation_count": 0.0,
            "repetition_rate_4gram": 0.0,
        }

    text_lower = trace_text.lower()
    tokens = text_lower.split()
    total_tokens = max(len(tokens), 1)

    # --- Wait-family token ratio ---
    # These metacognitive markers signal uncertainty/hesitation
    wait_words = {
        "wait", "wait,", "hmm", "hmm,", "hmmm", "hmmm,",
        "uh", "uh,", "um", "um,", "well,",
    }
    wait_count = sum(1 for t in tokens if t.strip(".,!?;:") in wait_words or t in wait_words)
    wait_ratio = wait_count / total_tokens

    # --- Question mark count ---
    # Self-questioning frequency correlates with uncertainty
    question_mark_count = trace_text.count("?")

    # --- Explicit negation count ---
    # Negation patterns indicating error acknowledgment
    negation_patterns = [
        r'\bno\b',
        r'\bnot\b',
        r'\bwrong\b',
        r'\bincorrect\b',
        r'\berror\b',
        r'\bmistake\b',
        r"\bcan't\b",
        r"\bdon't\b",
        r"\bdoesn't\b",
        r"\bisn't\b",
        r"\bwon't\b",
    ]
    negation_count = sum(
        len(re.findall(p, text_lower))
        for p in negation_patterns
    )

    # --- Token-level repetition rate (4-gram) ---
    # High repetition suggests rumination / circular reasoning
    if len(tokens) >= 4:
        four_grams = [tuple(tokens[i:i+4]) for i in range(len(tokens) - 3)]
        total_4grams = len(four_grams)
        unique_4grams = len(set(four_grams))
        repetition_rate = 1.0 - (unique_4grams / max(total_4grams, 1))
    else:
        repetition_rate = 0.0

    return {
        "wait_ratio": wait_ratio,
        "question_mark_count": float(question_mark_count),
        "negation_count": float(negation_count),
        "repetition_rate_4gram": repetition_rate,
    }


# =============================================================================
# UNIFIED FEATURE EXTRACTION
# =============================================================================

def extract_all_features(
    trace_text: str,
    episodes: list[CognitiveEpisode],
    use_embeddings: bool = False,
) -> dict[str, float]:
    """
    Extract all features from a parsed reasoning trace.

    This is the main entry point for feature extraction.

    Args:
        trace_text: Raw reasoning trace text.
        episodes: Parsed cognitive episodes.
        use_embeddings: Whether to use sentence embeddings for cycle detection.

    Returns:
        Dictionary with all 23 features.
    """
    features = {}
    features.update(extract_length_features(trace_text, episodes))
    features.update(extract_structural_features(episodes, use_embeddings=use_embeddings))
    features.update(extract_meta_features(trace_text))
    return features


def get_feature_names() -> list[str]:
    """Return the ordered list of all feature names."""
    return [
        # Group 1: Length & Proportion
        "total_tokens", "total_episodes",
        "prop_forward", "prop_verification", "prop_backtrack",
        "prop_restart", "prop_hesitation", "prop_subgoal", "prop_conclusion",
        # Group 2: Structural
        "backtrack_count", "verification_count", "restart_count",
        "vf_ratio", "bt_position_mean", "first_conclusion_pos",
        "v_clustering", "max_forward_run", "transition_entropy", "cycle_count",
        # Group 3: Meta
        "wait_ratio", "question_mark_count", "negation_count",
        "repetition_rate_4gram",
    ]


def get_feature_groups() -> dict[str, list[str]]:
    """Return feature names organized by group."""
    return {
        "group1_length": [
            "total_tokens", "total_episodes",
            "prop_forward", "prop_verification", "prop_backtrack",
            "prop_restart", "prop_hesitation", "prop_subgoal", "prop_conclusion",
        ],
        "group2_structural": [
            "backtrack_count", "verification_count", "restart_count",
            "vf_ratio", "bt_position_mean", "first_conclusion_pos",
            "v_clustering", "max_forward_run", "transition_entropy", "cycle_count",
        ],
        "group3_meta": [
            "wait_ratio", "question_mark_count", "negation_count",
            "repetition_rate_4gram",
        ],
    }


# =============================================================================
# BATCH FEATURE EXTRACTION
# =============================================================================

def extract_features_from_file(
    parsed_traces_path: str,
    output_path: str,
):
    """
    Extract features from a parsed traces JSONL file and save as CSV.

    Reads the parsed traces (output of behavior_classifier.parse_trace_file),
    extracts all features, and writes a CSV with one row per trace.

    Args:
        parsed_traces_path: Path to parsed JSONL file.
        output_path: Path to write the feature CSV.
    """
    import json
    import csv
    import os

    feature_names = get_feature_names()
    meta_fields = ["item_id", "dataset", "is_correct"]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    rows = []
    with open(parsed_traces_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            # Reconstruct episodes from stored data
            episodes_data = record.get("episodes", [])
            episodes = [CognitiveEpisode.from_dict(ep) for ep in episodes_data]

            trace_text = record.get("reasoning_trace", "")

            # Extract features
            features = extract_all_features(trace_text, episodes)

            # Build row
            row = {
                "item_id": record.get("item_id", f"item_{line_num}"),
                "dataset": record.get("dataset", "unknown"),
                "is_correct": int(record.get("is_correct", False)),
            }
            row.update(features)
            rows.append(row)

            if line_num % 200 == 0:
                logger.info(f"Extracted features for {line_num} traces...")

    # Write CSV
    all_fields = meta_fields + feature_names

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Feature extraction complete: {len(rows)} traces → {output_path}")
    logger.info(f"  Features per trace: {len(feature_names)}")

    return rows


# =============================================================================
# SELF-TEST
# =============================================================================

def run_feature_tests():
    """Test feature extraction on synthetic examples."""
    print("Running feature extraction self-tests...")

    tests_passed = 0
    tests_failed = 0

    def check(name, condition, msg=""):
        nonlocal tests_passed, tests_failed
        if condition:
            tests_passed += 1
        else:
            tests_failed += 1
            print(f"  FAIL: {name}: {msg}")

    # --- Test 1: Simple forward trace ---
    episodes = [
        CognitiveEpisode("Compute x.", BehaviorType.FORWARD, 0, 2),
        CognitiveEpisode("We get x=5.", BehaviorType.FORWARD, 1, 4),
        CognitiveEpisode("The answer is 5.", BehaviorType.CONCLUSION, 2, 4),
    ]
    trace = "Compute x. We get x=5. The answer is 5."
    features = extract_all_features(trace, episodes)

    check("f1_total_episodes", features["total_episodes"] == 3,
          f"expected 3, got {features['total_episodes']}")
    check("f1_prop_forward", abs(features["prop_forward"] - 2/3) < 0.01,
          f"expected ~0.667, got {features['prop_forward']}")
    check("f1_prop_conclusion", abs(features["prop_conclusion"] - 1/3) < 0.01,
          f"expected ~0.333, got {features['prop_conclusion']}")
    check("f1_backtrack_count", features["backtrack_count"] == 0,
          f"expected 0, got {features['backtrack_count']}")
    check("f1_max_forward_run", features["max_forward_run"] == 2,
          f"expected 2, got {features['max_forward_run']}")

    # --- Test 2: Trace with backtracking ---
    episodes2 = [
        CognitiveEpisode("Step 1.", BehaviorType.SUBGOAL, 0, 2),
        CognitiveEpisode("x = 3.", BehaviorType.FORWARD, 1, 3),
        CognitiveEpisode("Wait, wrong.", BehaviorType.BACKTRACK, 2, 2),
        CognitiveEpisode("x = 5.", BehaviorType.FORWARD, 3, 3),
        CognitiveEpisode("Checking.", BehaviorType.VERIFICATION, 4, 1),
        CognitiveEpisode("Answer is 5.", BehaviorType.CONCLUSION, 5, 3),
    ]
    trace2 = "Step 1. x = 3. Wait, wrong. x = 5. Checking. Answer is 5."
    features2 = extract_all_features(trace2, episodes2)

    check("f2_backtrack_count", features2["backtrack_count"] == 1)
    check("f2_verification_count", features2["verification_count"] == 1)
    check("f2_bt_position",
          0.3 < features2["bt_position_mean"] < 0.5,
          f"expected ~0.4, got {features2['bt_position_mean']}")
    check("f2_first_conclusion",
          features2["first_conclusion_pos"] == 1.0,
          f"expected 1.0, got {features2['first_conclusion_pos']}")

    # --- Test 3: Meta features ---
    trace3 = "Wait, hmm. Is this right? No, that's wrong. No, I mean wait."
    meta = extract_meta_features(trace3)
    check("f3_wait_ratio", meta["wait_ratio"] > 0,
          f"expected > 0, got {meta['wait_ratio']}")
    check("f3_question_marks", meta["question_mark_count"] >= 1,
          f"expected >= 1, got {meta['question_mark_count']}")
    check("f3_negations", meta["negation_count"] >= 2,
          f"expected >= 2, got {meta['negation_count']}")

    # --- Test 4: Empty trace ---
    empty_features = extract_all_features("", [])
    check("f4_empty_total_tokens", empty_features["total_tokens"] == 0)
    check("f4_empty_total_episodes", empty_features["total_episodes"] == 0)

    # --- Test 5: Feature count ---
    check("f5_feature_count",
          len(get_feature_names()) == 23,
          f"expected 23 features, got {len(get_feature_names())}")

    # --- Test 6: All features present ---
    for fname in get_feature_names():
        check(f"f6_has_{fname}", fname in features,
              f"missing feature: {fname}")

    print(f"\nResults: {tests_passed} passed, {tests_failed} failed "
          f"out of {tests_passed + tests_failed} tests")
    if tests_failed == 0:
        print("All feature extraction tests passed.")
    return tests_failed == 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_feature_tests()
