"""
behavior_classifier.py - Full trace parsing pipeline.

This is the central module that converts raw reasoning trace text into
structured cognitive episode sequences. It orchestrates:

  1. Sentence segmentation (sentence_segmenter.py)
  2. Behavior classification (taxonomy.py)
  3. Post-processing and quality checks

The output is a list of CognitiveEpisode objects, each tagged with a
behavior type, that can then be fed to the feature extraction pipeline.

Usage:
    from src.parsing.behavior_classifier import parse_trace

    episodes = parse_trace("Let me think about this... Wait, actually...")
    print(sequence_to_string(episodes))  # "FHBF..."
"""

import json
import os
import re
import logging
from collections import Counter
from typing import Optional

from src.parsing.taxonomy import (
    BehaviorType, CognitiveEpisode, classify_sentence,
    sequence_to_string
)
from src.parsing.sentence_segmenter import segment_trace

logger = logging.getLogger(__name__)


# =============================================================================
# MAIN PARSING PIPELINE
# =============================================================================

def parse_trace(
    trace: str,
    min_sentence_words: int = 4,
    merge_short: bool = True,
) -> list[CognitiveEpisode]:
    """
    Parse a reasoning trace into a sequence of cognitive episodes.

    This is the main entry point for trace parsing. It:
      1. Segments the trace into sentences
      2. Classifies each sentence into a behavior type
      3. Creates CognitiveEpisode objects with position metadata
      4. Applies post-processing rules

    Args:
        trace: Raw reasoning trace text (contents of <think> block).
        min_sentence_words: Minimum word count for standalone sentences.
        merge_short: Whether to merge short fragments.

    Returns:
        List of CognitiveEpisode objects in sequence order.
    """
    if not trace or not trace.strip():
        return []

    # Step 1: Segment into sentences
    sentences = segment_trace(
        trace,
        min_sentence_words=min_sentence_words,
        merge_short=merge_short,
    )

    if not sentences:
        return []

    # Step 2: Classify each sentence
    episodes = []
    for i, sentence in enumerate(sentences):
        behavior, confidence, pattern = classify_sentence(sentence)
        token_count = len(sentence.split())

        episode = CognitiveEpisode(
            text=sentence,
            behavior=behavior,
            position=i,
            token_count=token_count,
            confidence=confidence,
            matched_pattern=pattern,
        )
        episodes.append(episode)

    # Step 3: Post-processing
    episodes = _apply_contextual_rules(episodes)

    return episodes


# =============================================================================
# CONTEXTUAL POST-PROCESSING
# =============================================================================

def _apply_contextual_rules(episodes: list[CognitiveEpisode]) -> list[CognitiveEpisode]:
    """
    Apply context-dependent rules to refine classifications.

    These rules use the surrounding episode context to resolve ambiguities
    that can't be handled by single-sentence classification alone.

    Rules:
      1. BACKTRACK followed immediately by FORWARD should keep the
         BACKTRACK label (the forward reasoning is the correction).
      2. Multiple consecutive HESITATION episodes might indicate
         RUMINATION (a prolonged hesitation pattern).
      3. A FORWARD episode at the very end of the trace that contains
         answer-like content should be reclassified as CONCLUSION.
      4. "Wait" classified as BACKTRACK but followed by VERIFICATION
         content should be reconsidered.
    """
    if len(episodes) <= 1:
        return episodes

    # --- Rule 1: Last episode reclassification ---
    # If the final episode looks like an answer but was classified as FORWARD,
    # reclassify it as CONCLUSION.
    last = episodes[-1]
    if last.behavior == BehaviorType.FORWARD:
        answer_signals = [
            r'\b\d+\s*$',                  # Ends with a number
            r'\\boxed\{',                   # Has a boxed answer
            r'\b(?:answer|result)\b',       # Contains "answer" or "result"
        ]
        for pattern in answer_signals:
            if re.search(pattern, last.text, re.IGNORECASE):
                episodes[-1] = CognitiveEpisode(
                    text=last.text,
                    behavior=BehaviorType.CONCLUSION,
                    position=last.position,
                    token_count=last.token_count,
                    confidence=0.6,
                    matched_pattern=f"contextual_rule:last_episode:{pattern}",
                )
                break

    # --- Rule 2: Verification cluster detection ---
    # If we see V, V, V in a row, the middle ones have higher confidence
    # (this is a genuine verification phase, not incidental keyword matches)
    for i in range(1, len(episodes) - 1):
        if (episodes[i-1].behavior == BehaviorType.VERIFICATION and
            episodes[i].behavior == BehaviorType.VERIFICATION and
            episodes[i+1].behavior == BehaviorType.VERIFICATION):
            if episodes[i].confidence < 1.0:
                episodes[i] = CognitiveEpisode(
                    text=episodes[i].text,
                    behavior=BehaviorType.VERIFICATION,
                    position=episodes[i].position,
                    token_count=episodes[i].token_count,
                    confidence=min(1.0, episodes[i].confidence + 0.2),
                    matched_pattern=episodes[i].matched_pattern,
                )

    # --- Rule 3: First episode subgoal boost ---
    # The first 1-2 episodes are often subgoal decomposition even without
    # explicit markers ("I need to find the area of...")
    if (len(episodes) >= 3 and
        episodes[0].behavior == BehaviorType.FORWARD and
        episodes[0].confidence < 0.6):  # Low confidence FORWARD
        subgoal_hints = [
            r'\bfind\b', r'\bsolve\b', r'\bdetermine\b', r'\bcalculate\b',
            r'\bcompute\b', r'\bshow\s+that\b', r'\bprove\b',
        ]
        for pattern in subgoal_hints:
            if re.search(pattern, episodes[0].text, re.IGNORECASE):
                episodes[0] = CognitiveEpisode(
                    text=episodes[0].text,
                    behavior=BehaviorType.SUBGOAL,
                    position=0,
                    token_count=episodes[0].token_count,
                    confidence=0.6,
                    matched_pattern=f"contextual_rule:first_episode:{pattern}",
                )
                break

    return episodes


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def parse_trace_file(
    input_path: str,
    output_path: str,
    trace_field: str = "reasoning_trace",
):
    """
    Parse all traces in a JSONL file and write parsed results.

    Reads traces from the input JSONL, parses each one, and writes
    the results to a new JSONL file with the behavior sequence and
    episode details appended to each record.

    Args:
        input_path: Path to the input JSONL file (from generate_traces.py).
        output_path: Path to write the parsed JSONL output.
        trace_field: Field name containing the reasoning trace text.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    total = 0
    empty_traces = 0
    behavior_counts = Counter()

    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            trace = record.get(trace_field, "")

            # Parse the trace
            episodes = parse_trace(trace)

            # Compute summary stats
            behavior_seq = sequence_to_string(episodes)
            ep_counts = Counter(ep.behavior.value for ep in episodes)

            # Append parsing results to the record
            record["behavior_sequence"] = behavior_seq
            record["num_episodes"] = len(episodes)
            record["episode_counts"] = dict(ep_counts)
            record["episodes"] = [ep.to_dict() for ep in episodes]
            record["mean_confidence"] = (
                sum(ep.confidence for ep in episodes) / len(episodes)
                if episodes else 0.0
            )

            fout.write(json.dumps(record, ensure_ascii=False) + '\n')

            # Track statistics
            total += 1
            if not episodes:
                empty_traces += 1
            for ep in episodes:
                behavior_counts[ep.behavior.value] += 1

            if total % 100 == 0:
                logger.info(f"Parsed {total} traces...")

    # Summary
    logger.info(f"\nParsing complete: {total} traces processed")
    if empty_traces:
        logger.warning(f"  {empty_traces} traces produced no episodes")
    logger.info("  Behavior distribution:")
    total_episodes = sum(behavior_counts.values())
    for behavior_val in "FVBRSHHC":
        if behavior_val in behavior_counts:
            count = behavior_counts[behavior_val]
            pct = count / max(total_episodes, 1) * 100
            bt = BehaviorType(behavior_val)
            logger.info(f"    {bt.display_name:25s}: {count:6d} ({pct:5.1f}%)")


# =============================================================================
# TRACE ANALYSIS UTILITIES
# =============================================================================

def get_trace_summary(episodes: list[CognitiveEpisode]) -> dict:
    """
    Compute a summary of a parsed trace.

    Returns a dict with high-level statistics useful for EDA.
    """
    if not episodes:
        return {
            "num_episodes": 0,
            "behavior_sequence": "",
            "behavior_counts": {},
            "behavior_proportions": {},
            "total_tokens": 0,
            "mean_confidence": 0.0,
            "has_backtrack": False,
            "has_restart": False,
            "has_verification": False,
            "has_conclusion": False,
            "num_behavior_changes": 0,
        }

    seq = sequence_to_string(episodes)
    counts = Counter(ep.behavior.value for ep in episodes)
    total_eps = len(episodes)
    total_tokens = sum(ep.token_count for ep in episodes)

    # Count behavior transitions (changes in behavior type)
    changes = sum(
        1 for i in range(1, len(episodes))
        if episodes[i].behavior != episodes[i-1].behavior
    )

    return {
        "num_episodes": total_eps,
        "behavior_sequence": seq,
        "behavior_counts": dict(counts),
        "behavior_proportions": {k: v / total_eps for k, v in counts.items()},
        "total_tokens": total_tokens,
        "mean_confidence": sum(ep.confidence for ep in episodes) / total_eps,
        "has_backtrack": "B" in seq,
        "has_restart": "R" in seq,
        "has_verification": "V" in seq,
        "has_conclusion": "C" in seq,
        "num_behavior_changes": changes,
    }


def print_annotated_trace(
    episodes: list[CognitiveEpisode],
    max_text_length: int = 80,
    show_confidence: bool = True,
):
    """
    Pretty-print a parsed trace with behavior annotations.

    Useful for manual inspection and debugging.
    """
    if not episodes:
        print("  (empty trace)")
        return

    # Color codes for terminal output
    COLORS = {
        "F": "\033[0m",     # Default (no color)
        "V": "\033[94m",    # Blue
        "B": "\033[91m",    # Red
        "R": "\033[93m",    # Yellow
        "S": "\033[96m",    # Cyan
        "H": "\033[95m",    # Magenta
        "C": "\033[92m",    # Green
    }
    RESET = "\033[0m"

    seq = sequence_to_string(episodes)
    print(f"  Sequence: {seq}")
    print(f"  Episodes: {len(episodes)}")
    print()

    for ep in episodes:
        color = COLORS.get(ep.behavior.value, "")
        text_preview = ep.text[:max_text_length]
        if len(ep.text) > max_text_length:
            text_preview += "..."

        conf_str = f" (conf={ep.confidence:.1f})" if show_confidence else ""
        label = f"[{ep.behavior.value}] {ep.behavior.display_name}"
        print(f"  {color}{ep.position:3d}. {label:30s}{conf_str}{RESET}")
        print(f"       {text_preview}")
        print()


# =============================================================================
# SELF-TEST
# =============================================================================

def run_classifier_tests():
    """Test the full parsing pipeline on realistic trace examples."""
    print("Running behavior classifier self-tests...")

    tests_passed = 0
    tests_failed = 0

    def check(name, trace, expected_behaviors=None, min_episodes=None,
              max_episodes=None, must_contain=None, must_not_contain=None):
        nonlocal tests_passed, tests_failed
        episodes = parse_trace(trace)
        seq = sequence_to_string(episodes)
        ok = True

        if expected_behaviors is not None:
            if seq != expected_behaviors:
                print(f"  FAIL: {name}: expected seq '{expected_behaviors}', got '{seq}'")
                ok = False

        if min_episodes is not None and len(episodes) < min_episodes:
            print(f"  FAIL: {name}: expected >= {min_episodes} episodes, got {len(episodes)}")
            ok = False

        if max_episodes is not None and len(episodes) > max_episodes:
            print(f"  FAIL: {name}: expected <= {max_episodes} episodes, got {len(episodes)}")
            ok = False

        if must_contain:
            for b in must_contain:
                if b not in seq:
                    print(f"  FAIL: {name}: expected '{b}' in sequence, got '{seq}'")
                    ok = False

        if must_not_contain:
            for b in must_not_contain:
                if b in seq:
                    print(f"  FAIL: {name}: expected '{b}' NOT in sequence, got '{seq}'")
                    ok = False

        if ok:
            tests_passed += 1
        else:
            tests_failed += 1

    # --- Test 1: Simple forward reasoning ---
    check(
        "simple_forward",
        "We can compute the derivative of f(x) = x^2 as f'(x) = 2x. "
        "Evaluating at x=3 gives f'(3) = 6.",
        min_episodes=1, max_episodes=3,
        must_contain=["F"],
    )

    # --- Test 2: Backtracking pattern ---
    check(
        "backtrack_pattern",
        "Let me compute the integral. The integral of x^2 is x^3/3. "
        "Wait, that's wrong. I forgot the constant of integration. "
        "The correct answer is x^3/3 + C.",
        min_episodes=3,
        must_contain=["B", "F"],
    )

    # --- Test 3: Verification ---
    check(
        "verification",
        "The solution is x = 5. Let me verify by substituting back. "
        "If x = 5, then 2(5) - 3 = 7, which matches.",
        min_episodes=2,
        must_contain=["V"],
    )

    # --- Test 4: Restart ---
    check(
        "restart",
        "I'll try using integration by parts. Hmm, this gets complicated. "
        "Let me try a different approach. Using substitution instead.",
        min_episodes=2,
        must_contain=["R"],
    )

    # --- Test 5: Complex trace with multiple behaviors ---
    check(
        "complex_multi_behavior",
        "First, I need to find the roots of the polynomial. "
        "Using the quadratic formula, we get x = (-b ± √(b²-4ac))/2a. "
        "For our equation, a=1, b=-5, c=6. "
        "Let me check: b²-4ac = 25-24 = 1. "
        "So x = (5 ± 1)/2, giving x=3 or x=2. "
        "Therefore, the answer is x = 2 and x = 3.",
        min_episodes=4,
        must_contain=["S", "F", "C"],
    )

    # --- Test 6: Hesitation ---
    check(
        "hesitation",
        "Hmm, I'm not sure how to approach this. "
        "Maybe I should try completing the square. "
        "So x^2 + 6x + 9 = (x+3)^2.",
        min_episodes=2,
        must_contain=["H"],
    )

    # --- Test 7: Empty trace ---
    check("empty", "", min_episodes=0, max_episodes=0)

    # --- Test 8: Conclusion at end ---
    check(
        "conclusion_end",
        "Computing step by step. We get 2 + 3 = 5. "
        "So the final answer is 5.",
        min_episodes=2,
        must_contain=["C"],
    )

    # --- Test 9: Realistic DeepSeek-R1 style trace ---
    check(
        "realistic_trace",
        """The problem asks me to find the sum of the first 10 positive integers.

I know there's a formula for this: S = n(n+1)/2.

For n = 10, S = 10 × 11 / 2 = 55.

Let me verify: 1+2+3+4+5+6+7+8+9+10 = 55. Yes, that's correct.

Therefore, the answer is \\boxed{55}.""",
        min_episodes=4,
        must_contain=["V", "C"],
    )

    print(f"\nResults: {tests_passed} passed, {tests_failed} failed "
          f"out of {tests_passed + tests_failed} tests")

    if tests_failed == 0:
        print("All classifier tests passed.")
    else:
        print("WARNING: Some classifier tests failed!")

    return tests_failed == 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = run_classifier_tests()

    if success:
        # Show an annotated example
        print("\n" + "="*60)
        print("Example: Annotated realistic trace")
        print("="*60)
        example = """The problem asks me to find the area of a triangle with sides 3, 4, and 5.

First, I need to determine if this is a right triangle. Since 3^2 + 4^2 = 9 + 16 = 25 = 5^2, yes it is.

For a right triangle, the area is (base × height) / 2 = (3 × 4) / 2 = 6.

Wait, let me double-check this. Is 3-4-5 really a right triangle? Yes, it satisfies the Pythagorean theorem.

Hmm, actually I should verify: 3^2 + 4^2 = 9 + 16 = 25 = 5^2. Confirmed.

Therefore, the answer is \\boxed{6}."""

        episodes = parse_trace(example)
        print_annotated_trace(episodes)
