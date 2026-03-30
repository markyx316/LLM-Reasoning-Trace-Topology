"""
validate_pipeline.py - End-to-end Phase 1 pipeline validation.

This script exercises every component of the Phase 1 pipeline WITHOUT
requiring GPU access or model inference. It uses realistic synthetic
reasoning traces that mimic DeepSeek-R1-Distill output to validate:

  1. Scoring pipeline (math equiv, MC matching, GSM8K extraction)
  2. Trace extraction (<think> block parsing)
  3. Sentence segmentation (math-aware)
  4. Behavior classification (taxonomy + contextual rules)
  5. Feature extraction (all 23 features, 3 groups)
  6. Data serialization (JSONL read/write round-trip)
  7. Annotation template generation
  8. Statistical sanity checks

Run this BEFORE any GPU-based generation to catch bugs early.

Usage:
    cd reasoning-trace-uq
    PYTHONPATH=. python scripts/validate_pipeline.py
"""

import json
import os
import sys
import tempfile
import logging
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.parsing.taxonomy import BehaviorType


# =============================================================================
# SYNTHETIC TRACES — Realistic DeepSeek-R1-Distill outputs
# =============================================================================

SYNTHETIC_TRACES = [
    {
        "name": "correct_clean_math",
        "description": "Clean, correct solution with verification",
        "is_correct": True,
        "ground_truth": "6",
        "answer_type": "exact_match_math",
        "trace": """The problem asks me to find the area of a right triangle with legs 3 and 4.

I know the formula for the area of a triangle is A = (1/2) * base * height.

For a right triangle, the two legs serve as the base and height.

So A = (1/2) * 3 * 4 = (1/2) * 12 = 6.

Let me verify: a right triangle with legs 3 and 4 has a hypotenuse of 5 (since 3^2 + 4^2 = 9 + 16 = 25 = 5^2). The area using the leg formula gives 6 square units. This checks out.

Therefore, the answer is \\boxed{6}.""",
        "answer_text": "The area is \\boxed{6}.",
        "expected_behaviors": {"S", "F", "V", "C"},  # Must contain these
        "expected_no_behaviors": {"B", "R"},  # Must NOT contain these
    },
    {
        "name": "incorrect_with_backtracking",
        "description": "Incorrect solution with backtracking and rumination",
        "is_correct": False,
        "ground_truth": "120",
        "answer_type": "exact_match_math",
        "trace": """I need to find the value of 5 factorial, which is 5!.

Let me compute this step by step.

5! = 5 * 4 * 3 * 2 * 1

So 5 * 4 = 20.

20 * 3 = 60.

Wait, actually, I think I should double-check. Let me reconsider.

Hmm, 60 * 2 = 120. But wait, no, let me recalculate.

Actually, 5 * 4 = 20, and 20 * 3 = 60, and 60 * 2 = 120, and 120 * 1 = 120.

Wait, but the question might be asking for something else. I'm not sure if factorial is the right interpretation.

Hmm, this is tricky. Let me try a different approach.

Starting over. 5! means 5 factorial = 5 × 4 × 3 × 2 × 1 = 60.

Wait, that's wrong again. 5 * 4 * 3 * 2 * 1.

Step 1: 5 * 4 = 20.
Step 2: 20 * 3 = 60.
Step 3: 60 * 2 = 120.
Step 4: 120 * 1 = 120.

Hmm, so it is 120. But I keep second-guessing myself.

Therefore, the answer is \\boxed{60}.""",
        "answer_text": "The answer is \\boxed{60}.",
        "expected_behaviors": {"F", "B", "H", "R", "C"},
        "expected_no_behaviors": set(),
    },
    {
        "name": "mc_science_correct",
        "description": "Multiple choice science question, correct",
        "is_correct": True,
        "ground_truth": "B",
        "answer_type": "multiple_choice",
        "trace": """The question asks which planet is closest to the Sun.

Let me recall the order of planets from the Sun:
Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune.

So Mercury is the closest planet to the Sun.

Looking at the choices, Mercury corresponds to option B.

Let me verify: Mercury orbits at about 0.39 AU from the Sun, which is indeed the smallest orbital distance of any planet. This is correct.

The answer is B.""",
        "answer_text": "The answer is B.",
        "expected_behaviors": {"F", "V", "C"},  # No explicit subgoal markers in trace
        "expected_no_behaviors": {"B", "R"},
    },
    {
        "name": "short_confident_correct",
        "description": "Very short, confident, correct trace",
        "is_correct": True,
        "ground_truth": "4",
        "answer_type": "exact_match_math",
        "trace": """2 + 2 = 4. The answer is \\boxed{4}.""",
        "answer_text": "\\boxed{4}",
        "expected_behaviors": {"F"},
        "expected_no_behaviors": {"B", "R", "H"},
    },
    {
        "name": "gsm8k_style",
        "description": "GSM8K-style word problem with numeric answer",
        "is_correct": True,
        "ground_truth": "Janet sells 16 - 3 - 4921 = 9 duck eggs...\n#### 36",
        "answer_type": "exact_match_numeric",
        "answer_extraction": "gsm8k",
        "trace": """To solve this, I need to figure out how many duck eggs Janet sells at the farmers' market each day.

She has 16 eggs per day.
She eats 3 for breakfast.
She uses 4 for baking muffins.

So she sells 16 - 3 - 4 = 9 eggs per day.

At $2 per egg, she makes 9 * 2 = 18 dollars per day.

Wait, the question says she sells them at $2 per egg at the farmers' market. But she only goes to the market twice a week? No, let me re-read.

Actually, looking at the problem again, she sells the remaining eggs every day.

So daily revenue = 9 * $2 = $18. But wait, let me check: is it 9 eggs or did I make an error?

16 - 3 - 4 = 9. Yes, that's correct.

So the answer is $36. Wait, hmm, let me reconsider whether it's daily or some other time period.

The answer is 36.""",
        "answer_text": "The answer is 36.",
        "expected_behaviors": {"S", "F", "B", "V"},
        "expected_no_behaviors": set(),
    },
]


# =============================================================================
# VALIDATION STEPS
# =============================================================================

def validate_scoring():
    """Step 1: Validate the scoring pipeline."""
    print("\n" + "="*60)
    print("STEP 1: Scoring Pipeline Validation")
    print("="*60)

    from src.generation.scoring import run_scoring_tests
    success = run_scoring_tests()
    return success


def validate_trace_extraction():
    """Step 2: Validate <think> block extraction."""
    print("\n" + "="*60)
    print("STEP 2: Trace Extraction Validation")
    print("="*60)

    from src.generation.generate_traces import extract_think_block

    tests_passed = 0
    tests_failed = 0

    # Test 1: Standard format
    full = "<think>\nSome reasoning here.\n</think>\n\nThe answer is 42."
    trace, answer = extract_think_block(full)
    if "reasoning" in trace and "42" in answer:
        tests_passed += 1
    else:
        tests_failed += 1
        print(f"  FAIL: standard extraction")

    # Test 2: No think block
    full2 = "Just a plain answer."
    trace2, answer2 = extract_think_block(full2)
    if trace2 == "" and "plain answer" in answer2:
        tests_passed += 1
    else:
        tests_failed += 1
        print(f"  FAIL: no think block")

    # Test 3: Think block with math
    full3 = "<think>\nLet $f(x) = x^2$.\n</think>\n\n\\boxed{4}"
    trace3, answer3 = extract_think_block(full3)
    if "f(x)" in trace3 and "boxed" in answer3:
        tests_passed += 1
    else:
        tests_failed += 1
        print(f"  FAIL: math extraction")

    print(f"  Results: {tests_passed}/{tests_passed + tests_failed} passed")
    return tests_failed == 0


def validate_sentence_segmenter():
    """Step 3: Validate sentence segmentation."""
    print("\n" + "="*60)
    print("STEP 3: Sentence Segmenter Validation")
    print("="*60)

    from src.parsing.sentence_segmenter import run_segmenter_tests
    return run_segmenter_tests()


def validate_taxonomy():
    """Step 4: Validate behavior taxonomy."""
    print("\n" + "="*60)
    print("STEP 4: Behavior Taxonomy Validation")
    print("="*60)

    from src.parsing.taxonomy import run_taxonomy_tests
    return run_taxonomy_tests()


def validate_behavior_classifier():
    """Step 5: Validate full behavior classification."""
    print("\n" + "="*60)
    print("STEP 5: Behavior Classifier Validation")
    print("="*60)

    from src.parsing.behavior_classifier import run_classifier_tests
    return run_classifier_tests()


def validate_feature_extraction():
    """Step 6: Validate feature extraction."""
    print("\n" + "="*60)
    print("STEP 6: Feature Extraction Validation")
    print("="*60)

    from src.features.feature_pipeline import run_feature_tests
    return run_feature_tests()


def validate_synthetic_traces():
    """Step 7: Full pipeline on synthetic traces."""
    print("\n" + "="*60)
    print("STEP 7: Full Pipeline on Synthetic Traces")
    print("="*60)

    from src.generation.scoring import score_answer
    from src.parsing.behavior_classifier import parse_trace, get_trace_summary
    from src.parsing.taxonomy import sequence_to_string
    from src.features.feature_pipeline import extract_all_features, get_feature_names

    all_passed = True

    for test in SYNTHETIC_TRACES:
        print(f"\n  --- {test['name']}: {test['description']} ---")

        # Score the answer
        score_result = score_answer(
            model_answer_text=test["answer_text"],
            ground_truth=test["ground_truth"],
            answer_type=test["answer_type"],
            answer_extraction=test.get("answer_extraction"),
        )

        # Check scoring correctness
        if score_result["is_correct"] != test["is_correct"]:
            print(f"  FAIL: Scoring mismatch. "
                  f"Expected is_correct={test['is_correct']}, "
                  f"got {score_result['is_correct']}")
            print(f"    Extracted: {score_result['extracted_answer']}")
            print(f"    Ground truth: {score_result['ground_truth_normalized']}")
            all_passed = False
        else:
            print(f"  Scoring: {'CORRECT' if test['is_correct'] else 'INCORRECT'} "
                  f"(as expected)")

        # Parse the trace
        episodes = parse_trace(test["trace"])
        seq = sequence_to_string(episodes)
        summary = get_trace_summary(episodes)

        print(f"  Parsed:  {len(episodes)} episodes → {seq}")

        # Check expected behaviors
        seq_set = set(seq)
        for expected in test["expected_behaviors"]:
            if expected not in seq_set:
                print(f"  FAIL: Expected behavior '{expected}' not found in sequence '{seq}'")
                all_passed = False

        for unexpected in test["expected_no_behaviors"]:
            if unexpected in seq_set:
                print(f"  FAIL: Unexpected behavior '{unexpected}' found in sequence '{seq}'")
                all_passed = False

        # Extract features
        features = extract_all_features(test["trace"], episodes)
        feature_names = get_feature_names()

        # Sanity checks on features
        if len(features) != len(feature_names):
            print(f"  FAIL: Expected {len(feature_names)} features, got {len(features)}")
            all_passed = False

        for fname in feature_names:
            if fname not in features:
                print(f"  FAIL: Missing feature: {fname}")
                all_passed = False

        # Check feature values are reasonable
        if features["total_tokens"] <= 0:
            print(f"  FAIL: total_tokens should be > 0, got {features['total_tokens']}")
            all_passed = False

        if features["total_episodes"] != len(episodes):
            print(f"  FAIL: total_episodes mismatch")
            all_passed = False

        # Check proportions sum to ~1.0
        prop_sum = sum(
            features[f"prop_{b.value.lower()}" if f"prop_{b.value.lower()}" in features
                      else f"prop_{b.name.lower()}"]
            for b in [BehaviorType.FORWARD, BehaviorType.VERIFICATION,
                      BehaviorType.BACKTRACK, BehaviorType.RESTART,
                      BehaviorType.HESITATION, BehaviorType.SUBGOAL,
                      BehaviorType.CONCLUSION]
            if f"prop_{b.name.lower()}" in features
        )
        # Use the actual feature names
        prop_keys = [k for k in features if k.startswith("prop_")]
        prop_sum = sum(features[k] for k in prop_keys)
        if abs(prop_sum - 1.0) > 0.01 and len(episodes) > 0:
            print(f"  WARN: Proportions sum to {prop_sum:.3f}, expected ~1.0")

        print(f"  Features: {len(features)} extracted, all values reasonable")
        print(f"    Tokens={features['total_tokens']:.0f}, "
              f"BT={features['backtrack_count']:.0f}, "
              f"V={features['verification_count']:.0f}, "
              f"VF_ratio={features['vf_ratio']:.2f}, "
              f"Entropy={features['transition_entropy']:.2f}")

    return all_passed


def validate_data_roundtrip():
    """Step 8: Test JSONL serialization round-trip."""
    print("\n" + "="*60)
    print("STEP 8: Data Serialization Round-Trip")
    print("="*60)

    from src.parsing.behavior_classifier import parse_trace
    from src.parsing.taxonomy import CognitiveEpisode, sequence_to_string
    from src.features.feature_pipeline import extract_all_features

    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl',
                                     delete=False, dir='/tmp') as f:
        temp_path = f.name

        # Write synthetic records
        for test in SYNTHETIC_TRACES:
            episodes = parse_trace(test["trace"])
            features = extract_all_features(test["trace"], episodes)

            record = {
                "item_id": f"test_{test['name']}",
                "dataset": "synthetic",
                "reasoning_trace": test["trace"],
                "answer_text": test["answer_text"],
                "full_response": f"<think>\n{test['trace']}\n</think>\n\n{test['answer_text']}",
                "trace_length": len(test["trace"]),
                "trace_token_count": len(test["trace"].split()),
                "is_correct": test["is_correct"],
                "model_name": "synthetic-test-model",
                "model_short_name": "Synthetic",
                "backend": "test",
                "behavior_sequence": sequence_to_string(episodes),
                "episodes": [ep.to_dict() for ep in episodes],
                "features": features,
            }
            f.write(json.dumps(record) + '\n')

    # Read back and verify
    with open(temp_path, 'r') as f:
        records = [json.loads(line) for line in f if line.strip()]

    os.unlink(temp_path)

    if len(records) != len(SYNTHETIC_TRACES):
        print(f"  FAIL: Wrote {len(SYNTHETIC_TRACES)} records, "
              f"read back {len(records)}")
        return False

    # Verify episodes can be deserialized
    for record in records:
        episodes = [CognitiveEpisode.from_dict(ep) for ep in record["episodes"]]
        seq = sequence_to_string(episodes)
        if seq != record["behavior_sequence"]:
            print(f"  FAIL: Sequence mismatch after round-trip for {record['item_id']}")
            return False

    print(f"  {len(records)} records written and read back successfully")
    print(f"  All behavior sequences preserved through serialization")
    return True


def validate_annotation_template():
    """Step 9: Test annotation template creation."""
    print("\n" + "="*60)
    print("STEP 9: Annotation Template Generation")
    print("="*60)

    from src.parsing.behavior_classifier import parse_trace
    from src.parsing.taxonomy import sequence_to_string

    # Create a temporary traces file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl',
                                     delete=False, dir='/tmp') as f:
        temp_traces = f.name
        for test in SYNTHETIC_TRACES:
            record = {
                "item_id": f"test_{test['name']}",
                "dataset": "synthetic",
                "reasoning_trace": test["trace"],
                "is_correct": test["is_correct"],
            }
            f.write(json.dumps(record) + '\n')

    temp_csv = temp_traces.replace('.jsonl', '_annotations.csv')

    try:
        from src.parsing.parser_evaluation import create_annotation_template
        create_annotation_template(
            traces_path=temp_traces,
            output_csv=temp_csv,
            n_samples=5,
            per_dataset=5,
        )

        # Verify CSV was created and has correct structure
        import csv
        with open(temp_csv, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        expected_cols = {"item_id", "sentence_text", "parser_label", "manual_label"}
        actual_cols = set(reader.fieldnames or [])
        missing = expected_cols - actual_cols

        if missing:
            print(f"  FAIL: Missing columns in annotation CSV: {missing}")
            return False

        print(f"  Created annotation template with {len(rows)} rows")
        print(f"  Columns: {', '.join(reader.fieldnames or [])}")

        # Check parser labels are valid
        valid_labels = {'F', 'V', 'B', 'R', 'S', 'H', 'C'}
        for row in rows:
            if row['parser_label'] not in valid_labels:
                print(f"  FAIL: Invalid parser label: {row['parser_label']}")
                return False

        print(f"  All parser labels valid")
        return True

    finally:
        os.unlink(temp_traces)
        if os.path.exists(temp_csv):
            os.unlink(temp_csv)


def validate_api_client():
    """Step 10: Validate API client (without live API calls)."""
    print("\n" + "="*60)
    print("STEP 10: API Client Validation")
    print("="*60)

    from src.generation.api_client import run_api_client_tests
    return run_api_client_tests()


def print_feature_statistics():
    """Print statistics across all synthetic traces for sanity checking."""
    print("\n" + "="*60)
    print("FEATURE STATISTICS ACROSS SYNTHETIC TRACES")
    print("="*60)

    from src.parsing.behavior_classifier import parse_trace
    from src.features.feature_pipeline import extract_all_features, get_feature_names

    import numpy as np

    feature_names = get_feature_names()
    all_features = []
    labels = []

    for test in SYNTHETIC_TRACES:
        episodes = parse_trace(test["trace"])
        features = extract_all_features(test["trace"], episodes)
        all_features.append(features)
        labels.append(1 if test["is_correct"] else 0)

    # Print comparison: correct vs incorrect
    correct_feats = [f for f, l in zip(all_features, labels) if l == 1]
    incorrect_feats = [f for f, l in zip(all_features, labels) if l == 0]

    print(f"\n  {'Feature':30s} {'Correct':>10s} {'Incorrect':>10s} {'Signal?':>8s}")
    print(f"  {'-'*65}")

    for fname in feature_names:
        c_vals = [f[fname] for f in correct_feats]
        i_vals = [f[fname] for f in incorrect_feats]

        c_mean = np.mean(c_vals) if c_vals else 0
        i_mean = np.mean(i_vals) if i_vals else 0

        # Simple signal detection: is there a noticeable difference?
        diff = abs(c_mean - i_mean)
        max_val = max(abs(c_mean), abs(i_mean), 0.001)
        signal = "YES" if (diff / max_val) > 0.2 else ""

        print(f"  {fname:30s} {c_mean:10.2f} {i_mean:10.2f} {signal:>8s}")


# =============================================================================
# MAIN VALIDATION RUNNER
# =============================================================================

def main():
    print("=" * 60)
    print("PHASE 1 END-TO-END PIPELINE VALIDATION")
    print("=" * 60)
    print()
    print("This validates the entire Phase 1 pipeline without GPU access.")
    print("All components are tested with realistic synthetic traces.")
    print()

    results = {}

    steps = [
        ("1. Scoring Pipeline", validate_scoring),
        ("2. Trace Extraction", validate_trace_extraction),
        ("3. Sentence Segmenter", validate_sentence_segmenter),
        ("4. Behavior Taxonomy", validate_taxonomy),
        ("5. Behavior Classifier", validate_behavior_classifier),
        ("6. Feature Extraction", validate_feature_extraction),
        ("7. Synthetic Trace Pipeline", validate_synthetic_traces),
        ("8. Data Round-Trip", validate_data_roundtrip),
        ("9. Annotation Template", validate_annotation_template),
        ("10. API Client", validate_api_client),
    ]

    for step_name, step_fn in steps:
        try:
            success = step_fn()
            results[step_name] = "PASS" if success else "FAIL"
        except Exception as e:
            results[step_name] = f"ERROR: {e}"
            logger.error(f"Step {step_name} raised exception: {e}", exc_info=True)

    # Print feature statistics
    try:
        print_feature_statistics()
    except Exception as e:
        logger.error(f"Feature statistics failed: {e}")

    # Final summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_pass = True
    for step_name, result in results.items():
        status = "✓" if result == "PASS" else "✗"
        print(f"  {status} {step_name}: {result}")
        if result != "PASS":
            all_pass = False

    print()
    if all_pass:
        print("ALL STEPS PASSED — Pipeline is ready for generation.")
        print()
        print("Next steps:")
        print("  API workflow (recommended - pay as you go):")
        print("    export DEEPSEEK_API_KEY=sk-your-key-here")
        print("    ./scripts/run_generation.sh pilot")
        print("    ./scripts/run_generation.sh api-all")
        print()
        print("  HPC workflow (local GPU for distilled models):")
        print("    ./scripts/submit_hpc.sh pilot")
        print("    ./scripts/submit_hpc.sh all-datasets qwen7b")
        print("    ./scripts/submit_hpc.sh all-datasets llama8b")
        print("    ./scripts/submit_hpc.sh cross-model math500")
        print()
        print("  After generation:")
        print("    ./scripts/run_experiments.sh all")
    else:
        print("SOME STEPS FAILED — fix the issues above before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
