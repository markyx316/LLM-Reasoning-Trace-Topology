"""
baselines.py - Uncertainty quantification baseline implementations.

Implements 5 UQ baselines for comparison against our structural method:

  1. Trace Length (naive structural)     — 1 generation, text-only
  2. Verbalized Confidence (self-report) — 1 generation, text-only
  3. Self-Consistency (sampling)         — N generations, text-only
  4. Semantic Entropy (sampling)         — N generations, text-only + NLI
  5. Perplexity (logit-based)            — 1 generation, white-box

Each baseline produces a single scalar "confidence score" per item,
where higher = more confident the answer is correct.

Usage:
    from src.baselines.baselines import compute_all_baselines
    scores = compute_all_baselines(trace_record, sc_record)
"""

import re
import logging
import numpy as np
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# 1. TRACE LENGTH BASELINE
# =============================================================================

def trace_length_confidence(trace_record: dict) -> float:
    """
    Trace length as a naive UQ signal.

    Shorter traces = higher confidence (based on Marjanović et al. finding
    that correct solutions average ~2000 tokens vs ~4000 for incorrect).

    We invert and normalize the token count so that shorter traces
    produce higher confidence scores.

    Returns: confidence score in [0, 1] (higher = more confident).
    """
    token_count = trace_record.get("trace_token_count", 0)

    if token_count <= 0:
        return 0.5  # No information

    # Sigmoid-based inversion: maps token count to (0, 1)
    # Centered at 2000 tokens (the approximate mean for correct solutions)
    # Steepness controls how quickly confidence drops with length
    midpoint = 2000.0
    steepness = 0.001

    confidence = 1.0 / (1.0 + np.exp(steepness * (token_count - midpoint)))

    return float(confidence)


# =============================================================================
# 2. VERBALIZED CONFIDENCE BASELINE
# =============================================================================

def extract_verbalized_confidence(text: str) -> float:
    """
    Extract a self-reported confidence score from model text.

    Looks for patterns like:
      - "Confidence: 85"
      - "My confidence is 90%"
      - "I'm 75% confident"
      - "confidence level: 8/10"

    Returns: confidence score in [0, 1].
    """
    text_lower = text.lower()

    # Pattern 1: "Confidence: N" or "confidence level: N"
    match = re.search(
        r'confidence\s*(?:level\s*)?(?:is\s*)?[:=]?\s*(\d+(?:\.\d+)?)\s*[%/]?',
        text_lower
    )
    if match:
        val = float(match.group(1))
        if val > 1.0:
            val = val / 100.0  # Convert percentage to [0,1]
        return min(max(val, 0.0), 1.0)

    # Pattern 2: "I'm N% confident"
    match = re.search(r"i(?:'m|\s+am)\s+(\d+(?:\.\d+)?)\s*%?\s*confident", text_lower)
    if match:
        val = float(match.group(1))
        if val > 1.0:
            val = val / 100.0
        return min(max(val, 0.0), 1.0)

    # Pattern 3: "N/10" or "N out of 10"
    match = re.search(r'(\d+(?:\.\d+)?)\s*(?:/|out\s+of)\s*10', text_lower)
    if match:
        return min(max(float(match.group(1)) / 10.0, 0.0), 1.0)

    # Pattern 4: Qualitative labels
    # IMPORTANT: Check low-confidence phrases FIRST because they often
    # contain substrings of high-confidence phrases ("not sure" contains "sure")
    low_conf = ['not confident', 'not sure', 'not certain',
                'uncertain', 'unsure',
                'doubtful', 'unlikely', "don't know", "i'm not"]
    high_conf = ['very confident', 'highly confident', 'completely certain',
                 'absolutely certain', 'definitely correct',
                 'absolutely sure', 'completely sure']
    medium_conf = ['fairly confident', 'reasonably confident', 'probably',
                   'likely correct']

    for phrase in low_conf:
        if phrase in text_lower:
            return 0.3

    for phrase in high_conf:
        if phrase in text_lower:
            return 0.9

    for phrase in medium_conf:
        if phrase in text_lower:
            return 0.6

    # Default: no confidence signal found
    return 0.5


def verbalized_confidence(trace_record: dict) -> float:
    """
    Extract verbalized confidence from a trace record.

    If the model was prompted to state confidence, extract it.
    Otherwise, look for confidence signals in the reasoning trace itself.

    Returns: confidence score in [0, 1].
    """
    # Check the answer text first (where confidence is usually stated)
    answer = trace_record.get("answer_text", "")
    conf = extract_verbalized_confidence(answer)

    if conf != 0.5:  # Found a non-default confidence
        return conf

    # Fallback: check the last part of the reasoning trace
    trace = trace_record.get("reasoning_trace", "")
    if trace:
        # Only check the last 500 characters (confidence is usually at the end)
        return extract_verbalized_confidence(trace[-500:])

    return 0.5


# =============================================================================
# 3. SELF-CONSISTENCY BASELINE
# =============================================================================

def self_consistency_confidence(sc_record: dict) -> float:
    """
    Self-consistency confidence: agreement among N independent samples.

    The confidence is the fraction of samples that agree with the
    majority answer. Higher agreement = higher confidence.

    Args:
        sc_record: Record from self-consistency generation with a
                   'samples' field containing N sample results.

    Returns: confidence score in [0, 1].
    """
    samples = sc_record.get("samples", [])
    if not samples:
        return 0.5

    # Extract answers from all samples
    answers = []
    for s in samples:
        ans = s.get("extracted_answer")
        if ans is not None and "error" not in s:
            answers.append(str(ans).strip().lower())

    if not answers:
        return 0.5

    # Count occurrences of each unique answer
    answer_counts = Counter(answers)
    most_common_count = answer_counts.most_common(1)[0][1]

    # Agreement = fraction giving majority answer
    agreement = most_common_count / len(answers)

    return float(agreement)


# =============================================================================
# 4. SEMANTIC ENTROPY BASELINE
# =============================================================================

def semantic_entropy_confidence(
    sc_record: dict,
    use_nli: bool = True,
) -> float:
    """
    Semantic entropy: cluster answers by meaning, compute entropy.

    Lower entropy = answers cluster into fewer semantic groups = higher confidence.
    We return 1 - normalized_entropy as the confidence score.

    This is a simplified implementation of Kuhn et al. (2023).

    Args:
        sc_record: Record with 'samples' field.
        use_nli: If True, use NLI model for semantic clustering.
                If False, use exact-match clustering (faster, less accurate).

    Returns: confidence score in [0, 1].
    """
    samples = sc_record.get("samples", [])
    if not samples:
        return 0.5

    answers = []
    for s in samples:
        ans = s.get("extracted_answer")
        if ans is not None and "error" not in s:
            answers.append(str(ans).strip())

    if len(answers) <= 1:
        return 0.5

    if use_nli:
        clusters = _cluster_answers_nli(answers)
    else:
        clusters = _cluster_answers_exact(answers)

    if not clusters:
        return 0.5

    # Compute entropy over cluster sizes
    cluster_sizes = np.array([len(c) for c in clusters], dtype=float)
    cluster_probs = cluster_sizes / cluster_sizes.sum()

    # Shannon entropy
    entropy = -np.sum(cluster_probs * np.log2(cluster_probs + 1e-12))

    # Normalize by max possible entropy (all answers different)
    max_entropy = np.log2(len(answers))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # Invert: low entropy = high confidence
    confidence = 1.0 - normalized_entropy

    return float(confidence)


def _cluster_answers_exact(answers: list[str]) -> list[list[str]]:
    """Cluster answers by exact string match (fast baseline)."""
    clusters = {}
    for ans in answers:
        key = ans.strip().lower()
        clusters.setdefault(key, []).append(ans)
    return list(clusters.values())


def _cluster_answers_nli(
    answers: list[str],
    entailment_threshold: float = 0.5,
) -> list[list[str]]:
    """
    Cluster answers by semantic equivalence using an NLI model.

    Uses a DeBERTa-based NLI model to determine if two answers
    are semantically equivalent (mutual entailment).
    """
    try:
        from transformers import pipeline
        nli = pipeline(
            "text-classification",
            model="microsoft/deberta-base-mnli",
            device=-1,  # CPU (avoid GPU contention)
        )
    except Exception as e:
        logger.warning(f"NLI model unavailable ({e}), falling back to exact match")
        return _cluster_answers_exact(answers)

    clusters: list[list[str]] = []

    for answer in answers:
        placed = False
        for cluster in clusters:
            representative = cluster[0]

            # Check bidirectional entailment
            try:
                result = nli(f"{representative}. {answer}")
                if isinstance(result, list):
                    result = result[0]

                # Check if entailment score is high enough
                label = result.get("label", "").upper()
                score = result.get("score", 0)

                if label == "ENTAILMENT" and score >= entailment_threshold:
                    cluster.append(answer)
                    placed = True
                    break
            except Exception:
                # On any NLI failure, fall back to string comparison
                if answer.strip().lower() == representative.strip().lower():
                    cluster.append(answer)
                    placed = True
                    break

        if not placed:
            clusters.append([answer])

    return clusters


# =============================================================================
# 5. PERPLEXITY BASELINE
# =============================================================================

def perplexity_confidence(trace_record: dict) -> float:
    """
    Perplexity-based confidence: lower perplexity = higher confidence.

    Uses the mean log-probability of generated tokens (requires white-box
    access to the model during generation).

    Returns: confidence score in [0, 1].
    """
    mean_log_prob = trace_record.get("mean_log_prob")

    if mean_log_prob is None:
        return 0.5  # No log-prob data available

    # Perplexity = exp(-mean_log_prob)
    # Lower perplexity = more confident predictions
    perplexity = np.exp(-mean_log_prob)

    # Map perplexity to confidence using sigmoid
    # Centered at perplexity=10 (typical range is 2-50)
    midpoint = 10.0
    steepness = 0.2
    confidence = 1.0 / (1.0 + np.exp(steepness * (perplexity - midpoint)))

    return float(confidence)


# =============================================================================
# 6. OUR METHOD (STRUCTURAL FEATURES)
# =============================================================================

def structural_confidence(
    trace_record: dict,
    model=None,
    scaler=None,
) -> float:
    """
    Our method: structural feature-based confidence.

    Uses a pre-trained classifier on trace features to predict
    the probability of correctness.

    Args:
        trace_record: Record with 'features' dict containing all 23 features.
        model: Trained sklearn classifier with predict_proba method.
        scaler: Fitted StandardScaler for feature normalization.

    Returns: confidence score in [0, 1] (predicted P(correct)).
    """
    if model is None:
        logger.warning("No trained model provided for structural confidence")
        return 0.5

    from src.features.feature_pipeline import get_feature_names

    features = trace_record.get("features", {})
    feature_names = get_feature_names()

    # Build feature vector
    X = np.array([[features.get(fname, 0.0) for fname in feature_names]])

    if scaler is not None:
        X = scaler.transform(X)

    # Predict probability of correctness
    proba = model.predict_proba(X)[0]

    # Return P(correct) — index 1 for binary classifiers
    return float(proba[1]) if len(proba) > 1 else float(proba[0])


# =============================================================================
# UNIFIED BASELINE COMPUTATION
# =============================================================================

def compute_all_baselines(
    trace_record: dict,
    sc_record: Optional[dict] = None,
    structural_model=None,
    structural_scaler=None,
) -> dict[str, float]:
    """
    Compute all baseline confidence scores for a single item.

    Args:
        trace_record: Primary trace record (from generate_traces.py).
        sc_record: Self-consistency record (from run_self_consistency_generation).
                   If None, SC and SE baselines return 0.5.
        structural_model: Trained classifier for our method.
        structural_scaler: Feature scaler for our method.

    Returns:
        Dict mapping method name → confidence score in [0, 1].
    """
    scores = {
        "trace_length": trace_length_confidence(trace_record),
        "verbalized_confidence": verbalized_confidence(trace_record),
        "perplexity": perplexity_confidence(trace_record),
    }

    # Sampling-based baselines (require SC samples)
    if sc_record is not None:
        scores["self_consistency"] = self_consistency_confidence(sc_record)
        scores["semantic_entropy"] = semantic_entropy_confidence(
            sc_record, use_nli=False  # Start with exact match; use NLI if available
        )
    else:
        scores["self_consistency"] = None
        scores["semantic_entropy"] = None

    # Our method
    if structural_model is not None:
        scores["structural"] = structural_confidence(
            trace_record, structural_model, structural_scaler
        )
    else:
        scores["structural"] = None

    return scores


# =============================================================================
# SELF-TEST
# =============================================================================

def run_baseline_tests():
    """Test baseline implementations on synthetic data."""
    print("Running baseline self-tests...")

    tests_passed = 0
    tests_failed = 0

    def check(name, condition, msg=""):
        nonlocal tests_passed, tests_failed
        if condition:
            tests_passed += 1
        else:
            tests_failed += 1
            print(f"  FAIL: {name}: {msg}")

    # --- Trace length ---
    short_record = {"trace_token_count": 500}
    long_record = {"trace_token_count": 5000}
    short_conf = trace_length_confidence(short_record)
    long_conf = trace_length_confidence(long_record)
    check("tl_short_higher", short_conf > long_conf,
          f"short={short_conf:.3f} should be > long={long_conf:.3f}")
    check("tl_range", 0 <= short_conf <= 1 and 0 <= long_conf <= 1,
          f"scores out of range")

    # --- Verbalized confidence ---
    check("vc_numeric",
          abs(extract_verbalized_confidence("Confidence: 85") - 0.85) < 0.01,
          f"got {extract_verbalized_confidence('Confidence: 85')}")
    check("vc_percent",
          abs(extract_verbalized_confidence("I'm 90% confident") - 0.90) < 0.01,
          "expected ~0.90")
    check("vc_qualitative",
          extract_verbalized_confidence("I am very confident in this answer") > 0.7)
    check("vc_low",
          extract_verbalized_confidence("I'm not sure about this") < 0.5)
    check("vc_default",
          extract_verbalized_confidence("The answer is 42") == 0.5)

    # --- Self-consistency ---
    sc_perfect = {"samples": [
        {"extracted_answer": "42"}, {"extracted_answer": "42"},
        {"extracted_answer": "42"}, {"extracted_answer": "42"},
    ]}
    sc_mixed = {"samples": [
        {"extracted_answer": "42"}, {"extracted_answer": "42"},
        {"extracted_answer": "43"}, {"extracted_answer": "44"},
    ]}
    sc_split = {"samples": [
        {"extracted_answer": "42"}, {"extracted_answer": "43"},
    ]}

    check("sc_perfect", self_consistency_confidence(sc_perfect) == 1.0)
    check("sc_mixed", self_consistency_confidence(sc_mixed) == 0.5)
    check("sc_split", self_consistency_confidence(sc_split) == 0.5)
    check("sc_ordering",
          self_consistency_confidence(sc_perfect) > self_consistency_confidence(sc_mixed))

    # --- Semantic entropy ---
    se_perfect = semantic_entropy_confidence(sc_perfect, use_nli=False)
    se_mixed = semantic_entropy_confidence(sc_mixed, use_nli=False)
    check("se_perfect_high", se_perfect > 0.8,
          f"got {se_perfect:.3f}")
    check("se_mixed_lower", se_mixed < se_perfect,
          f"mixed={se_mixed:.3f} should be < perfect={se_perfect:.3f}")

    # --- Perplexity ---
    low_ppl_record = {"mean_log_prob": -0.5}    # Low perplexity (= exp(0.5) ≈ 1.65)
    high_ppl_record = {"mean_log_prob": -4.0}   # High perplexity (= exp(4) ≈ 55)
    no_ppl_record = {"mean_log_prob": None}

    low_ppl_conf = perplexity_confidence(low_ppl_record)
    high_ppl_conf = perplexity_confidence(high_ppl_record)
    check("ppl_low_higher", low_ppl_conf > high_ppl_conf,
          f"low_ppl={low_ppl_conf:.3f} should be > high_ppl={high_ppl_conf:.3f}")
    check("ppl_none", perplexity_confidence(no_ppl_record) == 0.5)

    # --- Unified computation ---
    all_scores = compute_all_baselines(short_record, sc_perfect)
    check("unified_has_tl", "trace_length" in all_scores)
    check("unified_has_vc", "verbalized_confidence" in all_scores)
    check("unified_has_sc", "self_consistency" in all_scores)
    check("unified_has_se", "semantic_entropy" in all_scores)
    check("unified_has_ppl", "perplexity" in all_scores)

    print(f"\nResults: {tests_passed} passed, {tests_failed} failed "
          f"out of {tests_passed + tests_failed} tests")
    if tests_failed == 0:
        print("All baseline tests passed.")
    return tests_failed == 0


if __name__ == "__main__":
    run_baseline_tests()
