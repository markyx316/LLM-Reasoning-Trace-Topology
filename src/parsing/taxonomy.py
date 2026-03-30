"""
taxonomy.py - Cognitive behavior taxonomy for reasoning traces.

Defines the behavior types used to characterize cognitive episodes in
reasoning traces. The taxonomy is adapted from:
  - Marjanović et al. (2025), "DeepSeek-R1 Thoughtology"
  - Gandhi et al. (2025), "Cognitive Behaviors in LLMs"

Design decisions:
  1. Priority ordering: When multiple patterns match a sentence, the
     highest-priority behavior wins. Priority reflects specificity:
     BACKTRACK > RESTART > VERIFICATION > CONCLUSION > SUBGOAL > HESITATION > FORWARD
  2. Forward is the default: any sentence that doesn't match a pattern
     is classified as forward reasoning (new claims, derivations, etc.)
  3. Pattern lists are ordered by specificity within each behavior type,
     with more specific patterns first.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import re


class BehaviorType(Enum):
    """
    Cognitive behavior types in reasoning traces.

    Each type represents a distinct cognitive operation the model performs
    during its reasoning process. The value is a short symbol used for
    compact sequence representation.
    """
    FORWARD = "F"       # Forward reasoning: new claims, derivations, calculations
    VERIFICATION = "V"  # Self-checking: confirming or validating previous work
    BACKTRACK = "B"     # Error correction: recognizing and fixing a mistake
    RESTART = "R"       # Approach abandonment: trying a completely different strategy
    SUBGOAL = "S"       # Decomposition: breaking the problem into sub-tasks
    HESITATION = "H"    # Uncertainty: expressing doubt, ruminating
    CONCLUSION = "C"    # Final answer: stating the solution

    @property
    def display_name(self) -> str:
        names = {
            "F": "Forward Reasoning",
            "V": "Verification",
            "B": "Backtracking",
            "R": "Restart/Abandonment",
            "S": "Sub-goal Decomposition",
            "H": "Hesitation/Rumination",
            "C": "Conclusion",
        }
        return names.get(self.value, self.value)

    @property
    def priority(self) -> int:
        """
        Priority for conflict resolution.
        Higher value = higher priority (wins when multiple behaviors match).

        Rationale:
          - BACKTRACK is highest because recognizing errors is the strongest
            signal and should never be overridden by a weaker pattern.
          - RESTART is second because approach abandonment is a strong,
            specific behavior (not just a keyword match).
          - FORWARD is lowest (default) because it's the catch-all.
        """
        priorities = {
            "B": 7,  # Backtrack: strongest signal
            "R": 6,  # Restart: strong, specific
            "V": 5,  # Verification: specific self-checking language
            "C": 4,  # Conclusion: specific answer-stating language
            "S": 3,  # Sub-goal: decomposition language
            "H": 2,  # Hesitation: weaker signals (hmm, maybe)
            "F": 1,  # Forward: default (no pattern matched)
        }
        return priorities.get(self.value, 0)


@dataclass
class CognitiveEpisode:
    """
    A single cognitive episode in a reasoning trace.

    An episode is typically one sentence (or a short group of closely
    related sentences) that performs a single cognitive function.
    """
    text: str                        # The raw text of this episode
    behavior: BehaviorType           # Classified behavior type
    position: int                    # 0-indexed position in the sequence
    token_count: int                 # Approximate token count (word-split)
    confidence: float = 1.0          # Classifier confidence (1.0 = rule-based match)
    matched_pattern: Optional[str] = None  # The regex pattern that triggered classification

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "behavior": self.behavior.value,
            "position": self.position,
            "token_count": self.token_count,
            "confidence": self.confidence,
            "matched_pattern": self.matched_pattern,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CognitiveEpisode":
        return cls(
            text=d["text"],
            behavior=BehaviorType(d["behavior"]),
            position=d["position"],
            token_count=d["token_count"],
            confidence=d.get("confidence", 1.0),
            matched_pattern=d.get("matched_pattern"),
        )


# =============================================================================
# BEHAVIOR DETECTION PATTERNS
# =============================================================================
# Each pattern list is ordered by specificity (most specific first).
# Patterns use raw strings and word boundaries to avoid false matches.
# The re.IGNORECASE flag is applied during matching, not in the patterns.

BEHAVIOR_PATTERNS: dict[BehaviorType, list[str]] = {
    BehaviorType.BACKTRACK: [
        # --- High confidence: explicit error acknowledgment ---
        r"(?:no|wait),?\s+that(?:'s|\s+is)\s+(?:wrong|incorrect|not\s+right|not\s+correct)",
        r"\bi\s+(?:made|have)\s+(?:a\s+|an\s+)?(?:error|mistake)",
        r"\bthat\s+(?:was|is)\s+(?:a\s+)?(?:mistake|error|wrong)",
        r"\blet\s+me\s+(?:reconsider|rethink|re-examine|redo\s+(?:this|that))",
        r"\bgoing\s+back\s+to\b",
        r"\bi\s+(?:was|am)\s+wrong\b",
        r"\bthat\s+(?:doesn't|does\s+not)\s+(?:work|seem\s+right|look\s+right|make\s+sense)",
        r"\bthis\s+(?:is|seems)\s+(?:wrong|incorrect)\b",
        r"\bcorrection\s*:",

        # --- Medium confidence: "wait" + error-related context ---
        r"\bwait\b.*?\b(?:wrong|mistake|error|incorrect|no\b)",
        r"\bwait\b.*?\b(?:actually|should\s+be|instead)\b",

        # --- Lower confidence: standalone "wait" or "actually" ---
        # NOTE: These are ambiguous. The "wait" token is famously
        # overloaded in DeepSeek-R1 traces. We assign BACKTRACK
        # only when followed by corrective language. Standalone
        # "wait" without correction goes to HESITATION.
        r"^wait\b(?!.*?(?:let\s+me\s+(?:check|verify)))",
        r"^actually\b",
    ],

    BehaviorType.RESTART: [
        # --- High confidence: explicit approach change ---
        r"\blet\s+me\s+try\s+(?:a\s+)?(?:different|another|new)\s+(?:approach|method|way|strategy)",
        r"\bstarting\s+(?:over|from\s+scratch|fresh|again)\b",
        r"\bscrap(?:ping)?\s+(?:this|that)\b",
        r"\babandoning?\s+(?:this|that)\s+(?:approach|method|idea)",
        r"\blet(?:'s|\s+me)\s+(?:take|use)\s+(?:a\s+)?(?:different|another|new)\s+approach",
        r"\binstead,?\s+(?:let\s+me|i(?:'ll|\s+will))\b",

        # --- Medium confidence ---
        r"\balternatively\b",
        r"\banother\s+(?:way|approach|method)\s+(?:to|would\s+be|is)\b",
        r"\bapproaching?\s+(?:this|it)\s+differently\b",
        r"\blet(?:'s|\s+me)\s+(?:try|consider)\s+(?:something\s+else|another)\b",
    ],

    BehaviorType.VERIFICATION: [
        # --- High confidence: explicit verification language ---
        r"\blet\s+me\s+(?:check|verify|confirm|validate|double[\s-]?check|make\s+sure)",
        r"\bto\s+(?:verify|confirm|check|validate)\b",
        r"\bdouble[\s-]?check(?:ing)?\b",
        r"\bsanity\s+check\b",
        r"\bchecking\s+(?:this|that|my|our|the)\b",
        r"\bis\s+(?:this|that)\s+(?:correct|right|true|valid)\?",
        r"\bverif(?:ying|ication)\b",

        # --- Medium confidence ---
        r"\blet(?:'s|\s+me)\s+(?:see|make\s+sure)\b",
        r"\blet\s+me\s+(?:re-?read|review)\b",
        r"\bsubstitut(?:e|ing)\s+(?:back|this|that)\s+(?:into|in)\b",
        r"\bplug(?:ging)?\s+(?:this|it|that)\s+(?:back\s+)?in\b",

        # --- "wait" + verification context ---
        r"\bwait\b.*?\blet\s+me\s+(?:check|verify)\b",
    ],

    BehaviorType.CONCLUSION: [
        # --- High confidence: answer-stating language ---
        r"\\boxed\{",                                    # LaTeX boxed answer
        r"\bthe\s+(?:final\s+)?answer\s+is\b",
        r"\btherefore,?\s+(?:the\s+)?(?:answer|solution|result)\b",
        r"\bin\s+conclusion\b",
        r"\bso\s+(?:the\s+)?(?:final\s+)?answer\b",
        r"\bthus,?\s+(?:the\s+)?(?:answer|solution|result)\b",
        r"\bhence,?\s+(?:the\s+)?(?:answer|solution|result)\b",
        r"\bour\s+(?:final\s+)?answer\s+is\b",
        r"\bfinal\s+answer\s*:",
    ],

    BehaviorType.SUBGOAL: [
        # --- Decomposition language ---
        r"\bfirst,?\s+(?:i\s+need\s+to|we\s+need\s+to|let(?:'s|\s+me))\b",
        r"\bstep\s+\d+\s*[:.]",
        r"\bbreaking\s+(?:this|it)\s+down\b",
        r"\bthe\s+key\s+(?:insight|idea|observation|step)\s+is\b",
        r"\bto\s+solve\s+this,?\s+(?:i|we)\s+(?:need|should|must|can)\b",
        r"\bmy\s+(?:plan|approach|strategy)\s+is\b",
        r"\blet(?:'s|\s+me)\s+(?:start|begin)\s+(?:by|with)\b",
        r"\b(?:i|we)\s+need\s+to\s+find\b",
        r"\bthe\s+(?:first|next)\s+step\s+is\b",
        r"\blet(?:'s|\s+me)\s+(?:set\s+up|formulate|define)\b",
    ],

    BehaviorType.HESITATION: [
        # --- Explicit uncertainty markers ---
        r"\bhmm+\b",
        r"\buhh?\b",
        r"\bumm?\b",
        r"\bi(?:'m|\s+am)\s+not\s+(?:sure|certain|confident)\b",
        r"\bnot\s+(?:sure|certain)\s+(?:if|whether|about|how)\b",
        r"\bthis\s+(?:is|seems)\s+(?:tricky|confusing|difficult|complicated|hard)\b",
        r"\bi\s+wonder\b",
        r"\bperhaps\b(?!.*\bapproach\b)",  # "perhaps" alone, not in "perhaps another approach"
        r"^maybe\b",                        # "Maybe" at sentence start
        r"\bi\s+(?:think|guess|suppose)\s+(?:so|that)\b",
        r"\bi(?:'m|\s+am)\s+(?:confused|unsure|uncertain)\b",
    ],
}


# =============================================================================
# BEHAVIOR CLASSIFICATION
# =============================================================================

def classify_sentence(sentence: str) -> tuple[BehaviorType, float, Optional[str]]:
    """
    Classify a single sentence into a cognitive behavior type.

    Uses the pattern dictionary with priority-based conflict resolution.
    If multiple behavior types match, the highest-priority one wins.

    Args:
        sentence: The sentence text to classify.

    Returns:
        (behavior_type, confidence, matched_pattern) tuple.
        If no pattern matches, returns (FORWARD, 0.5, None).
    """
    sentence_lower = sentence.lower().strip()

    if not sentence_lower:
        return BehaviorType.FORWARD, 0.0, None

    # Collect all matches
    matches: list[tuple[BehaviorType, str]] = []

    for behavior, patterns in BEHAVIOR_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, sentence_lower):
                matches.append((behavior, pattern))
                break  # Only the first (most specific) pattern per behavior

    if not matches:
        # Default: forward reasoning (no pattern matched)
        return BehaviorType.FORWARD, 0.5, None

    if len(matches) == 1:
        behavior, pattern = matches[0]
        return behavior, 1.0, pattern

    # Multiple behaviors matched — resolve by priority
    matches.sort(key=lambda x: x[0].priority, reverse=True)
    behavior, pattern = matches[0]
    # Lower confidence since there was ambiguity
    return behavior, 0.7, pattern


def sequence_to_string(episodes: list[CognitiveEpisode]) -> str:
    """
    Convert a list of cognitive episodes to a compact behavior string.

    Example: [F, F, V, B, F, F, C] → "FFVBFFC"
    """
    return "".join(ep.behavior.value for ep in episodes)


def string_to_behaviors(s: str) -> list[BehaviorType]:
    """
    Convert a compact behavior string back to a list of BehaviorType.

    Example: "FFVBFFC" → [FORWARD, FORWARD, VERIFICATION, BACKTRACK, ...]
    """
    return [BehaviorType(c) for c in s]


# =============================================================================
# SELF-TEST
# =============================================================================

def run_taxonomy_tests():
    """Test behavior classification on representative examples."""
    print("Running taxonomy self-tests...")

    test_cases = [
        # (sentence, expected_behavior)
        # --- BACKTRACK ---
        ("Wait, that's wrong. Let me reconsider.", BehaviorType.BACKTRACK),
        ("No, I made an error in the calculation.", BehaviorType.BACKTRACK),
        ("Actually, the formula should be different.", BehaviorType.BACKTRACK),
        ("That doesn't seem right.", BehaviorType.BACKTRACK),

        # --- RESTART ---
        ("Let me try a different approach.", BehaviorType.RESTART),
        ("Alternatively, we could use integration.", BehaviorType.RESTART),
        ("Starting over with a new method.", BehaviorType.RESTART),
        ("Instead, let me consider the problem from scratch.", BehaviorType.RESTART),

        # --- VERIFICATION ---
        ("Let me check this result.", BehaviorType.VERIFICATION),
        ("Let me verify by substituting back in.", BehaviorType.VERIFICATION),
        ("Double-checking the arithmetic.", BehaviorType.VERIFICATION),
        ("Is this correct?", BehaviorType.VERIFICATION),

        # --- CONCLUSION ---
        ("Therefore, the answer is 42.", BehaviorType.CONCLUSION),
        ("So the final answer is \\boxed{42}.", BehaviorType.CONCLUSION),
        ("In conclusion, x = 7.", BehaviorType.CONCLUSION),

        # --- SUBGOAL ---
        ("First, I need to find the derivative.", BehaviorType.SUBGOAL),
        ("Step 1: Set up the equation.", BehaviorType.SUBGOAL),
        ("To solve this, we need to factor the polynomial.", BehaviorType.SUBGOAL),
        ("Let me start by defining the variables.", BehaviorType.SUBGOAL),

        # --- HESITATION ---
        ("Hmm, this is tricky.", BehaviorType.HESITATION),
        ("I'm not sure about this step.", BehaviorType.HESITATION),
        ("I wonder if there's a simpler way.", BehaviorType.HESITATION),

        # --- FORWARD (no specific pattern) ---
        ("We can write f(x) = 3x^2 + 2x - 1.", BehaviorType.FORWARD),
        ("The eigenvalues of A are 1, 2, and 3.", BehaviorType.FORWARD),
        ("Applying the quadratic formula gives us.", BehaviorType.FORWARD),
    ]

    passed = 0
    failed = 0
    for sentence, expected in test_cases:
        behavior, confidence, pattern = classify_sentence(sentence)
        if behavior == expected:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: '{sentence[:50]}...'")
            print(f"    Expected: {expected.display_name}, Got: {behavior.display_name}")
            print(f"    Pattern: {pattern}")

    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    if failed == 0:
        print("All taxonomy tests passed.")
    else:
        print("WARNING: Some taxonomy tests failed!")

    return failed == 0


if __name__ == "__main__":
    run_taxonomy_tests()
