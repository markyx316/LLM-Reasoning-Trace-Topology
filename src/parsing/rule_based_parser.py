"""
rule_based_parser.py - Rule-based behavior parser for reasoning traces.

Classifies every sentence in a reasoning trace into one of 6 cognitive
behavior types using hand-crafted regex patterns:

  F  Forward   — new claims, derivations, calculations (default)
  V  Verify    — self-checking, confirming, double-checking
  X  Revise    — error correction, recognizing a mistake and fixing it
  R  Restart   — abandoning current approach, trying something different
  H  Hesitate  — expressing doubt, uncertainty, or confusion
  C  Conclude  — stating the final answer

Design:
  - Priority ordering (X > R > V > C > H > F) resolves conflicts when
    multiple patterns match the same sentence.
  - Forward is the default: any sentence that doesn't match a pattern is
    classified as Forward (new content being generated).
  - Pattern specificity: longer, more compositional patterns are listed
    first within each behavior; they take precedence over short wildcards.
  - The segmenter is math-aware (reuses sentence_segmenter.py) to avoid
    splitting sentences inside LaTeX expressions.

Usage:
    from src.parsing.rule_based_parser import parse_trace, print_annotated_trace

    episodes = parse_trace("Let me compute the integral... Wait, that's wrong.")
    print_annotated_trace(episodes)
    # F: "Let me compute the integral..."
    # X: "Wait, that's wrong."

    # Batch processing
    from src.parsing.rule_based_parser import parse_trace_file
    parse_trace_file("data/traces/math500_qwen7b_traces.jsonl",
                     "data/parsed/math500_qwen7b_rbp.jsonl")
"""

import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.parsing.sentence_segmenter import segment_trace

logger = logging.getLogger(__name__)


# =============================================================================
# TAXONOMY
# =============================================================================

class BehaviorType(Enum):
    """
    Six cognitive behavior types for reasoning traces.

    Symbol  Name       Cognitive function
    ------  ---------  --------------------------------------------------
    F       Forward    Generate new content: claims, steps, calculations
    V       Verify     Check or confirm previous work
    X       Revise     Acknowledge an error and correct it
    R       Restart    Abandon current approach, switch strategy
    H       Hesitate   Express doubt, uncertainty, or confusion
    C       Conclude   State the final answer / solution
    """
    FORWARD   = "F"
    VERIFY    = "V"
    REVISE    = "X"
    RESTART   = "R"
    HESITATE  = "H"
    CONCLUDE  = "C"

    @property
    def display_name(self) -> str:
        return {
            "F": "Forward",
            "V": "Verify",
            "X": "Revise",
            "R": "Restart",
            "H": "Hesitate",
            "C": "Conclude",
        }[self.value]

    @property
    def priority(self) -> int:
        """
        Conflict resolution priority (higher = wins).

        X > R > V > C > H > F

        Rationale:
          Revise (error correction) is the strongest, most specific signal —
          never override it with weaker patterns. Forward is the catch-all
          default and should only win when nothing else matches.
        """
        return {
            "X": 6,   # Revise: strongest, most specific
            "R": 5,   # Restart: explicit approach abandonment
            "V": 4,   # Verify: explicit self-checking language
            "C": 3,   # Conclude: answer-stating language
            "H": 2,   # Hesitate: uncertainty markers
            "F": 1,   # Forward: default
        }[self.value]


@dataclass
class Episode:
    """A single classified episode from a reasoning trace."""
    text: str                             # Raw sentence text
    behavior: BehaviorType                # Classified behavior
    position: int                         # 0-indexed position in sequence
    token_count: int                      # Approximate word count
    confidence: float                     # 1.0 = unambiguous match, 0.5 = default
    matched_pattern: Optional[str] = None # Pattern string that triggered this label

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "behavior": self.behavior.value,
            "behavior_name": self.behavior.display_name,
            "position": self.position,
            "token_count": self.token_count,
            "confidence": self.confidence,
            "matched_pattern": self.matched_pattern,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Episode":
        return cls(
            text=d["text"],
            behavior=BehaviorType(d["behavior"]),
            position=d["position"],
            token_count=d["token_count"],
            confidence=d.get("confidence", 1.0),
            matched_pattern=d.get("matched_pattern"),
        )


# =============================================================================
# PATTERNS
# =============================================================================
# Each list is ordered by specificity (most specific first).
# re.IGNORECASE is applied at match time — patterns are lowercase-safe.

PATTERNS: dict[BehaviorType, list[str]] = {

    # -------------------------------------------------------------------------
    # REVISE — error acknowledgment + correction
    # -------------------------------------------------------------------------
    BehaviorType.REVISE: [
        # Multi-word: strongest signals
        r"(?:no|wait|oh)[,.]?\s*that(?:'s|\s+is)\s+(?:wrong|incorrect|not\s+right|not\s+correct)",
        r"\bi\s+(?:made|have|had)\s+(?:a\s+|an\s+)?(?:error|mistake|miscalculation|blunder)",
        r"\bthat\s+(?:was|is)\s+(?:a\s+)?(?:mistake|error|wrong|incorrect)\b",
        r"\blet\s+me\s+(?:reconsider|rethink|re-?examine|re-?do|redo)\b",
        r"\bgoing\s+back\s+to\b",
        r"\bi\s+(?:was|am|had\s+been)\s+wrong\b",
        r"\bthat\s+(?:doesn't|does\s+not|won't|will\s+not)\s+(?:work|hold|make\s+sense|be\s+right|be\s+correct)",
        r"\bthis\s+(?:is|seems|appears)\s+(?:wrong|incorrect|off|invalid)\b",
        r"\bcorrection\s*[:—]",
        r"\bi\s+(?:need\s+to\s+)?(?:fix|correct)\s+(?:this|that|my\s+(?:error|mistake|calculation))\b",
        r"\bmy\s+(?:earlier|previous|prior)\s+(?:calculation|answer|reasoning|step)\s+(?:was|is)\s+(?:wrong|incorrect|off)\b",

        # "wait" + corrective context (medium confidence)
        r"\bwait\b.{0,60}\b(?:wrong|mistake|error|incorrect|no\b)",
        r"\bwait\b.{0,60}\b(?:actually|should\s+be|instead|but)\b",

        # standalone sentence-initial markers (lower confidence; must be
        # at the start so the leading ^ anchors help reduce false positives)
        r"^actually[,.]",
        r"^wait[,.!]?\s*(?!let\s+me\s+(?:check|verify|think))",
        r"^no[,.!]\s",
        r"^oops\b",
    ],

    # -------------------------------------------------------------------------
    # RESTART — approach abandonment / strategy switch
    # -------------------------------------------------------------------------
    BehaviorType.RESTART: [
        r"\blet\s+me\s+try\s+(?:a\s+)?(?:different|another|new|completely\s+different)\s+(?:approach|method|way|strategy|angle)",
        r"\blet(?:'s|\s+me)\s+(?:take|use)\s+(?:a\s+)?(?:different|another|new)\s+(?:approach|path|route|tack)",
        r"\bstart(?:ing)?\s+(?:over|from\s+scratch|from\s+the\s+beginning|fresh|again)\b",
        r"\bscrap(?:ping|ped)?\s+(?:this|that|everything\s+above)\b",
        r"\babandon(?:ing|ed)?\s+(?:this|that|the\s+current)\s+(?:approach|method|idea|strategy)\b",
        r"\binstead[,.]?\s+(?:let\s+me|i(?:'ll|\s+will|'d|'ve)|we\s+(?:can|could|should))\b",
        r"\balternatively[,.]?\s+(?:i|we|let(?:'s|\s+me))\b",
        r"\banother\s+(?:way|approach|method|strategy)\s+(?:to|would\s+be|is|might\s+be)\b",
        r"\bapproach(?:ing)?\s+(?:this|the\s+problem)\s+differently\b",
        r"\blet(?:'s|\s+me)\s+(?:try|consider)\s+(?:something\s+else|a\s+different|an\s+alternative)\b",
        r"\bpivoting?\s+to\b",
        r"\bswitching?\s+(?:to|approach|strategy|method)\b",
    ],

    # -------------------------------------------------------------------------
    # VERIFY — self-checking, confirmation
    # -------------------------------------------------------------------------
    BehaviorType.VERIFY: [
        # Explicit verification phrases
        r"\blet\s+me\s+(?:check|verify|confirm|validate|double[\s-]?check|make\s+sure)\b",
        r"\bto\s+(?:verify|confirm|check|validate|be\s+sure)\b",
        r"\bdouble[\s-]?check(?:ing|ed)?\b",
        r"\bsanity[\s-]?check\b",
        r"\bchecking\s+(?:this|that|my|our|the)\b",
        r"\bverif(?:y|ying|ied|ication)\b",
        r"\bconfirm(?:ing|ed|ation)?\s+(?:this|that|the|my)\b",
        r"\bis\s+(?:this|that)(?:\s+\w+)?\s+(?:correct|right|true|valid|consistent)\s*\?",

        # Plugging back in / substitution checks
        r"\bsubstitut(?:e|ing|ed)\s+(?:back|this|that)\s+(?:into|in|to\s+check)\b",
        r"\bplug(?:ging|ged)?\s+(?:back\s+)?(?:this|it|that|x\s*=|the\s+value)\s+(?:back\s+)?(?:in|into)\b",
        r"\bcheck(?:ing|ed)?\s+by\s+(?:substitut|plug|comput|calculat)\w+\b",

        # Reviewing / re-reading
        r"\blet\s+me\s+(?:re-?read|review|look\s+(?:at|over)\s+(?:this|that|my\s+work))\b",
        r"\breviewing?\s+(?:my|the)\s+(?:work|steps|calculation|answer)\b",

        # "wait" + verification (medium confidence)
        r"\bwait\b.{0,40}\blet\s+me\s+(?:check|verify|confirm)\b",

        # Rhetorical self-questions about correctness
        r"\bdoes\s+(?:this|that)\s+(?:make\s+sense|check\s+out|add\s+up|work)\s*\?",
        r"\bare\s+(?:these|those|my)\s+(?:values|numbers|answers|steps)\s+correct\s*\?",
    ],

    # -------------------------------------------------------------------------
    # CONCLUDE — final answer statement
    # -------------------------------------------------------------------------
    BehaviorType.CONCLUDE: [
        r"\\boxed\{",                                          # LaTeX boxed answer
        r"\bthe\s+(?:final\s+)?answer\s+is\b",
        r"\btherefore[,.]?\s+(?:the\s+)?(?:answer|solution|result|value)\b",
        r"\bso\s+(?:the\s+)?(?:final\s+)?(?:answer|solution|result)\b",
        r"\bthus[,.]?\s+(?:the\s+)?(?:answer|solution|result)\b",
        r"\bhence[,.]?\s+(?:the\s+)?(?:answer|solution|result)\b",
        r"\bin\s+conclusion[,.]?\s",
        r"\bour\s+(?:final\s+)?answer\s+is\b",
        r"\bfinal\s+answer\s*[:=]",
        r"\bthe\s+solution\s+(?:is|to\s+this\s+problem\s+is)\b",
        r"\bwe\s+(?:get|obtain|have|conclude)\s+(?:that\s+)?(?:the\s+)?(?:answer|solution)\b",
        r"\bto\s+summarize[,.]?\s",
    ],

    # -------------------------------------------------------------------------
    # HESITATE — uncertainty, doubt, confusion
    # -------------------------------------------------------------------------
    BehaviorType.HESITATE: [
        # Onomatopoeia / filler sounds
        r"\bhmm+\b",
        r"\buhh*\b",
        r"\bumm*\b",
        r"\bhuh\b",

        # Explicit uncertainty
        r"\bi(?:'m|\s+am)\s+not\s+(?:sure|certain|confident|quite\s+sure)\b",
        r"\bnot\s+(?:sure|certain)\s+(?:if|whether|about|how|what|why)\b",
        r"\bi(?:'m|\s+am)\s+(?:confused|unsure|uncertain|unclear)\b",
        r"\bi\s+(?:don't|do\s+not)\s+(?:know|understand|see)\s+(?:how|why|what|if|whether)\b",

        # Difficulty markers
        r"\bthis\s+(?:is|seems)\s+(?:tricky|confusing|difficult|complicated|hard|unclear)\b",
        r"\bi\s+(?:find\s+)?(?:this|it)\s+(?:confusing|unclear|hard\s+to)\b",

        # Wondering / musing
        r"\bi\s+wonder\s+(?:if|whether|how|what|why)\b",
        r"\bmaybe\b(?!.*\b(?:approach|method|way|instead)\b)",  # "maybe" not in approach context
        r"^perhaps\b",

        # Thinking aloud (tentative)
        r"\bi\s+(?:think|guess|suppose|believe)\s+(?:so\b|that\s+might|this\s+might|it\s+(?:might|could|may))\b",
        r"\bthis\s+(?:might|may|could)\s+(?:be|work|help)\b",
    ],
}


# =============================================================================
# SENTENCE-LEVEL CLASSIFIER
# =============================================================================

def classify_sentence(
    sentence: str,
) -> tuple[BehaviorType, float, Optional[str]]:
    """
    Classify one sentence into a behavior type.

    Tries all patterns for every behavior type.  When multiple behaviors
    match, the highest-priority one wins (Revise > Restart > Verify >
    Conclude > Hesitate > Forward).

    Args:
        sentence: Raw sentence text.

    Returns:
        (behavior, confidence, matched_pattern) where:
          - confidence=1.0 for unambiguous single match
          - confidence=0.7 for ambiguous (multiple behaviors matched)
          - confidence=0.5 for default Forward (no pattern matched)
    """
    text_lower = sentence.lower().strip()
    if not text_lower:
        return BehaviorType.FORWARD, 0.0, None

    matches: list[tuple[BehaviorType, str]] = []

    for behavior, pats in PATTERNS.items():
        for pat in pats:
            if re.search(pat, text_lower):
                matches.append((behavior, pat))
                break  # Only first (most specific) pattern per behavior

    if not matches:
        return BehaviorType.FORWARD, 0.5, None

    # Resolve by priority
    matches.sort(key=lambda x: x[0].priority, reverse=True)
    winner, winning_pattern = matches[0]

    confidence = 1.0 if len(matches) == 1 else 0.7
    return winner, confidence, winning_pattern


# =============================================================================
# TRACE-LEVEL PARSER
# =============================================================================

def parse_trace(
    trace: str,
    min_sentence_words: int = 4,
    merge_short: bool = True,
) -> list[Episode]:
    """
    Parse a full reasoning trace into a sequence of classified episodes.

    Steps:
      1. Segment into sentences (math-aware, via sentence_segmenter.py)
      2. Classify each sentence with classify_sentence()
      3. Apply contextual post-processing rules

    Args:
        trace: Raw reasoning trace text.
        min_sentence_words: Minimum word count for standalone sentences.
        merge_short: Whether to merge short fragments into neighbors.

    Returns:
        List of Episode objects in sequence order.
    """
    if not trace or not trace.strip():
        return []

    sentences = segment_trace(
        trace,
        min_sentence_words=min_sentence_words,
        merge_short=merge_short,
    )
    if not sentences:
        return []

    episodes: list[Episode] = []
    for i, sent in enumerate(sentences):
        behavior, confidence, pattern = classify_sentence(sent)
        episodes.append(Episode(
            text=sent,
            behavior=behavior,
            position=i,
            token_count=len(sent.split()),
            confidence=confidence,
            matched_pattern=pattern,
        ))

    episodes = _postprocess(episodes)
    return episodes


def _postprocess(episodes: list[Episode]) -> list[Episode]:
    """
    Context-dependent corrections that can't be handled sentence-by-sentence.

    Rules applied in order:
      1. Last-episode reclassification — if the final episode looks like an
         answer but was labeled Forward, relabel it Conclude.
      2. Verify cluster boost — three consecutive Verify episodes raise the
         middle one's confidence (confirms it's a genuine check phase).
    """
    if not episodes:
        return episodes

    # Rule 1: last episode → Conclude if it contains answer signals
    last = episodes[-1]
    if last.behavior == BehaviorType.FORWARD:
        answer_signals = [
            r'\\boxed\{',
            r'\b(?:answer|result|solution)\b',
            r'\b\d+(?:\.\d+)?\s*$',
        ]
        for pat in answer_signals:
            if re.search(pat, last.text, re.IGNORECASE):
                episodes[-1] = Episode(
                    text=last.text,
                    behavior=BehaviorType.CONCLUDE,
                    position=last.position,
                    token_count=last.token_count,
                    confidence=0.6,
                    matched_pattern=f"ctx:last_episode:{pat}",
                )
                break

    # Rule 2: consecutive Verify cluster → boost confidence
    for i in range(1, len(episodes) - 1):
        if (episodes[i - 1].behavior == BehaviorType.VERIFY and
                episodes[i].behavior == BehaviorType.VERIFY and
                episodes[i + 1].behavior == BehaviorType.VERIFY and
                episodes[i].confidence < 1.0):
            ep = episodes[i]
            episodes[i] = Episode(
                text=ep.text,
                behavior=BehaviorType.VERIFY,
                position=ep.position,
                token_count=ep.token_count,
                confidence=min(1.0, ep.confidence + 0.2),
                matched_pattern=ep.matched_pattern,
            )

    return episodes


# =============================================================================
# SEQUENCE UTILITIES
# =============================================================================

def sequence_to_string(episodes: list[Episode]) -> str:
    """Compact behavior string, e.g. [F,F,V,X,F,C] → 'FFVXFC'."""
    return "".join(ep.behavior.value for ep in episodes)


def get_trace_summary(episodes: list[Episode]) -> dict:
    """Return high-level statistics for a parsed trace."""
    if not episodes:
        return {
            "num_episodes": 0, "behavior_sequence": "",
            "behavior_counts": {}, "behavior_proportions": {},
            "total_tokens": 0, "mean_confidence": 0.0,
        }
    seq = sequence_to_string(episodes)
    counts = Counter(ep.behavior.value for ep in episodes)
    n = len(episodes)
    return {
        "num_episodes": n,
        "behavior_sequence": seq,
        "behavior_counts": dict(counts),
        "behavior_proportions": {k: v / n for k, v in counts.items()},
        "total_tokens": sum(ep.token_count for ep in episodes),
        "mean_confidence": sum(ep.confidence for ep in episodes) / n,
        "has_revise":   "X" in seq,
        "has_restart":  "R" in seq,
        "has_verify":   "V" in seq,
        "has_conclude": "C" in seq,
        "num_behavior_changes": sum(
            1 for i in range(1, n) if episodes[i].behavior != episodes[i - 1].behavior
        ),
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

_COLORS = {
    "F": "\033[0m",    # Default
    "V": "\033[94m",   # Blue
    "X": "\033[91m",   # Red
    "R": "\033[93m",   # Yellow
    "H": "\033[95m",   # Magenta
    "C": "\033[92m",   # Green
}
_RESET = "\033[0m"


def print_annotated_trace(
    episodes: list[Episode],
    max_text: int = 90,
    show_confidence: bool = True,
) -> None:
    """Pretty-print a parsed trace with color-coded behavior labels."""
    if not episodes:
        print("  (empty trace)")
        return

    seq = sequence_to_string(episodes)
    print(f"  Sequence : {seq}")
    print(f"  Episodes : {len(episodes)}")
    print()

    for ep in episodes:
        color = _COLORS.get(ep.behavior.value, "")
        preview = ep.text[:max_text] + ("…" if len(ep.text) > max_text else "")
        conf_str = f"  conf={ep.confidence:.1f}" if show_confidence else ""
        label = f"[{ep.behavior.value}] {ep.behavior.display_name}"
        print(f"  {color}{ep.position:3d}. {label:16s}{conf_str}{_RESET}")
        print(f"       {preview}")
        print()


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def parse_trace_file(
    input_path: str,
    output_path: str,
    trace_field: str = "reasoning_trace",
) -> dict:
    """
    Parse all traces in a JSONL file and write results to a new JSONL file.

    Each output record is the original record augmented with:
      - behavior_sequence  (str)  : e.g. "FFVXFFC"
      - num_episodes       (int)
      - episode_counts     (dict) : {"F": 12, "V": 3, ...}
      - episodes           (list) : list of Episode.to_dict()
      - mean_confidence    (float)

    Returns a summary dict with corpus-level stats.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    total = 0
    empty = 0
    corpus_counts: Counter = Counter()

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            trace = record.get(trace_field, "") or ""

            episodes = parse_trace(trace)
            seq = sequence_to_string(episodes)
            ep_counts = Counter(ep.behavior.value for ep in episodes)

            record["behavior_sequence"] = seq
            record["num_episodes"] = len(episodes)
            record["episode_counts"] = dict(ep_counts)
            record["episodes"] = [ep.to_dict() for ep in episodes]
            record["mean_confidence"] = (
                sum(ep.confidence for ep in episodes) / len(episodes)
                if episodes else 0.0
            )

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            total += 1
            if not episodes:
                empty += 1
            corpus_counts.update(ep_counts)

            if total % 200 == 0:
                logger.info(f"  Parsed {total} traces…")

    total_eps = sum(corpus_counts.values())
    logger.info(f"Parsing complete: {total} traces, {total_eps} episodes")
    if empty:
        logger.warning(f"  {empty} traces produced no episodes")

    summary = {
        "total_traces": total,
        "empty_traces": empty,
        "total_episodes": total_eps,
        "behavior_counts": dict(corpus_counts),
        "behavior_proportions": {
            k: v / max(total_eps, 1) for k, v in corpus_counts.items()
        },
    }
    return summary


# =============================================================================
# SELF-TESTS
# =============================================================================

def _run_tests() -> bool:
    print("Running rule_based_parser self-tests…")

    cases = [
        # Sentence, expected BehaviorType, description
        # --- REVISE ---
        ("Wait, that's wrong. Let me reconsider.",         BehaviorType.REVISE,   "explicit error + reconsider"),
        ("No, I made an error in step 2.",                 BehaviorType.REVISE,   "explicit mistake"),
        ("Actually, the formula should be different.",     BehaviorType.REVISE,   "sentence-initial 'actually'"),
        ("That doesn't work out correctly.",               BehaviorType.REVISE,   "that doesn't work"),
        ("I was wrong about the sign.",                    BehaviorType.REVISE,   "I was wrong"),
        ("My previous calculation was incorrect.",        BehaviorType.REVISE,   "previous was incorrect"),

        # --- RESTART ---
        ("Let me try a completely different approach.",    BehaviorType.RESTART,  "try different approach"),
        ("Starting over from scratch.",                    BehaviorType.RESTART,  "starting over"),
        ("Alternatively, I could use integration.",        BehaviorType.RESTART,  "alternatively"),
        ("Instead, let me consider the dual problem.",     BehaviorType.RESTART,  "instead let me"),
        ("Switching to a geometric approach.",             BehaviorType.RESTART,  "switching to"),

        # --- VERIFY ---
        ("Let me check this result.",                      BehaviorType.VERIFY,   "let me check"),
        ("Let me verify by substituting back.",            BehaviorType.VERIFY,   "verify substituting"),
        ("Double-checking the arithmetic now.",            BehaviorType.VERIFY,   "double-checking"),
        ("Is this answer correct?",                        BehaviorType.VERIFY,   "is this correct?"),
        ("To verify, let me plug x = 3 back in.",         BehaviorType.VERIFY,   "to verify"),
        ("Does this make sense?",                          BehaviorType.VERIFY,   "does this make sense"),

        # --- CONCLUDE ---
        ("Therefore, the answer is 42.",                   BehaviorType.CONCLUDE, "therefore answer"),
        ("So the final answer is \\boxed{42}.",            BehaviorType.CONCLUDE, "boxed answer"),
        ("In conclusion, x equals 7.",                     BehaviorType.CONCLUDE, "in conclusion"),
        ("Thus, the solution is x = 5.",                   BehaviorType.CONCLUDE, "thus solution"),
        ("Hence the answer is 12.",                        BehaviorType.CONCLUDE, "hence answer"),

        # --- HESITATE ---
        ("Hmm, this is quite tricky.",                     BehaviorType.HESITATE, "hmm tricky"),
        ("I'm not sure whether this is correct.",          BehaviorType.HESITATE, "not sure whether"),
        ("I wonder if there's a simpler way.",             BehaviorType.HESITATE, "I wonder if"),
        ("I'm confused about the sign convention here.",   BehaviorType.HESITATE, "I'm confused"),
        ("Uh, let me think about this for a moment.",      BehaviorType.HESITATE, "uh filler"),

        # --- FORWARD (default) ---
        ("We can write f(x) = 3x^2 + 2x - 1.",           BehaviorType.FORWARD,  "simple derivation"),
        ("Applying the quadratic formula gives us roots.", BehaviorType.FORWARD,  "quadratic formula"),
        ("The eigenvalues of A are 1, 2, and 3.",         BehaviorType.FORWARD,  "eigenvalue statement"),
    ]

    passed = failed = 0
    for sentence, expected, desc in cases:
        behavior, conf, pattern = classify_sentence(sentence)
        if behavior == expected:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL [{desc}]: '{sentence[:55]}…'")
            print(f"        Expected {expected.display_name}, got {behavior.display_name}")
            print(f"        Pattern: {pattern}")

    print(f"\n  {passed} passed, {failed} failed out of {len(cases)} tests")

    # --- Integration: parse a realistic multi-behavior trace ---
    sample = """First, I need to find the area of a right triangle with legs 3 and 4.

For a right triangle, the area is (base × height) / 2 = (3 × 4) / 2 = 6.

Wait, that's wrong. I forgot to check whether it's actually a right triangle.

Since 3² + 4² = 9 + 16 = 25 = 5², yes it is a right triangle.

Let me verify: area = (3 × 4) / 2 = 12 / 2 = 6.

Let me try a different method: using Heron's formula just to double check.

Hmm, this gets complicated but s = (3+4+5)/2 = 6, area = sqrt(6·3·2·1) = 6.

Therefore, the answer is \\boxed{6}."""

    episodes = parse_trace(sample)
    seq = sequence_to_string(episodes)
    must_contain = {"X", "V", "R", "H", "C"}
    missing = must_contain - set(seq)
    if missing:
        failed += 1
        print(f"  FAIL [integration]: sequence '{seq}' missing behaviors {missing}")
    else:
        passed += 1
        print(f"  PASS [integration]: sequence '{seq}'")

    print()
    if failed == 0:
        print("  All tests passed.")
    else:
        print(f"  WARNING: {failed} test(s) failed.")

    return failed == 0


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ok = _run_tests()

    if ok:
        print("\n" + "=" * 65)
        print("Demo: Annotated Trace")
        print("=" * 65)
        demo = """I need to compute the sum of squares of the first 5 positive integers.

The sum is 1² + 2² + 3² + 4² + 5² = 1 + 4 + 9 + 16 + 25.

Let me compute: 1 + 4 = 5, 5 + 9 = 14, 14 + 16 = 30, 30 + 25 = 55.

Wait, that's wrong. I was computing the sum of integers, not squares.

Actually, let me redo this. 1 + 4 + 9 + 16 + 25 = 55.

Hmm, but wait — is 1² + 2² + ... + n² = n(n+1)(2n+1)/6?

For n = 5: 5 × 6 × 11 / 6 = 55. Let me verify that.

To verify: 5 × 6 = 30, 30 × 11 = 330, 330 / 6 = 55. Correct.

Therefore, the answer is \\boxed{55}."""
        eps = parse_trace(demo)
        print_annotated_trace(eps)
        print("Summary:", get_trace_summary(eps))

    sys.exit(0 if ok else 1)
