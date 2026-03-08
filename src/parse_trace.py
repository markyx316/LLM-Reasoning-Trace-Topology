"""
parse_trace.py — Convert a raw reasoning trace into a behavior sequence.

Behavior taxonomy (from proposal Section 3.4):
  F = Forward reasoning (default)
  V = Verification
  B = Backtracking
  R = Restart / Abandonment
  S = Sub-goal decomposition
  H = Hesitation / Rumination
  C = Conclusion

Usage:
  from parse_trace import parse_trace
  seq = parse_trace("Let me first find the derivative... wait, that's wrong. ...")
  # Returns: ['S', 'F', 'B', 'F', ...]
"""

import re
from dataclasses import dataclass, field

# ─── Pattern definitions (priority order: higher index = lower priority) ──────

# Each entry: (behavior_symbol, list_of_regex_patterns)
# Patterns are applied to lowercased sentences. First match wins.
BEHAVIOR_PATTERNS: list[tuple[str, list[str]]] = [
    ("C", [
        r"\bthe (?:final )?answer is\b",
        r"\btherefore[,.]",
        r"\bin conclusion\b",
        r"\bthus[,.]",
        r"\bhence[,.]",
        r"\\boxed\{",
        r"\bso the answer\b",
        r"\bfinal answer\b",
    ]),
    ("B", [
        r"\bwait[,.]?\s",
        r"\bwait,?\s+(?:no|actually|that|let me|i )",
        r"\bactually[,.]",
        r"\bno,? that'?s? (?:wrong|incorrect|not right|a mistake)\b",
        r"\bi (?:made|made an|have a) (?:mistake|error|miscalculation)\b",
        r"\blet me (?:reconsider|re-?think|re-?examine|correct)\b",
        r"\bthat'?s? (?:not right|incorrect|wrong)\b",
        r"\bi (?:was wrong|made an error|need to redo)\b",
        r"\bhmm,?\s+(?:wait|actually|no)\b",
    ]),
    ("R", [
        r"\blet me try (?:a )?(?:different|another) approach\b",
        r"\bstarting over\b",
        r"\blet me start (?:over|again|fresh)\b",
        r"\balternatively[,.]",
        r"\banother (?:way|approach|method)\b",
        r"\binstead[,.]",
        r"\blet me (?:use|try|consider) a different\b",
    ]),
    ("V", [
        r"\blet me (?:check|verify|confirm|double.?check)\b",
        r"\bto (?:verify|confirm|check)\b",
        r"\bverif(?:y|ying|ied)\b",
        r"\blet'?s? (?:verify|check|confirm)\b",
        r"\bcheck(?:ing)? (?:this|that|my|the)\b",
        r"\bis this correct\b",
        r"\bdoes this (?:make sense|check out|work)\b",
        r"\bsanity check\b",
        r"\bplugging (?:in|back)\b",
        r"\bsubstitut(?:e|ing) back\b",
    ]),
    ("H", [
        r"^hmm[.,]?\s*$",
        r"^hmm[,.]?\s",
        r"\bi'?m not sure\b",
        r"\bthis is (?:tricky|confusing|unclear|complicated)\b",
        r"\bi (?:think|wonder|feel) (?:i'm|i am)?\s*(?:not sure|confused|unsure)\b",
        r"\bnot (?:quite|entirely|really) sure\b",
        r"\bthis seems? (?:off|wrong|strange|odd)\b",
    ]),
    ("S", [
        r"\bfirst[,.]?\s+(?:i (?:need|will|should)|let'?s?|we)\b",
        r"\bstep \d+[.:]",
        r"\bpart \d+[.:]",
        r"\bbreaking (?:this|it) down\b",
        r"\bthe key (?:insight|observation|idea) is\b",
        r"\bi (?:need|will|should) (?:find|compute|calculate|determine|solve)\b",
        r"\blet'?s? (?:first|start by|begin by|define|set up)\b",
        r"\bto (?:solve|find|compute) this[,.]",
        r"\bmy (?:plan|approach|strategy) is\b",
    ]),
]

# Compiled patterns for speed
_COMPILED: list[tuple[str, list[re.Pattern]]] = [
    (sym, [re.compile(p, re.IGNORECASE) for p in patterns])
    for sym, patterns in BEHAVIOR_PATTERNS
]


# ─── Sentence segmentation ───────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """
    Lightweight sentence splitter that does not require spaCy.
    Splits on sentence-ending punctuation followed by whitespace + capital letter.
    Also splits on newlines (common in reasoning traces).
    """
    # Split on newlines first
    lines = re.split(r"\n+", text)
    sentences = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Split on '. ', '! ', '? ' followed by uppercase
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'(])", line)
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences


def _classify_sentence(sentence: str) -> str:
    """Return behavior symbol for a single sentence."""
    s = sentence.lower()
    for sym, patterns in _COMPILED:
        for pat in patterns:
            if pat.search(s):
                return sym
    return "F"  # default: forward reasoning


# ─── Public API ──────────────────────────────────────────────────────────────

@dataclass
class ParsedTrace:
    sentences: list[str]
    behaviors: list[str]   # parallel list of behavior symbols

    @property
    def sequence(self) -> list[str]:
        return self.behaviors

    def __repr__(self) -> str:
        return f"ParsedTrace(n={len(self.behaviors)}, seq={''.join(self.behaviors)})"


def parse_trace(trace_text: str) -> ParsedTrace:
    """
    Parse a reasoning trace into a behavior sequence.

    Args:
        trace_text: The raw text of the <think> block.

    Returns:
        ParsedTrace with .sentences and .behaviors lists.
    """
    sentences = _split_sentences(trace_text)
    behaviors = [_classify_sentence(s) for s in sentences]
    return ParsedTrace(sentences=sentences, behaviors=behaviors)


# ─── Batch processing (for use in feature extraction) ────────────────────────

def parse_traces_from_jsonl(jsonl_path: str) -> list[dict]:
    """
    Load a JSONL trace file and parse each trace.
    Returns list of dicts with original record + 'parsed' key.
    """
    import json
    records = []
    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            record["parsed"] = parse_trace(record["trace"])
            records.append(record)
    return records


# ─── Validation helper ────────────────────────────────────────────────────────

def compute_parser_agreement(manual_annotations: list[dict], parsed_traces: list[ParsedTrace]) -> float:
    """
    Compute sentence-level accuracy between parser and manual annotations.

    manual_annotations: list of {"sentences": [...], "labels": [...]}
    parsed_traces: corresponding ParsedTrace objects
    """
    total = 0
    correct = 0
    for ann, parsed in zip(manual_annotations, parsed_traces):
        for gt_label, pred_label in zip(ann["labels"], parsed.behaviors):
            total += 1
            if gt_label == pred_label:
                correct += 1
    return correct / total if total > 0 else 0.0


# ─── CLI: parse a single trace for inspection ────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python parse_trace.py <trace_text_or_jsonl_path>")
        sys.exit(1)

    arg = sys.argv[1]
    if arg.endswith(".jsonl"):
        records = parse_traces_from_jsonl(arg)
        for r in records[:3]:
            print(f"\n=== {r['id']} ({'CORRECT' if r['correct'] else 'wrong'}) ===")
            p = r["parsed"]
            for sent, beh in zip(p.sentences[:10], p.behaviors[:10]):
                print(f"  [{beh}] {sent[:80]}")
            print(f"  ... sequence: {''.join(p.sequence)}")
    else:
        parsed = parse_trace(arg)
        for sent, beh in zip(parsed.sentences, parsed.behaviors):
            print(f"[{beh}] {sent}")
        print(f"\nSequence: {''.join(parsed.sequence)}")
