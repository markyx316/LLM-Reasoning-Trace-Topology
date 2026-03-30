"""
sentence_segmenter.py - Math-aware sentence segmentation for reasoning traces.

Standard sentence tokenizers (spaCy, NLTK) break on periods inside
mathematical expressions like "f(x) = 3.14" or "Eq. (1)". This module
uses a protect-segment-restore strategy:

  1. PROTECT: Replace mathematical expressions with placeholders that
     won't trigger sentence boundaries.
  2. SEGMENT: Apply sentence tokenization on the protected text.
  3. RESTORE: Replace placeholders with original expressions.

Additional handling:
  - Merge very short fragments (< 5 words) with adjacent sentences
  - Split on double newlines (paragraph breaks in reasoning traces)
  - Preserve LaTeX equations as part of their surrounding sentence

Design decisions:
  - We use regex-based segmentation rather than spaCy to avoid the
    dependency and because spaCy's sentence boundary detection is
    unreliable on mathematical text even with protections.
  - The segmenter is deliberately conservative: it's better to
    under-segment (keep related text together) than over-segment
    (split a sentence in the middle of a formula).
"""

import re
from typing import Optional


# =============================================================================
# PROTECTION PATTERNS
# =============================================================================

# Things that contain periods but are NOT sentence boundaries
PROTECTION_PATTERNS = [
    # LaTeX display math: \[...\] or $$...$$
    (r'\\\[.*?\\\]', 'DISPLAY_MATH'),
    (r'\$\$.*?\$\$', 'DISPLAY_MATH'),

    # LaTeX inline math: \(...\) or $...$
    (r'\\\(.*?\\\)', 'INLINE_MATH'),
    (r'\$[^$\n]+?\$', 'INLINE_MATH'),

    # Common LaTeX commands with dots: \cdots, \ldots, \dots
    (r'\\(?:cdots|ldots|dots|vdots|ddots)', 'LATEX_DOTS'),

    # Ellipsis: ... (must be before decimal protection)
    (r'\.{2,}', 'ELLIPSIS'),

    # Decimal numbers: 3.14, -0.5, 1.23e-4
    (r'-?\d+\.\d+(?:[eE][+-]?\d+)?', 'DECIMAL'),

    # Common abbreviations with periods
    (r'\b(?:e\.g|i\.e|etc|vs|Fig|Eq|eq|Thm|Def|Prop|Cor|Lem|Rem|Ex|No|Dr|Mr|Mrs|Ms|St|Jr|Sr)\.',
     'ABBREV'),

    # Function notation: f(x), g(x, y), P(A|B)
    (r'\b[a-zA-Z]\([^)]{1,30}\)', 'FUNCTION'),

    # Version numbers: v2.1, Python 3.10
    (r'\bv?\d+\.\d+(?:\.\d+)*', 'VERSION'),

    # URLs (rare in reasoning traces but possible)
    (r'https?://\S+', 'URL'),
]


def _protect_math(text: str) -> tuple[str, dict[str, str]]:
    """
    Replace mathematical expressions with placeholders.

    Returns:
        (protected_text, placeholder_map) where placeholder_map maps
        placeholder strings back to the original expressions.
    """
    placeholders = {}
    protected = text
    counter = 0

    for pattern, label in PROTECTION_PATTERNS:
        for match in re.finditer(pattern, protected, re.DOTALL):
            placeholder = f"__PROT_{label}_{counter}__"
            placeholders[placeholder] = match.group()
            protected = protected.replace(match.group(), placeholder, 1)
            counter += 1

    return protected, placeholders


def _restore_math(sentences: list[str], placeholders: dict[str, str]) -> list[str]:
    """Restore placeholders with original mathematical expressions."""
    restored = []
    for sent in sentences:
        for placeholder, original in placeholders.items():
            sent = sent.replace(placeholder, original)
        restored.append(sent)
    return restored


# =============================================================================
# SENTENCE SEGMENTATION
# =============================================================================

def segment_trace(
    trace: str,
    min_sentence_words: int = 4,
    merge_short: bool = True,
) -> list[str]:
    """
    Segment a reasoning trace into sentences.

    This is the main entry point for sentence segmentation. It handles
    mathematical notation, LaTeX expressions, and the specific formatting
    patterns found in DeepSeek-R1 reasoning traces.

    Args:
        trace: The reasoning trace text (contents of <think> block).
        min_sentence_words: Minimum words for a standalone sentence.
            Shorter fragments are merged with adjacent sentences.
        merge_short: Whether to merge very short fragments.

    Returns:
        List of sentence strings.
    """
    if not trace or not trace.strip():
        return []

    # Step 1: Normalize whitespace but preserve paragraph breaks
    # DeepSeek-R1 traces often use \n\n to separate reasoning phases
    text = trace.strip()

    # Step 2: Split on paragraph breaks first (double newlines)
    paragraphs = re.split(r'\n\s*\n', text)

    # We process each paragraph independently and merge short fragments
    # WITHIN paragraphs only, never across paragraph boundaries.
    all_sentences = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # Step 3: Protect mathematical expressions
        protected, placeholders = _protect_math(paragraph)

        # Step 4: Split on sentence boundaries
        # We use a regex that splits on:
        #   - Period followed by space and uppercase letter (or end of string)
        #   - Question mark followed by space
        #   - Exclamation mark followed by space
        #   - Newline (single newlines within a paragraph)
        # But NOT on:
        #   - Period inside a protected placeholder
        #   - Period in common abbreviations

        # First, replace single newlines with a sentence boundary marker
        protected = re.sub(r'\n', ' __NEWLINE__ ', protected)

        # Split on sentence-ending punctuation followed by space+uppercase
        # or end of string
        raw_sentences = re.split(
            r'(?<=[.!?])\s+(?=[A-Z__])|(?:__NEWLINE__)',
            protected
        )

        # Step 5: Restore placeholders
        raw_sentences = _restore_math(raw_sentences, placeholders)

        # Step 6: Clean up
        cleaned = []
        for s in raw_sentences:
            s = s.strip()
            s = s.replace('__NEWLINE__', '').strip()
            if s:
                cleaned.append(s)

        # Step 7: Merge very short fragments WITHIN this paragraph only
        if merge_short and min_sentence_words > 0:
            cleaned = _merge_short_fragments(cleaned, min_sentence_words)

        all_sentences.extend(cleaned)

    return all_sentences


def _merge_short_fragments(
    sentences: list[str],
    min_words: int,
) -> list[str]:
    """
    Merge very short sentence fragments with their neighbors.

    Short fragments like "Wait." or "Hmm." are kept standalone because
    they carry important behavioral signal. But fragments that are
    clearly incomplete (e.g., "So we get") are merged with the next sentence.
    """
    if len(sentences) <= 1:
        return sentences

    # Patterns that should remain standalone even if short
    # (they carry behavioral signal)
    standalone_patterns = [
        r'^\s*(?:wait|hmm+|actually|alternatively|therefore|thus|hence|so)\b',
        r'^\s*(?:let\s+me|let\'s)\b',
        r'^\s*(?:step\s+\d+)',
        r'^\s*(?:no,|yes,)',
        r'\?\s*$',  # Questions should stay standalone
    ]

    merged = []
    i = 0
    while i < len(sentences):
        sent = sentences[i]
        word_count = len(sent.split())

        if word_count < min_words:
            # Check if it should remain standalone
            is_standalone = any(
                re.search(p, sent, re.IGNORECASE)
                for p in standalone_patterns
            )

            if not is_standalone and i + 1 < len(sentences):
                # Merge with next sentence
                merged_sent = sent + " " + sentences[i + 1]
                merged.append(merged_sent)
                i += 2
                continue
            elif not is_standalone and merged:
                # Merge with previous sentence
                merged[-1] = merged[-1] + " " + sent
                i += 1
                continue

        merged.append(sent)
        i += 1

    return merged


# =============================================================================
# SELF-TEST
# =============================================================================

def run_segmenter_tests():
    """Test sentence segmentation on realistic examples."""
    print("Running segmenter self-tests...")

    tests_passed = 0
    tests_failed = 0

    def check(name, text, min_expected, max_expected=None):
        nonlocal tests_passed, tests_failed
        if max_expected is None:
            max_expected = min_expected
        sentences = segment_trace(text)
        count = len(sentences)
        if min_expected <= count <= max_expected:
            tests_passed += 1
        else:
            tests_failed += 1
            print(f"  FAIL: {name}: expected {min_expected}-{max_expected} sentences, "
                  f"got {count}")
            for j, s in enumerate(sentences):
                print(f"    [{j}] {s[:80]}...")

    # --- Basic sentence splitting ---
    check("two_sentences",
          "First, let me compute the derivative. Then I will evaluate it at x=0.",
          2)

    # --- Decimal numbers should NOT cause splits ---
    check("decimal_no_split",
          "The value is 3.14 and we need to compare it with 2.72.",
          1)

    # --- LaTeX should be protected ---
    check("latex_protected",
          "We have $f(x) = x^2 + 3.5x - 1$. Setting this equal to zero gives us the roots.",
          2)

    # --- Paragraph breaks should cause splits ---
    check("paragraph_break",
          "First paragraph here.\n\nSecond paragraph here.",
          2)

    # --- Single newlines within a paragraph ---
    check("single_newline",
          "Line one of reasoning.\nLine two of reasoning.",
          2)

    # --- Short fragments with behavioral signal should stay ---
    check("wait_standalone",
          "Wait. That's not right. Let me reconsider.",
          2, 3)

    # --- Multiple sentences with math ---
    check("math_heavy",
          "Let f(x) = x^2 + 2x + 1. We can factor this as (x+1)^2. "
          "Setting f(x) = 0, we get x = -1.",
          2, 3)

    # --- Abbreviations ---
    check("abbreviation",
          "From Eq. 5 we can see that the result follows. This proves the claim.",
          2)

    # --- Empty input ---
    check("empty", "", 0)

    # --- Ellipsis ---
    check("ellipsis",
          "So we have 1, 2, 3, ... and this continues. The pattern is clear.",
          2)

    print(f"\nResults: {tests_passed} passed, {tests_failed} failed "
          f"out of {tests_passed + tests_failed} tests")

    if tests_failed == 0:
        print("All segmenter tests passed.")
    else:
        print("WARNING: Some segmenter tests failed!")

    return tests_failed == 0


if __name__ == "__main__":
    run_segmenter_tests()
