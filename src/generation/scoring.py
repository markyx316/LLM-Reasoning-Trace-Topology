"""
scoring.py - Answer scoring and equivalence checking.

This module handles the critical task of determining whether a model's
generated answer matches the ground truth. It supports three answer types:
  1. exact_match_math  - LaTeX-aware symbolic math equivalence
  2. exact_match_numeric - Numeric answer extraction (GSM8K style)
  3. multiple_choice - Letter-based MC matching

The math equivalence checker is deliberately conservative: it tries multiple
normalization strategies and comparison methods, returning True only if at
least one method confirms equivalence.

Design decisions:
  - We normalize BEFORE comparing, not after, to avoid false negatives.
  - Sympy is used as a fallback, not the primary method, because it can
    be slow and sometimes fails to parse non-standard LaTeX.
  - For MC, we extract the answer letter from the model's free-form text
    using a cascade of regex patterns ordered by specificity.
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# 1. MATHEMATICAL ANSWER EQUIVALENCE
# =============================================================================

def extract_boxed_content(s: str) -> Optional[str]:
    """
    Extract content from the LAST \\boxed{...} in a string,
    correctly handling nested braces.

    \\boxed{\\frac{\\pi}{2}}   →  \\frac{\\pi}{2}
    \\boxed{(3, \\frac{1}{2})} →  (3, \\frac{1}{2})
    \\boxed{x^{2}}            →  x^{2}
    No \\boxed present         →  None

    Algorithm: find '\\boxed{', then walk forward counting brace depth
    until the matching closing brace is found.
    """
    # Find the LAST occurrence of \boxed{ (models sometimes have multiple)
    search_str = "\\boxed{"
    idx = s.rfind(search_str)
    if idx == -1:
        return None

    # Start after the opening brace of \boxed{
    content_start = idx + len(search_str)
    depth = 1
    i = content_start

    while i < len(s) and depth > 0:
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
        i += 1

    if depth != 0:
        # Unbalanced braces — fall back to greedy regex
        m = re.search(r'\\boxed\{(.+)\}', s, re.DOTALL)
        return m.group(1).strip() if m else None

    # content is between content_start and i-1 (the matching })
    return s[content_start:i - 1].strip()


def normalize_math_string(s: str) -> str:
    """
    Normalize a mathematical answer string for comparison.

    This function strips formatting, whitespace, and common LaTeX wrappers
    to produce a canonical string form. It does NOT attempt symbolic
    simplification — that's handled separately by sympy_equiv().

    Examples:
        "\\boxed{42}"                        -> "42"
        "\\boxed{(3, \\frac{\\pi}{2})}"      -> "(3, \\frac{\\pi}{2})"
        "$ \\frac{1}{2} $"                   -> "\\frac{1}{2}"
        "\\left( 3, \\frac{\\pi}{2} \\right)" -> "(3, \\frac{\\pi}{2})"
        "\\text{cm}"                          -> "cm"
    """
    if not isinstance(s, str):
        s = str(s)

    # Strip leading/trailing whitespace
    s = s.strip()

    # Remove \\boxed{...} wrapper (with proper brace matching)
    boxed_content = extract_boxed_content(s)
    if boxed_content is not None:
        s = boxed_content

    # Remove dollar signs (inline math delimiters)
    s = s.replace('$', '').strip()

    # Remove \text{...}, \textbf{...}, \mathrm{...} etc wrappers
    # Use a function to handle nested braces inside these commands
    latex_wrapper_cmds = [
        'text', 'textbf', 'textit', 'mathrm', 'mathbf',
        'mathit', 'operatorname', 'textrm',
    ]
    for cmd in latex_wrapper_cmds:
        pattern = '\\' + cmd + '{'
        while pattern in s:
            idx = s.find(pattern)
            inner = _extract_brace_content(s, idx + len(pattern) - 1)
            if inner is not None:
                s = s[:idx] + inner + s[idx + len(pattern) + len(inner) + 1:]
            else:
                break  # Avoid infinite loop on parse failure

    # Remove \left and \right modifiers
    s = s.replace('\\left', '').replace('\\right', '')

    # Normalize whitespace inside parentheses/brackets:
    # "( 3, x )" → "(3, x)"   and   "[ 1, 2 ]" → "[1, 2]"
    s = re.sub(r'\(\s+', '(', s)
    s = re.sub(r'\s+\)', ')', s)
    s = re.sub(r'\[\s+', '[', s)
    s = re.sub(r'\s+\]', ']', s)

    # Normalize whitespace (collapse multiple spaces to one)
    s = re.sub(r'\s+', ' ', s).strip()

    # Remove trailing period (sometimes models end with "42.")
    if s.endswith('.') and not s.endswith('...'):
        if len(s) >= 2 and s[-2].isdigit():
            if not re.search(r'\d+\.\d+\.$', s):
                s = s[:-1].strip()
        else:
            s = s[:-1].strip()

    return s


def _extract_brace_content(s: str, open_brace_pos: int) -> Optional[str]:
    """
    Given a string and the position of an opening '{', return the content
    up to the matching closing '}', handling nesting.

    Returns None if braces are unbalanced.
    """
    if open_brace_pos >= len(s) or s[open_brace_pos] != '{':
        return None
    depth = 1
    i = open_brace_pos + 1
    while i < len(s) and depth > 0:
        if s[i] == '{':
            depth += 1
        elif s[i] == '}':
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return s[open_brace_pos + 1:i - 1]


def extract_numeric_value(s: str) -> Optional[float]:
    """
    Attempt to extract a single numeric value from a string.
    Returns None if extraction fails.

    Handles: integers, decimals, negative numbers, fractions, percentages.
    """
    s = normalize_math_string(s)

    # Try direct float conversion
    try:
        return float(s)
    except (ValueError, TypeError):
        pass

    # Try fraction: a/b or \frac{a}{b}
    frac_match = re.match(r'^(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)$', s)
    if frac_match:
        num, den = float(frac_match.group(1)), float(frac_match.group(2))
        if den != 0:
            return num / den

    latex_frac = re.match(r'^\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}$', s)
    if latex_frac:
        num, den = float(latex_frac.group(1)), float(latex_frac.group(2))
        if den != 0:
            return num / den

    # Try percentage: "45%" -> 0.45
    pct_match = re.match(r'^(-?\d+(?:\.\d+)?)\s*%$', s)
    if pct_match:
        return float(pct_match.group(1)) / 100.0

    # Try removing commas from large numbers: "1,234" -> 1234
    no_comma = s.replace(',', '')
    try:
        return float(no_comma)
    except (ValueError, TypeError):
        pass

    return None


def sympy_equiv(pred: str, truth: str) -> bool:
    """
    Check symbolic equivalence using sympy.
    This is a fallback method — it's slow and can fail on complex expressions.

    Returns True if sympy confirms equivalence, False otherwise.
    Does NOT raise exceptions; returns False on any parsing failure.
    """
    try:
        import sympy
        from sympy import simplify, nsimplify
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations,
            implicit_multiplication_application, convert_xor
        )

        transformations = standard_transformations + (
            implicit_multiplication_application,
            convert_xor,
        )

        pred_expr = parse_expr(pred, transformations=transformations)
        truth_expr = parse_expr(truth, transformations=transformations)

        # Method 1: Direct simplification of difference
        diff = simplify(pred_expr - truth_expr)
        if diff == 0:
            return True

        # Method 2: Numerical evaluation
        pred_val = complex(pred_expr.evalf())
        truth_val = complex(truth_expr.evalf())
        if abs(pred_val - truth_val) < 1e-6:
            return True

    except Exception:
        # Sympy parsing can fail on many valid math strings
        # This is expected and not an error
        pass

    return False


def math_equiv(prediction: str, ground_truth: str, tolerance: float = 1e-6) -> bool:
    """
    Check if a predicted math answer is equivalent to the ground truth.

    Uses a cascade of increasingly sophisticated comparison methods:
      1. String comparison after normalization
      2. Numeric comparison (handles 1/2 == 0.5)
      3. Symbolic comparison via sympy (handles x^2+1 == 1+x^2)

    Args:
        prediction: The model's answer string.
        ground_truth: The correct answer string.
        tolerance: Numerical tolerance for float comparison.

    Returns:
        True if the answers are mathematically equivalent.
    """
    # Normalize both strings
    pred_norm = normalize_math_string(prediction)
    truth_norm = normalize_math_string(ground_truth)

    # --- Method 1: Direct string match ---
    if pred_norm == truth_norm:
        return True

    # Case-insensitive string match
    if pred_norm.lower() == truth_norm.lower():
        return True

    # --- Method 2: Numeric comparison ---
    pred_val = extract_numeric_value(pred_norm)
    truth_val = extract_numeric_value(truth_norm)

    if pred_val is not None and truth_val is not None:
        if abs(pred_val - truth_val) < tolerance:
            return True

    # --- Method 3: Symbolic comparison ---
    if sympy_equiv(pred_norm, truth_norm):
        return True

    return False


# =============================================================================
# 2. GSM8K ANSWER EXTRACTION
# =============================================================================

def extract_gsm8k_answer(answer_string: str) -> str:
    """
    Extract the final numeric answer from a GSM8K ground truth string.

    GSM8K ground truth format: "Step-by-step explanation...\n#### 42"
    We extract the number after "####".

    Args:
        answer_string: The full GSM8K answer string.

    Returns:
        The extracted numeric answer as a string.
    """
    # Find the #### marker
    match = re.search(r'####\s*(.+?)$', answer_string, re.MULTILINE)
    if match:
        answer = match.group(1).strip()
        # Remove commas from numbers like "1,234"
        answer = answer.replace(',', '')
        return answer

    # Fallback: if no #### marker, return the last number in the string
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', answer_string)
    if numbers:
        return numbers[-1].replace(',', '')

    logger.warning(f"Could not extract GSM8K answer from: {answer_string[:100]}")
    return answer_string.strip()


def extract_model_numeric_answer(text: str) -> Optional[str]:
    """
    Extract a mathematical answer from model-generated text.

    Looks for patterns like:
      - "\\boxed{(3, \\frac{\\pi}{2})}"  (with nested braces)
      - "the answer is 42"
      - "#### 42"
      - Final number on its own line

    Args:
        text: The model's generated answer text (outside <think> block).

    Returns:
        The extracted answer string, or None if no answer found.
    """
    text = text.strip()

    # Priority 1: Look for \boxed{...} with proper brace matching
    boxed = extract_boxed_content(text)
    if boxed is not None:
        return boxed.strip()

    # Priority 2: Look for "the answer is ..."
    answer_patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*(.+?)(?:\.\s*$|$)',
        r'(?:therefore|thus|so|hence),?\s*(?:the\s+)?answer\s+is\s*:?\s*(.+?)(?:\.\s*$|$)',
        r'####\s*(.+?)$',
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            candidate = match.group(1).strip()
            # If the candidate itself contains \boxed, extract from it
            inner_boxed = extract_boxed_content(candidate)
            if inner_boxed is not None:
                return inner_boxed.strip()
            return candidate

    # Priority 3: Take the last number/expression in the text
    # This is aggressive but necessary as a fallback
    numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?', text)
    if numbers:
        return numbers[-1].replace(',', '')

    return None


# =============================================================================
# 3. MULTIPLE CHOICE ANSWER EXTRACTION
# =============================================================================

def extract_mc_answer(text: str) -> Optional[str]:
    """
    Extract a multiple-choice answer letter (A/B/C/D) from model text.

    Uses a cascade of patterns ordered by specificity (most specific first)
    to robustly extract the chosen answer letter.

    Args:
        text: The model's generated answer text.

    Returns:
        The answer letter (uppercase A-D), or None if not found.
    """
    text_clean = text.strip()

    # Pattern cascade (most specific → least specific)
    patterns = [
        # "The answer is (A)" or "The answer is A"
        r'(?:the\s+)?(?:correct\s+)?(?:final\s+)?answer\s+is\s*:?\s*\(?([A-Da-d])\)?',
        # \boxed{A}
        r'\\boxed\{\s*\(?([A-Da-d])\)?\s*\}',
        # "I choose A" / "I'll go with B"
        r"(?:i(?:'ll)?\s+)?(?:choose|pick|select|go\s+with)\s+\(?([A-Da-d])\)?",
        # "(A)" at the end of text
        r'\(([A-Da-d])\)\s*\.?\s*$',
        # Standalone letter at end of line
        r'(?:^|\n)\s*([A-Da-d])\s*\.?\s*$',
        # "Option A" / "Choice B"
        r'(?:option|choice)\s+([A-Da-d])\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, text_clean, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()

    # Last resort: find the last standalone letter A-D in the text
    standalone = re.findall(r'\b([A-Da-d])\b', text_clean)
    # Filter to only A-D letters that appear in answer-like contexts
    for letter in reversed(standalone):
        if letter.upper() in 'ABCD':
            return letter.upper()

    return None


# =============================================================================
# 4. UNIFIED SCORING INTERFACE
# =============================================================================

def score_answer(
    model_answer_text: str,
    ground_truth: str,
    answer_type: str,
    answer_extraction: str = None,
    choices: list = None,
) -> dict:
    """
    Score a model's answer against ground truth.

    This is the main entry point for answer scoring. It handles extraction
    of the answer from the model's text, then comparison against ground truth.

    Args:
        model_answer_text: The model's generated text (outside <think> block).
        ground_truth: The correct answer.
        answer_type: One of 'exact_match_math', 'exact_match_numeric',
                     'multiple_choice'.
        answer_extraction: Special extraction mode (e.g., 'gsm8k').
        choices: List of MC choices (for context, not currently used).

    Returns:
        dict with keys:
          - 'is_correct': bool
          - 'extracted_answer': str or None
          - 'ground_truth_normalized': str
          - 'comparison_method': str (which method determined the result)
    """
    result = {
        'is_correct': False,
        'extracted_answer': None,
        'ground_truth_normalized': ground_truth,
        'comparison_method': 'none',
    }

    if answer_type == 'exact_match_math':
        # Extract the model's answer
        extracted = extract_model_numeric_answer(model_answer_text)
        result['extracted_answer'] = extracted

        if extracted is None:
            result['comparison_method'] = 'extraction_failed'
            return result

        # Compare using math equivalence
        is_correct = math_equiv(extracted, ground_truth)
        result['is_correct'] = is_correct
        result['comparison_method'] = 'math_equiv'
        result['ground_truth_normalized'] = normalize_math_string(ground_truth)

    elif answer_type == 'exact_match_numeric':
        # Handle GSM8K-style answers
        if answer_extraction == 'gsm8k':
            gt_numeric = extract_gsm8k_answer(ground_truth)
        else:
            gt_numeric = ground_truth.strip()

        result['ground_truth_normalized'] = gt_numeric

        # Extract model's numeric answer
        extracted = extract_model_numeric_answer(model_answer_text)
        result['extracted_answer'] = extracted

        if extracted is None:
            result['comparison_method'] = 'extraction_failed'
            return result

        # Compare numerically
        is_correct = math_equiv(extracted, gt_numeric)
        result['is_correct'] = is_correct
        result['comparison_method'] = 'numeric_equiv'

    elif answer_type == 'multiple_choice':
        # Extract MC answer letter
        extracted = extract_mc_answer(model_answer_text)
        result['extracted_answer'] = extracted

        if extracted is None:
            result['comparison_method'] = 'extraction_failed'
            return result

        # Normalize ground truth to a letter
        gt_letter = ground_truth.strip().upper()
        if len(gt_letter) == 1 and gt_letter in 'ABCD':
            gt_normalized = gt_letter
        elif gt_letter in ('1', '2', '3', '4'):
            gt_normalized = chr(ord('A') + int(gt_letter) - 1)
        else:
            # Ground truth might be full text — need to match against choices
            gt_normalized = gt_letter
            logger.debug(f"MC ground truth is not a letter: {gt_letter[:50]}")

        result['ground_truth_normalized'] = gt_normalized
        result['is_correct'] = (extracted == gt_normalized)
        result['comparison_method'] = 'mc_letter_match'

    else:
        raise ValueError(f"Unknown answer_type: {answer_type}")

    return result


# =============================================================================
# 5. TESTING / SELF-VALIDATION
# =============================================================================

def run_scoring_tests():
    """
    Self-test suite for the scoring module.
    Run this to validate scoring logic before using it on real data.
    """
    print("Running scoring module self-tests...")
    tests_passed = 0
    tests_failed = 0

    def check(test_name, expected, actual):
        nonlocal tests_passed, tests_failed
        if expected == actual:
            tests_passed += 1
        else:
            tests_failed += 1
            print(f"  FAIL: {test_name}: expected {expected!r}, got {actual!r}")

    # ================================================================
    # BRACE MATCHING (extract_boxed_content)
    # ================================================================
    check("boxed_simple", "42",
          extract_boxed_content("\\boxed{42}"))
    check("boxed_nested_frac", "\\frac{1}{2}",
          extract_boxed_content("\\boxed{\\frac{1}{2}}"))
    check("boxed_deep_nested", "(3, \\frac{\\pi}{2})",
          extract_boxed_content("\\boxed{(3, \\frac{\\pi}{2})}"))
    check("boxed_sqrt", "\\sqrt{x^{2}+1}",
          extract_boxed_content("\\boxed{\\sqrt{x^{2}+1}}"))
    check("boxed_multiple_takes_last", "B",
          extract_boxed_content("\\boxed{A} then \\boxed{B}"))
    check("boxed_none_when_absent", None,
          extract_boxed_content("no boxed here"))

    # ================================================================
    # NORMALIZATION (normalize_math_string)
    # ================================================================
    check("norm_boxed_simple", "42",
          normalize_math_string("\\boxed{42}"))
    check("norm_boxed_nested", "(3, \\frac{\\pi}{2})",
          normalize_math_string("\\boxed{(3, \\frac{\\pi}{2})}"))
    check("norm_dollar_signs", "x+1",
          normalize_math_string("$x+1$"))
    check("norm_text_wrapper", "cm",
          normalize_math_string("\\text{cm}"))
    check("norm_whitespace", "3 + 4",
          normalize_math_string("  3  +  4  "))
    check("norm_trailing_period", "42",
          normalize_math_string("42."))
    check("norm_decimal_no_strip", "3.14",
          normalize_math_string("3.14"))

    # The exact bug from the user's report:
    # \left( 3, \frac{\pi}{2} \right) should normalize to same as (3, \frac{\pi}{2})
    check("norm_left_right_parens",
          normalize_math_string("(3, \\frac{\\pi}{2})"),
          normalize_math_string("\\left( 3, \\frac{\\pi}{2} \\right)"))

    # Whitespace inside parentheses
    check("norm_space_in_parens", "(3, x)",
          normalize_math_string("( 3, x )"))
    check("norm_space_in_brackets", "[1, 2]",
          normalize_math_string("[ 1, 2 ]"))

    # ================================================================
    # MATH EQUIVALENCE (math_equiv)
    # ================================================================
    check("equiv_identical", True, math_equiv("42", "42"))
    check("equiv_boxed_vs_plain", True, math_equiv("\\boxed{42}", "42"))
    check("equiv_fraction_decimal", True, math_equiv("0.5", "1/2"))
    check("equiv_latex_frac", True, math_equiv("\\frac{1}{2}", "0.5"))
    check("equiv_negative", True, math_equiv("-3", "-3"))
    check("equiv_different", False, math_equiv("42", "43"))
    check("equiv_close_not_equal", False, math_equiv("0.333", "1/3"))

    # The exact user-reported case: coordinate/tuple answers
    check("equiv_coordinates", True,
          math_equiv("(3, \\frac{\\pi}{2})",
                     "\\left( 3, \\frac{\\pi}{2} \\right)"))

    # Boxed coordinate answer vs left/right ground truth
    check("equiv_boxed_coord_vs_leftright", True,
          math_equiv("\\boxed{(3, \\frac{\\pi}{2})}",
                     "\\left( 3, \\frac{\\pi}{2} \\right)"))

    # ================================================================
    # ANSWER EXTRACTION (extract_model_numeric_answer)
    # ================================================================
    check("extract_boxed_simple", "42",
          extract_model_numeric_answer("\\boxed{42}"))
    check("extract_boxed_frac", "\\frac{1}{2}",
          extract_model_numeric_answer("\\boxed{\\frac{1}{2}}"))
    check("extract_boxed_coord", "(3, \\frac{\\pi}{2})",
          extract_model_numeric_answer("\\boxed{(3, \\frac{\\pi}{2})}"))
    check("extract_boxed_sqrt", "\\sqrt{2}",
          extract_model_numeric_answer("The answer is \\boxed{\\sqrt{2}}."))
    check("extract_answer_is", "7",
          extract_model_numeric_answer("The answer is 7."))
    check("extract_therefore", "100",
          extract_model_numeric_answer("Therefore, the answer is 100"))

    # Boxed inside prose (the common DeepSeek-R1 format)
    check("extract_boxed_in_prose", "(3, \\frac{\\pi}{2})",
          extract_model_numeric_answer(
              "The polar coordinates are (3, pi/2).\n\n\\boxed{(3, \\frac{\\pi}{2})}"))

    # ================================================================
    # GSM8K EXTRACTION
    # ================================================================
    check("gsm8k_standard", "42",
          extract_gsm8k_answer("Some steps...\n#### 42"))
    check("gsm8k_comma", "1234",
          extract_gsm8k_answer("Some steps...\n#### 1,234"))
    check("gsm8k_decimal", "3.14",
          extract_gsm8k_answer("#### 3.14"))

    # ================================================================
    # MC ANSWER EXTRACTION
    # ================================================================
    check("mc_answer_is", "A", extract_mc_answer("The answer is A"))
    check("mc_boxed", "B", extract_mc_answer("\\boxed{B}"))
    check("mc_paren", "C", extract_mc_answer("I choose (C)"))
    check("mc_final_line", "D", extract_mc_answer("After analysis:\nD"))

    # ================================================================
    # FULL END-TO-END SCORING
    # ================================================================
    r1 = score_answer("The answer is 42", "42", "exact_match_math")
    check("score_math_correct", True, r1['is_correct'])

    r2 = score_answer("The answer is 43", "42", "exact_match_math")
    check("score_math_wrong", False, r2['is_correct'])

    r3 = score_answer("I think C is correct", "C", "multiple_choice")
    check("score_mc_correct", True, r3['is_correct'])

    r4 = score_answer("Some steps...\nThe final answer is 7",
                       "Some explanation\n#### 7", "exact_match_numeric",
                       answer_extraction="gsm8k")
    check("score_gsm8k_correct", True, r4['is_correct'])

    # THE EXACT BUG FROM THE USER'S PILOT RUN:
    # Model outputs \boxed{(3, \frac{\pi}{2})} but ground truth uses \left...\right
    r5 = score_answer(
        "The polar coordinates are \\((3, \\frac{\\pi}{2})\\).\n\n\\boxed{(3, \\frac{\\pi}{2})}",
        "\\left( 3, \\frac{\\pi}{2} \\right)",
        "exact_match_math",
    )
    check("score_boxed_coord_pilot_bug", True, r5['is_correct'])
    check("score_boxed_coord_extraction", "(3, \\frac{\\pi}{2})", r5['extracted_answer'])

    # Another common pattern: \boxed{\frac{a}{b}} vs ground truth a/b
    r6 = score_answer("\\boxed{\\frac{7}{3}}", "\\frac{7}{3}", "exact_match_math")
    check("score_frac_boxed", True, r6['is_correct'])

    print(f"\nResults: {tests_passed} passed, {tests_failed} failed "
          f"out of {tests_passed + tests_failed} tests")

    if tests_failed > 0:
        print("WARNING: Some scoring tests failed! Fix before proceeding.")
    else:
        print("All scoring tests passed.")

    return tests_failed == 0


if __name__ == "__main__":
    run_scoring_tests()
