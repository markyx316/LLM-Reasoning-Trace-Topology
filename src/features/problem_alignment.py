"""
problem_alignment.py - Four features measuring alignment between a problem
                       and its reasoning trace.

The existing features only examine the trace in isolation. They cannot detect
"off-topic" traces whose structure looks fine but don't actually answer the
given question. These features inject problem information so the hybrid can
flag traces that drift or miss the target.

Features (all in [0, 1], higher = better alignment EXCEPT problem_drift where
higher means more drift away from the problem):

  1. problem_conclusion_sim
     Cosine similarity between the problem embedding and the CONCLUSION step
     (the last step, or mean of last 3 if present). Low value = the stated
     answer doesn't relate to what was asked.

  2. problem_trace_max_sim
     Max cosine similarity between the problem and any step in the trace.
     Low value = trace never actually engaged with the problem.

  3. problem_drift
     Cosine similarity of FIRST 20% of steps to problem, minus similarity of
     LAST 20% of steps. High value = trace started on-topic then drifted.

  4. problem_keyword_coverage
     Fraction of content words (>3 chars, non-stopword) from the problem
     that appear somewhere in the trace.

Usage:
    from src.features.problem_alignment import extract_alignment_features

    feats = extract_alignment_features(
        problem_text=problem,
        step_embeddings=emb_array,   # (n_steps, 384) from build_step_embeddings
        trace_text=trace,
        model=embedder,
    )

Design notes:
  - Relies on MiniLM embeddings already precomputed per step (.npz files).
  - For the problem embedding itself we encode it once and compute similarity.
  - Stopwords: a small English list. No NLTK dependency.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


ALIGNMENT_FEATURE_NAMES = [
    "problem_conclusion_sim",
    "problem_trace_max_sim",
    "problem_drift",
    "problem_keyword_coverage",
]


# Minimal English stopwords (we don't need a full list; just to filter
# cheap function words that add noise to keyword coverage)
_STOPWORDS = frozenset("""
a an and are as at be but by do does for from had has have he her his i if
in into is it its me my no not of on or our out she so than that the their
them then there these they this those to was we were what when where which
while who why will with would you your can could should just some any all
over such one two three new like more most also may might must shall
""".split())


def _content_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9]{2,}", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 3}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalized or raw vectors."""
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))


# =============================================================================
# FEATURES
# =============================================================================

def problem_conclusion_sim(problem_emb: np.ndarray,
                           step_embs: np.ndarray,
                           n_tail: int = 3) -> float:
    """Sim between problem and (mean of) last `n_tail` steps."""
    if step_embs.shape[0] == 0:
        return 0.0
    tail = step_embs[-n_tail:] if step_embs.shape[0] >= n_tail else step_embs
    tail_mean = tail.mean(axis=0)
    return max(0.0, _cos(problem_emb, tail_mean))  # clamp at 0


def problem_trace_max_sim(problem_emb: np.ndarray,
                          step_embs: np.ndarray) -> float:
    """Max sim between problem and any step."""
    if step_embs.shape[0] == 0:
        return 0.0
    # Assume step_embs already normalized (from SentenceTransformer)
    # If problem_emb is also normalized, dot product = cosine sim.
    sims = step_embs @ problem_emb / (np.linalg.norm(problem_emb) + 1e-9)
    return float(np.clip(sims.max(), 0.0, 1.0))


def problem_drift(problem_emb: np.ndarray,
                  step_embs: np.ndarray,
                  head_frac: float = 0.20,
                  tail_frac: float = 0.20) -> float:
    """Sim(head steps, problem) - Sim(tail steps, problem).
    Positive = trace drifts AWAY from problem as it progresses."""
    n = step_embs.shape[0]
    if n < 5:
        return 0.0
    n_head = max(1, int(round(head_frac * n)))
    n_tail = max(1, int(round(tail_frac * n)))
    head = step_embs[:n_head].mean(axis=0)
    tail = step_embs[-n_tail:].mean(axis=0)
    sim_head = _cos(problem_emb, head)
    sim_tail = _cos(problem_emb, tail)
    return float(sim_head - sim_tail)


def problem_keyword_coverage(problem_text: str, trace_text: str) -> float:
    """Fraction of problem's content tokens that appear in trace."""
    p_toks = _content_tokens(problem_text or "")
    if not p_toks:
        return 0.0
    t_toks = _content_tokens(trace_text or "")
    hits = len(p_toks & t_toks)
    return hits / len(p_toks)


# =============================================================================
# TOP-LEVEL API
# =============================================================================

def extract_alignment_features(
    problem_text: str,
    step_embeddings: np.ndarray,    # (n_steps, d)
    trace_text: str,
    model,                           # SentenceTransformer
    problem_emb: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Compute 4 alignment features for one item. step_embeddings should be
    L2-normalized (SentenceTransformer normalize_embeddings=True default)."""
    if problem_emb is None:
        if not problem_text.strip():
            return {k: 0.0 for k in ALIGNMENT_FEATURE_NAMES}
        problem_emb = model.encode(
            [problem_text], convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )[0].astype(np.float32)

    return {
        "problem_conclusion_sim":    problem_conclusion_sim(problem_emb, step_embeddings),
        "problem_trace_max_sim":     problem_trace_max_sim(problem_emb, step_embeddings),
        "problem_drift":             problem_drift(problem_emb, step_embeddings),
        "problem_keyword_coverage":  problem_keyword_coverage(problem_text, trace_text),
    }


def encode_problems(problems: list[str], model, batch_size: int = 256) -> np.ndarray:
    """Batch-encode all problems for a dataset."""
    return model.encode(
        problems, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    ).astype(np.float32)


# =============================================================================
# SELF-TEST
# =============================================================================

def _run_tests():
    print("Running problem alignment tests...")
    rng = np.random.default_rng(0)

    # Fake aligned trace: problem emb === first step emb
    p_emb = rng.standard_normal(32).astype(np.float32)
    p_emb /= np.linalg.norm(p_emb) + 1e-9

    # Aligned: all steps are similar to problem
    step_aligned = np.tile(p_emb, (10, 1)) + 0.05 * rng.standard_normal((10, 32)).astype(np.float32)
    step_aligned /= (np.linalg.norm(step_aligned, axis=1, keepdims=True) + 1e-9)

    # Unaligned: random
    step_random = rng.standard_normal((10, 32)).astype(np.float32)
    step_random /= (np.linalg.norm(step_random, axis=1, keepdims=True) + 1e-9)

    f_a = {
        "problem_conclusion_sim": problem_conclusion_sim(p_emb, step_aligned),
        "problem_trace_max_sim":  problem_trace_max_sim(p_emb, step_aligned),
        "problem_drift":          problem_drift(p_emb, step_aligned),
    }
    f_r = {
        "problem_conclusion_sim": problem_conclusion_sim(p_emb, step_random),
        "problem_trace_max_sim":  problem_trace_max_sim(p_emb, step_random),
        "problem_drift":          problem_drift(p_emb, step_random),
    }
    print("  aligned :", f_a)
    print("  random  :", f_r)
    assert f_a["problem_conclusion_sim"] > 0.7, f_a
    assert f_r["problem_conclusion_sim"] < 0.5, f_r
    assert f_a["problem_trace_max_sim"] > 0.8
    print("  PASS  aligned > random")

    # Drift test: head sim to problem, tail drift away
    step_drift = np.vstack([
        np.tile(p_emb, (3, 1)),                                   # head -- on topic
        rng.standard_normal((7, 32)).astype(np.float32),          # tail -- off topic
    ])
    step_drift /= (np.linalg.norm(step_drift, axis=1, keepdims=True) + 1e-9)
    drift = problem_drift(p_emb, step_drift)
    print(f"  drift trace: {drift:.3f} (should be positive, high)")
    assert drift > 0.3, drift

    # Keyword coverage
    cov = problem_keyword_coverage(
        "What is the solution to the quadratic equation ax squared plus bx plus c?",
        "Let me solve this quadratic by using the discriminant and computing b squared minus four a c."
    )
    print(f"  keyword coverage: {cov:.3f}")
    assert 0.1 < cov < 1.0, cov

    print("All alignment tests passed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _run_tests()
