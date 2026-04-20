"""
recurrence_features.py - Semantic recurrence features for reasoning traces.

Computes 5 content-agnostic structural features at the *semantic* level,
using sentence embeddings rather than lexical overlap. These complement the
existing 25 handcrafted features and test the central hypothesis that
incorrect traces are structurally more recurrent (not just longer).

Features (all in [0, 1] after normalization, higher = more recurrent / less efficient):

  1. semantic_recurrence_rate
     Fraction of non-adjacent step pairs whose cosine similarity >= threshold.
     Measures how often the trace revisits semantically similar content.

  2. max_semantic_cycle_span
     Longest (normalized) distance between two high-similarity steps.
     Captures long-range recurrence (model loops back much later).

  3. progress_novelty (INVERTED: higher = less novelty = more repetition)
     1 - mean novelty of each step vs. its prior context, over all steps.
     Novelty of step i = 1 - max_{j<i} sim(s_i, s_j).

  4. termination_drift (INVERTED: higher = less drift = conclusion recycles)
     Mean cosine similarity between the final 20% of steps and the first 80%.
     High value = the conclusion just restates earlier content (unstable end).

  5. revision_effectiveness (INVERTED: higher = less effective revision)
     Mean similarity between each REVISE/RESTART step and the step it revises.
     High value = revisions don't meaningfully change semantic direction.

Usage:
    from src.features.recurrence_features import extract_recurrence_features

    feats = extract_recurrence_features(
        trace_text=trace,
        episodes=episodes,        # Optional, from rule_based_parser
        model=embedder,           # Preloaded SentenceTransformer
    )

Design notes:
  - Embeddings are computed once per trace; all 5 features reuse the matrix.
  - For traces with < 2 steps, features degrade gracefully to 0.0.
  - The recurrence threshold (default 0.70) is a hyperparameter; we report
    sensitivity in the length-controlled analysis script.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


RECURRENCE_FEATURE_NAMES = [
    "semantic_recurrence_rate",
    "max_semantic_cycle_span",
    "progress_repetition",          # 1 - novelty, so higher = more repetitive
    "termination_recycle",          # final-to-early similarity
    "revision_ineffectiveness",     # semantic similarity of revisions
]


# =============================================================================
# EMBEDDING
# =============================================================================

def load_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                  device: Optional[str] = None):
    """
    Load a SentenceTransformer embedding model; auto-uses CUDA if available.
    """
    from sentence_transformers import SentenceTransformer
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    logger.info(f"Embedder device: {device}")
    return SentenceTransformer(model_name, device=device)


def embed_steps(steps: list[str], model) -> np.ndarray:
    """
    Encode a list of step strings -> (n_steps, d) normalized embedding matrix.
    Returns an empty (0, d) array if steps is empty.
    """
    if not steps:
        return np.zeros((0, 384), dtype=np.float32)
    embs = model.encode(
        steps,
        convert_to_numpy=True,
        normalize_embeddings=True,       # so inner product == cosine sim
        show_progress_bar=False,
    )
    return embs.astype(np.float32)


# =============================================================================
# CORE FEATURE COMPUTATIONS (operate on a precomputed embedding matrix)
# =============================================================================

def _similarity_matrix(emb: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity matrix (n, n). emb must be L2-normalized."""
    if emb.shape[0] == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return emb @ emb.T


def semantic_recurrence_rate(
    sim: np.ndarray,
    threshold: float = 0.70,
    min_gap: int = 2,
) -> float:
    """
    Fraction of non-adjacent pairs (|i - j| >= min_gap) exceeding threshold.
    Excludes diagonal and near-diagonal (adjacent steps are always similar).
    """
    n = sim.shape[0]
    if n < min_gap + 1:
        return 0.0
    # upper triangle with gap
    total = 0
    hits = 0
    for k in range(min_gap, n):
        diag = np.diag(sim, k=k)
        total += len(diag)
        hits += int((diag >= threshold).sum())
    return float(hits / total) if total > 0 else 0.0


def max_semantic_cycle_span(
    sim: np.ndarray,
    threshold: float = 0.70,
    min_gap: int = 2,
) -> float:
    """
    Maximum normalized distance (|i-j|/n) between any two steps with sim >= threshold.
    Captures long-range loops. Returns 0.0 if no such pair.
    """
    n = sim.shape[0]
    if n < min_gap + 1:
        return 0.0
    max_span = 0
    for k in range(min_gap, n):
        diag = np.diag(sim, k=k)
        if (diag >= threshold).any():
            max_span = k
    return float(max_span / n) if n > 0 else 0.0


def progress_repetition(sim: np.ndarray) -> float:
    """
    1 - mean novelty, where novelty(i) = 1 - max_{j<i} sim(s_i, s_j).
    Higher = each step adds less novel semantic content.
    """
    n = sim.shape[0]
    if n < 2:
        return 0.0
    novelties = []
    for i in range(1, n):
        prior_sims = sim[i, :i]
        max_prior = float(prior_sims.max()) if len(prior_sims) else 0.0
        novelties.append(1.0 - max_prior)
    mean_novelty = float(np.mean(novelties)) if novelties else 1.0
    return float(1.0 - mean_novelty)


def termination_recycle(sim: np.ndarray, tail_frac: float = 0.20) -> float:
    """
    Mean similarity between the last tail_frac of steps and the preceding ones.
    High = conclusion semantically recycles earlier content (unstable termination).
    """
    n = sim.shape[0]
    if n < 4:
        return 0.0
    n_tail = max(1, int(round(tail_frac * n)))
    n_head = n - n_tail
    if n_head < 1:
        return 0.0
    block = sim[n_head:n, :n_head]     # (n_tail, n_head)
    return float(block.mean())


def revision_ineffectiveness(
    sim: np.ndarray,
    episodes: Optional[list] = None,
    revision_types: tuple = ("BACKTRACK", "RESTART", "REVISE", "X", "R"),
) -> float:
    """
    For each revision-labeled step i, measure sim to the step it revises (i-1).
    High value = revision says the same thing semantically (unproductive).

    If episodes is None or no revisions present, returns 0.0.
    """
    if episodes is None or sim.shape[0] < 2:
        return 0.0

    sims = []
    for i, ep in enumerate(episodes):
        if i == 0:
            continue
        btype = getattr(ep, "behavior", None)
        # Support both Enum and string
        label = btype.value if hasattr(btype, "value") else str(btype)
        if label.upper() in {s.upper() for s in revision_types}:
            if i < sim.shape[0]:
                sims.append(float(sim[i, i - 1]))
    return float(np.mean(sims)) if sims else 0.0


# =============================================================================
# TOP-LEVEL API
# =============================================================================

def extract_recurrence_features(
    trace_text: str,
    model,
    episodes: Optional[list] = None,
    steps: Optional[list[str]] = None,
    threshold: float = 0.70,
    min_gap: int = 2,
    tail_frac: float = 0.20,
) -> dict[str, float]:
    """
    Extract all 5 recurrence features for a single trace.

    Args:
        trace_text:  Raw reasoning trace.
        model:       Preloaded SentenceTransformer.
        episodes:    Optional list of CognitiveEpisode from rule_based_parser.
                     If given, we embed ep.text and use ep.behavior for
                     revision_ineffectiveness.
        steps:       Optional pre-segmented step strings (overrides episodes
                     for embedding). If neither is given, falls back to
                     sentence_segmenter.segment_trace.
        threshold:   Cosine-sim cutoff for "recurrent" pairs.
        min_gap:     Minimum step distance to count as non-adjacent.
        tail_frac:   Fraction of steps treated as "termination".

    Returns:
        dict[str, float] with the 5 feature names.
    """
    # --- Step segmentation ---
    if steps is not None:
        step_texts = steps
    elif episodes is not None and len(episodes) > 0:
        step_texts = [getattr(ep, "text", "") for ep in episodes]
    else:
        from src.parsing.sentence_segmenter import segment_trace
        step_texts = segment_trace(trace_text) if trace_text else []

    step_texts = [s for s in step_texts if s and s.strip()]

    # --- Embed ---
    emb = embed_steps(step_texts, model)
    sim = _similarity_matrix(emb)

    # --- Features ---
    return {
        "semantic_recurrence_rate": semantic_recurrence_rate(sim, threshold, min_gap),
        "max_semantic_cycle_span":  max_semantic_cycle_span(sim, threshold, min_gap),
        "progress_repetition":      progress_repetition(sim),
        "termination_recycle":      termination_recycle(sim, tail_frac),
        "revision_ineffectiveness": revision_ineffectiveness(sim, episodes),
    }


# =============================================================================
# SELF-TESTS
# =============================================================================

def _run_tests() -> bool:
    """Smoke tests with a tiny fake embedder (no model download needed)."""
    print("Running recurrence feature self-tests...")

    class FakeEmbedder:
        """Maps each unique step text to a deterministic random unit vector."""
        def __init__(self, d=16, seed=0):
            self.d = d
            self.seed = seed

        def encode(self, texts, convert_to_numpy=True,
                   normalize_embeddings=True, show_progress_bar=False):
            rng = np.random.default_rng(self.seed)
            vecs = []
            seen = {}
            for t in texts:
                if t not in seen:
                    v = rng.standard_normal(self.d).astype(np.float32)
                    v /= np.linalg.norm(v) + 1e-8
                    seen[t] = v
                vecs.append(seen[t])
            return np.stack(vecs) if vecs else np.zeros((0, self.d), dtype=np.float32)

    fake = FakeEmbedder(d=32, seed=42)

    # Test 1: no recurrence (all unique steps)
    steps_unique = [f"step {i} unique content" for i in range(10)]
    f1 = extract_recurrence_features("dummy", fake, steps=steps_unique)
    assert f1["semantic_recurrence_rate"] < 0.05, f1
    print("  PASS  unique-steps recurrence_rate low")

    # Test 2: heavy recurrence (repeated steps)
    steps_rep = ["A", "B", "A", "B", "A", "B", "A", "B"]
    f2 = extract_recurrence_features("dummy", fake, steps=steps_rep)
    assert f2["semantic_recurrence_rate"] > 0.3, f2
    print("  PASS  repeated-steps recurrence_rate high:",
          round(f2["semantic_recurrence_rate"], 2))

    # Test 3: progress_repetition monotone
    assert f2["progress_repetition"] > f1["progress_repetition"]
    print("  PASS  progress_repetition monotone")

    # Test 4: empty / degenerate
    f_empty = extract_recurrence_features("", fake, steps=[])
    for k, v in f_empty.items():
        assert v == 0.0, (k, v)
    print("  PASS  empty trace -> all zeros")

    # Test 5: feature names stable
    for k in RECURRENCE_FEATURE_NAMES:
        assert k in f1, k
    print("  PASS  all 5 feature names present")

    print("All recurrence feature tests passed.")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _run_tests()
