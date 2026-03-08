"""
baselines.py — UQ baselines for comparison.

Implements 5 baselines, all with a unified interface returning a confidence
score in [0, 1] per item (higher = more confident the answer is correct):

  1. verbalized_confidence  — model's self-reported confidence (0–100 → 0–1)
  2. trace_length           — normalized inverse token count
  3. self_consistency       — majority-vote agreement among N samples
  4. semantic_entropy       — negative semantic entropy of N samples (needs DeBERTa NLI)
  5. token_perplexity       — negative mean log-prob (white-box)

Usage:
  from baselines import self_consistency_score, trace_length_score
"""

import json
import math
import re
from pathlib import Path

import numpy as np


# ─── 1. Trace length (naive baseline) ────────────────────────────────────────

def trace_length_score(token_count: int, max_tokens: int = 16000) -> float:
    """
    Confidence = 1 - normalized_length.
    Shorter traces are (on average) more likely to be correct.
    """
    return 1.0 - min(token_count / max_tokens, 1.0)


def compute_trace_length_scores(feature_csv: str) -> dict[str, float]:
    """Load feature CSV and return {id: confidence} using trace length."""
    import pandas as pd
    df = pd.read_csv(feature_csv)
    max_tok = df["g1_token_count"].quantile(0.99)
    return {
        row["id"]: trace_length_score(row["g1_token_count"], max_tok)
        for _, row in df.iterrows()
    }


# ─── 2. Self-consistency ──────────────────────────────────────────────────────

def self_consistency_score(answers: list[str]) -> float:
    """
    Confidence = fraction of answers matching the plurality answer.
    """
    if not answers:
        return 0.0
    counts: dict[str, int] = {}
    for a in answers:
        a_norm = a.strip().lower()
        counts[a_norm] = counts.get(a_norm, 0) + 1
    majority_count = max(counts.values())
    return majority_count / len(answers)


def compute_self_consistency_scores(sc_jsonl: str, primary_jsonl: str) -> dict[str, float]:
    """
    Load self-consistency samples (sc_jsonl) and compute agreement per base item.

    sc_jsonl: path to e.g. math500_sc8.jsonl (each id has suffix _s0, _s1, ...)
    primary_jsonl: path to math500.jsonl (to get canonical item IDs)
    """
    # Group SC samples by base ID
    sc_by_item: dict[str, list[str]] = {}
    with open(sc_jsonl) as f:
        for line in f:
            r = json.loads(line)
            base_id = re.sub(r"_s\d+$", "", r["id"])
            sc_by_item.setdefault(base_id, []).append(r["answer"])

    scores = {}
    with open(primary_jsonl) as f:
        for line in f:
            r = json.loads(line)
            answers = sc_by_item.get(r["id"], [])
            scores[r["id"]] = self_consistency_score(answers) if answers else 0.5
    return scores


# ─── 3. Semantic entropy ──────────────────────────────────────────────────────

def _nli_entailment_score(premise: str, hypothesis: str, nli_pipeline) -> float:
    """Return P(entailment) from an NLI pipeline."""
    result = nli_pipeline(
        f"{premise} [SEP] {hypothesis}",
        truncation=True, max_length=512
    )
    for item in result:
        if item["label"].lower() in ("entailment", "contradiction"):
            if item["label"].lower() == "entailment":
                return item["score"]
    return 0.0


def semantic_entropy(answers: list[str], nli_pipeline=None) -> float:
    """
    Cluster answers into semantic equivalence classes using NLI,
    then compute entropy over the cluster distribution.

    nli_pipeline: HuggingFace pipeline("text-classification", model="cross-encoder/nli-deberta-v3-large")
    If None, falls back to exact-match clustering.
    Returns entropy (lower = more confident).
    """
    if not answers:
        return 0.0

    if nli_pipeline is None:
        # Fallback: exact-match clusters
        clusters: dict[str, int] = {}
        for a in answers:
            key = a.strip().lower()
            clusters[key] = clusters.get(key, 0) + 1
    else:
        # Greedy NLI clustering: assign each answer to first cluster it entails
        cluster_reps: list[str] = []
        cluster_counts: list[int] = []
        for ans in answers:
            assigned = False
            for ci, rep in enumerate(cluster_reps):
                if _nli_entailment_score(rep, ans, nli_pipeline) > 0.5:
                    cluster_counts[ci] += 1
                    assigned = True
                    break
            if not assigned:
                cluster_reps.append(ans)
                cluster_counts.append(1)
        clusters = {str(i): c for i, c in enumerate(cluster_counts)}

    total = sum(clusters.values())
    entropy = -sum(
        (c / total) * math.log2(c / total)
        for c in clusters.values()
        if c > 0
    )
    return entropy


def semantic_entropy_confidence(answers: list[str], nli_pipeline=None) -> float:
    """Convert semantic entropy to a confidence score in [0, 1]."""
    max_entropy = math.log2(max(len(answers), 1))
    if max_entropy == 0:
        return 1.0
    return 1.0 - (semantic_entropy(answers, nli_pipeline) / max_entropy)


# ─── 4. Verbalized confidence ─────────────────────────────────────────────────

def parse_verbalized_confidence(model_output: str) -> float:
    """
    Extract a 0–100 confidence score from model output and normalize to [0, 1].
    Looks for patterns like "confidence: 85", "I am 85% confident", etc.
    """
    # Look for explicit confidence statement
    patterns = [
        r"confidence[:\s]+(\d{1,3})",
        r"(\d{1,3})\s*(?:%|percent)\s+confident",
        r"i(?:'m| am)\s+(\d{1,3})\s*(?:%|percent)\s+(?:confident|sure)",
        r"certainty[:\s]+(\d{1,3})",
        r"\b(\d{1,3})/100\b",
    ]
    for p in patterns:
        m = re.search(p, model_output, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            return min(val, 100) / 100.0
    # If no explicit score found, return 0.5 (uncertain)
    return 0.5


VERBALIZED_PROMPT_SUFFIX = (
    "\n\nAfter providing your answer, rate your confidence on a scale from 0 to 100 "
    "(where 100 means completely certain). Format: 'Confidence: <number>'"
)


# ─── 5. Token perplexity (white-box) ─────────────────────────────────────────

def token_perplexity_confidence(mean_log_prob: float, scale: float = 5.0) -> float:
    """
    Convert mean log-probability to a confidence score in [0, 1].
    Higher log-prob (less negative) → higher confidence.
    Uses sigmoid normalization.
    """
    # mean_log_prob is typically in [-10, 0]; map to [0, 1]
    return 1.0 / (1.0 + math.exp(scale * (-mean_log_prob - 2.0)))


# ─── Unified interface ────────────────────────────────────────────────────────

class BaselineScorer:
    """Unified interface: given a record dict, return confidence in [0, 1]."""

    def __init__(self, name: str):
        self.name = name

    def score(self, record: dict) -> float:
        raise NotImplementedError

    def score_all(self, records: list[dict]) -> dict[str, float]:
        return {r["id"]: self.score(r) for r in records}


class TraceLengthScorer(BaselineScorer):
    def __init__(self, max_tokens: int = 16000):
        super().__init__("trace_length")
        self.max_tokens = max_tokens

    def score(self, record: dict) -> float:
        return trace_length_score(record.get("tokens", 0), self.max_tokens)


class VerbalizedConfidenceScorer(BaselineScorer):
    def __init__(self):
        super().__init__("verbalized_confidence")

    def score(self, record: dict) -> float:
        return parse_verbalized_confidence(record.get("verbalized_output", ""))


class TokenPerplexityScorer(BaselineScorer):
    def __init__(self):
        super().__init__("token_perplexity")

    def score(self, record: dict) -> float:
        lp = record.get("mean_log_prob", -5.0)
        return token_perplexity_confidence(lp)
