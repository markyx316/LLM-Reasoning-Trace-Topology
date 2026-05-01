"""
generation_uncertainty.py - Summary features of per-token logprob/entropy trajectory.

Given per-token log-probabilities and entropies (computed during a teacher-
forcing pass of the generator model over a trace), we summarize the
trajectory into a 10-d feature vector. These features are:

   1. mean_logprob              average log-probability of actual tokens
                                (less negative = more confident)
   2. std_logprob               spread of token-level confidence
   3. min_logprob               worst single-token surprise
   4. low_logprob_frac          fraction of tokens with logprob < -5
                                (how often the model was "guessing")
   5. mean_entropy              average vocab-level entropy
   6. std_entropy               spread of entropy
   7. max_entropy               peak entropy (hardest moment)
   8. max_entropy_pos           normalized position (0..1) of peak entropy
                                (0 = early, 1 = late confusion)
   9. entropy_in_conclusion     mean entropy in the LAST 20% of tokens
                                (signal-to-answer uncertainty)
  10. entropy_autocorr_lag1     first-order autocorrelation of entropy
                                (high = smooth reasoning; low = erratic)

This module only defines the summary; the trajectory itself is computed
inside scripts/extract_hidden_states.py while running the forward pass.
"""

from __future__ import annotations

import numpy as np


GENERATION_UNC_FEATURE_NAMES = [
    "mean_logprob",
    "std_logprob",
    "min_logprob",
    "low_logprob_frac",
    "mean_entropy",
    "std_entropy",
    "max_entropy",
    "max_entropy_pos",
    "entropy_in_conclusion",
    "entropy_autocorr_lag1",
]


def summarize_trajectory(logprobs: np.ndarray, entropies: np.ndarray,
                         low_logprob_threshold: float = -5.0,
                         tail_frac: float = 0.20) -> dict[str, float]:
    """Summarize per-token logprob + entropy trajectories into 10 features.

    Args:
        logprobs:  (L,) log p(token_i | token_<i) in nats.
        entropies: (L,) entropy of the distribution at position i in nats.
    """
    if logprobs is None or len(logprobs) == 0:
        return {k: 0.0 for k in GENERATION_UNC_FEATURE_NAMES}

    L = len(logprobs)
    mean_lp = float(np.mean(logprobs))
    std_lp = float(np.std(logprobs))
    min_lp = float(np.min(logprobs))
    low_frac = float(np.mean(logprobs < low_logprob_threshold))

    mean_e = float(np.mean(entropies))
    std_e = float(np.std(entropies))
    max_e = float(np.max(entropies))
    pos_max = float(np.argmax(entropies) / max(L - 1, 1))

    n_tail = max(1, int(round(tail_frac * L)))
    ent_conclusion = float(np.mean(entropies[-n_tail:]))

    # Autocorrelation at lag 1
    if L >= 2:
        e = entropies - np.mean(entropies)
        denom = np.sum(e * e) + 1e-9
        ac = float(np.sum(e[:-1] * e[1:]) / denom)
    else:
        ac = 0.0

    return {
        "mean_logprob":           mean_lp,
        "std_logprob":            std_lp,
        "min_logprob":            min_lp,
        "low_logprob_frac":       low_frac,
        "mean_entropy":           mean_e,
        "std_entropy":            std_e,
        "max_entropy":            max_e,
        "max_entropy_pos":        pos_max,
        "entropy_in_conclusion":  ent_conclusion,
        "entropy_autocorr_lag1":  ac,
    }


# =============================================================================
# Self-test
# =============================================================================

def _run_tests():
    print("Running generation_uncertainty tests...")
    # Test 1: high-confidence trace (all logprobs near 0, low entropy)
    lp_good = np.full(100, -0.5)
    e_good = np.full(100, 0.5)
    f_good = summarize_trajectory(lp_good, e_good)
    assert abs(f_good["mean_logprob"] + 0.5) < 1e-6
    assert f_good["low_logprob_frac"] == 0.0
    assert f_good["mean_entropy"] == 0.5

    # Test 2: confused trace (many low logprobs, spiky entropy)
    rng = np.random.default_rng(0)
    lp_bad = -5 + rng.standard_normal(100) * 3
    e_bad = 2 + np.abs(rng.standard_normal(100))
    f_bad = summarize_trajectory(lp_bad, e_bad)
    assert f_bad["mean_logprob"] < -3
    assert f_bad["low_logprob_frac"] > 0.3
    assert f_bad["mean_entropy"] > 1.5

    # Test 3: empty input
    f_e = summarize_trajectory(np.zeros(0), np.zeros(0))
    for k in GENERATION_UNC_FEATURE_NAMES:
        assert f_e[k] == 0.0

    print("All generation_uncertainty tests passed.")


if __name__ == "__main__":
    _run_tests()
