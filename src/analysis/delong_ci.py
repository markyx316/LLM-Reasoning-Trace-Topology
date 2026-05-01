"""DeLong 95% confidence intervals and paired tests for AUROC.

This module provides leakage-orthogonal uncertainty quantification for our
per-group AUROC reporting. Every AUROC we publish in the paper should be
accompanied by a CI so that (a) we can tell noise from signal on small
groups (gpqa_diamond has n<200 per group) and (b) downstream readers can
do their own multiple-comparison corrections.

Why DeLong and not bootstrap?
-----------------------------
Bootstrap CIs on AUROC are expensive (O(B * N log N)) and have subtle
bias when label marginals are very unbalanced. The DeLong (1988)
estimator gives an asymptotically exact variance for the empirical AUROC
as a single U-statistic in closed form, and Sun & Xu (2014) give an
O(N log N) algorithm based on midranks that is numerically stable.

Key facts we rely on:

  1. AUROC = (1 / (m * n)) * sum_{i in pos, j in neg} [I(X_i > Y_j) + 0.5 * I(X_i == Y_j)]
     where m = # positives, n = # negatives, X = positive scores,
     Y = negative scores. This is exactly the tie-corrected Mann-Whitney
     U statistic divided by m * n.

  2. DeLong expresses AUROC as a two-sample U-statistic and uses the
     influence functions (structural components):
         v_pos[i] = P(Y < X_i) + 0.5 * P(Y == X_i)     (for each positive)
         v_neg[j] = P(X > Y_j) + 0.5 * P(X == Y_j)     (for each negative)
     Then mean(v_pos) = mean(v_neg) = AUROC, and the DeLong estimator
     for Var(AUROC) is
         Var(AUROC) = var(v_pos, ddof=1) / m + var(v_neg, ddof=1) / n.

  3. Sun & Xu (2014) compute v_pos, v_neg from midranks of the positive
     scores, the negative scores, and the combined scores - all O(N log N).

  4. For confidence intervals we transform to the logit scale, which
     (a) keeps the CI inside [0, 1] even for AUROC near the edges, and
     (b) has better small-sample coverage than a Wald CI on the raw AUROC.
     eta = logit(AUROC), var(eta) = Var(AUROC) / (AUROC * (1 - AUROC))^2.
     CI on eta, then invert via the inverse logit.

  5. For paired comparisons between two models on the same data, DeLong
     gives the covariance of the influence functions, which lets us test
         H0: AUROC_1 = AUROC_2     vs.    H1: different,
     via the z-statistic
         z = (AUROC_1 - AUROC_2) / sqrt(Var(AUROC_1 - AUROC_2))
     where Var(diff) = [1, -1] @ (Cov_pos / m + Cov_neg / n) @ [1, -1]^T.
     This is MUCH more powerful than an unpaired test when the two models
     agree on a lot of examples (e.g. stacker vs. RoBERTa-only).

References:
  Sun & Xu, "Fast Implementation of DeLong's Algorithm for Comparing the
  Areas Under Correlated Receiver Operating Characteristic Curves",
  IEEE SPL, 2014. https://ieeexplore.ieee.org/document/6851192

  DeLong, DeLong, & Clarke-Pearson, "Comparing the Areas under Two or
  More Correlated Receiver Operating Characteristic Curves: A
  Nonparametric Approach", Biometrics 44(3), 1988.

Self-tests run via `PYTHONPATH=. python src/analysis/delong_ci.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Standard-normal helpers (so this module is scipy-optional).
# ---------------------------------------------------------------------------

# z_{0.975} = 1.959963984540054 — standard 95% two-sided quantile.
# We hard-code it so delong_ci() is importable even in a numpy-only env.
Z_97_5 = 1.959963984540054


def _norm_sf(z: float) -> float:
    """Survival function of the standard normal (1 - Phi(z)).

    Used for paired two-sided p-values. Uses numpy's erfc for numerical
    stability far in the tails; scipy.stats.norm.sf would give the same
    answer but we keep scipy optional.
    """
    from math import erfc, sqrt

    return 0.5 * erfc(z / sqrt(2.0))


# ---------------------------------------------------------------------------
# Core DeLong machinery.
# ---------------------------------------------------------------------------


def _midrank(x: np.ndarray) -> np.ndarray:
    """Return 1-indexed midranks of `x` (ties assigned the average rank).

    O(N log N) via argsort + linear scan. Equivalent to
    scipy.stats.rankdata(x, method='average'), which we avoid pulling in
    as a hard dep.

    >>> _midrank(np.array([0.1, 0.3, 0.3, 0.5, 0.5, 0.5, 0.9]))
    array([1. , 2.5, 2.5, 5. , 5. , 5. , 7. ])
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    order = np.argsort(x, kind="mergesort")  # stable sort, deterministic ties
    z = x[order]
    t = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        # ranks are 1-indexed: positions i..j-1 get midrank 0.5*(i+j-1)+1
        t[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    # scatter back
    out = np.empty(n, dtype=np.float64)
    out[order] = t
    return out


def _structural_components(scores: np.ndarray, labels: np.ndarray):
    """Compute DeLong structural components `v_pos` and `v_neg`.

    Returns
    -------
    auc : float
        Empirical AUROC with the standard tie correction
        (= mean(v_pos) = mean(v_neg)).
    v_pos : np.ndarray, shape (m,)
        Per-positive structural component.  v_pos[i] = P(Y < X_i) + 0.5 * P(Y == X_i).
    v_neg : np.ndarray, shape (n,)
        Per-negative structural component. v_neg[j] = P(X > Y_j) + 0.5 * P(X == Y_j).
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if labels.dtype != bool:
        labels = labels.astype(bool)

    pos = scores[labels]
    neg = scores[~labels]
    m = pos.size
    n = neg.size
    if m == 0 or n == 0:
        raise ValueError(
            f"DeLong needs both classes; got m={m} positives, n={n} negatives."
        )

    tx = _midrank(pos)
    ty = _midrank(neg)
    # concatenate with positives first, then negatives
    concat = np.concatenate([pos, neg])
    tz = _midrank(concat)

    # tz[:m] are midranks (in the combined sample) of the positive scores
    # tx are midranks of positives within positives only.
    # (tz[:m] - tx) counts (negatives with score < X_i) + 0.5 * (negatives tied with X_i),
    # i.e., the tie-corrected number of negatives below X_i. Divide by n to get a probability.
    v_pos = (tz[:m] - tx) / n
    # Symmetrically, (tz[m:] - ty) counts (positives with score < Y_j) + 0.5 * (positives tied with Y_j).
    # We want P(X > Y_j) + 0.5 * P(X == Y_j) = 1 - [P(X < Y_j) + 0.5 * P(X == Y_j)], which gives:
    v_neg = 1.0 - (tz[m:] - ty) / m

    auc = float(np.mean(v_pos))
    return auc, v_pos, v_neg


# ---------------------------------------------------------------------------
# Public API: single-model CI.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AurocCI:
    """AUROC with a confidence interval and its variance estimate."""

    auroc: float
    var_auroc: float
    ci_low: float
    ci_high: float
    n_pos: int
    n_neg: int
    method: str  # "logit" or "wald"
    alpha: float  # 1 - confidence level (e.g. 0.05 for 95%)

    def as_dict(self):
        return {
            "auroc": self.auroc,
            "var_auroc": self.var_auroc,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "method": self.method,
            "alpha": self.alpha,
        }


def delong_auroc_ci(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
    method: str = "logit",
) -> AurocCI:
    """DeLong (1988) confidence interval for AUROC.

    Parameters
    ----------
    y_true : 1-D array of 0/1 labels (any truthy/falsy accepted).
    y_score : 1-D array of predicted scores (any order; ties handled).
    alpha : two-sided significance level (default 0.05 => 95% CI).
    method : "logit" (default, keeps CI in [0,1]) or "wald" (raw normal CI).

    Returns
    -------
    AurocCI dataclass.

    Notes
    -----
    - For degenerate cases (AUROC exactly 0 or 1, or variance 0), the logit
      method falls back to a Wald CI to avoid division-by-zero.
    - For very small groups (n_pos + n_neg < ~30) the asymptotic normality
      assumption breaks down; treat the CI as nominal only and interpret
      accordingly.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.shape != y_score.shape:
        raise ValueError(
            f"y_true and y_score must have the same shape; "
            f"got {y_true.shape} vs {y_score.shape}"
        )
    if not np.all(np.isfinite(y_score)):
        raise ValueError("y_score contains non-finite values.")

    auc, v_pos, v_neg = _structural_components(y_score, y_true)
    m = v_pos.size
    n = v_neg.size

    # Unbiased sample variance of the structural components.
    s_pos = float(np.var(v_pos, ddof=1)) if m >= 2 else 0.0
    s_neg = float(np.var(v_neg, ddof=1)) if n >= 2 else 0.0
    var_auc = s_pos / m + s_neg / n

    # Normal quantile for the requested confidence level.
    if alpha == 0.05:
        z = Z_97_5  # hard-coded for the common case, avoids a scipy import
    else:
        try:
            from scipy.stats import norm

            z = float(norm.ppf(1.0 - alpha / 2.0))
        except ImportError as e:
            raise ImportError(
                f"scipy is needed for alpha != 0.05; got alpha={alpha}."
            ) from e

    se = float(np.sqrt(max(var_auc, 0.0)))

    if method == "wald" or se == 0.0 or auc <= 0.0 or auc >= 1.0:
        # Raw normal CI on the AUROC scale. Clip to [0,1] to stay meaningful.
        lo = max(0.0, auc - z * se)
        hi = min(1.0, auc + z * se)
        chosen = "wald" if method == "wald" else "wald-fallback"
    elif method == "logit":
        # Delta-method variance of logit(AUROC).
        eta = np.log(auc / (1.0 - auc))
        se_eta = se / (auc * (1.0 - auc))
        lo_eta = eta - z * se_eta
        hi_eta = eta + z * se_eta
        lo = float(1.0 / (1.0 + np.exp(-lo_eta)))
        hi = float(1.0 / (1.0 + np.exp(-hi_eta)))
        chosen = "logit"
    else:
        raise ValueError(f"Unknown method '{method}' (use 'logit' or 'wald').")

    return AurocCI(
        auroc=float(auc),
        var_auroc=float(var_auc),
        ci_low=float(lo),
        ci_high=float(hi),
        n_pos=int(m),
        n_neg=int(n),
        method=chosen,
        alpha=float(alpha),
    )


# ---------------------------------------------------------------------------
# Public API: paired comparison of two AUROCs on the same data.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedDelongResult:
    """Result of a paired DeLong test between two AUROCs on the same sample."""

    auroc_a: float
    auroc_b: float
    diff: float  # auroc_a - auroc_b
    var_diff: float
    z: float
    p_two_sided: float
    ci_low: float
    ci_high: float
    n_pos: int
    n_neg: int

    def as_dict(self):
        return {
            "auroc_a": self.auroc_a,
            "auroc_b": self.auroc_b,
            "diff": self.diff,
            "var_diff": self.var_diff,
            "z": self.z,
            "p_two_sided": self.p_two_sided,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
        }


def delong_paired_test(
    y_true: Sequence[int] | np.ndarray,
    y_score_a: Sequence[float] | np.ndarray,
    y_score_b: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
) -> PairedDelongResult:
    """Paired DeLong test: are AUROC_a and AUROC_b different on the same labels?

    Parameters
    ----------
    y_true : 1-D array of 0/1 labels.
    y_score_a, y_score_b : 1-D arrays of model predictions on the same items.
    alpha : two-sided significance level for the CI on (AUROC_a - AUROC_b).

    Returns
    -------
    PairedDelongResult dataclass with z, two-sided p, and a CI on the
    AUROC difference (on the raw scale, not logit — the difference is
    unbounded so no transform is needed).
    """
    y_true = np.asarray(y_true)
    y_score_a = np.asarray(y_score_a, dtype=np.float64)
    y_score_b = np.asarray(y_score_b, dtype=np.float64)
    if not (y_true.shape == y_score_a.shape == y_score_b.shape):
        raise ValueError("All three inputs must have the same shape.")

    auc_a, v_pos_a, v_neg_a = _structural_components(y_score_a, y_true)
    auc_b, v_pos_b, v_neg_b = _structural_components(y_score_b, y_true)
    m = v_pos_a.size
    n = v_neg_a.size

    # 2x2 covariance of the structural components (pos-side and neg-side).
    # np.cov returns unbiased (ddof=1) by default.
    cov_pos = np.cov(np.vstack([v_pos_a, v_pos_b])) if m >= 2 else np.zeros((2, 2))
    cov_neg = np.cov(np.vstack([v_neg_a, v_neg_b])) if n >= 2 else np.zeros((2, 2))
    cov_auc = cov_pos / m + cov_neg / n  # covariance of (AUC_a, AUC_b)
    contrast = np.array([1.0, -1.0])
    var_diff = float(contrast @ cov_auc @ contrast)

    diff = float(auc_a - auc_b)
    se = float(np.sqrt(max(var_diff, 0.0)))
    if se == 0.0:
        z = 0.0 if diff == 0.0 else np.sign(diff) * np.inf
        p = 1.0 if diff == 0.0 else 0.0
    else:
        z = diff / se
        p = 2.0 * _norm_sf(abs(z))

    z_crit = Z_97_5 if alpha == 0.05 else None
    if z_crit is None:
        from scipy.stats import norm

        z_crit = float(norm.ppf(1.0 - alpha / 2.0))

    return PairedDelongResult(
        auroc_a=float(auc_a),
        auroc_b=float(auc_b),
        diff=diff,
        var_diff=var_diff,
        z=float(z),
        p_two_sided=float(p),
        ci_low=float(diff - z_crit * se),
        ci_high=float(diff + z_crit * se),
        n_pos=int(m),
        n_neg=int(n),
    )


# ---------------------------------------------------------------------------
# Public API: per-group CIs in one call.
# ---------------------------------------------------------------------------


def per_group_auroc_ci(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    groups: Sequence | np.ndarray,
    alpha: float = 0.05,
    method: str = "logit",
    min_per_class: int = 5,
) -> dict:
    """Compute per-group AUROC + DeLong CI, plus a pooled AUROC + CI.

    Parameters
    ----------
    y_true, y_score : length-N arrays.
    groups : length-N array of group labels (any hashable type).
    alpha : two-sided significance level for CIs.
    method : "logit" or "wald" (see delong_auroc_ci).
    min_per_class : groups with fewer than this many positives or negatives
        are reported with `auroc=NaN`, `ci_low=NaN`, `ci_high=NaN` and a
        non-empty `note` field; the CI is simply not meaningful.

    Returns
    -------
    dict with keys:
        "pooled": AurocCI.as_dict() on all samples combined,
        "per_group": list of dicts, one per group, each with group name,
            n, n_pos, n_neg, label_rate, auroc, ci_low, ci_high, var_auroc,
            method, note.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    groups = np.asarray(groups)

    pooled_ci = delong_auroc_ci(y_true, y_score, alpha=alpha, method=method)

    out_rows = []
    for g in sorted(set(groups.tolist())):
        mask = groups == g
        y_g = y_true[mask]
        s_g = y_score[mask]
        n = int(mask.sum())
        n_pos = int((y_g == 1).sum())
        n_neg = n - n_pos
        label_rate = n_pos / n if n > 0 else float("nan")

        if n_pos < min_per_class or n_neg < min_per_class:
            out_rows.append(
                {
                    "group": g,
                    "n": n,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "label_rate": label_rate,
                    "auroc": float("nan"),
                    "var_auroc": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "method": method,
                    "alpha": alpha,
                    "note": (
                        f"too few per class for CI "
                        f"(n_pos={n_pos}, n_neg={n_neg}, min={min_per_class})"
                    ),
                }
            )
            continue

        ci = delong_auroc_ci(y_g, s_g, alpha=alpha, method=method)
        out_rows.append(
            {
                "group": g,
                "n": n,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "label_rate": label_rate,
                "auroc": ci.auroc,
                "var_auroc": ci.var_auroc,
                "ci_low": ci.ci_low,
                "ci_high": ci.ci_high,
                "method": ci.method,
                "alpha": ci.alpha,
                "note": "",
            }
        )

    return {"pooled": pooled_ci.as_dict(), "per_group": out_rows}


# ---------------------------------------------------------------------------
# Self-tests: run directly via `python src/analysis/delong_ci.py`.
# ---------------------------------------------------------------------------


def _test_midrank_matches_scipy():
    """Our midrank implementation must agree with scipy.stats.rankdata."""
    from scipy.stats import rankdata

    rng = np.random.default_rng(0)
    for size in (10, 100, 1000):
        x = rng.normal(size=size)
        # Inject ties so we exercise the midrank path.
        x[:: max(1, size // 5)] = 0.0
        ours = _midrank(x)
        theirs = rankdata(x, method="average")
        assert np.allclose(ours, theirs), (
            f"midrank mismatch at size={size}: "
            f"max |diff|={np.max(np.abs(ours - theirs))}"
        )
    print("  midrank matches scipy.stats.rankdata.")


def _test_auroc_matches_sklearn():
    """mean(v_pos) from our structural decomposition must equal sklearn's AUROC."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(1)
    for n in (50, 500, 5000):
        y = (rng.uniform(size=n) > 0.3).astype(int)
        # Give positives a noisy edge so AUROC is meaningful.
        s = rng.normal(loc=y * 0.7, scale=1.0)
        auc_ours, _, _ = _structural_components(s, y)
        auc_sk = roc_auc_score(y, s)
        assert abs(auc_ours - auc_sk) < 1e-10, (
            f"AUROC mismatch at n={n}: ours={auc_ours}, sklearn={auc_sk}"
        )
    print("  AUROC from structural components matches sklearn.roc_auc_score.")


def _test_ties_handled():
    """With many ties, AUROC should still agree with sklearn (which tie-corrects)."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(2)
    n = 400
    y = (rng.uniform(size=n) > 0.5).astype(int)
    # Categorical-ish scores with only a handful of values => lots of ties.
    s = rng.integers(low=0, high=5, size=n).astype(float)
    auc_ours, _, _ = _structural_components(s, y)
    auc_sk = roc_auc_score(y, s)
    assert abs(auc_ours - auc_sk) < 1e-10, (
        f"tie-handling mismatch: ours={auc_ours}, sklearn={auc_sk}"
    )
    print("  tie-corrected AUROC matches sklearn with many ties.")


def _test_variance_matches_bootstrap():
    """DeLong variance should agree with a stratified bootstrap on AUROC."""
    rng = np.random.default_rng(3)
    n = 2000
    y = (rng.uniform(size=n) > 0.4).astype(int)
    s = rng.normal(loc=y * 0.5, scale=1.0)

    ci = delong_auroc_ci(y, s, alpha=0.05, method="wald")  # compare on raw scale
    # Bootstrap (stratified by label) for a reference variance.
    B = 500
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    boot_aucs = np.empty(B)
    from sklearn.metrics import roc_auc_score

    for b in range(B):
        ps = rng.choice(pos_idx, size=pos_idx.size, replace=True)
        ns = rng.choice(neg_idx, size=neg_idx.size, replace=True)
        idx = np.concatenate([ps, ns])
        boot_aucs[b] = roc_auc_score(y[idx], s[idx])
    boot_var = float(np.var(boot_aucs, ddof=1))
    rel = abs(ci.var_auroc - boot_var) / boot_var
    # The two estimators agree asymptotically; a 30% relative tolerance
    # is very loose but guards against outright bugs at B=500.
    assert rel < 0.3, (
        f"DeLong variance {ci.var_auroc:.2e} vs bootstrap variance "
        f"{boot_var:.2e}; relative error {rel:.2%} exceeds tolerance."
    )
    print(
        f"  DeLong var={ci.var_auroc:.2e} matches bootstrap var={boot_var:.2e} "
        f"(rel err {rel:.2%})."
    )


def _test_logit_ci_inside_unit_interval():
    """Logit CI must stay inside [0, 1] even near AUROC = 1."""
    rng = np.random.default_rng(4)
    y = np.array([0] * 20 + [1] * 20)
    s = np.concatenate([rng.normal(0, 0.1, 20), rng.normal(5, 0.1, 20)])
    ci = delong_auroc_ci(y, s, alpha=0.05, method="logit")
    assert 0.0 <= ci.ci_low <= ci.auroc <= ci.ci_high <= 1.0, (
        f"logit CI escaped [0,1]: auc={ci.auroc}, "
        f"ci=[{ci.ci_low}, {ci.ci_high}]"
    )
    print(
        f"  logit CI stays in [0,1] near the edge: "
        f"auc={ci.auroc:.4f}, ci=[{ci.ci_low:.4f}, {ci.ci_high:.4f}]"
    )


def _test_paired_test_identifies_difference():
    """On two noisy scores with a real gap, the paired DeLong p-value
    should be small; on identical scores it should be exactly 1."""
    rng = np.random.default_rng(5)
    n = 2000
    y = (rng.uniform(size=n) > 0.4).astype(int)
    sa = rng.normal(loc=y * 0.8, scale=1.0)  # AUROC ~ 0.72
    sb = rng.normal(loc=y * 0.3, scale=1.0)  # AUROC ~ 0.58

    r = delong_paired_test(y, sa, sb)
    assert r.p_two_sided < 1e-5, (
        f"paired DeLong failed to detect a real difference: "
        f"p={r.p_two_sided:.3g}, diff={r.diff:.3f}"
    )

    r_same = delong_paired_test(y, sa, sa)
    assert abs(r_same.diff) < 1e-12
    assert r_same.p_two_sided == 1.0
    print(
        f"  paired DeLong: p={r.p_two_sided:.3g} for real diff, "
        f"p=1.0 for identical scores."
    )


def _test_per_group():
    """per_group_auroc_ci should return one row per unique group."""
    rng = np.random.default_rng(6)
    n = 500
    y = (rng.uniform(size=n) > 0.4).astype(int)
    s = rng.normal(loc=y * 0.5, scale=1.0)
    groups = rng.choice(["A", "B", "C"], size=n)
    res = per_group_auroc_ci(y, s, groups, alpha=0.05, method="logit")
    assert set(r["group"] for r in res["per_group"]) == {"A", "B", "C"}
    for r in res["per_group"]:
        assert 0.0 <= r["ci_low"] <= r["auroc"] <= r["ci_high"] <= 1.0
    print("  per_group_auroc_ci returns one well-formed row per group.")


def _run_self_tests():
    print("delong_ci self-tests:")
    _test_midrank_matches_scipy()
    _test_auroc_matches_sklearn()
    _test_ties_handled()
    _test_variance_matches_bootstrap()
    _test_logit_ci_inside_unit_interval()
    _test_paired_test_identifies_difference()
    _test_per_group()
    print("all delong_ci self-tests passed.")


if __name__ == "__main__":
    _run_self_tests()
