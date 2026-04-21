"""H3: cross-domain transfer (MATH500 -> {GPQA, ARC, GSM8K}) with DeLong 95% CIs.

The pre-committed H3 hypothesis (research_proposal.md:170-178):

    MATH500 -> GPQA-Diamond transfer AUROC >= 0.65.

The existing ``results/transfer_math500_*_to_gpqa_*.json`` files report
point estimates only. This script re-runs the transfer with
leakage-clean handcrafted + recurrence features (features_rec.csv),
records the full test-set probability vector for every transfer pair,
and computes the DeLong 95% CI so we can tell "0.664" (point estimate
above threshold) apart from "0.664 with CI [0.60, 0.72]" (CI straddles
threshold, evidence is weaker than it looks).

We evaluate three transfer axes, with per-model stratification
(qwen7b <-> qwen7b, llama8b <-> llama8b) since cross-model transfer
is a separate experiment:

    H3.a  MATH500  -> GPQA-Diamond     (literal H3 target)
    H3.b  MATH500  -> ARC-Challenge    (auxiliary)
    H3.c  MATH500  -> GSM8K            (auxiliary; in-distribution math)
    H3.d  GSM8K    -> MATH500          (reverse direction, useful sanity)
    H3.e  GSM8K    -> GPQA-Diamond     (math-reasoning -> science reasoning)

For each pair we fit two classifiers (to rule out classifier-choice
artifacts):

    - Logistic Regression  (C=1.0, L2, class_weight=None)  -- linear bar
    - Random Forest        (matches legacy transfer files; n_estimators=300,
                             max_depth=None, n_jobs=-1)    -- non-linear bar

and report AUROC + DeLong 95% CI on the target dataset.  We ALSO report
a length-only bar (univariate LR on total_tokens) on the same target
to tell structure-apart-from-length.

Outputs:
    reports/route_ab/h3_transfer_auroc_ci.csv
        One row per (source, target, classifier) with n, AUROC, CI.
    reports/route_ab/h3_transfer_details.json
        Full details including feature list and per-pair test-set size.

Usage:
    PYTHONPATH=. python scripts/h3_transfer_with_ci.py

Optional:
    --alpha 0.1          90 % CI instead of 95 %
    --out-dir reports/route_ab_vN
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# sklearn 1.8 deprecated `penalty=` on LogisticRegression (migration toward
# `l1_ratio=...`). The behavior is unchanged; only the parameter name will
# eventually change.  We silence the FutureWarning so CI output stays readable.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"'penalty' was deprecated in version 1.8",
)

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.analysis.delong_ci import delong_auroc_ci  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

RNG = 42


# ---------------------------------------------------------------------------
# Feature definitions (must match baseline_c_handcrafted + recurrence).
# ---------------------------------------------------------------------------
HANDCRAFTED_25 = [
    "total_tokens",
    "total_episodes",
    "prop_forward",
    "prop_verification",
    "prop_backtrack",
    "prop_restart",
    "prop_hesitation",
    "prop_subgoal",
    "prop_conclusion",
    "backtrack_count",
    "verification_count",
    "restart_count",
    "vf_ratio",
    "bt_position_mean",
    "first_conclusion_pos",
    "v_clustering",
    "max_forward_run",
    "transition_entropy",
    "cycle_count",
    "wait_ratio",
    "question_mark_count",
    "negation_count",
    "repetition_rate_4gram",
]
RECURRENCE_5 = [
    "semantic_recurrence_rate",
    "max_semantic_cycle_span",
    "progress_repetition",
    "termination_recycle",
    "revision_ineffectiveness",
]
FEATURE_COLS = HANDCRAFTED_25 + RECURRENCE_5  # length-23+5-ish; actual count depends on CSV


# ---------------------------------------------------------------------------
# Transfer pairs (same-model only; qwen<->qwen, llama<->llama).
# ---------------------------------------------------------------------------
def _pairs() -> list[tuple[str, str]]:
    """Return [(source, target), ...] for H3.a-e across both models."""
    out = []
    for model in ("qwen7b", "llama8b"):
        out.append((f"math500_{model}", f"gpqa_diamond_{model}"))  # H3.a literal
        out.append((f"math500_{model}", f"arc_challenge_{model}"))  # H3.b
        out.append((f"math500_{model}", f"gsm8k_{model}"))  # H3.c
        out.append((f"gsm8k_{model}", f"math500_{model}"))  # H3.d reverse
        out.append((f"gsm8k_{model}", f"gpqa_diamond_{model}"))  # H3.e math->science
    return out


def _load_feature_df(name: str, repo_root: Path) -> pd.DataFrame:
    """Load the features_rec CSV for a given {dataset}_{model} key."""
    path = repo_root / "data" / "features" / f"{name}_features_rec.csv"
    if not path.exists():
        # Fall back to the legacy features.csv if the _rec variant is missing
        path_legacy = repo_root / "data" / "features" / f"{name}_features.csv"
        if path_legacy.exists():
            return pd.read_csv(path_legacy)
        raise FileNotFoundError(f"No features CSV for '{name}' under data/features/")
    return pd.read_csv(path)


def _available_features(df: pd.DataFrame) -> list[str]:
    """Intersect the requested FEATURE_COLS with the columns actually present.

    Older CSVs may lack the 5 recurrence columns; newer ones have all 30.
    We keep whatever is available and record it in the output so the report
    is auditable.
    """
    return [c for c in FEATURE_COLS if c in df.columns]


def _fit_and_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    classifier: str,
) -> np.ndarray:
    """Fit `classifier` on (X_train, y_train) and return P(y=1 | X_test)."""
    if classifier == "lr":
        # Scale before LR so coefficients don't explode on unscaled inputs.
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        C=1.0,
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=2000,
                        class_weight=None,
                        random_state=RNG,
                    ),
                ),
            ]
        )
    elif classifier == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=RNG,
        )
    elif classifier == "length_only_lr":
        # Univariate LR on total_tokens (first column by our convention).
        X_train = X_train[:, :1]
        X_test = X_test[:, :1]
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        C=1.0,
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=2000,
                        random_state=RNG,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


# ---------------------------------------------------------------------------
# Main driver.
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default="reports/route_ab", help="Output directory.")
    ap.add_argument("--alpha", type=float, default=0.05, help="Two-sided alpha for CI.")
    args = ap.parse_args()

    out_dir = (_REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    classifiers = ["lr", "rf", "length_only_lr"]
    rows = []
    details = {
        "hypothesis": "H3: transfer AUROC >= 0.65 for MATH500 -> GPQA-Diamond",
        "alpha": args.alpha,
        "feature_cols_requested": FEATURE_COLS,
        "classifiers": classifiers,
        "seed": RNG,
        "pairs": [],
    }

    print(
        f"H3 transfer with {100 * (1 - args.alpha):.0f}% DeLong CIs  "
        f"(target=[n] | clf | auroc [ci_low, ci_high])"
    )
    print("=" * 78)

    for source, target in _pairs():
        try:
            df_src = _load_feature_df(source, _REPO_ROOT)
            df_tgt = _load_feature_df(target, _REPO_ROOT)
        except FileNotFoundError as e:
            print(f"[SKIP] {source} -> {target}  ({e})")
            continue

        feats_src = _available_features(df_src)
        feats_tgt = _available_features(df_tgt)
        # Use the intersection so the X shapes line up even if CSVs differ.
        feats = [c for c in feats_src if c in feats_tgt]
        if not feats:
            print(f"[SKIP] {source} -> {target}  (no shared feature columns)")
            continue

        # Enforce length-first ordering so length_only_lr reads total_tokens.
        if "total_tokens" in feats:
            feats = ["total_tokens"] + [c for c in feats if c != "total_tokens"]

        X_tr = df_src[feats].to_numpy(dtype=np.float64)
        y_tr = df_src["is_correct"].to_numpy(dtype=int)
        X_te = df_tgt[feats].to_numpy(dtype=np.float64)
        y_te = df_tgt["is_correct"].to_numpy(dtype=int)

        # Drop rows with NaNs in X/y (rare but can happen from upstream).
        ok_tr = np.isfinite(X_tr).all(axis=1) & np.isfinite(y_tr)
        ok_te = np.isfinite(X_te).all(axis=1) & np.isfinite(y_te)
        X_tr, y_tr = X_tr[ok_tr], y_tr[ok_tr]
        X_te, y_te = X_te[ok_te], y_te[ok_te]

        if len(set(y_tr)) < 2 or len(set(y_te)) < 2:
            print(f"[SKIP] {source} -> {target}  (single-class label column)")
            continue

        pair_detail = {
            "source": source,
            "target": target,
            "n_train": int(y_tr.size),
            "n_test": int(y_te.size),
            "n_features": len(feats),
            "features": feats,
            "label_rate_train": float(y_tr.mean()),
            "label_rate_test": float(y_te.mean()),
            "results": [],
        }

        for clf in classifiers:
            try:
                probs = _fit_and_predict(X_tr, y_tr, X_te, clf)
            except Exception as exc:
                print(f"[ERR]  {source} -> {target}  [{clf}]  {exc}")
                pair_detail["results"].append({"classifier": clf, "error": str(exc)})
                continue
            ci = delong_auroc_ci(y_te, probs, alpha=args.alpha, method="logit")
            row = {
                "source": source,
                "target": target,
                "classifier": clf,
                "n_train": int(y_tr.size),
                "n_test": int(y_te.size),
                "n_pos_test": int((y_te == 1).sum()),
                "n_neg_test": int((y_te == 0).sum()),
                "label_rate_test": float(y_te.mean()),
                "auroc": ci.auroc,
                "var_auroc": ci.var_auroc,
                "ci_low": ci.ci_low,
                "ci_high": ci.ci_high,
                "ci_width": ci.ci_high - ci.ci_low,
                "meets_h3_point": ci.auroc >= 0.65,
                "meets_h3_ci_lb": ci.ci_low >= 0.65,
                "method": ci.method,
                "alpha": ci.alpha,
                "n_features": len(feats),
            }
            rows.append(row)
            pair_detail["results"].append({"classifier": clf, **row})

            h3 = ""
            if target.startswith("gpqa_diamond"):
                if ci.ci_low >= 0.65:
                    h3 = "  H3 PASSES (CI LB >= 0.65)"
                elif ci.auroc >= 0.65:
                    h3 = "  H3 borderline (point >= 0.65 but CI spans)"
                else:
                    h3 = "  H3 REJECTED (point < 0.65)"
            print(
                f"{source:28s} -> {target:28s} "
                f"n={int(y_te.size):4d} | {clf:15s} | "
                f"{ci.auroc:.3f} [{ci.ci_low:.3f}, {ci.ci_high:.3f}]{h3}"
            )

        details["pairs"].append(pair_detail)

    # Save outputs.
    df_rows = pd.DataFrame(rows)
    csv_path = out_dir / "h3_transfer_auroc_ci.csv"
    df_rows.to_csv(csv_path, index=False)
    json_path = out_dir / "h3_transfer_details.json"
    with open(json_path, "w") as f:
        json.dump(details, f, indent=2, default=float)

    # Summary: H3 pass/fail by target=gpqa_diamond_{model}, best classifier.
    print("\n" + "=" * 78)
    print(
        "H3 summary (target = gpqa_diamond_*, best AUROC across "
        "LR/RF over handcrafted+recurrence):"
    )
    for model in ("qwen7b", "llama8b"):
        target = f"gpqa_diamond_{model}"
        sub = df_rows[(df_rows["target"] == target) & (df_rows["classifier"] != "length_only_lr")]
        if sub.empty:
            print(f"  {target}: no data")
            continue
        best = sub.loc[sub["auroc"].idxmax()]
        verdict = (
            "H3 PASSES (CI LB >= 0.65)"
            if best["ci_low"] >= 0.65
            else (
                "H3 borderline (point >= 0.65, CI spans)"
                if best["auroc"] >= 0.65
                else "H3 REJECTED (point < 0.65)"
            )
        )
        print(
            f"  {target}  best_clf={best['classifier']:3s}  "
            f"AUROC={best['auroc']:.3f}  "
            f"95%CI=[{best['ci_low']:.3f}, {best['ci_high']:.3f}]  "
            f"n={int(best['n_test'])}  => {verdict}"
        )

    # Also emit a compact markdown-style summary table so the report
    # is pasteable into the paper draft.
    md_path = out_dir / "h3_transfer_summary.md"
    with open(md_path, "w") as f:
        f.write("# H3 transfer results (AUROC with 95% DeLong CI)\n\n")
        f.write(
            f"Hypothesis H3: transfer AUROC >= 0.65 for "
            f"MATH500 -> GPQA-Diamond. Alpha={args.alpha}.\n\n"
        )
        f.write(
            "| source | target | n_test | clf | AUROC | 95% CI | verdict |\n"
        )
        f.write("|---|---|--:|---|--:|---|---|\n")
        for _, r in df_rows.iterrows():
            verdict = ""
            if r["target"].startswith("gpqa_diamond"):
                if r["ci_low"] >= 0.65:
                    verdict = "H3 PASSES"
                elif r["auroc"] >= 0.65:
                    verdict = "borderline"
                else:
                    verdict = "H3 REJECTED"
            f.write(
                f"| {r['source']} | {r['target']} | {int(r['n_test'])} | "
                f"{r['classifier']} | {r['auroc']:.3f} | "
                f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}] | {verdict} |\n"
            )
    print(f"\nwrote {csv_path}  ({len(df_rows)} rows)")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
