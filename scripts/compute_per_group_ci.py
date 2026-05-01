"""Compute per-group AUROC + DeLong 95% CI for every OOF artifact + paired tests.

Outputs (all under ``reports/route_ab/``):

    per_group_metrics_with_ci.csv
        One row per (model, group) with AUROC, 95 % DeLong CI, variance,
        n / n_pos / n_neg, label_rate, method, protocol, leaky.  The
        ``group="OVERALL"`` row per model is the pooled AUROC on *that
        model's own* sample (useful when OOFs cover slightly different
        subsets of items).

    pooled_metrics_with_ci.csv
        Compact one-row-per-model table of pooled AUROC + CI +
        provenance.  A convenience projection of the OVERALL rows above.

    paired_delong_stacker_vs_base.json
        For each base OOF, a paired DeLong test of
            H0 : AUROC(hybrid_stacker) == AUROC(base)
        evaluated on the *intersection* of item_ids (so the two models
        see exactly the same items).  Reports z, two-sided p, and a
        95 % CI on the AUROC difference on the raw scale.

Usage (from repo root, with any env that has numpy + scipy + sklearn +
pandas):

    PYTHONPATH=. python scripts/compute_per_group_ci.py

Optional flags let you point at a different stacker OOF (after the
re-tune on HPC completes) or write elsewhere:

    --stacker-oof results/route_ab/hybrid_tuned_hybrid_v2_truly_clean_pooled_oof.npz
    --out-dir     reports/route_ab_v2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure `src` is importable whether run from repo root or elsewhere.
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent  # scripts/ -> repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.analysis.delong_ci import (  # noqa: E402
    delong_auroc_ci,
    delong_paired_test,
    per_group_auroc_ci,
)


# ---------------------------------------------------------------------------
# Model registry. One canonical name per OOF artifact we CI.
# ---------------------------------------------------------------------------


def _default_registry(stacker_oof: str) -> list[dict]:
    """Return the list of (model_key, display_name, npz, provenance) entries.

    The stacker path is injected so a re-tuned stacker OOF can be swapped
    in from the command line. The others are stable.
    """
    return [
        {
            "model_key": "hybrid_stacker",
            "display_name": "Hybrid stacker (L1-LR on OOF+hand)",
            "npz": stacker_oof,
            "provenance": None,  # stacker has no .PROVENANCE.json sidecar
        },
        {
            "model_key": "roberta_pooled",
            "display_name": "RoBERTa-base (pooled)",
            "npz": "results/roberta_pooled_oof.npz",
            "provenance": "results/roberta_pooled_oof.npz.PROVENANCE.json",
        },
        {
            "model_key": "deberta_pooled",
            "display_name": "DeBERTa-v3-base (pooled)",
            "npz": "results/month2/deberta_pooled_oof.npz",
            "provenance": "results/month2/deberta_pooled_oof.npz.PROVENANCE.json",
        },
        {
            "model_key": "step_transformer",
            "display_name": "Step Transformer",
            "npz": "results/step_transformer_pooled_oof.npz",
            "provenance": "results/step_transformer_pooled_oof.npz.PROVENANCE.json",
        },
        {
            "model_key": "trace_gnn_structural",
            "display_name": "TraceGIN-structural (pooled)",
            "npz": "results/route_ab/trace_gnn_structural_pooled_oof.npz",
            "provenance": (
                "results/route_ab/trace_gnn_structural_pooled_oof.npz.PROVENANCE.json"
            ),
        },
        {
            "model_key": "trace_gnn_hybrid",
            "display_name": "TraceGIN-hybrid (pooled)",
            "npz": "results/route_ab/trace_gnn_hybrid_pooled_oof.npz",
            "provenance": (
                "results/route_ab/trace_gnn_hybrid_pooled_oof.npz.PROVENANCE.json"
            ),
        },
        {
            "model_key": "shapelet_pooled",
            "display_name": "Shapelet (LR, fold-local mining)",
            "npz": "results/route_ab/shapelet_oof.npz",
            "provenance": "results/route_ab/shapelet_oof.npz.PROVENANCE.json",
        },
    ]


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _load_oof(npz_path: Path):
    """Load an OOF .npz and return (item_ids, groups, y_true, oof_prob)."""
    if not npz_path.exists():
        raise FileNotFoundError(f"OOF not found: {npz_path}")
    z = np.load(npz_path, allow_pickle=True)
    required = {"item_ids", "groups", "y_true", "oof_prob"}
    missing = required - set(z.keys())
    if missing:
        raise ValueError(f"{npz_path} missing keys {missing}; has {list(z.keys())}")
    return (
        np.asarray(z["item_ids"]),
        np.asarray(z["groups"]),
        np.asarray(z["y_true"]).astype(int),
        np.asarray(z["oof_prob"]).astype(float),
    )


def _load_provenance(prov_path: Optional[Path]) -> dict:
    """Return {leaky, protocol, why_flagged} from a provenance sidecar or a
    stub dict when the sidecar is missing (e.g., for the stacker)."""
    if prov_path is None or not prov_path.exists():
        return {"leaky": None, "protocol": None, "why_flagged": None}
    with open(prov_path, "r") as f:
        d = json.load(f)
    return {
        "leaky": d.get("leaky"),
        "protocol": d.get("protocol"),
        "why_flagged": d.get("why_flagged"),
    }


def _per_model_rows(
    model_key: str,
    display_name: str,
    npz_path: Path,
    prov_path: Optional[Path],
    alpha: float,
    method: str,
) -> list[dict]:
    """Run DeLong per-group + pooled on one OOF artifact, return CSV rows."""
    item_ids, groups, y_true, oof_prob = _load_oof(npz_path)
    prov = _load_provenance(prov_path)

    res = per_group_auroc_ci(
        y_true=y_true,
        y_score=oof_prob,
        groups=groups,
        alpha=alpha,
        method=method,
    )

    pooled = res["pooled"]
    rows = [
        {
            "model_key": model_key,
            "model_name": display_name,
            "group": "OVERALL",
            "n": int(y_true.size),
            "n_pos": int((y_true == 1).sum()),
            "n_neg": int((y_true == 0).sum()),
            "label_rate": float((y_true == 1).mean()),
            "auroc": pooled["auroc"],
            "var_auroc": pooled["var_auroc"],
            "ci_low": pooled["ci_low"],
            "ci_high": pooled["ci_high"],
            "ci_width": pooled["ci_high"] - pooled["ci_low"],
            "method": pooled["method"],
            "alpha": pooled["alpha"],
            "protocol": prov["protocol"],
            "leaky": prov["leaky"],
            "why_flagged": prov["why_flagged"],
            "note": "",
        }
    ]
    for row in res["per_group"]:
        rows.append(
            {
                "model_key": model_key,
                "model_name": display_name,
                "group": row["group"],
                "n": row["n"],
                "n_pos": row["n_pos"],
                "n_neg": row["n_neg"],
                "label_rate": row["label_rate"],
                "auroc": row["auroc"],
                "var_auroc": row["var_auroc"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "ci_width": (
                    row["ci_high"] - row["ci_low"]
                    if not (np.isnan(row["ci_low"]) or np.isnan(row["ci_high"]))
                    else float("nan")
                ),
                "method": row["method"],
                "alpha": row["alpha"],
                "protocol": prov["protocol"],
                "leaky": prov["leaky"],
                "why_flagged": prov["why_flagged"],
                "note": row["note"],
            }
        )
    return rows


def _paired_stacker_vs_base(
    stacker_npz: Path,
    base_npz: Path,
    alpha: float,
) -> dict:
    """Run a paired DeLong test on the intersection of item_ids.

    Returns a dict with the paired test result, plus the aligned sample
    size and the per-group pooled stacker-vs-base diff (no CI; CI is
    overall).
    """
    s_ids, s_groups, s_y, s_p = _load_oof(stacker_npz)
    b_ids, b_groups, b_y, b_p = _load_oof(base_npz)

    # Intersect by (item_id, group); item_ids alone are NOT unique
    # (e.g. "arc_0000" appears in both arc_challenge_llama8b and
    # arc_challenge_qwen7b), so keying on item_id alone would produce
    # mis-aligned pairs with label disagreement on ~half the items.
    s_key = np.asarray(
        [f"{g}||{i}" for g, i in zip(s_groups.tolist(), s_ids.tolist())]
    )
    b_key = np.asarray(
        [f"{g}||{i}" for g, i in zip(b_groups.tolist(), b_ids.tolist())]
    )
    s_index = {k: idx for idx, k in enumerate(s_key)}
    common_idx_s = []
    common_idx_b = []
    for idx, k in enumerate(b_key):
        if k in s_index:
            common_idx_s.append(s_index[k])
            common_idx_b.append(idx)
    if not common_idx_s:
        return {
            "stacker_npz": str(stacker_npz),
            "base_npz": str(base_npz),
            "n_common": 0,
            "error": "no overlapping item_ids",
        }

    common_idx_s = np.asarray(common_idx_s)
    common_idx_b = np.asarray(common_idx_b)
    y_s = s_y[common_idx_s]
    y_b = b_y[common_idx_b]
    if not np.array_equal(y_s, y_b):
        n_disagree = int((y_s != y_b).sum())
        return {
            "stacker_npz": str(stacker_npz),
            "base_npz": str(base_npz),
            "n_common": int(common_idx_s.size),
            "error": f"label disagreement on {n_disagree} common items",
        }

    p_s = s_p[common_idx_s]
    p_b = b_p[common_idx_b]

    r = delong_paired_test(y_s, p_s, p_b, alpha=alpha)
    return {
        "stacker_npz": str(stacker_npz),
        "base_npz": str(base_npz),
        "n_common": int(common_idx_s.size),
        "auroc_stacker": r.auroc_a,
        "auroc_base": r.auroc_b,
        "diff_stacker_minus_base": r.diff,
        "var_diff": r.var_diff,
        "z": r.z,
        "p_two_sided": r.p_two_sided,
        "ci_low_diff": r.ci_low,
        "ci_high_diff": r.ci_high,
        "n_pos": r.n_pos,
        "n_neg": r.n_neg,
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--stacker-oof",
        default="results/route_ab/hybrid_tuned_hybrid_v1_clean_pooled_oof.npz",
        help="Path to the hybrid stacker's OOF .npz.",
    )
    p.add_argument(
        "--out-dir",
        default="reports/route_ab",
        help="Directory for CSV / JSON outputs (created if needed).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Two-sided significance level; default 0.05 -> 95%% CI.",
    )
    p.add_argument(
        "--method",
        default="logit",
        choices=("logit", "wald"),
        help="CI method: logit (default, keeps CI in [0,1]) or wald.",
    )
    p.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit nonzero if any registered OOF file is missing. Otherwise "
        "print a warning and skip.",
    )
    args = p.parse_args()

    repo_root = _REPO_ROOT
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = _default_registry(args.stacker_oof)

    # ---- per-group rows ----
    all_rows = []
    print(
        f"# Computing per-group AUROC + {100 * (1 - args.alpha):.0f}% DeLong CI "
        f"(method={args.method})\n"
    )
    for entry in registry:
        npz_path = (repo_root / entry["npz"]).resolve()
        prov_path = (
            (repo_root / entry["provenance"]).resolve() if entry["provenance"] else None
        )
        if not npz_path.exists():
            msg = f"[WARN] missing OOF, skipping: {npz_path}"
            if args.fail_on_missing:
                print(msg)
                sys.exit(1)
            print(msg)
            continue
        rows = _per_model_rows(
            model_key=entry["model_key"],
            display_name=entry["display_name"],
            npz_path=npz_path,
            prov_path=prov_path,
            alpha=args.alpha,
            method=args.method,
        )
        overall = rows[0]
        leaky_tag = (
            "leaky"
            if overall["leaky"] is True
            else ("clean" if overall["leaky"] is False else "unknown")
        )
        print(
            f"[{entry['model_key']:22s}] n={overall['n']:5d}  "
            f"AUROC={overall['auroc']:.4f}  "
            f"95%CI=[{overall['ci_low']:.4f}, {overall['ci_high']:.4f}]  "
            f"provenance={leaky_tag}"
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    per_group_path = out_dir / "per_group_metrics_with_ci.csv"
    df.to_csv(per_group_path, index=False)
    print(f"\nwrote {per_group_path}  ({len(df)} rows)")

    # ---- pooled-only projection ----
    pooled_df = df[df["group"] == "OVERALL"].copy()
    pooled_path = out_dir / "pooled_metrics_with_ci.csv"
    pooled_df.to_csv(pooled_path, index=False)
    print(f"wrote {pooled_path}  ({len(pooled_df)} rows)")

    # ---- paired DeLong: stacker vs each base ----
    stacker_entry = next(e for e in registry if e["model_key"] == "hybrid_stacker")
    stacker_npz = (repo_root / stacker_entry["npz"]).resolve()
    paired_results = []
    if stacker_npz.exists():
        print("\n# Paired DeLong: hybrid_stacker vs. each base OOF")
        print(
            "# (evaluated on intersection of item_ids; CI is on the "
            "raw AUROC difference)\n"
        )
        for entry in registry:
            if entry["model_key"] == "hybrid_stacker":
                continue
            base_npz = (repo_root / entry["npz"]).resolve()
            if not base_npz.exists():
                continue
            result = _paired_stacker_vs_base(
                stacker_npz=stacker_npz,
                base_npz=base_npz,
                alpha=args.alpha,
            )
            result["model_key"] = entry["model_key"]
            result["display_name"] = entry["display_name"]
            paired_results.append(result)
            if "error" in result:
                print(
                    f"[{entry['model_key']:22s}] "
                    f"ERROR: {result['error']}  "
                    f"(n_common={result['n_common']})"
                )
            else:
                sig = "***" if result["p_two_sided"] < 0.001 else (
                    "**" if result["p_two_sided"] < 0.01 else (
                        "*" if result["p_two_sided"] < 0.05 else "ns"
                    )
                )
                print(
                    f"[{entry['model_key']:22s}]  "
                    f"AUC_stacker={result['auroc_stacker']:.4f}  "
                    f"AUC_base={result['auroc_base']:.4f}  "
                    f"diff=+{result['diff_stacker_minus_base']:.4f}  "
                    f"z={result['z']:+.2f}  p={result['p_two_sided']:.2e}  {sig}  "
                    f"(n={result['n_common']})"
                )
    else:
        print(f"\n[WARN] stacker OOF missing, skipping paired tests: {stacker_npz}")

    paired_path = out_dir / "paired_delong_stacker_vs_base.json"
    with open(paired_path, "w") as f:
        json.dump(
            {
                "stacker_oof": str(stacker_npz),
                "alpha": args.alpha,
                "results": paired_results,
            },
            f,
            indent=2,
            default=float,
        )
    print(f"\nwrote {paired_path}  ({len(paired_results)} pairs)")


if __name__ == "__main__":
    main()
