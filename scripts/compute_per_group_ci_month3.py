"""Compute per-group AUROC + DeLong 95% CI for every Month 3 OOF artifact.

Adapted from ``scripts/compute_per_group_ci.py`` (youxuan-update) to the
peng-update Month 3 registry:

    * Month 2 base learners (deberta, deberta_conditioned, step_transformer)
    * Month 3 super-hybrid re-aggregates + LR/RF stackers
    * Month 3 hidden-state probes (pooled & Qwen-only)
    * Month 3 multi-layer probe cells (all 32)

Outputs (under ``reports/month3/`` by default):

    per_group_metrics_with_ci.csv
        One row per (model, group) with AUROC, 95 % DeLong CI, variance,
        n / n_pos / n_neg, label_rate, method, scope. The ``group="OVERALL"``
        row per model is the pooled AUROC on *that model's own* sample
        (useful when OOFs cover slightly different subsets of items, e.g.
        Qwen-only probes vs the pooled OOFs).

    pooled_metrics_with_ci.csv
        Compact one-row-per-model projection of the OVERALL rows.

    paired_delong_superhybrid_vs_base.json
        For each registered base OOF, a paired DeLong test:
            H0 : AUROC(SuperHybrid_LR) == AUROC(base)
            H0 : AUROC(SuperHybrid_RF) == AUROC(base)
        evaluated on the *intersection* of (group, item_id) pairs (so the
        two models see exactly the same items). Reports z, two-sided p,
        raw-scale CI on the difference.

Usage (from repo root, with any env that has numpy + scipy + sklearn +
pandas):

    PYTHONPATH=. python scripts/compute_per_group_ci_month3.py

Optional flags:

    --out-dir     reports/month3_v2       # override output directory
    --auto        true|false              # auto-discover unregistered OOFs
    --alpha       0.05                    # CI significance level
    --method      logit|wald              # CI method
"""

from __future__ import annotations

import argparse
import glob
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
    delong_paired_test,
    per_group_auroc_ci,
)


# ---------------------------------------------------------------------------
# Hand-curated model registry. Covers every Month 2/3 OOF artifact we have
# on peng-update that we want to assign a display-name + scope tag to.
#
# scope values:
#   "pooled_8"      - all 8 {dataset × model} groups, n≈6378
#   "qwen_only_4"   - 4 Qwen-only groups (MLP hidden-state probes fit only
#                      on Qwen because of hidden_dim differences)
#   "pooled_8_mlp"  - multi-layer probes, pooled 8-group (n=6378)
# ---------------------------------------------------------------------------


_REGISTRY: list[dict] = [
    # ---- Month 2 base learners (pooled_8) ----
    {
        "model_key": "deberta_pooled",
        "display_name": "DeBERTa-v3-base (trace tail, pooled)",
        "family": "text",
        "scope": "pooled_8",
        "npz": "results/month2/deberta_pooled_oof.npz",
    },
    {
        "model_key": "deberta_conditioned",
        "display_name": "DeBERTa-v3 [problem||trace] (pooled)",
        "family": "text",
        "scope": "pooled_8",
        "npz": "results/month2/deberta_conditioned_pooled_oof.npz",
    },
    {
        "model_key": "step_transformer",
        "display_name": "Step Transformer",
        "family": "structure",
        "scope": "pooled_8",
        "npz": "results/month2/step_transformer_pooled_oof.npz",
    },
    # ---- Month 3 super-hybrid re-aggregates + stackers (pooled_8) ----
    {
        "model_key": "superhybrid_deberta",
        "display_name": "SuperHybrid base: DeBERTa re-agg",
        "family": "text",
        "scope": "pooled_8",
        "npz": "results/month3/superhybrid_DeBERTa_oof.npz",
    },
    {
        "model_key": "superhybrid_deberta_cond",
        "display_name": "SuperHybrid base: DeBERTa_Cond re-agg",
        "family": "text",
        "scope": "pooled_8",
        "npz": "results/month3/superhybrid_DeBERTa_Cond_oof.npz",
    },
    {
        "model_key": "superhybrid_threeprobs",
        "display_name": "SuperHybrid base: 3-probs passthrough",
        "family": "fusion",
        "scope": "pooled_8",
        "npz": "results/month3/superhybrid_ThreeProbs_oof.npz",
    },
    {
        "model_key": "superhybrid_lr",
        "display_name": "SuperHybrid LR stacker (headline)",
        "family": "stacker",
        "scope": "pooled_8",
        "npz": "results/month3/superhybrid_SuperHybrid_LR_oof.npz",
    },
    {
        "model_key": "superhybrid_rf",
        "display_name": "SuperHybrid RF stacker (headline)",
        "family": "stacker",
        "scope": "pooled_8",
        "npz": "results/month3/superhybrid_SuperHybrid_RF_oof.npz",
    },
    # ---- Month 3 hidden-state probes, pooled_8 ----
    {
        "model_key": "probe_mlp_h_answer_pooled",
        "display_name": "Probe MLP h_answer (pooled)",
        "family": "hidden",
        "scope": "pooled_8",
        "npz": "results/month3/hidden_probe_pooled_mlp_h_answer_oof.npz",
    },
    {
        "model_key": "probe_mlp_concat_pooled",
        "display_name": "Probe MLP concat(h_last,h_think,h_answer) (pooled)",
        "family": "hidden",
        "scope": "pooled_8",
        "npz": "results/month3/hidden_probe_pooled_mlp_concat_oof.npz",
    },
    {
        "model_key": "probe_mlp_hidden_plus_genunc_pooled",
        "display_name": "Probe MLP hidden+genunc (pooled)",
        "family": "hidden",
        "scope": "pooled_8",
        "npz": "results/month3/hidden_probe_pooled_mlp_hidden_plus_genunc_oof.npz",
    },
    {
        "model_key": "probe_linear_concat_pooled",
        "display_name": "Probe Linear concat (pooled)",
        "family": "hidden",
        "scope": "pooled_8",
        "npz": "results/month3/hidden_probe_pooled_linear_concat_oof.npz",
    },
    {
        "model_key": "probe_linear_h_answer_pooled",
        "display_name": "Probe Linear h_answer (pooled)",
        "family": "hidden",
        "scope": "pooled_8",
        "npz": "results/month3/hidden_probe_pooled_linear_h_answer_oof.npz",
    },
    {
        "model_key": "probe_linear_h_last_pooled",
        "display_name": "Probe Linear h_last (pooled)",
        "family": "hidden",
        "scope": "pooled_8",
        "npz": "results/month3/hidden_probe_pooled_linear_h_last_oof.npz",
    },
    {
        "model_key": "probe_linear_h_think_pooled",
        "display_name": "Probe Linear h_think (pooled)",
        "family": "hidden",
        "scope": "pooled_8",
        "npz": "results/month3/hidden_probe_pooled_linear_h_think_oof.npz",
    },
    {
        "model_key": "probe_lr_genunc_pooled",
        "display_name": "Generation-uncertainty LR only (pooled)",
        "family": "genunc",
        "scope": "pooled_8",
        "npz": "results/month3/hidden_probe_pooled_lr_genunc_oof.npz",
    },
    {
        "model_key": "probe_rf_genunc_pooled",
        "display_name": "Generation-uncertainty RF only (pooled)",
        "family": "genunc",
        "scope": "pooled_8",
        "npz": "results/month3/hidden_probe_pooled_rf_genunc_oof.npz",
    },
    # ---- Month 3 hidden-state probes, qwen_only (4 Qwen groups) ----
    {
        "model_key": "probe_mlp_h_answer_qwen",
        "display_name": "Probe MLP h_answer (Qwen only)",
        "family": "hidden",
        "scope": "qwen_only_4",
        "npz": "results/month3/hidden_probe_qwen_mlp_h_answer_oof.npz",
    },
    {
        "model_key": "probe_mlp_concat_qwen",
        "display_name": "Probe MLP concat (Qwen only)",
        "family": "hidden",
        "scope": "qwen_only_4",
        "npz": "results/month3/hidden_probe_qwen_mlp_concat_oof.npz",
    },
    {
        "model_key": "probe_mlp_hidden_plus_genunc_qwen",
        "display_name": "Probe MLP hidden+genunc (Qwen only)",
        "family": "hidden",
        "scope": "qwen_only_4",
        "npz": "results/month3/hidden_probe_qwen_mlp_hidden_plus_genunc_oof.npz",
    },
    {
        "model_key": "probe_linear_concat_qwen",
        "display_name": "Probe Linear concat (Qwen only)",
        "family": "hidden",
        "scope": "qwen_only_4",
        "npz": "results/month3/hidden_probe_qwen_linear_concat_oof.npz",
    },
    {
        "model_key": "probe_linear_h_answer_qwen",
        "display_name": "Probe Linear h_answer (Qwen only)",
        "family": "hidden",
        "scope": "qwen_only_4",
        "npz": "results/month3/hidden_probe_qwen_linear_h_answer_oof.npz",
    },
    {
        "model_key": "probe_linear_h_last_qwen",
        "display_name": "Probe Linear h_last (Qwen only)",
        "family": "hidden",
        "scope": "qwen_only_4",
        "npz": "results/month3/hidden_probe_qwen_linear_h_last_oof.npz",
    },
    {
        "model_key": "probe_linear_h_think_qwen",
        "display_name": "Probe Linear h_think (Qwen only)",
        "family": "hidden",
        "scope": "qwen_only_4",
        "npz": "results/month3/hidden_probe_qwen_linear_h_think_oof.npz",
    },
    {
        "model_key": "probe_lr_genunc_qwen",
        "display_name": "Generation-uncertainty LR only (Qwen)",
        "family": "genunc",
        "scope": "qwen_only_4",
        "npz": "results/month3/hidden_probe_qwen_lr_genunc_oof.npz",
    },
    {
        "model_key": "probe_rf_genunc_qwen",
        "display_name": "Generation-uncertainty RF only (Qwen)",
        "family": "genunc",
        "scope": "qwen_only_4",
        "npz": "results/month3/hidden_probe_qwen_rf_genunc_oof.npz",
    },
]


# ---------------------------------------------------------------------------
# Auto-discovery: adds any *_oof.npz under results/month{2,3}/ that isn't
# already in the registry. Given the fixed file naming this is robust.
# ---------------------------------------------------------------------------


def _auto_discover(repo_root: Path) -> list[dict]:
    known = {e["npz"] for e in _REGISTRY}
    extras: list[dict] = []
    for d in ("results/month2", "results/month3"):
        for p in sorted(glob.glob(str(repo_root / d / "*_oof.npz"))):
            rel = str(Path(p).relative_to(repo_root))
            if rel in known:
                continue
            stem = Path(p).stem
            # Classify family / scope from naming convention
            if "multi_layer_probe" in stem:
                family = "multi_layer"
                scope = "pooled_8_mlp"
            elif "hidden_probe_qwen" in stem:
                family = "hidden"
                scope = "qwen_only_4"
            elif "hidden_probe_pooled" in stem:
                family = "hidden"
                scope = "pooled_8"
            elif "superhybrid" in stem:
                family = "stacker"
                scope = "pooled_8"
            else:
                family = "other"
                scope = "pooled_8"
            extras.append(
                {
                    "model_key": stem,
                    "display_name": stem,
                    "family": family,
                    "scope": scope,
                    "npz": rel,
                }
            )
    return extras


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


def _per_model_rows(entry: dict, npz_path: Path, alpha: float, method: str) -> list[dict]:
    """Run DeLong per-group + pooled on one OOF artifact, return CSV rows."""
    item_ids, groups, y_true, oof_prob = _load_oof(npz_path)

    res = per_group_auroc_ci(
        y_true=y_true,
        y_score=oof_prob,
        groups=groups,
        alpha=alpha,
        method=method,
    )

    pooled = res["pooled"]
    rows: list[dict] = [
        {
            "model_key": entry["model_key"],
            "model_name": entry["display_name"],
            "family": entry["family"],
            "scope": entry["scope"],
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
            "note": "",
        }
    ]
    for row in res["per_group"]:
        rows.append(
            {
                "model_key": entry["model_key"],
                "model_name": entry["display_name"],
                "family": entry["family"],
                "scope": entry["scope"],
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
                "note": row["note"],
            }
        )
    return rows


def _paired_vs_base(
    stacker_npz: Path,
    base_npz: Path,
    alpha: float,
) -> dict:
    """Run a paired DeLong test on the intersection of (group, item_id) pairs."""
    s_ids, s_groups, s_y, s_p = _load_oof(stacker_npz)
    b_ids, b_groups, b_y, b_p = _load_oof(base_npz)

    # Intersect on (group, item_id). item_id alone is not unique
    # (e.g. "arc_0000" exists in both arc_challenge_llama8b and
    # arc_challenge_qwen7b groups).
    s_key = np.asarray(
        [f"{g}||{i}" for g, i in zip(s_groups.tolist(), s_ids.tolist())]
    )
    b_key = np.asarray(
        [f"{g}||{i}" for g, i in zip(b_groups.tolist(), b_ids.tolist())]
    )
    s_index = {k: idx for idx, k in enumerate(s_key)}
    common_idx_s, common_idx_b = [], []
    for idx, k in enumerate(b_key):
        if k in s_index:
            common_idx_s.append(s_index[k])
            common_idx_b.append(idx)

    if not common_idx_s:
        return {
            "stacker_npz": str(stacker_npz.name),
            "base_npz": str(base_npz.name),
            "n_common": 0,
            "error": "no overlapping (group, item_id) pairs",
        }

    common_idx_s = np.asarray(common_idx_s)
    common_idx_b = np.asarray(common_idx_b)
    y_s = s_y[common_idx_s]
    y_b = b_y[common_idx_b]
    if not np.array_equal(y_s, y_b):
        n_disagree = int((y_s != y_b).sum())
        return {
            "stacker_npz": str(stacker_npz.name),
            "base_npz": str(base_npz.name),
            "n_common": int(common_idx_s.size),
            "error": f"label disagreement on {n_disagree} common items",
        }

    p_s = s_p[common_idx_s]
    p_b = b_p[common_idx_b]

    r = delong_paired_test(y_s, p_s, p_b, alpha=alpha)
    return {
        "stacker_npz": str(stacker_npz.name),
        "base_npz": str(base_npz.name),
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
        "--out-dir",
        default="reports/month3",
        help="Directory for CSV / JSON outputs (created if needed).",
    )
    p.add_argument("--alpha", type=float, default=0.05, help="95%% CI by default.")
    p.add_argument(
        "--method",
        default="logit",
        choices=("logit", "wald"),
        help="CI method (logit keeps CI inside [0,1]).",
    )
    p.add_argument(
        "--auto",
        type=lambda s: s.lower() in ("true", "1", "yes"),
        default=True,
        help="If true, also CI every unregistered *_oof.npz under results/month{2,3}/.",
    )
    p.add_argument(
        "--stacker-lr",
        default="results/month3/superhybrid_SuperHybrid_LR_oof.npz",
        help="SuperHybrid_LR OOF path for paired tests.",
    )
    p.add_argument(
        "--stacker-rf",
        default="results/month3/superhybrid_SuperHybrid_RF_oof.npz",
        help="SuperHybrid_RF OOF path for paired tests.",
    )
    p.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit nonzero if any registered OOF file is missing.",
    )
    args = p.parse_args()

    out_dir = (_REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build full model list
    registry: list[dict] = list(_REGISTRY)
    if args.auto:
        extras = _auto_discover(_REPO_ROOT)
        registry.extend(extras)
        print(f"# Auto-discovered {len(extras)} additional OOFs")
    print(f"# Total models to CI: {len(registry)}\n")

    # ---- per-group rows ----
    all_rows: list[dict] = []
    print(
        f"# Computing per-group AUROC + {100 * (1 - args.alpha):.0f}% DeLong CI "
        f"(method={args.method})\n"
    )
    for entry in registry:
        npz_path = (_REPO_ROOT / entry["npz"]).resolve()
        if not npz_path.exists():
            msg = f"[WARN] missing OOF, skipping: {npz_path}"
            if args.fail_on_missing:
                print(msg)
                sys.exit(1)
            print(msg)
            continue
        try:
            rows = _per_model_rows(entry, npz_path, args.alpha, args.method)
        except Exception as exc:
            print(f"[ERROR] {entry['model_key']}: {exc}")
            continue
        overall = rows[0]
        print(
            f"[{entry['model_key']:48s}] n={overall['n']:5d}  "
            f"AUROC={overall['auroc']:.4f}  "
            f"95%CI=[{overall['ci_low']:.4f}, {overall['ci_high']:.4f}]  "
            f"{entry['scope']}"
        )
        all_rows.extend(rows)

    if not all_rows:
        print("No rows produced; aborting.")
        sys.exit(2)

    df = pd.DataFrame(all_rows)
    per_group_path = out_dir / "per_group_metrics_with_ci.csv"
    df.to_csv(per_group_path, index=False)
    print(f"\nwrote {per_group_path}  ({len(df)} rows)")

    # ---- pooled-only projection ----
    pooled_df = df[df["group"] == "OVERALL"].copy()
    pooled_df = pooled_df.sort_values(["scope", "family", "auroc"],
                                      ascending=[True, True, False])
    pooled_path = out_dir / "pooled_metrics_with_ci.csv"
    pooled_df.to_csv(pooled_path, index=False)
    print(f"wrote {pooled_path}  ({len(pooled_df)} rows)")

    # ---- paired DeLong: each stacker vs each base ----
    paired_results = []
    for stacker_key, stacker_label, stacker_path in [
        ("superhybrid_lr", "SuperHybrid_LR", args.stacker_lr),
        ("superhybrid_rf", "SuperHybrid_RF", args.stacker_rf),
    ]:
        stacker_npz = (_REPO_ROOT / stacker_path).resolve()
        if not stacker_npz.exists():
            print(f"\n[WARN] stacker OOF missing, skipping paired tests: {stacker_npz}")
            continue
        print(f"\n# Paired DeLong: {stacker_label} vs. each base OOF")
        for entry in registry:
            if entry["model_key"] == stacker_key:
                continue
            base_npz = (_REPO_ROOT / entry["npz"]).resolve()
            if not base_npz.exists():
                continue
            result = _paired_vs_base(
                stacker_npz=stacker_npz,
                base_npz=base_npz,
                alpha=args.alpha,
            )
            result["stacker_key"] = stacker_key
            result["base_key"] = entry["model_key"]
            result["base_family"] = entry["family"]
            result["base_scope"] = entry["scope"]
            paired_results.append(result)
            if "error" in result:
                print(
                    f"[{stacker_label} vs {entry['model_key']:44s}]  "
                    f"ERROR: {result['error']}  (n_common={result['n_common']})"
                )
            else:
                sig = (
                    "***" if result["p_two_sided"] < 0.001
                    else "**" if result["p_two_sided"] < 0.01
                    else "*" if result["p_two_sided"] < 0.05
                    else "ns"
                )
                print(
                    f"[{stacker_label} vs {entry['model_key']:44s}]  "
                    f"AUC_S={result['auroc_stacker']:.4f} "
                    f"AUC_B={result['auroc_base']:.4f} "
                    f"diff={result['diff_stacker_minus_base']:+.4f} "
                    f"z={result['z']:+.2f} p={result['p_two_sided']:.2e} {sig} "
                    f"(n={result['n_common']})"
                )

    paired_path = out_dir / "paired_delong_superhybrid_vs_base.json"
    with open(paired_path, "w") as f:
        json.dump(
            {
                "stacker_lr": args.stacker_lr,
                "stacker_rf": args.stacker_rf,
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
