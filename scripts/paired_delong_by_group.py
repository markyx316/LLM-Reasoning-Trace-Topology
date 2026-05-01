"""Paired DeLong AUROC comparison between any two OOF artifacts, broken
down by group / model family / dataset family / overall.

Motivated by Phase 1.3: we want to know whether SuperHybrid_LR is
robustly preferable to SuperHybrid_RF, or whether the relative ordering
flips across {qwen, llama} model-family cuts and across per-dataset
cuts. A pooled test can mask opposing signals inside slices.

Inputs:
    --a   path/to/A_oof.npz   # "challenger"
    --b   path/to/B_oof.npz   # "baseline"

Both .npz files are expected to follow the project contract:
    {item_ids, groups, y_true, oof_prob, seed, n_splits}

Outputs (JSON+CSV under --out-dir):

    paired_<tag>_by_group.csv
        One row per slice (overall, per-group, per-model-family,
        per-dataset-family) with paired AUROC_a, AUROC_b, diff, z, p,
        95% DeLong CI on diff, n_common, slice-definition metadata.

    paired_<tag>_by_group.json
        Same rows, JSON-structured + run metadata.

Usage:

    PYTHONPATH=. python scripts/paired_delong_by_group.py \
        --a results/month3/superhybrid_SuperHybrid_LR_oof.npz \
        --b results/month3/superhybrid_SuperHybrid_RF_oof.npz \
        --tag sh_lr_vs_sh_rf \
        --out-dir reports/month3

Group naming convention assumed: "{dataset}_{model_short}" where
dataset ∈ {math500, gsm8k, gpqa_diamond, arc_challenge} and model_short
∈ {qwen7b, llama8b}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.analysis.delong_ci import delong_paired_test  # noqa: E402


def _load_oof(npz_path: Path):
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


def _intersect(
    a_ids, a_groups, a_y, a_p,
    b_ids, b_groups, b_y, b_p,
):
    """Intersect on (group, item_id), return aligned arrays (a/b) + the
    group array for the common items."""
    a_key = np.asarray([f"{g}||{i}" for g, i in zip(a_groups.tolist(), a_ids.tolist())])
    b_key = np.asarray([f"{g}||{i}" for g, i in zip(b_groups.tolist(), b_ids.tolist())])
    a_index = {k: idx for idx, k in enumerate(a_key)}
    a_sel, b_sel = [], []
    for idx, k in enumerate(b_key):
        if k in a_index:
            a_sel.append(a_index[k])
            b_sel.append(idx)
    if not a_sel:
        return None
    a_sel = np.asarray(a_sel)
    b_sel = np.asarray(b_sel)
    if not np.array_equal(a_y[a_sel], b_y[b_sel]):
        n_dis = int((a_y[a_sel] != b_y[b_sel]).sum())
        raise ValueError(f"label disagreement on {n_dis} common items (bug upstream)")
    return {
        "y": a_y[a_sel],
        "p_a": a_p[a_sel],
        "p_b": b_p[b_sel],
        "groups": a_groups[a_sel],
        "item_ids": a_ids[a_sel],
    }


def _parse_group(g: str):
    """Split a group like 'arc_challenge_qwen7b' into (dataset, model_short)."""
    MODEL_SHORTS = ("qwen7b", "llama8b", "qwen14b", "deepseek_r1")
    for m in MODEL_SHORTS:
        suffix = f"_{m}"
        if g.endswith(suffix):
            return (g[: -len(suffix)], m)
    return (g, "unknown")


def _safe_paired(y, p_a, p_b, alpha, min_pos=2, min_neg=2):
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos < min_pos or n_neg < min_neg:
        return {
            "auroc_a": float("nan"),
            "auroc_b": float("nan"),
            "diff": float("nan"),
            "var_diff": float("nan"),
            "z": float("nan"),
            "p_two_sided": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "note": f"skipped (n_pos={n_pos} n_neg={n_neg})",
        }
    r = delong_paired_test(y, p_a, p_b, alpha=alpha)
    return {
        "auroc_a": r.auroc_a,
        "auroc_b": r.auroc_b,
        "diff": r.diff,
        "var_diff": r.var_diff,
        "z": r.z,
        "p_two_sided": r.p_two_sided,
        "ci_low": r.ci_low,
        "ci_high": r.ci_high,
        "n_pos": r.n_pos,
        "n_neg": r.n_neg,
        "note": "",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--a", required=True, help="OOF A (challenger)")
    p.add_argument("--b", required=True, help="OOF B (baseline)")
    p.add_argument(
        "--tag", required=True,
        help="Short identifier for output filenames, e.g. sh_lr_vs_sh_rf.",
    )
    p.add_argument("--out-dir", default="reports/month3")
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    out_dir = (_REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    a_path = (_REPO_ROOT / args.a).resolve()
    b_path = (_REPO_ROOT / args.b).resolve()

    a_ids, a_groups, a_y, a_p = _load_oof(a_path)
    b_ids, b_groups, b_y, b_p = _load_oof(b_path)

    aligned = _intersect(a_ids, a_groups, a_y, a_p, b_ids, b_groups, b_y, b_p)
    if aligned is None:
        print("No common (group, item_id) pairs, aborting.")
        sys.exit(2)
    y = aligned["y"]; p_a = aligned["p_a"]; p_b = aligned["p_b"]
    groups = aligned["groups"]

    rows: list[dict] = []

    # ---- Overall ----
    r = _safe_paired(y, p_a, p_b, args.alpha)
    rows.append({
        "slice_kind": "overall",
        "slice_value": "OVERALL",
        "n": int(y.size),
        **r,
    })

    # ---- Per group ----
    for g in sorted(np.unique(groups).tolist()):
        mask = groups == g
        r = _safe_paired(y[mask], p_a[mask], p_b[mask], args.alpha)
        rows.append({
            "slice_kind": "group",
            "slice_value": g,
            "n": int(mask.sum()),
            **r,
        })

    # ---- Per model_family (qwen, llama) ----
    parsed = np.asarray([_parse_group(g) for g in groups.tolist()])  # (n, 2)
    model_fam = parsed[:, 1]
    for m in sorted(np.unique(model_fam).tolist()):
        mask = model_fam == m
        r = _safe_paired(y[mask], p_a[mask], p_b[mask], args.alpha)
        rows.append({
            "slice_kind": "model_family",
            "slice_value": m,
            "n": int(mask.sum()),
            **r,
        })

    # ---- Per dataset_family (math500, gsm8k, ...) ----
    dset_fam = parsed[:, 0]
    for d in sorted(np.unique(dset_fam).tolist()):
        mask = dset_fam == d
        r = _safe_paired(y[mask], p_a[mask], p_b[mask], args.alpha)
        rows.append({
            "slice_kind": "dataset_family",
            "slice_value": d,
            "n": int(mask.sum()),
            **r,
        })

    df = pd.DataFrame(rows)
    df["sig"] = df["p_two_sided"].apply(
        lambda p: (
            "" if np.isnan(p)
            else "***" if p < 0.001
            else "**" if p < 0.01
            else "*" if p < 0.05
            else "ns"
        )
    )

    csv_path = out_dir / f"paired_{args.tag}_by_group.csv"
    json_path = out_dir / f"paired_{args.tag}_by_group.json"
    df.to_csv(csv_path, index=False)

    payload = {
        "a_oof": str(args.a),
        "b_oof": str(args.b),
        "tag": args.tag,
        "alpha": args.alpha,
        "n_total_common": int(y.size),
        "rows": df.to_dict(orient="records"),
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)

    print(f"\nPaired DeLong ({args.tag}): A={os.path.basename(args.a)}  B={os.path.basename(args.b)}")
    print(f"n_common={y.size}")
    print()
    # Pretty print
    pretty = df.copy()
    pretty["auroc_a"] = pretty["auroc_a"].round(4)
    pretty["auroc_b"] = pretty["auroc_b"].round(4)
    pretty["diff"] = pretty["diff"].round(4)
    pretty["ci_low"] = pretty["ci_low"].round(4)
    pretty["ci_high"] = pretty["ci_high"].round(4)
    pretty["p_two_sided"] = pretty["p_two_sided"].apply(
        lambda v: "nan" if np.isnan(v) else f"{v:.2e}"
    )
    cols = ["slice_kind", "slice_value", "n", "auroc_a", "auroc_b",
            "diff", "ci_low", "ci_high", "p_two_sided", "sig"]
    print(pretty[cols].to_string(index=False))
    print(f"\nwrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
