#!/usr/bin/env python
"""
build_hybrid_table.py — Join all 6 base-model OOFs + all 6 Route-A feature
families + labels into a single wide parquet table keyed by (group, item_id),
ready for hyperparameter tuning of the meta-stacker.

Design rationale
----------------
The hybrid meta-learner takes as input:
  * 6 base-model OOF probabilities (RoBERTa, DeBERTa, StepTF, GIN-hyb,
    GIN-str, Shapelet)
  * Raw Route-A structural features: handcrafted+recurrence, n-gram,
    graph, timing, structural persistent-homology (+ optionally raw
    shapelet distances, deferred)
  * Group metadata (dataset_model) and the canonical fold assignment
We want all of these in a single wide table so that tuning trials only pay
disk I/O once.

Key hazards we handle
~~~~~~~~~~~~~~~~~~~~~
1. **OOF fold disagreement.** RoBERTa/DeBERTa/StepTF share folds exactly;
   GIN-hyb/GIN-str share a different fold partition; shapelet has no fold
   column at all. We canonicalize on RoBERTa's `oof_fold` (best-calibrated
   base, 100% coverage) and fill the rest via the (group, item_id) join.
   This is leakage-safe because each base-model's OOF prob is still a
   leave-this-item-out prediction under that base's scheme; the meta-CV
   fold we use for tuning is separate from the base's own CV scheme.

2. **`dataset` column inconsistency across CSV families.** `features_rec`
   says `dataset='math500'` while `ngram`/`graph`/`timing`/`structural_ph`
   say `dataset='math500_qwen7b'`. We IGNORE every CSV's internal
   `dataset` column and derive `group` from the filename glob. This gives
   us one canonical grouping across everything.

3. **Item-id collisions across models.** `math500_0000` exists for both
   `math500_qwen7b` and `math500_llama8b`. Joining on `item_id` alone
   would mix them. We always use (group, item_id) as the composite key.

4. **Partial coverage.** GIN OOFs cover 9 groups (includes
   `math500_deepseek_r1`, n=20); RoBERTa covers 8. We do an INNER JOIN
   on base OOFs first — that naturally drops the deepseek group, which
   is too small (per-fold AUROC std 0.327) for defensible stacking.
   Route-A CSVs are LEFT-joined; missing cells are left as NaN and the
   tuner can impute or mask.

5. **Leakage provenance.** 5 of 6 base OOFs are currently tainted by the
   best-epoch-on-val pattern (patched in source 2026-04-20, but existing
   .npz files were produced before the patch). We emit a summary of
   each OOF's provenance into the parquet's sibling JSON and gate
   consumption via `--leaky-policy`.

Output contract
---------------
  data/hybrid_table.parquet  — one row per (group, item_id); columns:
      group          str    e.g. 'math500_qwen7b'
      item_id        str    e.g. 'math500_0000'
      label          int    0/1 (agrees across all sources or build fails)
      fold           int    canonical meta-CV fold (from RoBERTa)
      oof_roberta    float  RoBERTa pooled OOF prob
      oof_deberta    float  DeBERTa pooled OOF prob
      oof_step       float  Step Transformer pooled OOF prob
      oof_gin_hyb    float  TraceGIN-hybrid pooled OOF prob
      oof_gin_str    float  TraceGIN-structural pooled OOF prob
      oof_shapelet   float  Shapelet LR pooled OOF prob
      hand_*         float  25 + 5 recurrence handcrafted features (features_rec)
      ng_*           float  ~230 n-gram transition count/rate features
      graph_*        float  16 graph-structure features
      timing_*       float  ~46 timing/IEI/burstiness features
      ph_*           float  ~36 persistent-homology features

  data/hybrid_table.META.json — per-column origin, provenance list per OOF,
      row counts per group, inner/left-join diagnostics, what was dropped.

Usage
-----
    PYTHONPATH=. python scripts/build_hybrid_table.py              # build
    PYTHONPATH=. python scripts/build_hybrid_table.py --dry-run    # inspect
    PYTHONPATH=. python scripts/build_hybrid_table.py \\
        --leaky-policy fail                                         # refuse tainted OOFs
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG: OOF paths + column names
# =============================================================================

# Ordered so `oof_roberta` is first — we use it as the canonical fold source.
OOF_SOURCES = [
    ("oof_roberta",  "results/roberta_pooled_oof.npz"),
    ("oof_deberta",  "results/month2/deberta_pooled_oof.npz"),
    ("oof_step",     "results/step_transformer_pooled_oof.npz"),
    ("oof_gin_hyb",  "results/route_ab/trace_gnn_hybrid_pooled_oof.npz"),
    ("oof_gin_str",  "results/route_ab/trace_gnn_structural_pooled_oof.npz"),
    ("oof_shapelet", "results/route_ab/shapelet_oof.npz"),
]

# Which OOF provides the canonical meta-CV fold. Must have `oof_fold` key.
CANONICAL_FOLD_SOURCE = "oof_roberta"

# Route-A feature CSV families.
#   prefix: new column prefix in the hybrid table (replaces raw col names)
#   suffix: file suffix following the group name (e.g. math500_qwen7b_{suffix}.csv)
FEATURE_FAMILIES = [
    dict(prefix="hand",   suffix="features_rec"),
    dict(prefix="ng",     suffix="ngram"),
    dict(prefix="graph",  suffix="graph"),
    dict(prefix="timing", suffix="timing"),
    dict(prefix="ph",     suffix="structural_ph"),
]

# Columns in feature CSVs that are identity/label metadata — never prefixed,
# never kept as feature columns.
META_COLS = {"item_id", "dataset", "is_correct"}

# Standard 8 groups (exclude math500_deepseek_r1 which is too small).
CANONICAL_GROUPS = {
    "arc_challenge_llama8b",
    "arc_challenge_qwen7b",
    "gpqa_diamond_llama8b",
    "gpqa_diamond_qwen7b",
    "gsm8k_llama8b",
    "gsm8k_qwen7b",
    "math500_llama8b",
    "math500_qwen7b",
}


# =============================================================================
# HELPERS
# =============================================================================

def _group_from_filename(fname: str, suffix: str) -> Optional[str]:
    """Derive group name from a CSV filename. Returns None if no match.

    E.g. ('math500_qwen7b_features_rec.csv', 'features_rec') -> 'math500_qwen7b'.
    """
    stem = Path(fname).stem  # e.g. math500_qwen7b_features_rec
    expected = f"_{suffix}"
    if not stem.endswith(expected):
        return None
    return stem[: -len(expected)]


def _load_oof(path: Path, name: str) -> pd.DataFrame:
    """Load a single OOF .npz into a long DataFrame indexed by
    (group, item_id)."""
    z = np.load(path, allow_pickle=True)
    if not {"item_ids", "y_true", "oof_prob", "groups"}.issubset(z.files):
        raise ValueError(f"{path}: missing required keys. Got {list(z.files)}")
    df = pd.DataFrame({
        "group":   [str(g) for g in z["groups"]],
        "item_id": [str(i) for i in z["item_ids"]],
        "label":   z["y_true"].astype(np.int8),
        name:      z["oof_prob"].astype(np.float32),
    })
    if "oof_fold" in z.files:
        df[f"_fold_{name}"] = z["oof_fold"].astype(np.int8)
    return df


def _load_provenance(path: Path) -> dict:
    sidecar = path.with_suffix(path.suffix + ".PROVENANCE.json")
    if not sidecar.exists():
        return {"sidecar_missing": True}
    return json.loads(sidecar.read_text())


def _load_feature_family(prefix: str, suffix: str,
                         groups: set[str]) -> pd.DataFrame:
    """Load every `{group}_{suffix}.csv` under data/features/, concatenate,
    prefix non-meta cols with `prefix_`, and return (group, item_id) + feature
    columns."""
    fdir = REPO / "data" / "features"
    pattern = f"*_{suffix}.csv"
    paths = sorted(fdir.glob(pattern))
    if not paths:
        logger.warning(f"  No CSV matching {pattern} under data/features/")
        return pd.DataFrame()

    parts = []
    feature_cols_union = None
    for p in paths:
        g = _group_from_filename(p.name, suffix)
        if g is None or g not in groups:
            continue
        df = pd.read_csv(p)
        # Drop meta cols we don't want in the feature table (we carry our own
        # group/label from the OOF join).
        feat_cols = [c for c in df.columns if c not in META_COLS]
        if feature_cols_union is None:
            feature_cols_union = set(feat_cols)
        else:
            mismatch = feature_cols_union.symmetric_difference(feat_cols)
            if mismatch:
                logger.warning(f"    {p.name}: feature-column mismatch vs "
                               f"union ({len(mismatch)} cols); taking union")
                feature_cols_union |= set(feat_cols)

        sub = df[["item_id"] + feat_cols].copy()
        sub["group"] = g
        # Prefix feature cols to avoid collisions across families
        rename = {c: f"{prefix}_{c}" for c in feat_cols}
        sub = sub.rename(columns=rename)
        parts.append(sub)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, axis=0, ignore_index=True, sort=False)
    # Order cols: group, item_id, then prefixed features
    feat_cols = sorted([c for c in out.columns if c.startswith(f"{prefix}_")])
    out = out[["group", "item_id"] + feat_cols]
    # Dedup on (group,item_id) in case a file was listed twice
    n_before = len(out)
    out = out.drop_duplicates(subset=["group", "item_id"], keep="first")
    if len(out) != n_before:
        logger.warning(f"    {prefix}: dropped {n_before - len(out)} "
                       f"duplicates on (group,item_id)")
    return out


# =============================================================================
# MAIN BUILD
# =============================================================================

def build_hybrid_table(output_parquet: str,
                       output_meta: str,
                       leaky_policy: str = "warn",
                       dry_run: bool = False) -> dict:
    """Build the unified hybrid table. Returns a diagnostics dict."""
    diagnostics: dict = {
        "output_parquet": output_parquet,
        "oofs": [],
        "features": [],
        "counts": {},
        "provenance_summary": {},
    }

    # ------------------------------------------------------------------
    # Step 1: Inner-join all 6 OOFs on (group, item_id)
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("Step 1: Loading + inner-joining 6 base-model OOFs")
    logger.info("=" * 70)

    merged: Optional[pd.DataFrame] = None
    oof_provs = {}
    any_leaky = False

    for name, rel in OOF_SOURCES:
        full = REPO / rel
        if not full.exists():
            msg = f"OOF MISSING: {rel}"
            logger.error(msg)
            diagnostics["oofs"].append({"name": name, "path": rel,
                                         "status": "missing"})
            raise FileNotFoundError(msg)

        prov = _load_provenance(full)
        oof_provs[name] = prov
        is_leaky = bool(prov.get("leaky", False))
        any_leaky = any_leaky or is_leaky
        status_tag = "LEAKY" if is_leaky else "clean"

        df = _load_oof(full, name)
        logger.info(f"  {name:14s}  n={len(df):5d}  groups={df['group'].nunique()}  "
                    f"AUROC~(see metrics file)  [{status_tag}]")
        diagnostics["oofs"].append({
            "name": name, "path": rel, "n_rows": len(df),
            "n_groups": int(df["group"].nunique()),
            "has_fold": f"_fold_{name}" in df.columns,
            "leaky": is_leaky,
            "protocol": prov.get("protocol", "unknown"),
        })

        if merged is None:
            merged = df
            continue

        # Verify label agreement on the join keys, then drop the RHS label
        tmp = merged.merge(df, on=["group", "item_id"],
                            how="inner", suffixes=("", "_R"))
        n_before = len(tmp)
        label_disagree = int((tmp["label"] != tmp["label_R"]).sum())
        if label_disagree:
            msg = (f"Label disagreement between previous merge and {name}: "
                   f"{label_disagree}/{n_before} rows")
            logger.error(msg)
            raise ValueError(msg)
        tmp = tmp.drop(columns=["label_R"])
        logger.info(f"    after inner-join: n={len(tmp):5d}")
        merged = tmp

    assert merged is not None
    # Filter to canonical 8 groups (drops deepseek)
    before = len(merged)
    merged = merged[merged["group"].isin(CANONICAL_GROUPS)].reset_index(drop=True)
    n_dropped = before - len(merged)
    if n_dropped:
        logger.info(f"  Filtered to canonical 8 groups: dropped {n_dropped} "
                    f"rows (most likely math500_deepseek_r1)")
    diagnostics["counts"]["after_oof_inner_join"] = len(merged)

    # ------------------------------------------------------------------
    # Step 2: Canonical fold
    # ------------------------------------------------------------------
    fold_src_col = f"_fold_{CANONICAL_FOLD_SOURCE}"
    if fold_src_col not in merged.columns:
        raise ValueError(f"{CANONICAL_FOLD_SOURCE} has no oof_fold key; "
                         f"cannot canonicalize.")
    merged["fold"] = merged[fold_src_col].astype(np.int8)
    # Drop the per-source _fold_* columns (they've served their role); the
    # fold-disagreement stats will go in META.
    fold_cols = [c for c in merged.columns if c.startswith("_fold_")]
    fold_disagree = {}
    for fc in fold_cols:
        if fc == fold_src_col:
            continue
        agree = int((merged[fc] == merged["fold"]).sum())
        fold_disagree[fc.replace("_fold_", "")] = {
            "agree": agree, "total": len(merged),
            "pct": round(agree / max(len(merged), 1), 4),
        }
    diagnostics["fold_disagreement_vs_canonical"] = fold_disagree
    logger.info(f"  Canonical fold = {CANONICAL_FOLD_SOURCE}")
    for src, s in fold_disagree.items():
        logger.info(f"    {src:14s} agrees with canonical on "
                    f"{s['agree']}/{s['total']} ({s['pct']:.3f})")
    merged = merged.drop(columns=fold_cols)

    # ------------------------------------------------------------------
    # Step 3: Left-join each feature family
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("Step 3: Left-joining Route-A feature families")
    logger.info("=" * 70)

    groups = set(merged["group"].unique())
    for fam in FEATURE_FAMILIES:
        logger.info(f"  {fam['prefix']:7s}  ({fam['suffix']})")
        ff = _load_feature_family(fam["prefix"], fam["suffix"], groups)
        if ff.empty:
            logger.warning(f"    No data for {fam['prefix']}; filling NaN")
            diagnostics["features"].append({
                "prefix": fam["prefix"], "suffix": fam["suffix"],
                "status": "missing", "n_cols": 0,
            })
            continue
        before = len(merged)
        merged = merged.merge(ff, on=["group", "item_id"], how="left")
        matched = int(merged[[c for c in merged.columns
                              if c.startswith(fam["prefix"] + "_")][0]].notna().sum())
        n_cols = len([c for c in merged.columns
                      if c.startswith(fam["prefix"] + "_")])
        logger.info(f"    merged: n={len(merged)} (unchanged={len(merged)==before}) "
                    f"| matched {matched}/{len(merged)} rows on this family "
                    f"| {n_cols} feature cols added")
        diagnostics["features"].append({
            "prefix": fam["prefix"], "suffix": fam["suffix"],
            "status": "ok",
            "n_cols": int(n_cols),
            "n_matched": matched,
            "coverage_pct": round(matched / max(len(merged), 1), 4),
        })

    # ------------------------------------------------------------------
    # Step 4: Column ordering + label sanity + leaky gate
    # ------------------------------------------------------------------
    oof_cols = [n for n, _ in OOF_SOURCES]
    key_cols = ["group", "item_id", "label", "fold"]
    feat_cols = sorted([c for c in merged.columns
                        if c not in set(key_cols + oof_cols)])
    merged = merged[key_cols + oof_cols + feat_cols]

    # Per-group counts
    per_group = merged.groupby("group").size().to_dict()
    diagnostics["counts"]["per_group"] = {k: int(v) for k, v in per_group.items()}
    diagnostics["counts"]["total"] = int(len(merged))
    diagnostics["counts"]["n_feature_cols"] = int(len(feat_cols))
    diagnostics["counts"]["n_oof_cols"] = int(len(oof_cols))
    logger.info("=" * 70)
    logger.info(f"Final table: {len(merged)} rows x {len(merged.columns)} cols")
    logger.info(f"  key cols  : {key_cols}")
    logger.info(f"  oof cols  : {oof_cols}")
    logger.info(f"  feat cols : {len(feat_cols)} (see META.json for details)")
    logger.info("  per-group :")
    for g, n in sorted(per_group.items()):
        logger.info(f"    {g:28s} {n}")

    # Leaky gate
    if any_leaky and leaky_policy == "fail":
        raise RuntimeError("One or more OOFs have leaky provenance; "
                           "--leaky-policy=fail. Re-run base models with "
                           "the patched (last-epoch) protocol.")
    if any_leaky:
        logger.warning("=" * 70)
        logger.warning("⚠  WARNING: one or more base OOFs are LEAKY "
                       "(best-epoch-on-val).")
        logger.warning("⚠  Downstream stacker AUROC will be inflated by "
                       "~0.01-0.025.")
        logger.warning("⚠  See .PROVENANCE.json sidecars for rerun commands.")
        logger.warning("=" * 70)

    diagnostics["any_leaky"] = any_leaky
    diagnostics["leaky_policy"] = leaky_policy
    diagnostics["provenance_summary"] = {
        n: {"leaky": bool(p.get("leaky")),
            "protocol": p.get("protocol", "unknown")}
        for n, p in oof_provs.items()
    }

    # ------------------------------------------------------------------
    # Step 5: Write
    # ------------------------------------------------------------------
    if dry_run:
        logger.info("DRY-RUN: not writing files")
        return diagnostics

    os.makedirs(Path(output_parquet).parent, exist_ok=True)
    merged.to_parquet(output_parquet, engine="pyarrow", index=False)
    size_mb = Path(output_parquet).stat().st_size / 1e6
    logger.info(f"✓ Wrote {output_parquet}  ({size_mb:.1f} MB)")

    with open(output_meta, "w") as f:
        json.dump(diagnostics, f, indent=2, default=str)
    logger.info(f"✓ Wrote {output_meta}")
    return diagnostics


# =============================================================================
# CLI
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", default="data/hybrid_table.parquet",
                    help="Output parquet path")
    ap.add_argument("--meta-output", default="data/hybrid_table.META.json")
    ap.add_argument("--leaky-policy",
                    choices=["allow", "warn", "fail"],
                    default="warn",
                    help=("How to handle OOFs with leaky provenance. "
                          "'allow'=silent, 'warn'=log, 'fail'=raise"))
    ap.add_argument("--dry-run", action="store_true",
                    help="Build and log stats but do not write parquet")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    out_parquet = str(REPO / args.output) if not os.path.isabs(args.output) \
        else args.output
    out_meta = str(REPO / args.meta_output) if not os.path.isabs(args.meta_output) \
        else args.meta_output

    diagnostics = build_hybrid_table(
        output_parquet=out_parquet,
        output_meta=out_meta,
        leaky_policy=args.leaky_policy,
        dry_run=args.dry_run,
    )

    # Exit nonzero if anything looks drastically wrong
    if diagnostics["counts"]["total"] < 5000:
        logger.error("Suspiciously small table; review diagnostics")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
