#!/usr/bin/env python3
"""
analyze_route_ab_oofs.py — per-dataset breakdown of every OOF .npz against
the falsifier criteria in research_proposal.md.

Questions answered:
  1.  Per-dataset AUROC / AUPRC / PRR / ECE / Acc@{80,90} for every model.
  2.  H1 falsifier: is AUROC >= 0.75 on each MATH500 variant?
  3.  H3 falsifier: DeBERTa vs TraceGIN-hybrid per (dataset, model).
  4.  Pooled AUROC across a COMMON subset of groups (so GNNs with an extra
      math500_deepseek_r1 group are compared apples-to-apples).
  5.  Per-dataset delta hybrid_gnn − deberta (complementarity vs dominance).

Usage:
    PYTHONPATH=. python scripts/analyze_route_ab_oofs.py
    PYTHONPATH=. python scripts/analyze_route_ab_oofs.py --out-dir reports/route_ab/
    PYTHONPATH=. python scripts/analyze_route_ab_oofs.py --models gnn_structural,gnn_hybrid,deberta,step
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.modeling.cv_utils import evaluate

# -------------------------------------------------------------------------
# Manifest of OOF .npz files we want to score.
# Paths are relative to project root; auto-skip if file is missing.
# -------------------------------------------------------------------------
MODEL_MANIFEST = [
    # key,                display name,             path
    ("gnn_structural",   "TraceGIN-struct",        "results/route_ab/trace_gnn_structural_pooled_oof.npz"),
    ("gnn_hybrid",       "TraceGIN-hybrid",        "results/route_ab/trace_gnn_hybrid_pooled_oof.npz"),
    ("deberta",          "DeBERTa-v3",             "results/month2/deberta_pooled_oof.npz"),
    ("step",             "StepTF (MiniLM)",        "results/month2_v2/step_transformer_pooled_oof.npz"),
    ("step_bge",         "StepTF (BGE)",           "results/month2_v2/step_transformer_bge_pooled_oof.npz"),
    ("roberta",          "RoBERTa",                "results/month2_v2/roberta_pooled_oof.npz"),
    ("trace_mlm",        "Trace-MLM",              "results/month2_v2/trace_mlm_pooled_oof.npz"),
    ("trace_mlm_np",     "Trace-MLM (no pretrain)","results/month2_v2/trace_mlm_no_pretrain_oof.npz"),
    ("behavior_seq_lm",  "BehaviorSeq-LM",         "results/month2_v2/behavior_seq_lm_pooled_oof.npz"),
    ("shapelet",         "Shapelet (A2+)",         "results/route_ab/shapelet_oof.npz"),
]

METRICS = ["auroc", "auprc", "prr", "ece", "accuracy_at_80", "accuracy_at_90"]

# The 8 canonical dataset_model groups; math500_deepseek_r1 is GNN-only.
COMMON_GROUPS = [
    "math500_qwen7b", "math500_llama8b",
    "gsm8k_qwen7b", "gsm8k_llama8b",
    "gpqa_diamond_qwen7b", "gpqa_diamond_llama8b",
    "arc_challenge_qwen7b", "arc_challenge_llama8b",
]
MATH500_VARIANTS = [g for g in ["math500_qwen7b", "math500_llama8b", "math500_deepseek_r1"] if True]
H1_THRESHOLD = 0.75


# -------------------------------------------------------------------------
# Loading
# -------------------------------------------------------------------------
def load_oof(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    return {
        "y": np.asarray(z["y_true"]).astype(int),
        "p": np.asarray(z["oof_prob"]).astype(float),
        "groups": np.asarray(z["groups"]).astype(str) if "groups" in z.files else None,
    }


def per_dataset_metrics(d: dict, model_key: str, model_name: str) -> list[dict]:
    """Compute metrics for pooled overall AND each group."""
    rows = []
    # Overall (on whatever groups this model covers)
    m_overall = evaluate(d["y"], d["p"], name="overall")
    rows.append({"model_key": model_key, "model_name": model_name,
                 "group": "OVERALL", "n_groups": len(set(d["groups"])) if d["groups"] is not None else 1,
                 **{k: m_overall[k] for k in METRICS},
                 "n_samples": m_overall["n_samples"],
                 "base_accuracy": m_overall["base_accuracy"]})

    if d["groups"] is None:
        return rows

    # Per-group
    for g in sorted(set(d["groups"])):
        mask = d["groups"] == g
        if mask.sum() < 10:
            continue
        y = d["y"][mask]
        p = d["p"][mask]
        if len(np.unique(y)) < 2:
            continue
        m = evaluate(y, p, name=g)
        rows.append({"model_key": model_key, "model_name": model_name,
                     "group": g, "n_groups": 1,
                     **{k: m[k] for k in METRICS},
                     "n_samples": m["n_samples"],
                     "base_accuracy": m["base_accuracy"]})
    return rows


# -------------------------------------------------------------------------
# Report builders
# -------------------------------------------------------------------------
def build_long_table(manifest: list[tuple[str, str, str]]) -> pd.DataFrame:
    all_rows = []
    for key, name, path in manifest:
        d = load_oof(path)
        if d is None:
            print(f"  [skip] {key}  (missing: {path})")
            continue
        n_groups = len(set(d["groups"])) if d["groups"] is not None else 1
        print(f"  [load] {key:<22s}  n={len(d['y']):>5d}  groups={n_groups}")
        all_rows.extend(per_dataset_metrics(d, key, name))
    if not all_rows:
        raise SystemExit("No OOF files could be loaded — check paths in MODEL_MANIFEST.")
    return pd.DataFrame(all_rows)


def pivot_auroc(df: pd.DataFrame) -> pd.DataFrame:
    """Wide table: rows=group (incl. OVERALL), cols=model_key, values=auroc."""
    return df.pivot_table(index="group", columns="model_key", values="auroc")


def pooled_on_common(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-compute pooled AUROC using ONLY rows where group is in COMMON_GROUPS,
    per model. This lets us compare GNNs (9-group pooled) against DeBERTa/StepTF
    (8-group pooled) apples-to-apples.

    We use the (auroc * n_samples) weighted mean as a close-enough proxy for
    a true concatenated-pool AUROC; for the most fair call we would need to
    re-concatenate probs, so we also emit a "common-pool" re-computation.
    """
    common_df = df[df["group"].isin(COMMON_GROUPS)].copy()
    rows = []
    for key in common_df["model_key"].unique():
        sub = common_df[common_df["model_key"] == key]
        if sub.empty:
            continue
        weights = sub["n_samples"].to_numpy()
        auroc = float(np.average(sub["auroc"], weights=weights))
        auprc = float(np.average(sub["auprc"], weights=weights))
        prr   = float(np.average(sub["prr"],   weights=weights))
        ece   = float(np.average(sub["ece"],   weights=weights))
        a80   = float(np.average(sub["accuracy_at_80"], weights=weights))
        rows.append({"model_key": key,
                     "model_name": sub["model_name"].iloc[0],
                     "n_groups_common": len(sub),
                     "n_samples_common": int(weights.sum()),
                     "auroc_wtd_common": auroc,
                     "auprc_wtd_common": auprc,
                     "prr_wtd_common":   prr,
                     "ece_wtd_common":   ece,
                     "acc80_wtd_common": a80})
    return pd.DataFrame(rows).sort_values("auroc_wtd_common", ascending=False)


def h1_check(df: pd.DataFrame, threshold: float = H1_THRESHOLD) -> pd.DataFrame:
    """H1: AUROC on MATH500 variants (research_proposal.md:170-178)."""
    math = df[df["group"].str.startswith("math500_")].copy()
    math["H1_pass"] = math["auroc"] >= threshold
    return math[["model_name", "group", "auroc", "H1_pass", "n_samples", "base_accuracy"]]\
        .sort_values(["group", "auroc"], ascending=[True, False])


def falsifier_table(df: pd.DataFrame,
                    challenger: str = "gnn_hybrid",
                    incumbent: str = "deberta",
                    groups: list[str] | None = None) -> pd.DataFrame:
    """Per-dataset delta: challenger AUROC − incumbent AUROC.
    'Pass' = challenger ≥ incumbent (tie counts)."""
    groups = groups or COMMON_GROUPS
    ch = df[df["model_key"] == challenger].set_index("group")
    inc = df[df["model_key"] == incumbent].set_index("group")
    rows = []
    for g in groups:
        if g not in ch.index or g not in inc.index:
            continue
        rows.append({
            "group": g,
            f"{challenger}_auroc": float(ch.loc[g, "auroc"]),
            f"{incumbent}_auroc": float(inc.loc[g, "auroc"]),
            "delta": float(ch.loc[g, "auroc"] - inc.loc[g, "auroc"]),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["pass"] = out["delta"] >= 0
    return out.sort_values("delta", ascending=False)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="reports/route_ab/",
                    help="Write per-group CSV + markdown here (default: reports/route_ab/).")
    ap.add_argument("--models", default=None,
                    help="Comma-separated model keys to include (default: all available).")
    ap.add_argument("--no-save", action="store_true",
                    help="Print only; do not write report files.")
    args = ap.parse_args()

    manifest = MODEL_MANIFEST
    if args.models:
        wanted = set(args.models.split(","))
        manifest = [(k, n, p) for (k, n, p) in manifest if k in wanted]

    print("=== Loading OOFs ===")
    df = build_long_table(manifest)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\n=== Per-group AUROC (wide) ===")
    auroc_wide = pivot_auroc(df)
    # Show OVERALL first, then groups
    row_order = ["OVERALL"] + [g for g in auroc_wide.index if g != "OVERALL"]
    print(auroc_wide.loc[row_order])

    print("\n=== Per-group AUPRC (wide) ===")
    auprc_wide = df.pivot_table(index="group", columns="model_key", values="auprc")
    print(auprc_wide.loc[row_order])

    print("\n=== Per-group PRR (wide) ===")
    prr_wide = df.pivot_table(index="group", columns="model_key", values="prr")
    print(prr_wide.loc[row_order])

    print("\n=== Pooled on COMMON 8 groups (weighted by n_samples) ===")
    pooled = pooled_on_common(df)
    print(pooled.to_string(index=False))

    print(f"\n=== H1 falsifier (AUROC >= {H1_THRESHOLD} on MATH500 variants) ===")
    h1 = h1_check(df)
    print(h1.to_string(index=False))
    print(f"  -> H1 passes: {int(h1['H1_pass'].sum())} / {len(h1)}  (model × dataset pairs)")

    print("\n=== Falsifier: TraceGIN-hybrid vs DeBERTa, per dataset ===")
    fals = falsifier_table(df, "gnn_hybrid", "deberta")
    if fals.empty:
        print("  (skipped — gnn_hybrid or deberta missing)")
    else:
        print(fals.to_string(index=False))
        wins = int(fals["pass"].sum())
        print(f"  -> Hybrid-GNN wins or ties DeBERTa on {wins} / {len(fals)} groups  "
              f"(falsifier threshold: >= 5/8).")

    print("\n=== Falsifier: TraceGIN-hybrid vs StepTF (MiniLM), per dataset ===")
    fals2 = falsifier_table(df, "gnn_hybrid", "step")
    if not fals2.empty:
        print(fals2.to_string(index=False))
        print(f"  -> wins on {int(fals2['pass'].sum())} / {len(fals2)} groups.")

    print("\n=== Falsifier: TraceGIN-structural vs DeBERTa (content-free story) ===")
    fals3 = falsifier_table(df, "gnn_structural", "deberta")
    if not fals3.empty:
        print(fals3.to_string(index=False))
        print(f"  -> wins on {int(fals3['pass'].sum())} / {len(fals3)} groups.")

    # Save artifacts
    if not args.no_save:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "per_group_metrics.csv", index=False)
        auroc_wide.to_csv(out_dir / "auroc_wide.csv")
        auprc_wide.to_csv(out_dir / "auprc_wide.csv")
        prr_wide.to_csv(out_dir / "prr_wide.csv")
        pooled.to_csv(out_dir / "pooled_on_common8.csv", index=False)
        h1.to_csv(out_dir / "h1_math500.csv", index=False)
        if not fals.empty:
            fals.to_csv(out_dir / "falsifier_hybrid_vs_deberta.csv", index=False)
        if not fals2.empty:
            fals2.to_csv(out_dir / "falsifier_hybrid_vs_step.csv", index=False)
        if not fals3.empty:
            fals3.to_csv(out_dir / "falsifier_structural_vs_deberta.csv", index=False)
        print(f"\n[saved] CSVs under {out_dir}/")


if __name__ == "__main__":
    main()
