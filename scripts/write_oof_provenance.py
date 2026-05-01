#!/usr/bin/env python
"""
write_oof_provenance.py — Emit sidecar *.PROVENANCE.json files flagging each
base-model OOF whose training protocol had the best-epoch-on-val leakage.

Why this script exists:
  The OOF .npz files in results/ were produced before we fixed the soft
  leakage in src/modeling/{deberta_baseline,step_transformer,trace_gnn,
  trace_mlm_encoder,behavior_seq_lm}.py. Those files are still useful for
  local exploration (building tables, running tuning studies) but their OOF
  AUROCs are inflated by ~0.01–0.025 because epoch selection was done on
  held-out fold labels.

  To prevent silent use in "clean" analyses, we leave the .npz files in
  place but drop a provenance sidecar next to each. Downstream scripts
  (build_hybrid_table, tune_hybrid) check for these sidecars and can warn
  or refuse to load tainted OOFs based on a --leaky-policy flag.

  Sidecar filename convention:  <oof_path>.PROVENANCE.json

Usage:
    python scripts/write_oof_provenance.py            # writes all sidecars
    python scripts/write_oof_provenance.py --dry-run  # preview
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

# Repo-relative paths. Absolute prefix is computed at runtime.
REPO_ROOT = Path(__file__).resolve().parents[1]

# (path_in_results, kind, protocol, note, rerun_module, rerun_args)
TAINTED = [
    # RoBERTa pooled (best single-base AUROC)
    dict(
        path="results/roberta_pooled_oof.npz",
        kind="roberta_pooled",
        protocol="leaky_best_epoch_on_val",
        note=("Pooled RoBERTa fine-tune; OOF fold predictions were picked by "
              "best val_AUROC across 3 epochs (test-set model selection)."),
        rerun_module="src.modeling.deberta_baseline",
        rerun_args=("--model roberta-base "
                    "--traces-glob 'data/traces/*_traces.jsonl' "
                    "--output results/roberta_pooled_clean.json "
                    "--epochs 3 --batch-size 16 --lr 2e-5 --seed 42"),
    ),
    # DeBERTa pooled
    dict(
        path="results/month2/deberta_pooled_oof.npz",
        kind="deberta_pooled",
        protocol="leaky_best_epoch_on_val",
        note=("Pooled DeBERTa-v3-base fine-tune; OOF fold predictions were "
              "picked by best val_AUROC across 3 epochs."),
        rerun_module="src.modeling.deberta_baseline",
        rerun_args=("--model microsoft/deberta-v3-base "
                    "--traces-glob 'data/traces/*_traces.jsonl' "
                    "--output results/month2/deberta_pooled_clean.json "
                    "--epochs 3 --batch-size 8 --lr 2e-5 --seed 42"),
    ),
    # Step Transformer pooled
    dict(
        path="results/step_transformer_pooled_oof.npz",
        kind="step_transformer_pooled",
        protocol="leaky_best_epoch_on_val",
        note=("Pooled Step Transformer; OOF fold predictions picked by best "
              "val_AUROC per epoch."),
        rerun_module="src.modeling.step_transformer",
        rerun_args=("--emb-glob 'data/step_embeddings/*.npz' "
                    "--output results/step_transformer_pooled_clean.json "
                    "--epochs 10 --batch-size 16 --lr 3e-4 --seed 42"),
    ),
    # TraceGIN hybrid (features + behavior graph)
    dict(
        path="results/route_ab/trace_gnn_hybrid_pooled_oof.npz",
        kind="tracegin_hybrid_pooled",
        protocol="leaky_best_epoch_on_val_plus_earlystop",
        note=("Pooled TraceGIN-hybrid; OOF picked by best val_AUROC AND "
              "early-stopped when val_AUROC plateaued. Double leak."),
        rerun_module="src.modeling.trace_gnn",
        rerun_args=("--variant hybrid "
                    "--graph-glob 'data/graph_datasets/*.npz' "
                    "--output results/route_ab/trace_gnn_hybrid_pooled_clean.json "
                    "--epochs 40 --batch-size 32 --lr 1e-3 --seed 42"),
    ),
    # TraceGIN structural-only
    dict(
        path="results/route_ab/trace_gnn_structural_pooled_oof.npz",
        kind="tracegin_structural_pooled",
        protocol="leaky_best_epoch_on_val_plus_earlystop",
        note=("Pooled TraceGIN-structural; OOF picked by best val_AUROC + "
              "early stop. Double leak."),
        rerun_module="src.modeling.trace_gnn",
        rerun_args=("--variant structural "
                    "--graph-glob 'data/graph_datasets/*.npz' "
                    "--output results/route_ab/trace_gnn_structural_pooled_clean.json "
                    "--epochs 40 --batch-size 32 --lr 1e-3 --seed 42"),
    ),
]

# Non-tainted OOFs (recorded so we can emit a clean provenance for them too,
# documenting that they are leakage-safe by construction).
CLEAN = [
    dict(
        path="results/route_ab/shapelet_oof.npz",
        kind="shapelet_pooled",
        protocol="clean_no_epoch_selection",
        note=("Fold-local info-gain mining + LR. No epoch concept, no test-"
              "set peeking. Leakage-safe by construction."),
    ),
]


def _summarize(npz_path: Path) -> dict:
    if not npz_path.exists():
        return {"exists": False}
    z = np.load(npz_path, allow_pickle=True)
    out = {
        "exists": True,
        "n_samples": int(len(z["item_ids"])) if "item_ids" in z.files else -1,
        "has_oof_fold": "oof_fold" in z.files,
        "has_groups": "groups" in z.files,
        "keys": list(z.files),
        "sha256_prefix": hashlib.sha256(npz_path.read_bytes()).hexdigest()[:16],
        "size_bytes": npz_path.stat().st_size,
    }
    if "y_true" in z.files and "oof_prob" in z.files:
        try:
            from sklearn.metrics import roc_auc_score
            out["pooled_auroc"] = float(roc_auc_score(z["y_true"], z["oof_prob"]))
        except Exception as e:
            out["pooled_auroc_error"] = str(e)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be written without touching files")
    args = ap.parse_args()

    n_written = 0
    n_missing = 0

    for spec in TAINTED:
        full = REPO_ROOT / spec["path"]
        summ = _summarize(full)
        if not summ.get("exists"):
            print(f"MISSING: {spec['path']}")
            n_missing += 1
            continue
        prov = {
            "path": spec["path"],
            "kind": spec["kind"],
            "protocol": spec["protocol"],
            "leaky": True,
            "why_flagged": spec["note"],
            "expected_auroc_inflation": "0.01-0.025 (bounded by n_epochs and fold size)",
            "fix_status": "patched in source (2026-04-20); rerun required to produce clean OOF",
            "rerun_module": spec["rerun_module"],
            "rerun_cli": (f"PYTHONPATH=. python -m {spec['rerun_module']} "
                          f"{spec['rerun_args']}"),
            "summary": summ,
            "schema_notes": ("If 'oof_fold' is absent the sidecar-consuming "
                             "script will fall back to RoBERTa's fold "
                             "assignment via (group,item_id) join."),
        }
        sidecar = full.with_suffix(full.suffix + ".PROVENANCE.json")
        if args.dry_run:
            print(f"[dry-run] would write {sidecar}")
            print(json.dumps(prov, indent=2))
        else:
            sidecar.write_text(json.dumps(prov, indent=2))
            print(f"WROTE {sidecar}")
            n_written += 1

    for spec in CLEAN:
        full = REPO_ROOT / spec["path"]
        summ = _summarize(full)
        if not summ.get("exists"):
            continue
        prov = {
            "path": spec["path"],
            "kind": spec["kind"],
            "protocol": spec["protocol"],
            "leaky": False,
            "why_flagged": spec["note"],
            "summary": summ,
        }
        sidecar = full.with_suffix(full.suffix + ".PROVENANCE.json")
        if args.dry_run:
            print(f"[dry-run] would write {sidecar} (clean)")
        else:
            sidecar.write_text(json.dumps(prov, indent=2))
            print(f"WROTE {sidecar} (clean)")
            n_written += 1

    print(f"\nTotal: wrote {n_written} sidecars; {n_missing} missing.")
    return 0 if n_missing == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
