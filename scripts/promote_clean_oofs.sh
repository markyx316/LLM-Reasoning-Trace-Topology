#!/bin/bash
# =============================================================================
# promote_clean_oofs.sh
#
# Atomically swap the 5 "clean" (leakage-safe, last-epoch-protocol) base-model
# OOF .npz files over their tainted (best-epoch-on-val) production originals,
# then rebuild the hybrid_table.parquet so downstream tuning ingests the
# honest OOFs.
#
# Safety:
#   - Every tainted OOF is backed up as "<orig>.LEAKY.npz" before overwrite.
#   - Each clean OOF is validated BEFORE swap: n_samples non-zero, AUROC not
#     exactly 0.5 (that signature means training aborted — see the DeBERTa
#     BF16 bug fixed 2026-04-20 via `torch_dtype=torch.float32`).
#   - `--check-only` mode validates without touching any file.
#   - `--skip-rebuild` skips the parquet rebuild (just does the swap).
#
# Usage:
#     bash scripts/promote_clean_oofs.sh                 # full pipeline
#     bash scripts/promote_clean_oofs.sh --check-only    # validate only
#     bash scripts/promote_clean_oofs.sh --skip-rebuild  # swap, no rebuild
#
# After this script succeeds, the next step is:
#     sbatch scripts/sbatch_tune_hybrid.sh
#   (with STUDY_NAME bumped to hybrid_v2_truly_clean or similar so the
#   Optuna storage doesn't clash with the v1 study).
# =============================================================================

set -Eeo pipefail
trap 'ec=$?; echo "==== FAILED at line $LINENO (exit $ec): $BASH_COMMAND" >&2; exit $ec' ERR

CHECK_ONLY=0
SKIP_REBUILD=0
for arg in "$@"; do
    case "$arg" in
        --check-only)   CHECK_ONLY=1 ;;
        --skip-rebuild) SKIP_REBUILD=1 ;;
        -h|--help)
            sed -n '3,30p' "$0"; exit 0 ;;
        *)
            echo "Unknown arg: $arg"; exit 2 ;;
    esac
done

# Resolve repo root (works from anywhere)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# -----------------------------------------------------------------------------
# pairs of "clean -> tainted" paths
# -----------------------------------------------------------------------------
PAIRS=(
    "results/roberta_pooled_clean_oof.npz:results/roberta_pooled_oof.npz"
    "results/month2/deberta_pooled_clean_oof.npz:results/month2/deberta_pooled_oof.npz"
    "results/step_transformer_pooled_clean_oof.npz:results/step_transformer_pooled_oof.npz"
    "results/route_ab/trace_gnn_structural_pooled_clean_oof.npz:results/route_ab/trace_gnn_structural_pooled_oof.npz"
    "results/route_ab/trace_gnn_hybrid_pooled_clean_oof.npz:results/route_ab/trace_gnn_hybrid_pooled_oof.npz"
)

# -----------------------------------------------------------------------------
# STAGE 1: validate every clean OOF before touching anything
# -----------------------------------------------------------------------------
echo "=========================================================================="
echo "STAGE 1: validate clean OOFs (non-destructive)"
echo "=========================================================================="

python - "${PAIRS[@]}" <<'PY'
import json, os, sys
import numpy as np
from sklearn.metrics import roc_auc_score

pairs = sys.argv[1:]
bad = []
print(f"{'model':<40s} {'n':>6s}  {'AUROC':>7s}  {'label_rate':>10s}  {'status':<8s}")
print("-" * 80)
for pair in pairs:
    src, dst = pair.split(":")
    if not os.path.exists(src):
        print(f"{os.path.basename(src):<40s} {'?':>6s}  {'?':>7s}  {'?':>10s}  MISSING")
        bad.append((src, "file missing"))
        continue
    z = np.load(src, allow_pickle=True)
    y = z["y_true"]; p = z["oof_prob"]
    n = int(len(y))
    # AUROC can fail if all labels are same class — guard it
    try:
        auroc = float(roc_auc_score(y, p))
    except Exception as e:
        auroc = float("nan")
    lr = float(np.mean(y))
    # Validation: AUROC in [0.55, 0.99] is the acceptable band. 0.500 exactly
    # is the DeBERTa abort signature; < 0.55 is suspicious for any of our
    # base models. > 0.99 means label leakage in the training itself.
    status = "OK"
    if not np.isfinite(auroc) or auroc < 0.55 or auroc > 0.99:
        status = "FAIL"
        bad.append((src, f"AUROC={auroc:.4f}"))
    # Also check probs are finite, in [0,1], and not all identical
    if not np.isfinite(p).all():
        status = "FAIL"
        bad.append((src, "non-finite probs"))
    elif p.min() == p.max():
        status = "FAIL"
        bad.append((src, f"constant probs = {p.min():.4f}"))
    elif p.min() < 0 or p.max() > 1:
        status = "FAIL"
        bad.append((src, f"probs out of [0,1]: min={p.min():.4f} max={p.max():.4f}"))
    print(f"{os.path.basename(src):<40s} {n:>6d}  {auroc:>7.4f}  {lr:>10.4f}  {status}")

if bad:
    print()
    print(f"VALIDATION FAILED for {len(bad)} clean OOF(s):")
    for src, reason in bad:
        print(f"  - {src}  ({reason})")
    print()
    print("Action: re-run the failing model(s) with the patched code and")
    print("re-invoke this script. Common root causes:")
    print("  - DeBERTa AUROC=0.5000 : BF16 safetensors load. Fix applied in")
    print("    src/modeling/deberta_baseline.py 2026-04-20. Re-run via")
    print("    STEPS=\"deberta\" FORCE=1 sbatch scripts/sbatch_rerun_clean_oofs.sh")
    print("  - Any other AUROC < 0.55 : training didn't converge. Check the")
    print("    corresponding .err log for NaN warnings or the ABORTING FOLD")
    print("    message.")
    sys.exit(1)

print()
print("All 5 clean OOFs validated. Safe to promote.")
PY

if [[ "$CHECK_ONLY" == "1" ]]; then
    echo ""
    echo "=== --check-only mode: no files were modified. ==="
    exit 0
fi

# -----------------------------------------------------------------------------
# STAGE 2: atomic swap with backup
# -----------------------------------------------------------------------------
echo ""
echo "=========================================================================="
echo "STAGE 2: backup tainted originals as *.LEAKY.npz and swap clean -> prod"
echo "=========================================================================="

for pair in "${PAIRS[@]}"; do
    src="${pair%:*}"   # clean_oof
    dst="${pair#*:}"   # tainted_oof
    if [[ ! -f "$src" ]]; then
        echo "SKIP: $src missing (validation should have caught this)"
        continue
    fi
    if [[ -f "$dst" ]]; then
        backup="${dst%.npz}.LEAKY.npz"
        if [[ -f "$backup" ]]; then
            echo "NOTE: $backup already exists — keeping the earlier backup"
        else
            cp -v "$dst" "$backup"
        fi
    fi
    cp -v "$src" "$dst"
    # Copy the PROVENANCE sidecar for the clean file if present
    if [[ -f "${src}.PROVENANCE.json" ]]; then
        cp -v "${src}.PROVENANCE.json" "${dst}.PROVENANCE.json"
    fi
done

# -----------------------------------------------------------------------------
# STAGE 3: rewrite provenance sidecars to reflect the clean state
# -----------------------------------------------------------------------------
echo ""
echo "=========================================================================="
echo "STAGE 3: rewrite *.PROVENANCE.json sidecars"
echo "=========================================================================="

if [[ -f scripts/write_oof_provenance.py ]]; then
    python scripts/write_oof_provenance.py
else
    echo "scripts/write_oof_provenance.py not found — skipping"
fi

# -----------------------------------------------------------------------------
# STAGE 4: rebuild hybrid_table.parquet on the clean OOFs
# -----------------------------------------------------------------------------
if [[ "$SKIP_REBUILD" == "1" ]]; then
    echo ""
    echo "=== --skip-rebuild: leaving data/hybrid_table.parquet untouched ==="
    exit 0
fi

echo ""
echo "=========================================================================="
echo "STAGE 4: rebuild data/hybrid_table.parquet with --leaky-policy fail"
echo "=========================================================================="

# Use --leaky-policy fail so if any residual tainted OOF slipped through
# (e.g. missing PROVENANCE sidecar), the build halts rather than silently
# using leaky data.
if [[ -z "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="$REPO_ROOT"
elif [[ ":$PYTHONPATH:" != *":$REPO_ROOT:"* ]]; then
    export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
fi

python scripts/build_hybrid_table.py --leaky-policy fail

# -----------------------------------------------------------------------------
# STAGE 5: summary
# -----------------------------------------------------------------------------
echo ""
echo "=========================================================================="
echo "DONE. Summary of current state:"
echo "=========================================================================="

python - <<'PY'
import json, os
import numpy as np
from sklearn.metrics import roc_auc_score

meta_path = "data/hybrid_table.META.json"
if os.path.exists(meta_path):
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"hybrid_table.META.json:")
    print(f"  any_leaky        : {meta.get('any_leaky')}")
    print(f"  rows             : {meta.get('counts', {}).get('total')}")
    print(f"  n_feature_cols   : {meta.get('counts', {}).get('n_feature_cols')}")
    print(f"  n_oof_cols       : {meta.get('counts', {}).get('n_oof_cols')}")
    prov = meta.get("provenance_summary", {})
    for n, p in prov.items():
        print(f"    {n:<14s} leaky={str(p.get('leaky','?')):<6s}  protocol={p.get('protocol')}")
else:
    print(f"  (no {meta_path} — rebuild may have been skipped)")

print()
print("OOF files in production + backup check:")
pairs = [
    ("results/roberta_pooled_oof.npz",              "results/roberta_pooled_oof.LEAKY.npz"),
    ("results/month2/deberta_pooled_oof.npz",        "results/month2/deberta_pooled_oof.LEAKY.npz"),
    ("results/step_transformer_pooled_oof.npz",      "results/step_transformer_pooled_oof.LEAKY.npz"),
    ("results/route_ab/trace_gnn_structural_pooled_oof.npz", "results/route_ab/trace_gnn_structural_pooled_oof.LEAKY.npz"),
    ("results/route_ab/trace_gnn_hybrid_pooled_oof.npz",     "results/route_ab/trace_gnn_hybrid_pooled_oof.LEAKY.npz"),
]
for prod, backup in pairs:
    def auroc_of(path):
        if not os.path.exists(path): return None
        z = np.load(path, allow_pickle=True)
        try:
            return float(roc_auc_score(z["y_true"], z["oof_prob"]))
        except Exception:
            return float("nan")
    a_prod = auroc_of(prod)
    a_back = auroc_of(backup)
    astr_prod = f"{a_prod:.4f}" if a_prod is not None else "MISS "
    astr_back = f"{a_back:.4f}" if a_back is not None else "MISS "
    print(f"  prod={astr_prod}  backup(LEAKY)={astr_back}  {os.path.basename(prod)}")
PY

echo ""
echo "=========================================================================="
echo "Next step — re-tune on the newly-clean table:"
echo "=========================================================================="
cat <<'EOF'
    # CPU-only, 8 cores, 12h cap (same sbatch as the v1 run)
    STUDY_NAME=hybrid_v2_truly_clean \
    STORAGE="sqlite:///data/optuna_hybrid_v2_truly_clean.db" \
    N_TRIALS=2000 \
    REBUILD_PARQUET=0 \
    sbatch scripts/sbatch_tune_hybrid.sh
EOF
