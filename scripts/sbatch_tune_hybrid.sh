#!/bin/bash
# =============================================================================
# sbatch_tune_hybrid.sh
#
# Run the hybrid-stacker Optuna hyperparameter search as a batch job and
# automatically refit the best trial at the end. Designed for CPU partitions
# (tuning is CPU-bound; per-trial cost is dominated by xgboost/lightgbm
# tree-building, not GPU work).
#
# Why this lives separately from sbatch_rerun_clean_oofs.sh:
#   - No GPU needed (faster-to-schedule CPU partition).
#   - Different wall-clock budget (500 trials ≈ 2-6 h on 8 cores).
#   - Runs against the parquet *already built locally*; doesn't re-invoke
#     build_hybrid_table unless explicitly requested via REBUILD_PARQUET=1.
#   - Safe to resume: set the same STUDY_NAME and the job picks up where it
#     left off via the Optuna storage.
#
# Submit with:
#     cd /home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology
#     sbatch scripts/sbatch_tune_hybrid.sh
#
# Env-var overrides (all optional; defaults shown in parens):
#     CONDA_ENV=torch311
#     STUDY_NAME=hybrid_v1_clean              # name persisted in STORAGE
#     STORAGE=sqlite:///data/optuna_hybrid_v1_clean.db
#                                             # SQLite = HPC Linux (fast +
#                                             # queryable via optuna-dashboard).
#                                             # Use journal:///... on WSL only.
#     N_TRIALS=500                            # trials THIS submission
#     TRIAL_TIMEOUT=180                       # seconds per trial, SIGALRM cap
#     LEAKY_POLICY=fail                       # fail | warn | allow
#                                             # fail = refuse tainted OOFs;
#                                             # switch to 'warn' only if
#                                             # deliberately tuning on tainted.
#     N_JOBS=8                                # threads per xgb/lgbm fit;
#                                             # should match --cpus-per-task
#     REBUILD_PARQUET=0                       # 1 = run build_hybrid_table
#                                             #     before tuning
#     DO_REFIT=1                              # 1 = refit best after tuning,
#                                             #     saving OOF + per-group
#     SEED=42                                 # TPE seed
#
# Wall-clock budget (very rough, 8 cores):
#     500 trials with mixed fam × subset (median trial ~5s, p90 ~30s):
#       total ~45 min - 2 h. Add refit (~30 s) and overhead. We request 12 h
#       for headroom (big feature subsets + timeouts can inflate it).
#
# Output contract (relative to repo root):
#     results/route_ab/hybrid_tuning_trials_${STUDY_NAME}.csv
#     results/route_ab/hybrid_tuned_best_${STUDY_NAME}.json
#     results/route_ab/hybrid_tuned_${STUDY_NAME}_pooled_oof.npz   [if DO_REFIT=1]
#     results/route_ab/hybrid_tuned_${STUDY_NAME}_pooled.json      [if DO_REFIT=1]
#     reports/route_ab/hybrid_tuned_${STUDY_NAME}_per_group.csv    [if DO_REFIT=1]
#     ${STORAGE path}                                              (persistent study)
#     logs/tune_hybrid_${SLURM_JOB_ID}.{out,err}
# =============================================================================

#SBATCH --job-name=tune_hybrid
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology
#SBATCH --output=/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology/logs/tune_hybrid_%j.out
#SBATCH --error=/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology/logs/tune_hybrid_%j.err
#SBATCH --mail-type=ALL

set -Eeo pipefail
trap 'ec=$?; echo "==== FAILED at line $LINENO (exit $ec): $BASH_COMMAND" >&2; exit $ec' ERR

# =============================================================================
# CONFIG / ENV
# =============================================================================
PROJECT_DIR="${PROJECT_DIR:-/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology}"
CONDA_ENV="${CONDA_ENV:-torch311}"
STUDY_NAME="${STUDY_NAME:-hybrid_v1_clean}"
STORAGE="${STORAGE:-sqlite:///data/optuna_hybrid_v1_clean.db}"
N_TRIALS="${N_TRIALS:-500}"
TRIAL_TIMEOUT="${TRIAL_TIMEOUT:-180}"
LEAKY_POLICY="${LEAKY_POLICY:-fail}"
N_JOBS="${N_JOBS:-8}"
REBUILD_PARQUET="${REBUILD_PARQUET:-0}"
DO_REFIT="${DO_REFIT:-1}"
SEED="${SEED:-42}"

cd "$PROJECT_DIR"
mkdir -p logs results/route_ab reports/route_ab data

echo "=========================================================================="
echo "Hybrid HP tuning (Optuna TPE)"
echo "Started:        $(date)"
echo "Node:           $(hostname)"
echo "JobID:          ${SLURM_JOB_ID:-<not-slurm>}"
echo "PWD:            $PWD"
echo "STUDY_NAME:     $STUDY_NAME"
echo "STORAGE:        $STORAGE"
echo "N_TRIALS:       $N_TRIALS"
echo "TRIAL_TIMEOUT:  ${TRIAL_TIMEOUT}s"
echo "N_JOBS:         $N_JOBS  (matches --cpus-per-task for best throughput)"
echo "LEAKY_POLICY:   $LEAKY_POLICY"
echo "REBUILD_PARQUET:$REBUILD_PARQUET"
echo "DO_REFIT:       $DO_REFIT"
echo "SEED:           $SEED"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-?}"
echo "=========================================================================="

# =============================================================================
# Conda activation (same pattern as sbatch_rerun_clean_oofs.sh)
# =============================================================================
trap - ERR
set +e
module load miniconda 2>/dev/null || true
conda activate "$CONDA_ENV" || { echo "FATAL: conda activate $CONDA_ENV failed"; exit 2; }
set -e
trap 'ec=$?; echo "==== FAILED at line $LINENO (exit $ec): $BASH_COMMAND" >&2; exit $ec' ERR

export PYTHONPATH="$PROJECT_DIR"

# Give xgboost/lightgbm a stable thread cap matching the allocation. Otherwise
# lightgbm sees the full physical core count and oversubscribes.
export OMP_NUM_THREADS="$N_JOBS"
export MKL_NUM_THREADS="$N_JOBS"
export OPENBLAS_NUM_THREADS="$N_JOBS"
export LIGHTGBM_NUM_THREADS="$N_JOBS"

# =============================================================================
# Sanity: imports + parquet exist
# =============================================================================
python - <<'PY'
import optuna, xgboost, lightgbm, pandas, sklearn, numpy
print(f"  optuna  {optuna.__version__}")
print(f"  xgb     {xgboost.__version__}")
print(f"  lgbm    {lightgbm.__version__}")
print(f"  pandas  {pandas.__version__}")
print(f"  sklearn {sklearn.__version__}")
print(f"  numpy   {numpy.__version__}")
PY

# =============================================================================
# Optional rebuild of the parquet (if base OOFs changed since last build)
# =============================================================================
if [[ "$REBUILD_PARQUET" == "1" ]]; then
    echo ""
    echo "=== REBUILD_PARQUET=1 → running build_hybrid_table.py ==="
    python scripts/build_hybrid_table.py --leaky-policy "$LEAKY_POLICY"
else
    if [[ ! -f data/hybrid_table.parquet ]]; then
        echo "FATAL: data/hybrid_table.parquet missing and REBUILD_PARQUET=0."
        echo "       Either set REBUILD_PARQUET=1 or run"
        echo "       PYTHONPATH=. python scripts/build_hybrid_table.py"
        echo "       on the login node first."
        exit 3
    fi
fi

# =============================================================================
# Pre-flight diagnostics — read parquet, report counts + leakage state
# =============================================================================
python - <<'PY'
import json, pandas as pd
df = pd.read_parquet("data/hybrid_table.parquet")
try:
    meta = json.load(open("data/hybrid_table.META.json"))
    any_leaky = meta.get("any_leaky", "unknown")
    prov = meta.get("provenance_summary", {})
except Exception as e:
    any_leaky = f"META.json unreadable ({e})"
    prov = {}
print(f"  rows             : {len(df):>6}")
print(f"  groups           : {df['group'].nunique():>6}  ({sorted(df['group'].unique())})")
print(f"  label rate       : {df['label'].mean():.3f}")
print(f"  fold distribution: {dict(df['fold'].value_counts().sort_index())}")
# any_leaky is a JSON bool (True/False) — no format spec so Python's
# implicit str() is fine. Don't add :5s here; it will ValueError on bool.
print(f"  any_leaky        : {any_leaky}")
# Provenance dict values are JSON bools/strings. The :Ns format spec rejects
# non-strings, so coerce to str() first before applying any width padding.
for n, p in prov.items():
    leaky_str    = str(p.get("leaky", "?"))
    protocol_str = str(p.get("protocol", "?"))
    print(f"    {n:<14s} leaky={leaky_str:<6s}  protocol={protocol_str}")
PY

# =============================================================================
# OUTPUT PATHS — anchored by STUDY_NAME so parallel studies don't collide
# =============================================================================
TRIALS_CSV="results/route_ab/hybrid_tuning_trials_${STUDY_NAME}.csv"
BEST_JSON="results/route_ab/hybrid_tuned_best_${STUDY_NAME}.json"
REFIT_OOF="results/route_ab/hybrid_tuned_${STUDY_NAME}_pooled_oof.npz"
REFIT_METRICS="results/route_ab/hybrid_tuned_${STUDY_NAME}_pooled.json"
REFIT_PER_GROUP="reports/route_ab/hybrid_tuned_${STUDY_NAME}_per_group.csv"

# =============================================================================
# STAGE 1 — tune
# =============================================================================
echo ""
echo "=========================================================================="
echo "STAGE 1: Optuna TPE tuning (up to $N_TRIALS trials this submission)"
echo "STARTED at $(date)"
echo "=========================================================================="
t0=$(date +%s)

python scripts/tune_hybrid.py \
    --parquet         "data/hybrid_table.parquet" \
    --meta            "data/hybrid_table.META.json" \
    --n-trials        "$N_TRIALS" \
    --study-name      "$STUDY_NAME" \
    --storage         "$STORAGE" \
    --trial-timeout   "$TRIAL_TIMEOUT" \
    --n-jobs          "$N_JOBS" \
    --leaky-policy    "$LEAKY_POLICY" \
    --trials-csv      "$TRIALS_CSV" \
    --best-json       "$BEST_JSON" \
    --seed            "$SEED" \
    --log-level       INFO

t1=$(date +%s)
echo "STAGE 1 complete in $((t1 - t0))s"

# =============================================================================
# STAGE 2 — refit (optional)
# =============================================================================
if [[ "$DO_REFIT" == "1" ]]; then
    echo ""
    echo "=========================================================================="
    echo "STAGE 2: Refit best trial → OOF + per-group breakdown"
    echo "STARTED at $(date)"
    echo "=========================================================================="

    python scripts/tune_hybrid.py --refit \
        --parquet         "data/hybrid_table.parquet" \
        --meta            "data/hybrid_table.META.json" \
        --study-name      "$STUDY_NAME" \
        --storage         "$STORAGE" \
        --n-jobs          "$N_JOBS" \
        --leaky-policy    "$LEAKY_POLICY" \
        --refit-oof       "$REFIT_OOF" \
        --refit-metrics   "$REFIT_METRICS" \
        --refit-per-group "$REFIT_PER_GROUP" \
        --seed            "$SEED" \
        --log-level       INFO

    echo ""
    echo "=== Refit metrics ==="
    python - <<PY
import json
m = json.load(open("$REFIT_METRICS"))
print(f"  pooled_auroc   : {m['pooled_auroc']:.4f}")
print(f"  weighted_auroc : {m['weighted_auroc']:.4f}")
print(f"  best trial     : {m['best_trial_number']}")
print(f"  feature subset size: {m['feature_subset_size']}")
print()
print("  per-group:")
for row in m["per_group"]:
    print(f"    {row['group']:28s}  n={row['n']:>4}  "
          f"AUROC={row['auroc']:.4f}  label_rate={row['label_rate']:.3f}")
PY
else
    echo "DO_REFIT=0 → skipping refit. Run it separately with:"
    echo "    python scripts/tune_hybrid.py --refit \\"
    echo "        --study-name '$STUDY_NAME' \\"
    echo "        --storage    '$STORAGE' \\"
    echo "        --n-jobs     $N_JOBS \\"
    echo "        --refit-oof       '$REFIT_OOF' \\"
    echo "        --refit-metrics   '$REFIT_METRICS' \\"
    echo "        --refit-per-group '$REFIT_PER_GROUP'"
fi

# =============================================================================
# STAGE 3 — summary (always)
# =============================================================================
echo ""
echo "=========================================================================="
echo "STAGE 3: Top-10 trial summary"
echo "=========================================================================="
python - <<PY
import pandas as pd
df = pd.read_csv("$TRIALS_CSV")
# Sort by weighted_auroc, take top 10; fall back to 'value' if user_attrs is absent
sort_col = "user_attrs_weighted_auroc" if "user_attrs_weighted_auroc" in df.columns \
    else ("value" if "value" in df.columns else df.columns[0])
keep = [c for c in ("number", "state", "value", sort_col,
                    "user_attrs_pooled_auroc", "user_attrs_n_features",
                    "user_attrs_elapsed_sec",
                    "params_family", "params_feature_subset",
                    "params_scaler", "params_add_group_dummies",
                    "params_isotonic")
        if c in df.columns]
top = df.sort_values(sort_col, ascending=False).head(10)[keep]
print(top.to_string(index=False))
print()
print(f"Total trials in CSV: {len(df)}")
if "state" in df.columns:
    print(f"  completed: {(df['state']=='COMPLETE').sum()}")
    print(f"  pruned   : {(df['state']=='PRUNED').sum()}  (e.g. trial_timeout)")
    print(f"  failed   : {(df['state']=='FAIL').sum()}")
PY

echo ""
echo "=========================================================================="
echo "Finished at $(date)"
echo "=========================================================================="
echo ""
echo "Artifacts:"
echo "  $TRIALS_CSV"
echo "  $BEST_JSON"
if [[ "$DO_REFIT" == "1" ]]; then
    echo "  $REFIT_OOF"
    echo "  $REFIT_METRICS"
    echo "  $REFIT_PER_GROUP"
fi
echo ""
echo "To resume / add more trials:"
echo "  sbatch --export=ALL,N_TRIALS=500,STUDY_NAME=$STUDY_NAME,STORAGE=$STORAGE \\"
echo "         scripts/sbatch_tune_hybrid.sh"
echo ""
echo "To inspect interactively (on login node):"
echo "  python -c \"import optuna; s=optuna.load_study(study_name='$STUDY_NAME', storage='$STORAGE'); \\"
echo "             print(f'n={len(s.trials)} best={s.best_value:.4f}')\""
