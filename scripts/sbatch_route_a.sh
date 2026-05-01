#!/bin/bash
# =============================================================================
# sbatch_route_a.sh
#
# Route-A content-free feature extraction on Yale Bouchet (cpsc4770_ym466).
# CPU-only; no GPU needed.
#
# Submit with:
#     cd /home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology
#     sbatch scripts/sbatch_route_a.sh
#
# What this job does (in order):
#   1.  A1 n-gram motifs           -> data/features/{group}_ngram.csv
#       A3 trace-graph descriptors -> data/features/{group}_graph.csv
#       A5 inter-event timing      -> data/features/{group}_timing.csv
#       A4 structural PH           -> data/features/{group}_structural_ph.csv
#       A2 shapelet distance matrix-> data/features/{group}_shapelet_distmat.npz
#   2.  A2+ shapelet OOF predictor -> results/route_ab/shapelet_oof.npz
#                                     results/route_ab/shapelet_pooled.json
#   3.  PRR backfill on results/route_ab/*_oof.npz
#
# Outputs land under results/route_ab/ (Route-A + B1 isolated tree).
#
# Estimated wall-clock: ~1-3 hr (dominated by A2 shapelet distmat, N * M * L).
# 12 hr requested has headroom.
# =============================================================================

# ---------- Hard-coded absolute paths (so --output/--error don't depend on submit dir) ----------
#SBATCH --job-name=route_a
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology
#SBATCH --output=/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology/logs/route_a_%j.out
#SBATCH --error=/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology/logs/route_a_%j.err
#SBATCH --mail-type=ALL
# If your account / partition differs, override at submit time:
#   sbatch --partition=education --time=06:00:00 scripts/sbatch_route_a.sh

# ---------- Debug / error-trap prelude (helps diagnose silent failures) ----------
# NOTE: we use -e + pipefail but NOT -u (nounset) because conda's shell hook
# references several unset variables as part of normal operation. We also
# briefly disable -e around the activation block so failing-but-recoverable
# steps (e.g. probing which conda module exists) don't kill the job.
set -Eeo pipefail
trap 'ec=$?; echo "==== FAILED at line $LINENO (exit $ec): $BASH_COMMAND" >&2; exit $ec' ERR
# Uncomment the next line if the script fails silently again — floods the log with every command:
#set -x

# =============================================================================
# CLUSTER / ENV CONFIGURATION  (override at submit time via env vars if needed)
# =============================================================================
PROJECT_DIR="${PROJECT_DIR:-/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology}"
CONDA_ENV="${CONDA_ENV:-torch311}"      # conda env on Bouchet; override via sbatch --export=...
HF_HOME_OVERRIDE="${HF_HOME:-$PROJECT_DIR/.cache/huggingface}"

# Which feature families to run. Default = all 5. Split-job strategy:
#   Stage 1 (fast):    FAMILIES="ngram graph timing structural_ph"
#   Stage 2 (heavy):   FAMILIES="shapelet"
# Override via:  sbatch --export=ALL,FAMILIES="shapelet" scripts/sbatch_route_a.sh
FAMILIES="${FAMILIES:-ngram graph timing structural_ph shapelet}"
# If FAMILIES does not include "shapelet" we still want step-2 (OOF eval) to
# run iff the distmat .npz files already exist; it auto-skips otherwise.
RUN_SHAPELET_OOF="${RUN_SHAPELET_OOF:-auto}"   # auto|yes|no

cd "$PROJECT_DIR" || { echo "FATAL: cd $PROJECT_DIR failed"; exit 1; }
mkdir -p logs data/features results/route_ab .cache/huggingface

echo "=========================================================================="
echo "Job:         ${SLURM_JOB_NAME:-?}  (id=${SLURM_JOB_ID:-?})"
echo "Started:     $(date)"
echo "Node:        $(hostname)"
echo "Submit dir:  ${SLURM_SUBMIT_DIR:-?}"
echo "PWD:         $PWD"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "CONDA_ENV:   $CONDA_ENV"
echo "PATH:        $PATH"
echo "=========================================================================="

# =============================================================================
# Activate environment — the condabin directory is already on $PATH via the
# user's login profile (see PATH printed above), so we do NOT need to source
# /etc/profile.d/{modules,lmod}.sh or call 'module load miniconda'. Those system
# scripts contain zsh-oriented snippets (e.g. 'compinit -c') that return 127
# in bash and would trigger our ERR trap.
# =============================================================================
echo ""
echo "=== Activating conda env '$CONDA_ENV' ==="

# IMPORTANT: 'trap - ERR' actually disables the ERR trap in bash. 'set +e'
# alone does NOT — the trap fires on non-zero exits regardless of -e. The
# conda shell hook + 'conda activate' legitimately produce non-zero exits
# along informational paths; we silence the trap here and re-install it
# after activation completes.
trap - ERR
set +e

module load miniconda || { echo "FATAL: 'module load miniconda' failed"; exit 2; }

echo "  conda binary: $(command -v conda)"

echo "  conda envs available:"
conda env list 2>&1 | sed 's/^/    /' | head -20

conda activate "$CONDA_ENV"
act_rc=$?
if [ $act_rc -ne 0 ]; then
    echo "FATAL: 'conda activate $CONDA_ENV' failed (rc=$act_rc). Env list above."
    exit 4
fi
echo "  Active env: ${CONDA_DEFAULT_ENV:-?}"

if ! command -v python >/dev/null 2>&1; then
    echo "FATAL: python still not on PATH after 'conda activate $CONDA_ENV'"
    exit 5
fi
echo "  Python:     $(command -v python)  $(python -V 2>&1)"

# Re-enable strict error handling for the science workload below
set -e
trap 'ec=$?; echo "==== FAILED at line $LINENO (exit $ec): $BASH_COMMAND" >&2; exit $ec' ERR

# =============================================================================
# Project paths and env vars
# =============================================================================
export PYTHONPATH="$PROJECT_DIR"
export HF_HOME="$HF_HOME_OVERRIDE"
mkdir -p "$HF_HOME"
echo "PYTHONPATH: $PYTHONPATH"
echo "HF_HOME:    $HF_HOME"

# Source project .env (HF_TOKEN, DEEPSEEK_API_KEY, etc.) without overriding existing vars
if [ -f "$PROJECT_DIR/.env" ]; then
    echo "Loading .env from $PROJECT_DIR/.env"
    set -a; source "$PROJECT_DIR/.env"; set +a
fi

# =============================================================================
# Lightweight dependency check (don't try to install — that's a setup step)
# =============================================================================
echo ""
echo "=== Dependency check ==="
python - <<'PYCHECK' || { echo "FATAL: missing dependency above; activate the right conda env"; exit 3; }
import importlib, sys
required = ["numpy", "pandas", "sklearn", "networkx", "scipy", "tqdm"]
optional = ["ripser", "persim", "community", "xgboost"]
missing = []
for m in required:
    try: importlib.import_module(m); print(f"  OK   {m}")
    except Exception as e: print(f"  MISS {m}: {e}"); missing.append(m)
for m in optional:
    try: importlib.import_module(m); print(f"  OK   {m} (optional)")
    except Exception: print(f"  WARN {m} (optional; install with pip --user if needed)")
sys.exit(1 if missing else 0)
PYCHECK

# Install optional ripser/persim/python-louvain on-demand into the active
# conda env. We DO NOT use `--user` here: Bouchet's policy disables user
# site-packages ("Can not perform a '--user' install. User site-packages are
# disabled for this Python."), and since torch311 is already user-owned the
# plain install targets the env lib/ directly.
echo ""
echo "=== Ensuring optional deps for A3/A4 ==="
python -c "import ripser" 2>/dev/null || pip install ripser persim || \
    echo "  WARNING: ripser install failed; A4 (structural PH) will be skipped"
python -c "import community" 2>/dev/null || pip install python-louvain || \
    echo "  WARNING: python-louvain install failed; A3 falls back to greedy_modularity"

# =============================================================================
# 1. Route A: extract selected content-free feature families
#    (FAMILIES env var controls which; default = all 5)
# =============================================================================
echo ""
echo "=========================================================================="
echo "Step 1: Route A feature extraction"
echo "    FAMILIES = $FAMILIES"
echo "=========================================================================="
# Pass FAMILIES as --families <list>. Split the env var into positional args.
# shellcheck disable=SC2086
python scripts/build_route_a_features.py \
    --parsed-glob "data/parsed/*_parsed.jsonl" \
    --output-dir  data/features/ \
    --families    $FAMILIES || exit 11

echo ""
echo "--- Route-A feature CSVs produced: ---"
ls -la data/features/*_ngram.csv data/features/*_graph.csv \
       data/features/*_timing.csv data/features/*_structural_ph.csv \
       data/features/*_shapelet_distmat.npz 2>/dev/null | tail -40 || true

# =============================================================================
# 2. A2 shapelet OOF predictor (fold-aware mining, emits OOF probs .npz)
# =============================================================================
echo ""
echo "=========================================================================="
echo "Step 2: A2 shapelet OOF predictor (fold-aware mining)"
echo "=========================================================================="
# NB: don't use `ls ... | wc -l` here. Under `set -eo pipefail`, an empty
# glob causes ls to exit 2, which poisons the whole pipeline and trips the
# ERR trap even though we redirected stderr. `find` returns 0 on no match.
N_DISTMAT=$(find data/features -maxdepth 1 -name "*_shapelet_distmat.npz" 2>/dev/null | wc -l)
echo "  RUN_SHAPELET_OOF=$RUN_SHAPELET_OOF  N_DISTMAT=$N_DISTMAT"
if [ -f results/route_ab/shapelet_oof.npz ] && [ -f results/route_ab/shapelet_pooled.json ]; then
    echo "  (already done, skip)"
elif [ "$RUN_SHAPELET_OOF" = "no" ]; then
    echo "  (RUN_SHAPELET_OOF=no; skip)"
elif [ "$N_DISTMAT" -eq 0 ]; then
    echo "  (no *_shapelet_distmat.npz yet; skip — rerun after shapelet family)"
else
    python src/modeling/shapelet_eval.py \
        --distmat-glob "data/features/*_shapelet_distmat.npz" \
        --output       results/route_ab/shapelet_pooled.json \
        --top-k 40 --n-splits 5 --seed 42 || exit 12
fi

# =============================================================================
# 3. PRR backfill on every new OOF
# =============================================================================
echo ""
echo "=========================================================================="
echo "Step 3: PRR backfill on results/route_ab/*_oof.npz"
echo "=========================================================================="
python scripts/backfill_prr.py --glob "results/route_ab/*_oof.npz" || true

# =============================================================================
# Done
# =============================================================================
echo ""
echo "=========================================================================="
echo "Finished at $(date)"
echo "=========================================================================="
ls -la data/features/ | tail -25
echo ""
ls -la results/route_ab/ | tail -15
