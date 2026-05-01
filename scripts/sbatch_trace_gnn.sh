#!/bin/bash
# =============================================================================
# sbatch_trace_gnn.sh
#
# Route-B1 Trace-GNN training on Yale Bouchet (cpsc4770_ym466).
# GPU-bound; uses gpu_rtx6000 partition.
#
# Submit with:
#     cd /home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology
#     sbatch scripts/sbatch_trace_gnn.sh                # structural only
#     WITH_CONTENT=1 sbatch scripts/sbatch_trace_gnn.sh # + hybrid (+MiniLM)
#
# What this job does (in order):
#   1a. Build per-dataset graph .npz (structural 10-d node features)
#       -> data/graphs/{group}_graph.npz
#   1b. [Optional, WITH_CONTENT=1] Build hybrid graphs (+MiniLM episode embeddings)
#       -> data/graphs/hybrid/{group}_graph.npz
#   2a. Train 2-layer GIN, structural variant, 5-fold OOF CV
#       -> results/route_ab/trace_gnn_structural_pooled.json
#          results/route_ab/trace_gnn_structural_oof.npz
#   2b. [Optional, WITH_CONTENT=1] Train hybrid variant
#       -> results/route_ab/trace_gnn_hybrid_pooled.json
#          results/route_ab/trace_gnn_hybrid_oof.npz
#   3.  PRR backfill on results/route_ab/trace_gnn_*_oof.npz
#
# Estimated wall-clock on 1× RTX6000: ~30 min structural, ~60 min +hybrid.
# 6 hr requested has generous headroom.
# =============================================================================

# ---------- Hard-coded absolute paths (so --output/--error don't depend on submit dir) ----------
#SBATCH --job-name=trace_gnn
#SBATCH --partition=gpu_rtx6000
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --chdir=/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology
#SBATCH --output=/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology/logs/trace_gnn_%j.out
#SBATCH --error=/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology/logs/trace_gnn_%j.err
#SBATCH --mail-type=ALL
# If your account / partition / GPU type differs, override at submit time:
#   sbatch --partition=gpu_devel --gres=gpu:a100:1 --time=06:00:00 \
#          scripts/sbatch_trace_gnn.sh

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
WITH_CONTENT="${WITH_CONTENT:-0}"

cd "$PROJECT_DIR" || { echo "FATAL: cd $PROJECT_DIR failed"; exit 1; }
mkdir -p logs data/graphs data/graphs/hybrid results/route_ab .cache/huggingface

echo "=========================================================================="
echo "Job:         ${SLURM_JOB_NAME:-?}  (id=${SLURM_JOB_ID:-?})"
echo "Started:     $(date)"
echo "Node:        $(hostname)"
echo "Submit dir:  ${SLURM_SUBMIT_DIR:-?}"
echo "PWD:         $PWD"
echo "PROJECT_DIR: $PROJECT_DIR"
echo "CONDA_ENV:   $CONDA_ENV"
echo "WITH_CONTENT: $WITH_CONTENT"
echo "PATH:        $PATH"
echo "GPUs:        $(nvidia-smi -L 2>/dev/null || echo 'NONE DETECTED')"
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

# Verify torch imports and CUDA is visible BEFORE re-enabling trap
if ! python -c 'import torch; print("  Torch:     ", torch.__version__, " cuda=", torch.cuda.is_available())'; then
    echo "FATAL: 'import torch' failed in env '$CONDA_ENV' — is torch installed there?"
    exit 6
fi

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
required = ["torch", "numpy", "pandas", "sklearn", "tqdm"]
optional = ["sentence_transformers", "transformers", "sentencepiece", "xgboost"]
missing = []
for m in required:
    try: importlib.import_module(m); print(f"  OK   {m}")
    except Exception as e: print(f"  MISS {m}: {e}"); missing.append(m)
for m in optional:
    try: importlib.import_module(m); print(f"  OK   {m} (optional)")
    except Exception: print(f"  WARN {m} (optional; needed only for WITH_CONTENT=1)")
sys.exit(1 if missing else 0)
PYCHECK

# =============================================================================
# 1a. Build structural graph .npz (CPU, ~minute)
# =============================================================================
echo ""
echo "=========================================================================="
echo "Step 1a: Build structural trace graphs"
echo "=========================================================================="
# Rebuild is cheap (~minute), but skip if outputs already exist unless forced.
# Force rebuild: `REBUILD_STRUCTURAL_GRAPHS=1 sbatch ...`
REBUILD_STRUCTURAL_GRAPHS="${REBUILD_STRUCTURAL_GRAPHS:-0}"
EXISTING_STRUCT=$(ls data/graphs/*_graph.npz 2>/dev/null | wc -l)
if [ "$REBUILD_STRUCTURAL_GRAPHS" = "1" ] || [ "$EXISTING_STRUCT" -eq 0 ]; then
    python scripts/build_trace_graphs.py \
        --parsed-glob "data/parsed/*_parsed.jsonl" \
        --output-dir  data/graphs/ || exit 11
else
    echo "  (found $EXISTING_STRUCT existing structural graphs in data/graphs/; skip)"
    echo "  (set REBUILD_STRUCTURAL_GRAPHS=1 or 'rm data/graphs/*_graph.npz' to force rebuild)"
fi
ls -la data/graphs/ | tail -15

# =============================================================================
# 1b. Optional hybrid graphs (+MiniLM episode embeddings)
# =============================================================================
if [ "$WITH_CONTENT" = "1" ]; then
    echo ""
    echo "=========================================================================="
    echo "Step 1b: Build hybrid trace graphs (+MiniLM content embeddings)"
    echo "=========================================================================="
    python -c "import sentence_transformers" 2>/dev/null || \
        pip install --user sentence-transformers || \
        { echo "FATAL: sentence-transformers install failed; cannot build hybrid graphs"; exit 12; }

    # MiniLM encoding is expensive (~30 min / 8 datasets). Skip if already built
    # unless the user forces a rebuild. The defensive fix in collate_graphs
    # handles legacy hybrid .npz files that had empty traces stored as (0, 10).
    # Force rebuild: `REBUILD_HYBRID_GRAPHS=1 WITH_CONTENT=1 sbatch ...`
    REBUILD_HYBRID_GRAPHS="${REBUILD_HYBRID_GRAPHS:-0}"
    EXISTING_HYBRID=$(ls data/graphs/hybrid/*_graph.npz 2>/dev/null | wc -l)
    if [ "$REBUILD_HYBRID_GRAPHS" = "1" ] || [ "$EXISTING_HYBRID" -eq 0 ]; then
        python scripts/build_trace_graphs.py \
            --parsed-glob "data/parsed/*_parsed.jsonl" \
            --output-dir  data/graphs/hybrid/ \
            --with-content --device cuda || exit 13
    else
        echo "  (found $EXISTING_HYBRID existing hybrid graphs in data/graphs/hybrid/; skip)"
        echo "  (set REBUILD_HYBRID_GRAPHS=1 or 'rm data/graphs/hybrid/*_graph.npz' to force rebuild)"
    fi
    ls -la data/graphs/hybrid/ | tail -15
fi

# =============================================================================
# 2a. Train GIN (structural variant)
# =============================================================================
echo ""
echo "=========================================================================="
echo "Step 2a: TraceGIN structural (content-free), 5-fold OOF CV"
echo "=========================================================================="
if [ -f results/route_ab/trace_gnn_structural_pooled.json ] && \
   [ -f results/route_ab/trace_gnn_structural_oof.npz ]; then
    echo "  (already done, skip)"
else
    python src/modeling/trace_gnn.py \
        --graph-glob "data/graphs/*_graph.npz" \
        --variant    structural \
        --output     results/route_ab/trace_gnn_structural_pooled.json \
        --epochs 30 --batch-size 32 --lr 3e-4 --hidden 128 \
        --patience 5 --device cuda --seed 42 || exit 21
fi

# =============================================================================
# 2b. Train GIN (hybrid variant, optional)
# =============================================================================
if [ "$WITH_CONTENT" = "1" ]; then
    echo ""
    echo "=========================================================================="
    echo "Step 2b: TraceGIN hybrid (+content, MiniLM node features)"
    echo "=========================================================================="
    if [ -f results/route_ab/trace_gnn_hybrid_pooled.json ] && \
       [ -f results/route_ab/trace_gnn_hybrid_oof.npz ]; then
        echo "  (already done, skip)"
    else
        python src/modeling/trace_gnn.py \
            --graph-glob "data/graphs/hybrid/*_graph.npz" \
            --variant    hybrid \
            --output     results/route_ab/trace_gnn_hybrid_pooled.json \
            --epochs 30 --batch-size 32 --lr 3e-4 --hidden 128 \
            --patience 5 --device cuda --seed 42 || exit 22
    fi
fi

# =============================================================================
# 3. PRR backfill on every GNN OOF
# =============================================================================
echo ""
echo "=========================================================================="
echo "Step 3: PRR backfill on results/route_ab/trace_gnn_*_oof.npz"
echo "=========================================================================="
python scripts/backfill_prr.py --glob "results/route_ab/trace_gnn_*_oof.npz" || true

# =============================================================================
# Done
# =============================================================================
echo ""
echo "=========================================================================="
echo "Finished at $(date)"
echo "=========================================================================="
ls -la results/route_ab/ | tail -15
