#!/bin/bash
# =============================================================================
# sbatch_rerun_clean_oofs.sh
#
# Re-produce all 5 neural base-model OOFs after the 2026-04-20 leakage fix
# (removal of best-epoch-on-val model selection in deberta_baseline.py,
# step_transformer.py, trace_gnn.py; last-epoch OOF protocol).
#
# Why this script:
#   The existing .npz files at
#       results/roberta_pooled_oof.npz
#       results/month2/deberta_pooled_oof.npz
#       results/step_transformer_pooled_oof.npz
#       results/route_ab/trace_gnn_structural_pooled_oof.npz
#       results/route_ab/trace_gnn_hybrid_pooled_oof.npz
#   were produced before the fix and carry inflated OOF AUROC (~0.01-0.025)
#   due to per-fold epoch selection on the held-out fold. Each one has a
#   .PROVENANCE.json sidecar flagging the issue. This job re-runs all 5 end
#   to end with the patched code and writes outputs to *_clean.json /
#   *_clean_oof.npz, leaving the tainted originals untouched. After the job
#   succeeds, promote the clean files over the originals (see PROMOTE step
#   below) and re-run build_hybrid_table.py.
#
# Total GPU time on 1× RTX6000 / A100:
#   RoBERTa pooled   : ~2 h
#   DeBERTa pooled   : ~2 h
#   StepTF pooled    : ~30 min
#   GIN structural   : ~30 min
#   GIN hybrid       : ~1 h
#   total            : ~6-7 h (request 12 h for headroom)
#
# Submit with:
#     cd /home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology
#     sbatch scripts/sbatch_rerun_clean_oofs.sh
#
# Env-var overrides (optional):
#     CONDA_ENV=torch311               Python environment
#     STEPS="roberta deberta step gin"  Subset to run (default: all)
#     FORCE=1                           Re-run even if *_clean.* exists
# =============================================================================

#SBATCH --job-name=clean_oofs
#SBATCH --partition=gpu_rtx6000
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology
#SBATCH --output=/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology/logs/clean_oofs_%j.out
#SBATCH --error=/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology/logs/clean_oofs_%j.err
#SBATCH --mail-type=ALL

set -Eeo pipefail
trap 'ec=$?; echo "==== FAILED at line $LINENO (exit $ec): $BASH_COMMAND" >&2; exit $ec' ERR

# =============================================================================
# CLUSTER / ENV
# =============================================================================
PROJECT_DIR="${PROJECT_DIR:-/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology}"
CONDA_ENV="${CONDA_ENV:-torch311}"
STEPS="${STEPS:-roberta deberta step gin_structural gin_hybrid}"
FORCE="${FORCE:-0}"

cd "$PROJECT_DIR"
mkdir -p logs results/month2 results/route_ab .cache/huggingface

echo "=========================================================================="
echo "Clean-OOF re-run (leakage-safe last-epoch protocol)"
echo "Started:     $(date)"
echo "Node:        $(hostname)"
echo "PWD:         $PWD"
echo "STEPS:       $STEPS"
echo "FORCE:       $FORCE"
echo "GPUs:        $(nvidia-smi -L 2>/dev/null || echo 'NONE DETECTED')"
echo "=========================================================================="

# =============================================================================
# Conda activation (same pattern as sbatch_trace_gnn.sh)
# =============================================================================
trap - ERR
set +e
module load miniconda
conda activate "$CONDA_ENV" || { echo "FATAL: conda activate $CONDA_ENV failed"; exit 2; }
set -e
trap 'ec=$?; echo "==== FAILED at line $LINENO (exit $ec): $BASH_COMMAND" >&2; exit $ec' ERR

export PYTHONPATH="$PROJECT_DIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"

python -c "import torch; print('Torch', torch.__version__, 'cuda=', torch.cuda.is_available())"

# =============================================================================
# Verify patches are in place — grep for the telltale comment
# =============================================================================
echo ""
echo "=== Verifying leakage patches are in source ==="
for f in src/modeling/deberta_baseline.py \
         src/modeling/step_transformer.py \
         src/modeling/trace_gnn.py \
         src/modeling/behavior_seq_lm.py \
         src/modeling/trace_mlm_encoder.py; do
    if grep -q "LEAKAGE-SAFE OOF PROTOCOL\|last-epoch protocol" "$f"; then
        echo "  OK   $f"
    else
        echo "  FAIL $f  — leakage patch not applied; abort"
        exit 3
    fi
done

run_if_needed() {
    local name="$1"
    local out_json="$2"
    shift 2
    if [[ "$STEPS" != *"$name"* ]]; then
        echo "--- SKIP $name (not in STEPS='$STEPS')"
        return
    fi
    if [[ "$FORCE" != "1" && -f "$out_json" ]]; then
        echo "--- SKIP $name (output exists: $out_json; set FORCE=1 to re-run)"
        return
    fi
    echo ""
    echo "=========================================================================="
    echo "STEP: $name  ->  $out_json"
    echo "STARTED at $(date)"
    echo "=========================================================================="
    "$@"
    echo "STEP: $name DONE at $(date)"
}

# =============================================================================
# 1. RoBERTa pooled (uses deberta_baseline.py with --model roberta-base)
# =============================================================================
run_if_needed "roberta" "results/roberta_pooled_clean.json" \
    python src/modeling/deberta_baseline.py \
        --model roberta-base \
        --traces-glob "data/traces/*_traces.jsonl" \
        --output      results/roberta_pooled_clean.json \
        --epochs 3 --batch-size 16 --lr 2e-5 --seed 42

# =============================================================================
# 2. DeBERTa pooled
# =============================================================================
run_if_needed "deberta" "results/month2/deberta_pooled_clean.json" \
    python src/modeling/deberta_baseline.py \
        --model microsoft/deberta-v3-base \
        --traces-glob "data/traces/*_traces.jsonl" \
        --output      results/month2/deberta_pooled_clean.json \
        --epochs 3 --batch-size 8 --lr 2e-5 --seed 42

# =============================================================================
# 3. Step Transformer pooled
# =============================================================================
# Prereq: step embeddings must exist. If not, build them first.
if [[ "$STEPS" == *"step"* ]]; then
    EXISTING_EMB=$(ls data/step_embeddings/*.npz 2>/dev/null | wc -l)
    if [ "$EXISTING_EMB" -lt 8 ]; then
        echo "Only $EXISTING_EMB step embeddings found (<8); rebuilding..."
        python scripts/build_step_embeddings.py --all --batch-size 256
    fi
fi
run_if_needed "step" "results/step_transformer_pooled_clean.json" \
    python src/modeling/step_transformer.py \
        --npz-glob "data/step_embeddings/*.npz" \
        --output   results/step_transformer_pooled_clean.json \
        --epochs 15 --batch-size 32 --lr 3e-4

# =============================================================================
# 4. TraceGIN structural
# =============================================================================
# Prereq: structural trace graphs
if [[ "$STEPS" == *"gin"* ]]; then
    EXISTING_G=$(ls data/graphs/*_graph.npz 2>/dev/null | wc -l)
    if [ "$EXISTING_G" -lt 8 ]; then
        echo "Only $EXISTING_G structural graphs found (<8); rebuilding..."
        python scripts/build_trace_graphs.py \
            --parsed-glob "data/parsed/*_parsed.jsonl" \
            --output-dir  data/graphs/
    fi
fi
run_if_needed "gin_structural" "results/route_ab/trace_gnn_structural_pooled_clean.json" \
    python src/modeling/trace_gnn.py \
        --graph-glob "data/graphs/*_graph.npz" \
        --variant    structural \
        --output     results/route_ab/trace_gnn_structural_pooled_clean.json \
        --epochs 30 --batch-size 32 --lr 3e-4 --hidden 128 \
        --device cuda --seed 42

# =============================================================================
# 5. TraceGIN hybrid (requires +MiniLM content graphs)
# =============================================================================
if [[ "$STEPS" == *"gin_hybrid"* ]]; then
    EXISTING_H=$(ls data/graphs/hybrid/*_graph.npz 2>/dev/null | wc -l)
    if [ "$EXISTING_H" -lt 8 ]; then
        echo "Only $EXISTING_H hybrid graphs found (<8); rebuilding (includes MiniLM encoding, ~30 min)..."
        python -c "import sentence_transformers" 2>/dev/null || \
            pip install --user sentence-transformers
        python scripts/build_trace_graphs.py \
            --parsed-glob "data/parsed/*_parsed.jsonl" \
            --output-dir  data/graphs/hybrid/ \
            --with-content --device cuda
    fi
fi
run_if_needed "gin_hybrid" "results/route_ab/trace_gnn_hybrid_pooled_clean.json" \
    python src/modeling/trace_gnn.py \
        --graph-glob "data/graphs/hybrid/*_graph.npz" \
        --variant    hybrid \
        --output     results/route_ab/trace_gnn_hybrid_pooled_clean.json \
        --epochs 30 --batch-size 32 --lr 3e-4 --hidden 128 \
        --device cuda --seed 42

# =============================================================================
# 6. Summary — read each clean OOF's pooled AUROC to compare vs tainted
# =============================================================================
echo ""
echo "=========================================================================="
echo "CLEAN OOF inventory (produced with last-epoch protocol)"
echo "=========================================================================="
python - <<'PY'
import json, os, numpy as np
from sklearn.metrics import roc_auc_score
pairs = [
    ("RoBERTa",  "results/roberta_pooled_oof.npz",
                 "results/roberta_pooled_clean_oof.npz"),
    ("DeBERTa",  "results/month2/deberta_pooled_oof.npz",
                 "results/month2/deberta_pooled_clean_oof.npz"),
    ("StepTF",   "results/step_transformer_pooled_oof.npz",
                 "results/step_transformer_pooled_clean_oof.npz"),
    ("GIN-str",  "results/route_ab/trace_gnn_structural_pooled_oof.npz",
                 "results/route_ab/trace_gnn_structural_pooled_clean_oof.npz"),
    ("GIN-hyb",  "results/route_ab/trace_gnn_hybrid_pooled_oof.npz",
                 "results/route_ab/trace_gnn_hybrid_pooled_clean_oof.npz"),
]
print(f"{'model':12s}  {'tainted_AUROC':14s}  {'clean_AUROC':14s}  {'delta':>7s}")
print("-"*55)
for name, t, c in pairs:
    def auc(p):
        if not os.path.exists(p): return None
        z = np.load(p, allow_pickle=True)
        return float(roc_auc_score(z["y_true"], z["oof_prob"]))
    a, b = auc(t), auc(c)
    delta = (b - a) if (a is not None and b is not None) else None
    astr = f"{a:.4f}" if a is not None else "MISS"
    bstr = f"{b:.4f}" if b is not None else "MISS"
    dstr = f"{delta:+.4f}" if delta is not None else "   ?   "
    print(f"{name:12s}  {astr:14s}  {bstr:14s}  {dstr}")
PY

# =============================================================================
# 7. Promote instructions
# =============================================================================
echo ""
echo "=========================================================================="
echo "NEXT STEPS — manual promotion of clean OOFs over tainted ones"
echo "=========================================================================="
cat <<'EOF'
After reviewing the AUROC deltas above, promote the clean OOFs if they look
reasonable (expect ~0.01-0.025 DROP vs tainted — that's the leakage coming
out). One-liner:

    for pair in \
        "results/roberta_pooled_clean_oof.npz:results/roberta_pooled_oof.npz" \
        "results/month2/deberta_pooled_clean_oof.npz:results/month2/deberta_pooled_oof.npz" \
        "results/step_transformer_pooled_clean_oof.npz:results/step_transformer_pooled_oof.npz" \
        "results/route_ab/trace_gnn_structural_pooled_clean_oof.npz:results/route_ab/trace_gnn_structural_pooled_oof.npz" \
        "results/route_ab/trace_gnn_hybrid_pooled_clean_oof.npz:results/route_ab/trace_gnn_hybrid_pooled_oof.npz"
    do
        src="${pair%:*}"; dst="${pair#*:}"
        [ -f "$src" ] || { echo "SKIP $src (missing)"; continue; }
        cp -v "$dst" "${dst%.npz}.LEAKY.npz"   # backup tainted
        cp -v "$src" "$dst"
        cp -v "$src.PROVENANCE.json" "$dst.PROVENANCE.json" 2>/dev/null || true
    done

Then rebuild + retune:

    python scripts/write_oof_provenance.py
    PYTHONPATH=. python scripts/build_hybrid_table.py  --leaky-policy fail
    PYTHONPATH=. python scripts/tune_hybrid.py --n-trials 500 \
        --study-name hybrid_v1_clean \
        --storage "sqlite:///data/optuna_hybrid_clean.db"

EOF
echo ""
echo "=========================================================================="
echo "Finished at $(date)"
echo "=========================================================================="
