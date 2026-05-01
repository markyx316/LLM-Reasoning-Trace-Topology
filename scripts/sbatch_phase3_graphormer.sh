#!/bin/bash
# =============================================================================
# Phase 3 (T5) — Graphormer over enriched trace DAGs.
#
# Stage A  scripts/phase3/build_trace_dags.py   → data/graphs_v3/
#          (same schema as build_trace_graphs.py + a new `edge_types` array;
#           includes REVISION edges from X/R episodes to nearest prior F/V.)
#
# Stage B  scripts/phase3/train_graphormer.py
#          Trains a small Graphormer (d=192, 4 layers, 6 heads) with
#          centrality + SPD + edge-type-on-path attention biases.
#          Writes results/month3/graphormer_v3.json + _oof.npz.
#
# Submit
# ------
#   sbatch scripts/sbatch_phase3_graphormer.sh
#
# Notes
# -----
#   - Assumes `data/parsed/*_parsed.jsonl` already exist. Run
#     `./scripts/run_parsing.sh parse` first if not.
#   - Floyd-Warshall on L≤256 graphs is ~5-10 ms each; for ~50k graphs that's
#     well under an hour of pre-processing, mostly CPU-bound.
# =============================================================================
#SBATCH --job-name=p3_graph
#SBATCH --partition=gpu_b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=9:00:00
#SBATCH --chdir=/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology
#SBATCH --output=logs/p3_graph_%j.out
#SBATCH --error=logs/p3_graph_%j.err

trap - ERR
set +e

mkdir -p logs data/graphs_v3 results/month3

echo "=== Phase 3 / T5 Graphormer  start $(date) on $(hostname) ==="

module load miniconda || exit 2
conda activate torch311 || exit 3

export PYTHONPATH=$PWD

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 10
nvidia-smi -L | head -3

echo ""
echo "=== Stage A: Build enriched trace DAGs (adds revision-reference edges) ==="
python scripts/phase3/build_trace_dags.py \
    --parsed-glob 'data/parsed/*_parsed.jsonl' \
    --output-dir data/graphs_v3
RCA=$?
if [ $RCA -ne 0 ]; then
    echo "[FATAL] build_trace_dags failed with $RCA"
    exit 20
fi
ls -la data/graphs_v3/ | head
du -sh data/graphs_v3

echo ""
echo "=== Stage B: Train Graphormer with 5-fold OOF ==="
python scripts/phase3/train_graphormer.py \
    --npz-glob 'data/graphs_v3/*_graph_v3.npz' \
    --output results/month3/graphormer_v3.json \
    --epochs 25 \
    --batch-size 16 \
    --lr 2e-4 \
    --n-splits 5 \
    --seed 42 \
    --d 192 \
    --n-heads 6 \
    --n-layers 4
RCB=$?
if [ $RCB -ne 0 ]; then
    echo "[FATAL] train_graphormer failed with $RCB"
    exit 21
fi

echo ""
echo "=== Output summary ==="
ls -la results/month3/graphormer_v3*

echo ""
echo "=== DONE at $(date) ==="
exit 0
