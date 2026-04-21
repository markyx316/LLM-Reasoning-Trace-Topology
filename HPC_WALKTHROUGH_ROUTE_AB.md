# HPC Walkthrough — Route A + B1

Step-by-step runbook for executing the Route A + B1 extension on
**Yale Bouchet** (course cluster `cpsc4770_ym466`,
path `/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology`).

This walkthrough mirrors the execution pattern used by
[sbatch_phase0_phase1_hpc.sh](scripts/sbatch_phase0_phase1_hpc.sh) — the script
that is known to run cleanly in this environment.

**Prereqs (already in place):**
- Parsed JSONLs at `data/parsed/{group}_parsed.jsonl` on HPC.
- Working miniconda env `torch311` (name overridable via `CONDA_ENV=`).
- Legacy artifacts on HPC: `data/step_embeddings/`, `data/traces/`,
  `results/month2_v2/deberta_pooled_oof.npz`,
  `results/month2_v2/step_transformer_pooled_oof.npz`.
- `module load miniconda` available on the compute node (Bouchet default).

**What stays local (laptop / WSL):**
- Code editing.
- Final hybrid meta-learning (`hybrid_route_ab.py`; CPU; minutes).
- Plotting / summary / interpretation.

**What runs on HPC:**
- A1 / A3 / A5 content-free feature extraction (CPU, fast).
- A2 shapelet distance-matrix (CPU-heavy: O(N·M·L)).
- A4 structural persistent homology (CPU, ripser).
- A2+ shapelet OOF predictor.
- B1 GIN training (GPU).
- Optional B1+ hybrid-GIN variant (GPU, +MiniLM).

---

## Step 0 — Sync the code to HPC

From your laptop (WSL) working tree:

```bash
cd /home/marky/LLM-Reasoning-Trace-Topology
git status                                   # confirm on branch `update`, clean

# Option A — push via git, then pull on HPC
git add src/features/ngram_features.py \
        src/features/graph_features.py \
        src/features/timing_features.py \
        src/features/structural_ph_features.py \
        src/features/shapelet_features.py \
        src/modeling/shapelet_eval.py \
        src/modeling/trace_gnn.py \
        src/modeling/hybrid_route_ab.py \
        scripts/build_route_a_features.py \
        scripts/build_trace_graphs.py \
        scripts/sbatch_route_a.sh \
        scripts/sbatch_trace_gnn.sh \
        PLAN_ROUTE_AB.md \
        HPC_WALKTHROUGH_ROUTE_AB.md
git commit -m "Route A + B1 implementation (Bouchet-ready sbatch)"
git push origin update

# On HPC
ssh cpsc4770_ym466@bouchet.ycrc.yale.edu     # adjust hostname if your login alias differs
cd /home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology
git fetch --all
git checkout update
git pull origin update
```

Option B — rsync directly (skip git):

```bash
rsync -av --exclude='data/' --exclude='results/' --exclude='__pycache__' \
    /home/marky/LLM-Reasoning-Trace-Topology/ \
    cpsc4770_ym466@bouchet.ycrc.yale.edu:/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology/
```

If parsed JSONLs are **only on laptop**, push them too:

```bash
rsync -av /home/marky/LLM-Reasoning-Trace-Topology/data/parsed/ \
    cpsc4770_ym466@bouchet.ycrc.yale.edu:/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology/data/parsed/
```

---

## Step 1 — Smoke test on HPC login node (no SLURM)

Before submitting batch jobs, confirm imports + self-tests work in the
miniconda env. The Bouchet pattern is `module load miniconda` → `conda activate`,
**not** sourcing `/etc/profile.d/conda.sh` directly (the system lmod scripts
contain zsh-oriented snippets that break bash).

```bash
ssh cpsc4770_ym466@bouchet.ycrc.yale.edu
cd /home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology

module load miniconda
conda activate torch311

# Confirm core deps
python -c "import numpy, pandas, sklearn, networkx, torch; \
           print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Install optional deps (user-site; the sbatch job will also try this)
python -c "import ripser"      2>/dev/null || pip install --user ripser persim
python -c "import community"   2>/dev/null || pip install --user python-louvain

# Run every module's self-test (each is ~seconds)
export PYTHONPATH="$PWD"
for m in src/features/ngram_features.py \
         src/features/graph_features.py \
         src/features/timing_features.py \
         src/features/structural_ph_features.py \
         src/features/shapelet_features.py \
         src/modeling/shapelet_eval.py \
         scripts/build_trace_graphs.py \
         src/modeling/trace_gnn.py ; do
    echo "=== $m ==="
    python "$m" || echo "FAIL: $m"
done
```

If any module fails its self-test, fix before submitting batch jobs.
The GNN self-test (`trace_gnn.py`) needs CUDA for the full path; on a CPU
login node it just verifies the forward pass.

---

## Step 2 — Submit Route-A feature extraction (CPU batch)

```bash
cd /home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology
mkdir -p logs
sbatch scripts/sbatch_route_a.sh
```

This submits to `partition=day`, 8 CPUs, 48 G RAM, 12 h time limit. The script:
1. Activates conda (`module load miniconda` → `conda activate torch311`).
2. Runs dependency check via heredoc (fails fast if anything's missing).
3. Sources `.env` (HF_TOKEN, etc.) without overriding existing vars.
4. Runs `build_route_a_features.py --with-ph --with-shapelet` (all 5 families).
5. Runs `shapelet_eval.py` for the OOF predictor.
6. Runs `backfill_prr.py` on every new `results/route_ab/*_oof.npz`.

Expected wall time: ~1–3 hours (dominated by A2: `N * M * L` shapelet distmat ops).

**Override examples:**
```bash
# Different partition (e.g. faster short queue)
sbatch --partition=education --time=06:00:00 scripts/sbatch_route_a.sh

# Different conda env
sbatch --export=ALL,CONDA_ENV=torch312 scripts/sbatch_route_a.sh

# Different project dir
sbatch --export=ALL,PROJECT_DIR=/path/to/other/checkout scripts/sbatch_route_a.sh
```

**Monitor:**
```bash
squeue -u $USER
tail -f logs/route_a_<JOBID>.out
```

**Outputs (once done):**
- `data/features/{group}_ngram.csv`            (~231 cols × 8 datasets)
- `data/features/{group}_graph.csv`            (~15 cols × 8 datasets)
- `data/features/{group}_timing.csv`           (~46 cols × 8 datasets)
- `data/features/{group}_structural_ph.csv`    (~36 cols × 8 datasets)
- `data/features/{group}_shapelet_distmat.npz` (~2k candidates × N items)
- `data/features/ngram_vocab.json`             (corpus-wide vocab manifest)
- `results/route_ab/shapelet_oof.npz`          (5-fold OOF probs)
- `results/route_ab/shapelet_pooled.json`      (A2 metrics)
- `results/route_ab/shapelet_metrics_prr.json` (PRR backfill)

**Sanity check:**
```bash
ls -la data/features/*_ngram.csv data/features/*_graph.csv \
       data/features/*_timing.csv data/features/*_structural_ph.csv \
       data/features/*_shapelet_distmat.npz
python - <<'PY'
import pandas as pd, glob
for f in sorted(glob.glob('data/features/*_ngram.csv')):
    d = pd.read_csv(f)
    print(f, d.shape, round(d['is_correct'].mean(), 3))
PY
```

---

## Step 3 — Submit Trace GNN training (GPU batch)

```bash
cd /home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology
sbatch scripts/sbatch_trace_gnn.sh                       # structural only
# OR, also train the content-augmented variant (adds ~30 min):
WITH_CONTENT=1 sbatch scripts/sbatch_trace_gnn.sh
```

This submits to `partition=gpu_rtx6000`, 1× GPU, 4 CPUs, 48 G RAM, 6 h.
The script:
1. Same conda activation pattern as above.
2. Verifies `torch.cuda.is_available()` and exits early if not.
3. Builds graph .npz per dataset (CPU, ~minute).
4. Trains 2-layer GIN, 30 epochs / patience 5 / 5-fold OOF CV across 8 datasets.
5. If `WITH_CONTENT=1`: also builds hybrid graphs (+MiniLM) and trains the hybrid GIN.
6. Runs `backfill_prr.py` on `trace_gnn_*_oof.npz`.

Expected wall time: ~30 min structural, ~60 min +hybrid.

**Override examples:**
```bash
# Different GPU partition / type
sbatch --partition=gpu_devel --gres=gpu:a100:1 --time=03:00:00 \
    scripts/sbatch_trace_gnn.sh

# Explicit hybrid run with content
WITH_CONTENT=1 sbatch scripts/sbatch_trace_gnn.sh
```

**Outputs:**
- `data/graphs/{group}_graph.npz`             (structural, 10-d node features)
- `data/graphs/hybrid/{group}_graph.npz`      (hybrid 394-d if `WITH_CONTENT=1`)
- `results/route_ab/trace_gnn_structural_pooled.json`
- `results/route_ab/trace_gnn_structural_oof.npz`
- `results/route_ab/trace_gnn_structural_metrics_prr.json`
- `results/route_ab/trace_gnn_hybrid_{pooled,oof}.{json,npz}` if requested
- `results/route_ab/trace_gnn_hybrid_metrics_prr.json` if requested

**Sanity check:**
```bash
python - <<'PY'
import numpy as np
from sklearn.metrics import roc_auc_score
z = np.load('results/route_ab/trace_gnn_structural_oof.npz', allow_pickle=True)
print('n_items:', len(z['item_ids']))
print('pooled AUROC:', roc_auc_score(z['y_true'], z['oof_prob']))
PY
```

---

## Step 4 — Pull artifacts back to local (optional)

Meta-learning is CPU-light and fast locally. Pull the artifacts:

```bash
# On laptop
mkdir -p /home/marky/LLM-Reasoning-Trace-Topology/results/route_ab/
rsync -av \
    cpsc4770_ym466@bouchet.ycrc.yale.edu:/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology/results/route_ab/ \
    /home/marky/LLM-Reasoning-Trace-Topology/results/route_ab/

# Feature CSVs (for the hybrid to merge with)
rsync -av --include='*_ngram.csv' --include='*_graph.csv' \
          --include='*_timing.csv' --include='*_structural_ph.csv' \
          --exclude='*' \
    cpsc4770_ym466@bouchet.ycrc.yale.edu:/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology/data/features/ \
    /home/marky/LLM-Reasoning-Trace-Topology/data/features/
```

Legacy DeBERTa / StepTF OOFs (if only on HPC — note `results/month2_v2/`,
which is the honest-single-seed output of `sbatch_phase0_phase1_hpc.sh`,
**not** the older buggy `results/month2/`):

```bash
rsync -av \
    cpsc4770_ym466@bouchet.ycrc.yale.edu:/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology/results/month2_v2/ \
    /home/marky/LLM-Reasoning-Trace-Topology/results/month2_v2/
```

---

## Step 5 — Run the hybrid meta-learner

Fast enough to run locally (under a minute for 8 × ~800 items × 15 variants × 3
classifiers). Can also be run on the HPC login node.

```bash
cd /home/marky/LLM-Reasoning-Trace-Topology
source /home/marky/miniconda3/etc/profile.d/conda.sh
conda activate torch311

PYTHONPATH=. python src/modeling/hybrid_route_ab.py \
    --features-glob      "data/features/*_features_rec.csv" \
    --ngram-glob         "data/features/*_ngram.csv" \
    --graph-glob         "data/features/*_graph.csv" \
    --timing-glob        "data/features/*_timing.csv" \
    --structural-ph-glob "data/features/*_structural_ph.csv" \
    --shapelet-oof       results/route_ab/shapelet_oof.npz \
    --gnn-structural-oof results/route_ab/trace_gnn_structural_oof.npz \
    --deberta-oof        results/month2_v2/deberta_pooled_oof.npz \
    --step-oof           results/month2_v2/step_transformer_pooled_oof.npz \
    --output             results/route_ab/route_ab_pooled.json \
    --clf all --seed 42
```

Adjust the OOF globs to match where your artifacts actually live. If you ran
`WITH_CONTENT=1` on HPC, also pass `--gnn-hybrid-oof
results/route_ab/trace_gnn_hybrid_oof.npz`.

**Outputs:**
- `results/route_ab/route_ab_pooled.json`             — pooled metrics per variant
- `results/route_ab/route_ab_pooled_per_dataset.csv`  — per (variant, group) metrics
- `results/route_ab/route_ab_pooled_falsifier.csv`    — ROUTE_AB_TOTAL vs DeBERTa, per dataset

**Falsifier rule.** ROUTE_AB_TOTAL must beat or tie DeBERTa on ≥ 5/8 datasets to
salvage the "structure beats content" claim. Otherwise fall back to the
complementarity framing (already supported: `ROUTE_AB+deberta` > `deberta_only`).

---

## Step 6 — Interpret the ablation

Primary questions and where to look:

1. **Which Route-A family individually helps most?**
   Compare `C+rec+{ngram,graph,timing,structural_ph}` vs `baselineC+rec`
   in the pooled JSON. Expect `ngram` to contribute most (A1 captures
   motif patterns DeBERTa cannot see).

2. **Does Route A stacked beat StepTF alone?**
   `ROUTE_A_FULL` vs `step_only` AUROC in the per-dataset CSV. If yes,
   pure structural features match text-adjacent learned embeddings.

3. **Does shapelet OOF + GNN add beyond Route A?**
   `ROUTE_AB_TOTAL` vs `ROUTE_A_FULL` delta. Expect +0.01 to +0.04 AUROC.

4. **Does structure add to DeBERTa?**
   `ROUTE_AB+deberta` vs `deberta_only`. Even a +0.01 AUROC gain supports
   the complementarity framing (fallback-paper claim).

5. **Falsifier.** If ROUTE_AB_TOTAL is under DeBERTa on > 3/8 datasets,
   the headline claim is not rescued; pivot to "structure is
   content-complementary" (already safe).

---

## Troubleshooting

### `module load miniconda` returns a non-zero code

The sbatch scripts deliberately disable the ERR trap around `module load`
because the system lmod scripts (`/etc/profile.d/lmod.sh`) contain zsh
snippets that `exit 127` under bash. This is handled automatically by
`trap - ERR; set +e; ... ; set -e; trap ...`. If you still see a FATAL
here, `module avail miniconda` should list the available versions — pin
one explicitly:

```bash
module load miniconda/23.5.2     # or whichever version is present
```

### `conda activate torch311` fails

```bash
conda env list                   # is torch311 actually present?
# If not — create it once:
conda create -n torch311 python=3.11 -y
conda activate torch311
pip install torch numpy pandas scikit-learn networkx tqdm ripser persim \
            python-louvain sentence-transformers transformers xgboost
```

### `ripser` install fails on HPC

Cython needs a working C toolchain. Try:
```bash
module load foss                 # provides gcc
pip install --user --no-build-isolation ripser
```

If that still fails, A4 (structural PH) will be silently skipped by the
orchestrator; the pipeline continues without the `h0_*/h1_*` features. The
hybrid meta-learner will auto-skip variants that depend on them.

### GNN training OOMs

The padded-batch GIN scales as `B * N^2`. If `max_nodes > 256` causes memory
issues, lower it at graph-build time and re-submit:
```bash
python scripts/build_trace_graphs.py --max-nodes 128 \
    --parsed-glob "data/parsed/*_parsed.jsonl" --output-dir data/graphs/
sbatch scripts/sbatch_trace_gnn.sh
```
Traces longer than the cap are truncated (<3% of traces).

### Shapelet distmat job too slow

Reduce candidate budget (set on the underlying extractor):
```bash
PYTHONPATH=. python src/features/shapelet_features.py \
    --parsed-glob "data/parsed/*_parsed.jsonl" \
    --output-dir data/features/ \
    --max-candidates-per-k 1000        # default is 2000
```

### Join returns 0 rows in `hybrid_route_ab.py`

Almost always a dataset-column mismatch. Verify filenames encode the
`{dataset}_{model}` suffix (e.g. `math500_qwen7b_ngram.csv`, not
`math500_ngram.csv`). The `_group_name_from_path` helper strips suffixes
to derive the canonical group name; every family must agree.

### `np.array(..., dtype=object)` ValueError on graph save

Caused by numpy trying to coerce ragged object arrays. Both
`build_trace_graphs.py` and `trace_gnn.py` use a `_as_object` helper
that preallocates a Python-object array and fills per slot. Apply the
same pattern if you see the error elsewhere.

### Mail notifications are noisy

The sbatch scripts set `--mail-type=ALL` to match the working
`sbatch_phase0_phase1_hpc.sh` pattern. To silence, override at submit:
```bash
sbatch --mail-type=END,FAIL scripts/sbatch_route_a.sh
# or
sbatch --mail-type=NONE scripts/sbatch_route_a.sh
```

---

## Quick-reference execution order

```bash
# --- laptop ---
git push origin update

# --- HPC (Bouchet) ---
ssh cpsc4770_ym466@bouchet.ycrc.yale.edu
cd /home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology
git pull

sbatch scripts/sbatch_route_a.sh                     # ~1-3 h   CPU
sbatch scripts/sbatch_trace_gnn.sh                   # ~0.5 h   GPU
# Optional hybrid GNN (+MiniLM):
# WITH_CONTENT=1 sbatch scripts/sbatch_trace_gnn.sh  # ~1 h     GPU

squeue -u $USER                                      # monitor

# --- back on laptop (after both jobs complete) ---
BOUCHET="cpsc4770_ym466@bouchet.ycrc.yale.edu:/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology"
rsync -av "$BOUCHET/results/route_ab/" ./results/route_ab/
rsync -av --include='*_ngram.csv' --include='*_graph.csv' \
          --include='*_timing.csv' --include='*_structural_ph.csv' \
          --exclude='*' "$BOUCHET/data/features/" ./data/features/

PYTHONPATH=. python src/modeling/hybrid_route_ab.py \
    --features-glob      "data/features/*_features_rec.csv" \
    --ngram-glob         "data/features/*_ngram.csv" \
    --graph-glob         "data/features/*_graph.csv" \
    --timing-glob        "data/features/*_timing.csv" \
    --structural-ph-glob "data/features/*_structural_ph.csv" \
    --shapelet-oof       results/route_ab/shapelet_oof.npz \
    --gnn-structural-oof results/route_ab/trace_gnn_structural_oof.npz \
    --deberta-oof        results/month2_v2/deberta_pooled_oof.npz \
    --step-oof           results/month2_v2/step_transformer_pooled_oof.npz \
    --output             results/route_ab/route_ab_pooled.json --clf all

# Read the three output tables:
cat  results/route_ab/route_ab_pooled_falsifier.csv
head -20 results/route_ab/route_ab_pooled_per_dataset.csv
```
