# Phase 3 HPC Runbook

End-to-end line-by-line instructions for running every Phase 3 experiment on
Yale HPC (or any SLURM cluster with the same module stack). Every sbatch
uses the user-verified template:

```
#SBATCH --partition=gpu_b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --chdir=/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology
module load miniconda
conda activate torch311
```

If you need to change the partition, chdir path, or conda env, edit the
`#SBATCH --partition`, `--chdir`, and `conda activate` lines at the top of
each `scripts/sbatch_phase3_*.sh`.

---

## Prereqs (run once on the HPC after pulling the branch)

```bash
# SSH into the cluster, then:
cd /home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology

# 1. Sanity-check the env
module load miniconda
conda activate torch311
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 2. Ensure logs/ exists
mkdir -p logs

# 3. Put DEEPSEEK_API_KEY in .env (needed for T1 LLM-judge)
#    The key looks like 'sk-...'. Never commit .env.
grep -q '^DEEPSEEK_API_KEY=' .env || echo 'DEEPSEEK_API_KEY=sk-your-key' >> .env

# 4. Ensure parsed JSONLs exist (needed for T5 Graphormer). If missing:
# bash scripts/run_parsing.sh parse
ls data/parsed/*_parsed.jsonl | wc -l   # should be 8
```

---

## Experiment submission order

Phase 3 tests are grouped into three tiers by expected-value (EV) per GPU-hour.
Run Tier 1 in parallel, wait for them, then run Tier 2 in parallel, etc.

### Tier 1 — Highest EV, independent of other Phase 3 artifacts

| # | Task | Script | Est. GPU-h | Produces |
|---|---|---|---|---|
| T1 | LLM-as-judge OOF | `sbatch_phase3_llm_judge.sh` | 1–2 h (I/O bound) | `results/month3/llm_judge_deepseek_v3_oof.npz` |
| T2 | Token-logprob features | `sbatch_phase3_token_logprobs.sh` | 3–5 h | `data/features/*_features_steplp.csv`, `data/token_logprobs/*_raw.npz` |
| T3 | Multi-layer probes | `sbatch_phase3_multilayer_probes.sh` | 4–6 h | `data/hidden_atlas/*.npz`, `results/month3/multi_layer_probe*_oof.npz` |

Submit all three at once; they do not depend on one another:

```bash
sbatch scripts/sbatch_phase3_llm_judge.sh
sbatch scripts/sbatch_phase3_token_logprobs.sh
sbatch scripts/sbatch_phase3_multilayer_probes.sh
squeue --me
```

### Tier 2 — Depends on Tier 1 (step embeddings) or existing artifacts

| # | Task | Script | Est. GPU-h | Depends on | Produces |
|---|---|---|---|---|---|
| T4 | Step-Transformer v2 (MPNet / E5) | `sbatch_phase3_step_tf_v2.sh` | 2–4 h | nothing new | `results/month3/step_tf_v2_<enc>_oof.npz` |

```bash
# MPNet (default, 768-d):
sbatch scripts/sbatch_phase3_step_tf_v2.sh

# E5-large (1024-d):
ENCODER=e5-large sbatch scripts/sbatch_phase3_step_tf_v2.sh

# BGE-large (1024-d):
ENCODER=bge-large sbatch scripts/sbatch_phase3_step_tf_v2.sh
```

### Tier 3 — Most speculative, lowest EV

| # | Task | Script | Est. GPU-h | Depends on | Produces |
|---|---|---|---|---|---|
| T5 | Graphormer over enriched DAGs | `sbatch_phase3_graphormer.sh` | 3–5 h | `data/parsed/*_parsed.jsonl` | `results/month3/graphormer_v3_oof.npz` |

```bash
sbatch scripts/sbatch_phase3_graphormer.sh
```

---

## Local-only tasks (do NOT submit to HPC)

These run in seconds to a few minutes on a laptop and do not need GPU.
They consume the OOFs produced by the HPC jobs above.

### T6 — Answer-trace consistency features

Uses existing `data/step_embeddings/*.npz` (MiniLM-L6) + a MiniLM encode
of each trace's final answer.

```bash
# Local, from repo root, with torch311 env:
conda activate torch311
PYTHONPATH=. python scripts/phase3/build_answer_trace_features.py \
    --traces-glob 'data/traces/*_traces.jsonl' \
    --step-emb-dir data/step_embeddings \
    --out-dir data/features
# Produces data/features/{group}_ans_trace.csv  (12 features per trace)
```

### T7 — Selective-prediction curves

```bash
PYTHONPATH=. python scripts/phase3/selective_prediction_curves.py \
    --oofs \
      ULTRA_TEXT_ONLY_LR=results/month3/ultra_hybrid/ultrahybrid_ULTRA_TEXT_ONLY__lr_oof.npz \
      SH_LR=results/month3/superhybrid_SuperHybrid_LR_oof.npz \
      SH_RF=results/month3/superhybrid_SuperHybrid_RF_oof.npz \
    --out-dir reports/month3
# Produces reports/month3/phase3_selective.csv + phase3_selective_summary.csv
```

After the T1 judge OOF is back from HPC you can add it to the compare:

```bash
PYTHONPATH=. python scripts/phase3/selective_prediction_curves.py \
    --oofs \
      ULTRA_TEXT_ONLY_LR=results/month3/ultra_hybrid/ultrahybrid_ULTRA_TEXT_ONLY__lr_oof.npz \
      LLM_JUDGE=results/month3/llm_judge_deepseek_v3_oof.npz \
      P3_STACK_LR=results/month3/phase3_stack_LR_oof.npz \
    --out-dir reports/month3
```

---

## Stacking the Phase 3 OOFs

Once Tiers 1–3 have all finished (or whichever subset you want to combine),
stack them with the existing `super_hybrid.py` driver. Every Phase 3 artifact
follows the standard OOF schema (`item_ids`, `groups`, `y_true`, `oof_prob`,
`oof_fold`), so the stacker treats them as drop-in predictor views.

Example (on the HPC, via a CPU-only sbatch; or just run locally — it's fast):

```bash
PYTHONPATH=. python src/modeling/super_hybrid.py \
    --deberta-oof     results/month2/deberta_pooled_oof.npz \
    --cond-oof        results/month2/deberta_conditioned_pooled_oof.npz \
    --probe-oof       results/month3/multi_layer_probe_L_spread_6_P_answer_oof.npz \
    --judge-oof       results/month3/llm_judge_deepseek_v3_oof.npz \
    --steptfv2-oof    results/month3/step_tf_v2_mpnet_oof.npz \
    --graphormer-oof  results/month3/graphormer_v3_oof.npz \
    --features-glob   'data/features/*_features_rec.csv' \
    --genunc-glob     'data/features/*_features_genunc.csv' \
    --steplp-glob     'data/features/*_features_steplp.csv' \
    --ans-trace-glob  'data/features/*_ans_trace.csv' \
    --output          results/month3/phase3_stack.json
```

> Note: `super_hybrid.py` may need new `--judge-oof / --steptfv2-oof /
> --graphormer-oof / --steplp-glob / --ans-trace-glob` flags. If they're
> missing, either add them to the CLI or use `src/modeling/hybrid_route_ab.py`
> / `src/modeling/save_super_hybrid_oof.py` as a template.

---

## Troubleshooting

### DEEPSEEK_API_KEY not found
Check `.env` in the repo root:
```bash
grep 'DEEPSEEK_API_KEY' .env
```
If missing, add it. The key is sourced by both the sbatch wrapper and the
Python driver.

### `module load miniconda` exits non-zero
Verify that the cluster exposes a miniconda module. If your cluster uses a
different name (e.g. `anaconda3`), edit the `module load` line in every
`scripts/sbatch_phase3_*.sh` file.

### OOM on the Graphormer job
Graphormer dense attention is `O(L^2)` per batch element. Reduce batch size
in `sbatch_phase3_graphormer.sh`:
```
--batch-size 8
```
Or reduce model width: `--d 128 --n-heads 4`.

### Step-TF v2 fails with "sentence_transformers not found"
In the torch311 env:
```bash
conda activate torch311
pip install sentence-transformers
```

### Partial job re-runs
Every major script is idempotent: it skips already-written outputs (atlas,
step embeddings, graphs) and the LLM-judge resumes from a JSONL cache.
To force a re-run, delete the corresponding output file / directory first.

---

## Where to look when everything finishes

| Artifact | Path |
|---|---|
| Per-task OOFs | `results/month3/*_oof.npz` |
| Per-task metrics | `results/month3/*.json` |
| Phase 3 stack | `results/month3/phase3_stack.json` (+ `_oof.npz`) |
| Selective-prediction curves | `reports/month3/phase3_selective*.csv` |
| Summary narrative | `reports/month3/PHASE3_SUMMARY.md` (update as jobs land) |
