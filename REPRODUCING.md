# Reproducing the SuperHybrid Headline Numbers

This walkthrough takes you from a fresh clone to the **AUROC 0.815 / ECE 0.042**
SuperHybrid pooled result reported in `reports/FINAL_REPORT.pdf`. Steps 1-3
require GPUs (we used Yale's Grace HPC cluster with NVIDIA A100/H100); steps
4-6 are CPU-only and fast.

If you are reviewing the project and only want to reproduce the **headline
table and ablation** without re-running the heavy generation/training, **skip to
step 4** — all base-model OOF probabilities are committed under `results/`.

---

## 0. Prerequisites

```bash
# Repo-root on PYTHONPATH
export PYTHONPATH=$PWD

# Python deps (see README.md for full install)
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

For trace generation you also need either:
- A GPU + the HuggingFace cache for `DeepSeek-R1-Distill-Qwen-7B` and
  `DeepSeek-R1-Distill-Llama-8B`, **OR**
- `DEEPSEEK_API_KEY` in `.env` (much slower for 6,378 traces but no GPU
  needed)

For DeBERTa fine-tuning and hidden-state probing you need a GPU with at
least 24 GB of memory (A100/H100 used in our experiments).

---

## 1. Trace generation (GPU; ~8-12 hours total)

```bash
# Local + GPU (one model at a time)
./scripts/run_generation.sh hpc-all qwen7b
./scripts/run_generation.sh hpc-all llama8b

# OR via DeepSeek API (slower, no local GPU needed)
./scripts/run_generation.sh api-all
```

Outputs: `data/traces/{dataset}_{model}_traces.jsonl` (8 files, one per
cell). Each line is a JSON record containing `item_id`, `dataset`, the full
`reasoning_trace`, the extracted `answer_text`, the gold answer, and the
binary `is_correct` label.

Sanity check after generation:
```bash
./scripts/run_generation.sh status
```

---

## 2. Behavior parsing + handcrafted features (CPU; ~5-10 min)

```bash
# Parse traces into 6-class behavior episodes
./scripts/run_parsing.sh all

# Extract the 28-dim behavioral feature vector per trace
PYTHONPATH=. python src/features/feature_extractor.py \
    --traces-glob "data/traces/*_traces.jsonl" \
    --output-dir  data/features/
```

Outputs:
- `data/parsed/{dataset}_{model}_parsed.jsonl` — episode sequences
- `data/features/{dataset}_{model}_features.csv` — 28-dim feature CSVs

---

## 3. Train the four base predictors (GPU; ~6-10 hours total)

Each base predictor is trained with 5-fold stratified CV; out-of-fold (OOF)
probabilities are saved to `results/` as `.npz` files for the meta-learner.

### 3a. DeBERTa-v3 on the trace tail (signal P_text)

```bash
sbatch scripts/sbatch_phase0_phase1_hpc.sh        # HPC SLURM
# or:
PYTHONPATH=. python src/modeling/deberta_baseline.py \
    --traces-glob "data/traces/*_traces.jsonl" \
    --output-oof  results/deberta_pooled_oof.npz
```

### 3b. Problem-conditioned DeBERTa (signal P_cond)

```bash
sbatch scripts/sbatch_problem_cond.sh
# or:
PYTHONPATH=. python src/modeling/deberta_conditioned.py \
    --traces-glob "data/traces/*_traces.jsonl" \
    --output-oof  results/deberta_conditioned_pooled_oof.npz
```

### 3c. Multi-layer hidden-state probe (signal P_probe)

First extract hidden states (one teacher-forcing pass per trace):
```bash
sbatch scripts/sbatch_extract_llama.sh        # Llama-8B side
sbatch scripts/sbatch_hidden_probe.sh         # Qwen-7B side
```

Then train the multi-layer probe:
```bash
sbatch scripts/sbatch_multi_layer.sh
# or:
PYTHONPATH=. python src/modeling/multi_layer_probe.py \
    --hidden-states-dir data/hidden_states/ \
    --layers 8,16,24,32 \
    --output-oof results/multi_layer_probe_oof.npz
```

### 3d. Step Transformer (additional signal used in the wider ablation)

```bash
sbatch scripts/sbatch_month2.sh
# or:
PYTHONPATH=. python src/modeling/step_transformer.py \
    --step-embeddings-dir data/step_embeddings/ \
    --output-oof results/step_transformer_pooled_oof.npz
```

(Step embeddings are themselves regenerable via
`scripts/build_step_embeddings.py`.)

---

## 4. Build the SuperHybrid OOF stack (CPU; ~1 min)

This is the core meta-learning step. With all base-model OOFs in place
(or using the ones we ship in `results/`), fit the LR / RF / XGBoost
meta-learners on a fresh outer 5-fold split:

```bash
PYTHONPATH=. python src/modeling/hybrid_route_ab.py \
    --variant ULTRA_TEXT_ONLY \
    --deberta-oof    results/month3/deberta_pooled_oof.npz \
    --deberta-cond-oof results/month3/deberta_conditioned_pooled_oof.npz \
    --roberta-oof    results/roberta_pooled_oof.npz \
    --step-oof       results/month3/step_transformer_pooled_oof.npz \
    --probe-oof      results/month3/hidden_probe_pooled_mlp_hidden_plus_genunc_oof.npz \
    --output         results/month3/superhybrid_pooled.json \
    --oof-out-dir    results/month3/
```

This produces:
- `results/month3/superhybrid_pooled.json` — pooled metrics for LR/RF/XGB
- `results/month3/<variant>__lr_oof.npz`, `*__rf_oof.npz`, `*__xgb_oof.npz`
  — per-meta-learner OOF probability arrays (used downstream for paired
  DeLong tests)

You should see pooled AUROC near **0.815 (LR)** / **0.807 (RF)** with
**ECE 0.042** for RF, matching Table 1 of the report.

---

## 5. Statistical inference: pooled CIs and paired DeLong tests (CPU; ~30 s)

Compute logit-space DeLong 95% CIs for every OOF, plus paired challenger-vs-
baseline tests:

```bash
# Pooled and per-group CIs for the 13 base models + stacker
PYTHONPATH=. python scripts/compute_per_group_ci_month3.py

# Paired DeLong: SuperHybrid (RF) vs Plain DeBERTa, fanned out by slice
PYTHONPATH=. python scripts/paired_delong_by_group.py \
    --a   results/month3/superhybrid__rf_oof.npz \
    --b   results/month3/deberta_pooled_oof.npz \
    --tag superhybrid_rf_vs_plain_deberta \
    --out-dir reports/month3/paired/
```

Outputs:
- `reports/month3/pooled_metrics_with_ci.csv`
- `reports/month3/per_group_metrics_with_ci.csv`
- `reports/month3/paired/*_by_group.{csv,json}`

---

## 6. Reproduce the report tables (CPU; ~10 s)

The summary CSVs feeding Tables 1-3 of the report are produced by:

```bash
PYTHONPATH=. python scripts/analyze_ultra_hybrid.py        # ULTRA-hybrid sweep summary
PYTHONPATH=. python scripts/summarize_phase2_paired.py     # paired-test matrix
PYTHONPATH=. python scripts/analyze_route_ab_oofs.py       # per-cell breakdown
```

Cross-domain transfer (Table 3 of the report):
```bash
PYTHONPATH=. python scripts/h3_transfer_with_ci.py
# Outputs: reports/route_ab/h3_transfer_auroc_ci.csv
#          reports/route_ab/h3_transfer_summary.md
```

The handcrafted-features-only ablation (text-saturation finding):
```bash
PYTHONPATH=. python scripts/roberta_only_ablation.py
# Outputs: reports/route_ab/roberta_only_ablation.{csv,json}
```

---

## Expected numbers (sanity check)

Pooled AUROC (n = 6,344-6,378 across the 8 dataset x model cells):

| Method                          | AUROC (expected)        |
|---------------------------------|-------------------------|
| Length-only LR                  | 0.595 +/- 0.003         |
| Plain DeBERTa-v3 (trace tail)   | 0.762 +/- 0.005         |
| Problem-conditioned DeBERTa     | 0.788 +/- 0.005         |
| Multi-layer hidden-state probe  | 0.776 +/- 0.005         |
| **SuperHybrid (LR)**            | **0.815 +/- 0.005**     |
| **SuperHybrid (RF)** (ECE 0.042) | **0.807 +/- 0.005**    |

Per-cell peak: **0.929** on MATH500-Qwen7B.

If the numbers you see deviate from these by more than 0.01 AUROC, the most
common cause is a stale OOF — re-run step 4 against the OOFs in
`results/month3/` rather than older OOFs in `results/`.

---

## Reproducing the optional XGBoost Optuna tune (~6 GPU-hours)

The XGBoost meta-learner ships with hyperparameters tuned via a 2,000-trial
Optuna study; the study log is committed at
`data/optuna_hybrid_v1_clean.db`. To re-run from scratch:

```bash
sbatch scripts/sbatch_tune_hybrid.sh
# or locally:
PYTHONPATH=. python scripts/tune_hybrid.py \
    --study-name hybrid_v2 \
    --n-trials 2000 \
    --hybrid-table data/hybrid_table.parquet
```

The final selected XGBoost configuration is documented in **Appendix A of the
final report**.

---

## Files NOT shipped (re-generated locally)

The following large binary artifacts are excluded from git via `.gitignore`
(each up to ~165 MB) and re-generated on demand:

| Artifact                          | Regeneration script                        |
|-----------------------------------|--------------------------------------------|
| `data/step_embeddings/`           | `scripts/build_step_embeddings.py`         |
| `data/step_embeddings_bge/`       | `scripts/build_step_embeddings.py --encoder bge` |
| `data/graphs/`                    | `scripts/build_trace_graphs.py`            |
| `data/hidden_atlas/`, `hidden_states/` | `scripts/sbatch_extract_*.sh`         |

All of these are intermediate artifacts; the **final OOF probability arrays
they produce are committed** under `results/` so the meta-learning step does
not require them.

---

## Troubleshooting

- **DeBERTa AUROC saturates at 0.5** — load the model in fp32 explicitly
  (`torch_dtype=torch.float32`); a bf16 fallback in some HuggingFace versions
  causes silent overflow on the classification head. The fix is in
  `src/modeling/deberta_baseline.py`.
- **`numpy ValueError: could not broadcast (2,N) into (2,)` when building
  trace DAGs** — `numpy >= 1.24` rejects ragged arrays even when the leading
  dim is constant. Use `np.empty(N, dtype=object)` and fill per slot
  (`scripts/build_trace_graphs.py` uses this pattern).
- **`REVISE` events look like `PAD` in step embeddings** — the legacy
  7-class taxonomy in `src/parsing/taxonomy.py` lacks a `REVISE` member. Make
  sure `scripts/build_step_embeddings.py` imports `BehaviorType` from
  `src/parsing/rule_based_parser.py` (the 6-class enum). Already patched in
  this branch.

For deeper context, see the developer notes:
- `RESEARCH_GUIDE.md` — high-level reproduction tips
- `HPC_WALKTHROUGH_ROUTE_AB.md` — HPC submission protocol
- `HPC_HYBRID_TUNING.md` — Optuna tuning protocol
- `PLAN_ROUTE_AB.md` — design of the Route A/B feature blocks
