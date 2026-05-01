# Phase 3 Plan — Novel Signals, Better Encoders, HPC Scale

**Branch:** `peng-update` &nbsp;&nbsp; **Date:** 2026-04-21 &nbsp;&nbsp; **Owner:** Phase 3 (post-ULTRA_HYBRID)

---

## 0. Why this plan exists

Phases 1-2 produced a +0.008 AUROC lift over the pre-existing `peng-update`
Phase-1 numbers (0.8051 → 0.8128). That's statistically real (p = 5e-04)
but too small to carry a paper. The Phase-2 ablation identified WHY:

1. **Text encoders + hidden-state probe saturate the stack.** Adding 328
   handcrafted structural features on top contributes zero or negative
   signal.
2. **Every OOF in the stack shares the same underlying signal source:
   surface text and one probe layer.** We have no *orthogonal* signal.
3. **The weakest links in the stack (Step-TF at 0.697, GIN-GNN at 0.71,
   shapelet at ~0.65) are artificially held back by small encoders
   (MiniLM-L6-v2 at 384d), single-layer probes, and limited training.**

Conclusion: we need **new kinds of signal**, **bigger encoders trained
on HPC**, and **more experimentation throughput** than we've been doing.

---

## 1. Design principles

1. **Orthogonality first**: every new OOF must plausibly capture a signal
   the existing stack does not. Merely retraining a DeBERTa variant won't
   move the stacker.
2. **Scale the weakest encoders**: MiniLM→MPNet/E5, GIN→Graphormer, single
   probe→multi-layer probe bank.
3. **Grow the corpus**: 6k traces is small. Increase coverage via HPC
   generation at higher sampling diversity and (optionally) larger
   datasets.
4. **Commit to falsifiable criteria** for each experiment upfront (see §6).
5. **HPC is the default compute substrate** for all model training and
   hidden-state / logprob extraction. Local machine is only for analysis,
   stacker fitting, and API-light tasks.

---

## 2. The seven Phase-3 experiments (prioritized by EV)

### T1 — LLM-as-Judge OOF (🎯 highest-EV, new orthogonal signal)

**Hypothesis**: An independently trained frontier LLM, given the problem
and the trace, predicts correctness at AUROC ≥ 0.80 on its own and
adds ≥ +0.03 pooled AUROC to the best Phase-2 stacker.

**Why this is likely strong**:
- It's the most content-aware signal possible; it reads the whole trace.
- It's orthogonal to our DeBERTa/RoBERTa/Step OOFs because it's a different
  architecture with different training data, accessed via API.
- Literature (LLM-as-judge, self-verification) shows frontier models score
  well above 0.80 AUROC on math-verification tasks.

**What's new**: We've never done this. DeepSeek API is wired only for
generation.

**Implementation**: `scripts/phase3/build_llm_judge_oof.py`
- For each of 6378 traces, send `(problem, reasoning_trace, answer)` to a
  judge model.
- Ask for a calibrated correctness probability plus a categorical
  judgment (CORRECT / INCORRECT / UNCERTAIN).
- Parse and save as an OOF-compatible `.npz`.
- Use `ThreadPoolExecutor` with 8-16 workers and exponential-backoff retry.
- Judge models, in order of preference:
  1. DeepSeek-V3 chat (fastest, cheapest, already keyed)
  2. Claude-Sonnet-4.5 (if ANTHROPIC_API_KEY set)
  3. GPT-4o-mini (if OPENAI_API_KEY set)
  4. Local Llama-3.1-70B via vLLM (HPC fallback if APIs unavailable)

**Compute**: light CPU + API I/O. **Runnable locally** with parallel
workers (~30 min for all 6378 traces @ 16 workers). HPC option included.

**Falsifier**: If judge standalone AUROC < 0.72 on MATH500-qwen7b (the
strongest group), the approach is defeated. Stop and log as null.

---

### T2 — Token-logprob features via HPC re-scoring

**Hypothesis**: Per-token logprob profiles (perplexity spikes, flat
regions, end-of-trace collapse) carry uncertainty signal that's not
captured by the aggregate `mean_log_prob` we currently save, and
standalone AUROC ≥ 0.70.

**Why this is likely strong**:
- Raw per-token logprobs are the generator's own confidence timecourse.
- Spike count / variance / final-token logprob are strong OOD signals in
  the uncertainty literature.
- Orthogonal to DeBERTa (which only sees the text, not the logprob
  stream).

**What's new**: No per-token logprobs are saved in our JSONLs.
Re-extraction means running the existing traces back through the
generator with the model in teacher-forcing mode.

**Implementation**:
- Patch `src/generation/generate_traces.py` to save per-token logprobs
  under `token_logprobs` on new runs (future-proofing).
- New script `scripts/phase3/extract_token_logprobs.py`: loads each
  HF generator (R1-Distill-Qwen-7B, Llama-3.1-8B), teacher-forces the
  saved `full_response`, extracts logprobs per token.
- `scripts/phase3/build_logprob_features.py`: collapses the per-token
  vector into 40-ish scalar features (mean, std, quantiles, spike
  count at k thresholds, trailing-k statistics, entropy).
- `scripts/phase3/fit_logprob_oof.py`: 5-fold stratified CV, saves an OOF.

**Compute**: **HPC REQUIRED**. ~6378 traces × ~2000 tokens × 2 models.
Teacher-forcing is O(1) forward pass per trace on GPU. Estimated
**2-3 GPU-hours on an H100, 6-10 on A100**.

**Falsifier**: If logprob-derived features alone underperform `mean_log_prob`
(which is already in Baseline A with 0.60 AUROC), discard.

---

### T3 — Multi-layer hidden-state probe bank (HPC re-extraction)

**Hypothesis**: Probes at different transformer layers capture
different uncertainty signals; stacking probes across {8, 16, 24, 32}
layers adds ≥ +0.02 over the single-layer v2 probe.

**Why this is likely strong**:
- `multi_layer_probe.py` + `layer_atlas.py` already exist. Infrastructure
  is half-done.
- Literature on probing shows that mid-layers (16-24) often hold the
  best generalization-relevant features; last-layer probes emphasize
  output-specific signal.
- Cheap relative to T2: same hidden-state extraction pass yields all
  layers; only storage grows.

**Implementation**:
- Modify `scripts/extract_hidden_states.py` to optionally save
  `{h_last_l8, h_last_l16, h_last_l24, h_last_l32}` in addition to the
  existing three keyframes.
- `scripts/phase3/build_layered_probes.py`: runs `multi_layer_probe.py`
  per layer, aggregates OOFs.
- New OOF family: `hidden_probe_l{8,16,24,32}__{linear,mlp}_oof.npz`
  (8 new OOFs).

**Compute**: **HPC REQUIRED**. Hidden-state extraction with the full
layer stack is ~1.5× the current extraction cost. Estimated
**3 GPU-hours total**.

**Falsifier**: If the best multi-layer probe fails to beat the current
v2 probe (0.676 AUROC pooled) by ≥ +0.01, multi-layer buys nothing.

---

### T4 — Step-Transformer v2 with MPNet / E5 encoder

**Hypothesis**: Replacing MiniLM-L6-v2 (384d) with MPNet-base-v2 (768d)
or E5-large-v2 (1024d) as the frozen per-step embedder lifts pooled
Step-TF AUROC from 0.697 → ≥ 0.75, enough to matter in the ULTRA stack.

**Why**:
- MiniLM is known to be a weak sentence embedder; MPNet-v2 beats it by
  ~3-5 points on STS benchmarks, and E5-large by ~6-8 points.
- Step-Transformer's upper layers are already a trained Transformer; the
  ceiling is set by the frozen embeddings.
- Adding attention pooling + mean pooling + CLS concat has ~0.01 lift
  in similar tasks.

**Implementation**:
- `scripts/phase3/build_step_embeddings_v2.py`: re-runs step embedding
  with MPNet-base-v2 or E5-large-v2 via `sentence-transformers`.
- `src/modeling/step_transformer_v2.py`: same Transformer stack as v1
  but with:
  - Variable embedding dim (384/768/1024)
  - Multi-pool concat: `[CLS; mean; max; attn]`
  - LayerNorm + Dropout + 2-layer MLP head
  - AdamW + cosine schedule, 30 epochs instead of 15
- New HPC sbatch: `scripts/sbatch_phase3_step_tf_v2.sh`.

**Compute**: **HPC REQUIRED**. Embedding extraction ≈ 1 GPU-hour;
Step-TF training ≈ 2 GPU-hours per (8-dataset CV run).

**Falsifier**: If Step-TF v2 ≤ 0.72 pooled AUROC, abort. If ≥ 0.75,
replace v1 in every downstream stack.

---

### T5 — Graphormer over trace DAGs

**Hypothesis**: A Graphormer with learned positional encodings over
proper trace DAGs (with revision-reference edges, not just sequential
adjacency) captures structural signal the GIN misses, lifting GNN
standalone from 0.71 → ≥ 0.75.

**Why**:
- Current GNN is a 3-layer GIN over sequential-adjacency graphs. It
  cannot see long-range back-references.
- Revision-reference edges (step j references step i with i < j-1) are
  a first-class notion in the taxonomy.
- Graphormer's attention over graph tokens + learned spatial PE captures
  long-range interactions.

**Implementation**:
- `src/parsing/trace_dag_builder.py`: extend existing graph construction
  to add revision-reference edges. Matching is simple: at each REVISE
  step, find the most-recent FORWARD step with max token overlap.
- `src/modeling/graph_transformer.py`: PyG `TransformerConv` stack +
  learned node feats + Laplacian PE.
- `scripts/phase3/train_trace_graphormer.py`: 5-fold CV, OOF out.

**Compute**: **HPC REQUIRED** (PyG + GPU for throughput). Estimated
**2 GPU-hours**.

**Falsifier**: Must beat GIN standalone (0.71) by ≥ +0.03.

---

### T6 — Answer-trace semantic-consistency features (local, cheap)

**Hypothesis**: A rich set of features derived from comparing the
*final answer embedding* to *each step embedding* captures
"convergence" dynamics that handcrafted structural features do not,
adding standalone AUROC ~0.68 and ~+0.003 to the stacker.

**Why**:
- Cheap to compute from existing MiniLM step embeddings.
- Novel feature family: nothing in Phase-1/2 uses answer-anchored
  similarity structure.
- Fails gracefully: if it adds nothing to the stacker, no compute wasted.

**Implementation**: `scripts/phase3/build_answer_trace_features.py`.
Features per trace:
- `ans_step_max_cos` — highest cosine similarity between answer and any
  step (high = trajectory found the right region)
- `ans_step_final_cos` — cosine similarity between answer and the LAST
  step (high = convergent)
- `ans_step_argmax_idx` — normalized index of the best-matching step
  (late = convergent, early = model locked in early and drifted)
- `ans_step_slope` — regression slope of similarity vs step index
  (positive slope = increasingly on-topic)
- `ans_step_variance` — std of answer-step similarity (low = focused)

**Compute**: negligible, **runs locally**.

**Falsifier**: Must add ≥ +0.01 AUROC as a feature family to the
handcrafted-25 baseline; if not, discard.

---

### T7 — Selective-prediction reframing + per-dataset stackers

**Hypothesis**: At fixed coverage (e.g., 70%), ULTRA_TEXT_ONLY achieves
accuracy ≥ 95% on MATH500 (vs self-consistency N=8 at ~92%), turning
the paper's pitch from "AUROC competitive" to "deployed-at-threshold
compute-efficient."

**Why**:
- Even a small AUROC bump translates to meaningfully better selective-
  prediction curves at high-trust thresholds.
- Per-dataset stackers can squeeze +0.02-0.03 on weak groups (e.g.,
  gpqa_diamond_qwen7b) where Phase-1 Cond+Probe still beats us pooled.

**Implementation**:
- `scripts/phase3/selective_prediction_curves.py`: for each top-3 ULTRA
  variant, compute accuracy-at-coverage at {50%, 70%, 85%, 90%, 95%}
  cutoffs. Compare to self-consistency N=8 baseline numbers.
- `scripts/phase3/per_dataset_stacker.py`: fit 8 independent stackers
  (one per dataset-model combo), evaluate per-group.
- Weighted hybrid: `w * pooled_pred + (1-w) * per_dataset_pred`, tune w
  per-group on validation.

**Compute**: negligible, **runs locally**.

**Falsifier**: If selective-prediction at 70% coverage does not exceed
self-consistency N=8 accuracy on MATH500, the efficiency story is dead.

---

## 3. Cross-cutting: bigger corpus (HPC re-generation)

**T8 (optional, HPC-only)**: Generate 2-3× more traces per dataset on
HPC to stabilize every downstream model.

- Current: 500 MATH500 / 1319 GSM8K / 198 GPQA / 1172 ARC per model = 6378
- Target: 1500 / 2500 / 500 / 2000 = ~13k traces
- **HPC needed**: GPU + 4-6 hours per model per dataset.
- **Only triggered if T1-T4 collectively plateau below target AUROC.**

---

## 4. HPC workflow

All HPC jobs will land in `scripts/sbatch_phase3_*.sh` and be invokable
from a single runbook: `scripts/PHASE3_HPC_RUNBOOK.md`.

### Submission order (dependencies noted)

```
Step 1 (independent, in parallel):
  sbatch scripts/sbatch_phase3_extract_token_logprobs.sh
       # → data/token_logprobs/{dataset}_{model}.npz
  sbatch scripts/sbatch_phase3_extract_multilayer_hs.sh
       # → data/hidden_states/layered/{dataset}_{model}.npz
  sbatch scripts/sbatch_phase3_step_emb_mpnet.sh
       # → data/step_embeddings_mpnet/{dataset}_{model}.npz

Step 2 (depends on Step 1):
  sbatch --dependency=afterok:$JOB1 scripts/sbatch_phase3_logprob_oof.sh
  sbatch --dependency=afterok:$JOB2 scripts/sbatch_phase3_multilayer_probes.sh
  sbatch --dependency=afterok:$JOB3 scripts/sbatch_phase3_step_tf_v2.sh
  sbatch scripts/sbatch_phase3_trace_graphormer.sh   # independent

Step 3 (depends on Step 2):
  sbatch --dependency=afterok:... scripts/sbatch_phase3_mega_stacker.sh
```

### Environment requirements

- `PyTorch/2.2.0` or equivalent with CUDA
- `sentence-transformers`, `transformers>=4.40`, `torch_geometric`
- For LLM judge via local vLLM: `vllm>=0.5`

### Storage

- Token logprobs: ~200MB/dataset/model = ~3.2 GB total
- Multi-layer hidden states: ~1GB/dataset/model = ~16GB total
- MPNet step embeddings: ~1GB/dataset/model = ~16GB total
- Fits under the 500GB $WORK allocation easily.

---

## 5. Local work (does not need HPC)

1. **T1 LLM-as-judge**: implementation + API calls (I/O-bound, runs on
   laptop with 16 workers).
2. **T6 answer-trace features**: runs on existing step embeddings.
3. **T7 selective-prediction reframing + per-dataset stackers**: uses
   existing OOFs.
4. **Analysis, plots, paired DeLong, meta-stackers**: all local.

---

## 6. Falsification matrix

| Experiment | Success threshold | Failure = |
|---|---|---|
| T1 LLM-Judge | Judge AUROC ≥ 0.80 standalone; +0.03 over Phase-2 in stack | Drop from stack; record as null. |
| T2 Token logprobs | Logprob-family AUROC ≥ 0.70; +0.01 in stack | Keep the `mean_log_prob` baseline, drop new features. |
| T3 Multi-layer probes | Best layer probe ≥ v2 probe + 0.01 | Single-layer v2 stays; no change. |
| T4 Step-TF v2 | Pooled AUROC ≥ 0.75 (vs 0.697) | Keep v1; report null. |
| T5 Graphormer | Standalone AUROC ≥ 0.74 (vs GIN 0.71) | Keep GIN; GNN contribution stays neutral. |
| T6 Answer-trace | +0.01 to handcrafted-25 baseline | Drop feature family. |
| T7 Selective pred. | At 70% coverage, accuracy > self-consistency N=8 | Reframe paper without efficiency claim. |

**Overall Phase-3 target**: pooled AUROC ≥ **0.84** with a MEGA_HYBRID
stacker that includes LLM-judge + token-logprob + multi-layer-probe +
Step-TF v2 + Graphormer OOFs on top of Phase-2's ULTRA_TEXT_ONLY base.

Minimum acceptable: ≥ **0.82** pooled with ≥ 2 new families contributing
significant positive deltas (p<0.05 paired DeLong).

---

## 7. What I'll do without waiting

I will now:

1. Write `scripts/phase3/build_llm_judge_oof.py` + small-prompt
   template (T1). Ready to run locally against DeepSeek-V3.
2. Write `scripts/phase3/build_answer_trace_features.py` (T6). Ready
   to run locally on existing step embeddings.
3. Write `scripts/phase3/selective_prediction_curves.py` (T7). Ready.
4. Draft HPC sbatch scripts for T2, T3, T4, T5 with all env + path
   assumptions documented. Not submitted.
5. Write `scripts/PHASE3_HPC_RUNBOOK.md` with line-by-line submit
   instructions.
6. Create `reports/month3/PHASE3_SUMMARY.md` scaffold.

See `reports/month3/PHASE3_HPC_CHECKLIST.md` once HPC drafting
completes.
