# Phase 3 Summary

> **Status:** SCAFFOLD — fill in as each HPC job lands. This file is the
> single source of truth for "did Phase 3 do anything on top of the
> ULTRA_TEXT_ONLY 0.8128 baseline?"

Baselines to beat (pooled 8 dataset×model, from Phase 2):

| Variant | AUROC | ECE | Notes |
|---|---|---|---|
| ULTRA_TEXT_ONLY-LR (Phase 2 winner) | 0.8128 | 0.042 | 5-OOF text stack |
| SuperHybrid_LR (Phase 1) | 0.8051 | 0.042 | baseline ensemble |
| SuperHybrid_RF (Phase 1) | 0.8045 | 0.044 | baseline ensemble |

**Pre-committed success bar (PHASE3_PLAN.md):**
- Target: pooled AUROC ≥ **0.84**.
- Minimum acceptable: ≥ **0.82** with ≥ 2 new signal families contributing
  (positive paired-DeLong vs. ULTRA_TEXT_ONLY at p < 0.05).

---

## 1. Headline result

| Variant | AUROC | ΔAUROC vs ULTRA_TEXT_ONLY | p (paired DeLong) | ECE |
|---|---|---|---|---|
| P3 stack (all T1–T5) | _pending_ | _pending_ | _pending_ | _pending_ |
| P3 stack (text only: +T1 +T2) | _pending_ | _pending_ | _pending_ | _pending_ |
| P3 stack (structural only: +T4 +T5) | _pending_ | _pending_ | _pending_ | _pending_ |

---

## 2. Per-experiment standalone AUROCs

Each row is a **single** OOF compared to the pre-existing baselines.
"Standalone" = the model's OOF alone, no stacking.

| Task | Family | Expected AUROC (literature) | Observed | p vs ULTRA_TEXT_ONLY | Orthogonal to which? |
|---|---|---|---|---|---|
| T1 LLM-as-judge (DeepSeek-V3) | External LLM | 0.78–0.88 | _pending_ | _pending_ | DeBERTa, step-TF, probe |
| T2 Token-logprob (step-LP features) | Token-level | 0.65–0.72 | _pending_ | _pending_ | text encoders |
| T3 Multi-layer probe (best variant) | Hidden-state | 0.72–0.80 | _pending_ | _pending_ | surface text, recurrence |
| T4 Step-TF v2 (MPNet-768) | Step-text | 0.70–0.76 | _pending_ | _pending_ | DeBERTa |
| T4 Step-TF v2 (E5-large-1024) | Step-text | 0.70–0.78 | _pending_ | _pending_ | DeBERTa |
| T5 Graphormer over DAGs | Graph | 0.68–0.75 | _pending_ | _pending_ | handcrafted recurrence |
| T6 Answer-trace features (local) | Semantic | 0.60–0.68 | _pending_ | _pending_ | length, surface |

Fill in when OOFs are in (semantics: `--a` = challenger, `--b` = baseline):
```
PYTHONPATH=. python scripts/paired_delong_by_group.py \
    --a   results/month3/multi_layer_probe_<variant>_oof.npz \
    --b   results/month3/ultra_hybrid/ultrahybrid_ULTRA_TEXT_ONLY__lr_oof.npz \
    --tag t3_probe_vs_ultra_text_only \
    --out-dir reports/month3
```

---

## 3. Selective-prediction curves

Output: `reports/month3/phase3_selective.csv` and `phase3_selective_summary.csv`.

Accuracy at coverage (pooled):

| Variant | 100% | 95% | 90% | 85% | 70% | 50% | 30% |
|---|---|---|---|---|---|---|---|
| ULTRA_TEXT_ONLY_LR | _pending_ | | | | | | |
| P3_STACK_LR | _pending_ | | | | | | |
| LLM_JUDGE | _pending_ | | | | | | |

**Reference:** self-consistency N=8 on MATH500 is reported as ~82% accuracy
at coverage=100% (2025 literature). If our stack hits ≥ 82% at 100% or ≥ 85%
at 90%, it is competitive at 1/8 the compute.

---

## 4. What changed vs Phase 2

- New OOFs on disk: list paths here once they land.
- New features on disk: `data/features/*_features_steplp.csv` (T2),
  `data/features/*_ans_trace.csv` (T6).
- Super-hybrid stack refreshed: `results/month3/phase3_stack.json`.

---

## 5. Falsifiers

Pre-committed in PHASE3_PLAN.md:

- [ ] **LLM-judge must beat ULTRA_TEXT_ONLY as a STANDALONE predictor**
      (not just in the stack) — otherwise the "orthogonal frontier signal"
      framing is false.
- [ ] **Multi-layer probe and step-TF v2 must show positive paired-DeLong
      p vs their Phase-2 counterparts** (single-layer probe, MiniLM step-TF).
- [ ] **Graphormer must beat GIN (`trace_gnn_oof.npz`) by ≥ 0.01 AUROC.**
      If it doesn't, the revision-reference edges and centrality encoding
      add nothing and T5 is dropped from the stack.
- [ ] **The final stack's gain over ULTRA_TEXT_ONLY must survive ablation:**
      removing any one new OOF should drop the stack by ≤ 0.01 AUROC. If a
      single OOF carries all the gain, it IS the new baseline (so the
      framing collapses to "LLM-judge is the winner", etc.).

---

## 6. Compute budget

| Task | GPU-h (actual) | GPU-h (planned) |
|---|---|---|
| T1 | _pending_ | 1–2 |
| T2 | _pending_ | 3–5 |
| T3 | _pending_ | 4–6 |
| T4 ×1 (mpnet) | _pending_ | 2–4 |
| T4 ×1 (e5-large) | _pending_ | 2–4 |
| T5 | _pending_ | 3–5 |
| **Total** | _pending_ | **15–26** |

---

## 7. Decisions & next-steps log

- [date, TBD] Phase 3 jobs submitted to HPC (T1, T2, T3 tier 1).
- [date, TBD] T4 submitted after Tier 1 results.
- [date, TBD] T5 submitted last.
- [date, TBD] First stack refresh.
- [date, TBD] Selective-prediction table updated.
- [date, TBD] Decision: does Phase 3 warrant an incremental paper update, a
  new subsection, or a retraction of the Phase 2 framing?

---

## Appendix — files produced by Phase 3

### Scripts
- `scripts/phase3/build_llm_judge_oof.py`
- `scripts/phase3/build_answer_trace_features.py`
- `scripts/phase3/selective_prediction_curves.py`
- `scripts/phase3/extract_token_logprob_features.py`
- `scripts/phase3/build_trace_dags.py`
- `scripts/phase3/train_graphormer.py`

### HPC sbatch wrappers
- `scripts/sbatch_phase3_llm_judge.sh`        (T1)
- `scripts/sbatch_phase3_token_logprobs.sh`   (T2)
- `scripts/sbatch_phase3_multilayer_probes.sh`(T3)
- `scripts/sbatch_phase3_step_tf_v2.sh`       (T4)
- `scripts/sbatch_phase3_graphormer.sh`       (T5)

### Runbook
- `reports/month3/PHASE3_HPC_RUNBOOK.md`

### Plan
- `reports/month3/PHASE3_PLAN.md`
