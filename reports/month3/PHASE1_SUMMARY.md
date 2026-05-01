# Phase 1 — Month 3 Consolidation: Summary

**Branch:** `peng-update` &nbsp;&nbsp; **Date closed:** 2026-04-21 &nbsp;&nbsp; **Python env:** `miniconda3/envs/torch311`

Phase 1 of the approved 5-phase plan. Objective: consolidate the existing
Month 3 OOF artefacts (DeBERTa, hidden-state probes, SuperHybrid stacker,
per-dataset analysis) with rigorous inference — DeLong 95% CIs on every
pooled and per-group AUROC, and paired DeLong tests between head-to-head
competitor pairs — and to fix two dormant bugs that compromised the
structural signal.

---

## 1. Deliverables checklist

| # | Sub-task | Status | Key output |
|---|---|---|---|
| 1.1 | Port `delong_ci.py` from `youxuan-update` | ✅ | `src/analysis/delong_ci.py` — 7/7 self-tests pass |
| 1.2 | Compute pooled + per-group DeLong CIs for every Month 3 OOF | ✅ | `reports/month3/pooled_metrics_with_ci.csv` (58 models), `reports/month3/per_group_metrics_with_ci.csv` (≈500 rows) |
| 1.3 | Paired DeLong: SH_LR vs SH_RF (and two companion pairs) | ✅ | 3 × `paired_*_by_group.{csv,json}` — 114 paired tests |
| 1.4 | Re-run per-dataset analysis against v2 probe OOF | ✅ | `results/month3/per_dataset_analysis_v2.json` |
| 1.5 | Fix REVISE BehaviorType bug in `build_step_embeddings.py` | ✅ | Source fix committed; regeneration deferred to Phase 4 (needs GPU) |
| 1.6 | Fix empty `cells` dict in `layer_probe_sweep.py` | ✅ | Source fix + existing `layer_atlas.json` patched (64 cells) |
| 1.7 | Write this summary + flag missing HPC files | ✅ | this file |

**Scripts added under `scripts/`:**

- `compute_per_group_ci_month3.py` — ≈460 lines, Month-3 adapted version of the `youxuan-update` CI computer. Hand-curated registry of 27 core models + auto-discovery of 31 multi-layer probes = **58** total.
- `paired_delong_by_group.py` — generic per-slice paired test with `(dataset, model_family, overall)` fan-out. Uses composite `(group, item_id)` key to align OOFs.

**Source files modified:**

- `src/modeling/layer_probe_sweep.py` — populated `results["cells"]` by iterating over `pooled_cells` before `save_results`. Prior version wrote an empty dict, breaking downstream lookup.
- `scripts/build_step_embeddings.py` — changed `BehaviorType` import from the legacy 7-class `src/parsing/taxonomy.py` to the 6-class `src/parsing/rule_based_parser.py` (which is the parser actually producing the episodes). Before the fix, every `REVISE (X)` step silently mapped to the PAD ordinal (0) because the legacy enum has no `REVISE` member, so Month-2's Step Transformer never observed a revise event. Existing `data/step_embeddings/*.npz` files still reflect the buggy ordinalisation and need GPU regeneration (scheduled for Phase 4).

**Data patched without recomputation:**

- `results/month3/layer_atlas.json` — its pre-Phase-1 version had `cells: {}` despite a fully-populated `heatmap`. The two are derivable from each other, so I repopulated `cells` from `heatmap` and verified the best-cell pick (`dim3584/L20/answer_marker` AUROC=0.7886) matches the number previously cited in `CLAUDE.md`. Backup kept at `layer_atlas.json.bak_empty_cells`.

---

## 2. Headline numbers

All AUROCs and CIs below are pooled across the 8 dataset×model combos
(`n=6378`, `base_acc=0.614`). 95% CIs use logit-transformed DeLong; paired
tests report two-sided p-values.

### 2.1 Top 10 pooled models (AUROC ordering)

| model_key | family | AUROC | CI_low | CI_high |
|---|---|---:|---:|---:|
| `superhybrid_lr` (SH_LR) | stacker | **0.8051** | 0.7942 | 0.8156 |
| `superhybrid_rf` (SH_RF) | stacker | 0.8045 | 0.7935 | 0.8151 |
| `superhybrid_threeprobs` (ThreeProbs) | fusion | 0.8026 | 0.7916 | 0.8133 |
| `superhybrid_deberta_cond` | text | 0.7882 | 0.7766 | 0.7994 |
| `deberta_conditioned` | text | 0.7868 | 0.7752 | 0.7980 |
| `deberta_pooled` | text | 0.7626 | 0.7505 | 0.7743 |
| `superhybrid_deberta` | text | 0.7622 | 0.7501 | 0.7739 |
| `probe_mlp_hidden_plus_genunc_pooled` | hidden | 0.7557 | 0.7436 | 0.7674 |
| `probe_mlp_h_answer_pooled` | hidden | 0.7556 | 0.7435 | 0.7673 |
| `probe_mlp_concat_pooled` | hidden | 0.7533 | 0.7411 | 0.7652 |

(`step_transformer` pooled **0.6778**, still well below the text bar; the
revise-event bug documented above may have depressed it.)

### 2.2 Paired DeLong: the three critical head-to-heads

All tests below aligned on `(group, item_id)` composite key, n_common = 6378.

**a) SH_LR vs SH_RF — "Can we drop one of them?"**

| Slice | n | AUROC_a | AUROC_b | diff | p | sig |
|---|---:|---:|---:|---:|---:|---|
| OVERALL | 6378 | 0.8051 | 0.8045 | +0.0006 | 0.7431 | ns |
| gsm8k_llama8b | 1319 | 0.7770 | 0.7599 | +0.0171 | <0.0001 | *** |
| (all 7 other groups) | — | — | — | — | >0.08 | ns |

Pooled they are indistinguishable. Only `gsm8k_llama8b` shows a
significant LR>RF at group level. **Use either (the paper should cite SH_LR
because it carries the better ECE story across the top variants, 0.079
vs 0.042 — see §2.4 caveat).**

**b) SH_LR vs DeBERTa+Cond — "Does the stack still pay off over the best single text model?"**

| Slice | n | AUROC_SH | AUROC_DC | diff | p | sig |
|---|---:|---:|---:|---:|---:|---|
| OVERALL | 6378 | 0.8051 | 0.7868 | +0.0183 | <0.0001 | *** |
| gsm8k_llama8b | 1319 | 0.7770 | 0.7532 | +0.0238 | 0.0001 | *** |
| arc_challenge_qwen7b | 1172 | 0.7362 | 0.7181 | +0.0181 | 0.0277 | * |
| gsm8k_qwen7b | 1319 | 0.8494 | 0.8367 | +0.0127 | 0.0301 | * |
| math500_qwen7b | 500 | 0.9339 | 0.9203 | +0.0136 | 0.0531 | ns |
| gpqa_diamond_qwen7b | 198 | 0.7751 | 0.7954 | **−0.0202** | 0.198 | ns |
| gpqa_diamond_llama8b | 198 | 0.6734 | 0.6911 | **−0.0177** | 0.523 | ns |

Pooled win is real and highly significant. SH_LR is directionally better
in 6/8 groups, significantly better in 3/8. The two groups where SH_LR
*loses* directionally are both GPQA — unsurprising given GPQA has only
n=198 per group and is the hardest-signal dataset; the stacker may be
mildly over-fitting on the scarce GPQA-specific probe dimensions.

**c) SH_RF vs ThreeProbs (unweighted averaging) — "Does stacking buy anything over mean of 3 probs?"**

| Slice | n | AUROC_RF | AUROC_3P | diff | p | sig |
|---|---:|---:|---:|---:|---:|---|
| OVERALL | 6378 | 0.8045 | 0.8026 | +0.0018 | 0.4011 | ns |
| gsm8k_llama8b | 1319 | 0.7599 | 0.7771 | **−0.0172** | <0.0001 | *** |
| (all 7 other groups) | — | — | — | — | >0.10 | ns |

**Punchline for the paper:** pooled, the RF stacker is statistically
indistinguishable from simply averaging DeBERTa+Cond, Probe, and 3-probs'
output-head. On `gsm8k_llama8b` it is *significantly worse*. This is a
strong argument for framing the stacker as "a marginal refinement,
primarily for calibration (ECE)," not as "the key modelling innovation."

### 2.3 Per-dataset winners (v2 probe OOF)

| Group | Best method | AUROC |
|---|---|---:|
| arc_challenge_llama8b | Cond+Probe | 0.6652 |
| arc_challenge_qwen7b | ThreeProbs | 0.7432 |
| gpqa_diamond_llama8b | DeBERTa+Cond | 0.7049 |
| gpqa_diamond_qwen7b | Cond+Probe | 0.8171 |
| gsm8k_llama8b | ThreeProbs | 0.7788 |
| gsm8k_qwen7b | SuperHybrid_LR | 0.8485 |
| math500_llama8b | SuperHybrid_RF | 0.8665 |
| math500_qwen7b | SuperHybrid_RF | **0.9344** |

No single method sweeps. The deeper the reasoning task (MATH500 > GSM8K >
GPQA > ARC) the more the stacker pays off; on easier / shorter tasks,
simple averaging or even DeBERTa+Cond is competitive. This is the
empirical shape that Phase 2's ULTRA_HYBRID should be designed against.

### 2.4 Calibration (ECE @ 10 uniform bins, pooled)

| Method | AUROC | ECE |
|---|---:|---:|
| DeBERTa | 0.7622 | 0.0904 |
| DeBERTa+Cond | 0.7882 | 0.0857 |
| Cond+Probe | 0.8030 | 0.0835 |
| ThreeProbs | 0.8052 | 0.0820 |
| **SuperHybrid_LR** | 0.8074 | 0.0806 |
| **SuperHybrid_RF** | 0.8066 | **0.0421** |

RF stacking halves the ECE at essentially the same AUROC — this is the
one concrete advantage of the stacker over averaging, and where the
paper should lean on SH_RF rather than SH_LR. RF's well-known tree-mean
calibration is doing the work.

### 2.5 v1 vs v2 probe OOF deltas (pooled)

| Method | v1 AUROC | v2 AUROC | Δ |
|---|---:|---:|---:|
| DeBERTa | 0.7622 | 0.7622 | +0.0000 |
| DeBERTa+Cond | 0.7882 | 0.7882 | +0.0000 |
| Cond+Probe | 0.8005 | 0.8030 | **+0.0025** |
| ThreeProbs | 0.8026 | 0.8052 | **+0.0026** |
| SuperHybrid_LR | 0.8051 | 0.8074 | **+0.0023** |
| SuperHybrid_RF | 0.8045 | 0.8066 | **+0.0021** |

Exactly the right shape: methods that ingest the probe pick up a
uniform +0.002 to +0.003 AUROC; methods that do not are byte-identical.
The v2 probe (`hidden_probe_pooled_mlp_hidden_plus_genunc_oof.npz`,
AUROC 0.7557 alone) is consistently strictly better than v1 and should
be the default going forward.

---

## 3. Research-question status after Phase 1

- **H1 (AUROC ≥ 0.75 on MATH500):** crushed. Pooled-within-MATH500
  AUROC is **0.9003** (SH_LR) / 0.9061 (SH_RF); per-group
  `math500_qwen7b` hits **0.9344** with SH_RF and `math500_llama8b`
  hits 0.8665.
- **H3 (cross-domain transfer AUROC ≥ 0.65):** out-of-scope for this
  Phase, but the per-group CI table includes the per-dataset AUROCs a
  future transfer analysis will need.
- **Length-control falsifier:** preserved (handcrafted+recurrence still
  beats length-only in every quintile; Month-1 artefact, untouched).
- **Text-encoder bar:** SH_LR beats DeBERTa+Cond pooled by +0.0183
  AUROC, *p* < 0.001 DeLong paired — structural/probe layer adds
  content-free signal over the text encoder, which is the claim the
  project hangs on.

**Main revisions to the pre-committed story (must land in the paper):**

1. The stacker and the simple average of 3 probs are statistically
   indistinguishable pooled. Frame the stacker as a calibration tool,
   not a modelling advance.
2. The winning method is dataset-dependent; reporting a single
   "pooled" number masks that `Cond+Probe` beats the stacker on both
   ARC-Llama and GPQA-Qwen, and `DeBERTa+Cond` beats it on GPQA-Llama.
3. RF is the calibration winner (ECE 0.042); LR is the AUROC winner by
   a whisker (0.8051 vs 0.8045, ns). Paper should present both and
   document that they agree statistically.

---

## 4. Files rescued + remaining gaps (post user-sync)

The audit at Phase-1 start flagged several HPC-generated artefacts as
absent. The user has since rsynced some of them; the remainder have
been recovered from the `youxuan-update` branch in-tree. Final
accounting:

### 4.1 User-rsynced from HPC ✅

- `data/hybrid_table.parquet` (3.7 MB, 6344 × 366)
- `data/optuna_hybrid_v1_clean.db` (4.1 MB)
- `reports/route_ab/` (full dir, prior Route A/B reports + CIs)
- `data/graphs/` — per-dataset PyG trace-graph artifacts (9 .npz:
  8 combos + `math500_deepseek_r1`), plus a `hybrid` subdir. Route B1
  (trace GNN) input, confirms the GNN pipeline is runnable end-to-end.
- `results/roberta_pooled_oof.npz` (37 KB, 6378 samples, full OOF
  contract `{item_ids, groups, y_true, oof_prob, oof_fold, seed,
  n_splits}`) — the third text-encoder base. Landed at
  `results/` rather than `results/month2_v2/` as originally expected.

### 4.2 Restored in-tree from `youxuan-update` branch ✅

The user noted no literal `data/features_routeA*.csv` files exist —
they were never so named on HPC either. "Route A features" is an
umbrella for **five content-free feature families** that land as
separate CSV families. Restoring everything:

**Feature data (62 files):**

```
data/features/
  *_graph.csv              × 9 combos  (Route A3 — graph topology, 15 feats each)
  *_ngram.csv              × 9 combos  (Route A1 — behavior motifs, 231 feats each)
  *_structural_ph.csv      × 9 combos  (Route A4 — content-free PH, 36 feats each)
  *_timing.csv             × 9 combos  (Route A5 — inter-event timing, 46 feats each)
  *_shapelet_distmat.npz   × 9 combos  (Route A2 — shapelet distance matrices)
  ngram_vocab.json                     (reproducible trigram vocab manifest)
  v2/
    *_features_ph.csv      × 8 combos  (v1 PH from MiniLM point clouds, 7 feats)
    *_features_phimg.csv   × 8 combos  (v2 length-norm PH images, 36 feats)
```

**Feature extractor modules (5 + 2 deps):**

```
src/features/graph_features.py         (A3)
src/features/ngram_features.py         (A1)
src/features/timing_features.py        (A5)
src/features/structural_ph_features.py (A4, content-free PH)
src/features/shapelet_features.py      (A2)
src/features/topology_features.py      (v1 PH, dep of A4)
src/features/topology_features_v2.py   (v2 PH-image, dep of A4)
```

**Route AB modeling + scripts:**

```
src/modeling/hybrid_route_ab.py        (ULTRA_HYBRID stacker)
src/modeling/shapelet_eval.py          (Route A2 training loop)
src/modeling/trace_gnn.py              (Route B1 GNN)
scripts/build_route_a_features.py      (orchestrator for A1..A5)
scripts/build_trace_graphs.py          (PyG graph artifact dumper for B1)
scripts/analyze_route_ab_oofs.py       (OOF analysis for Route A/B)
scripts/sbatch_route_a.sh              (HPC batch for Route A)
scripts/sbatch_trace_gnn.sh            (HPC batch for trace GNN)
```

**Planning docs:**

```
PLAN_ROUTE_AB.md
HPC_WALKTHROUGH_ROUTE_AB.md
HPC_HYBRID_TUNING.md
```

**Verification (executed at Phase 1 close):**

- All 10 Route-A modules import cleanly under `torch311`
  (`graph_features`, `ngram_features`, `timing_features`,
  `topology_features`, `topology_features_v2`, `structural_ph_features`,
  `shapelet_features`, `hybrid_route_ab`, `shapelet_eval`, `trace_gnn`).
- All 3 orchestrator scripts compile without syntax errors.
- Route A CSVs cover all 8 pooled_8 combos at the same row counts as
  `hybrid_table.parquet` — item-ID intersection on `math500_qwen7b` is
  499 / 499 perfect, zero asymmetric diff. Column prefixes from CSV
  (`g_`, `ng2_`/`ng3_`/`ng_`, `h0_`/`h1_`, `t_`) map 1:1 to the
  hybrid table's `graph_g_`, `ng_`, `ph_`, `timing_t_` families
  (15+231+36+46 = 328 feature columns restored).
- Feature extractors are consistent: `graph` and `ngram` CSVs drop
  traces with <2 episodes (e.g. math500_qwen7b has 499 vs
  structural_ph's 500), matching the hybrid_table's intersected 499.

### 4.3 Remaining gaps (user confirmed non-recoverable)

- `data/hidden_atlas/` — raw `(N, L, P, H)` hidden tensors across
  layers/positions. User confirmed: does not exist on HPC or locally.
  **Phase 4 can still proceed** using the already-patched
  `results/month3/layer_atlas.json` (heatmap + 64 cells). New probe
  architectures would need fresh extraction (requires GPU + model
  cache; schedule in Phase 4).
- `data/hidden_states/` — per-layer extraction cache. Same status as
  above; not blocking Phase 4.
- `data/step_embeddings/*.npz` — still stale (need REVISE-fix
  regeneration). Local `data/step_embeddings_bge/` exists as a
  zipped/unzipped BGE-based variant; may or may not be affected by
  the REVISE bug depending on which parser the generator used.
  Regeneration scheduled for Phase 4 (≈15 min GPU).
(All previously-flagged gaps that were blocking Phase 2 have been
closed.)

**Phase 2 readiness:** `xgboost 3.2.0` is installed in `torch311` ✅.
Route A inputs in place ✅. `hybrid_table.parquet` already contains
the joined feature matrix so Phase 2 can start against the parquet
directly without re-running feature extraction.

---

## 5. Inventory of Phase 1 outputs

```
src/analysis/delong_ci.py                                           (23 KB, 645 lines, ported)
src/modeling/layer_probe_sweep.py                                   (source fix for empty cells)
scripts/build_step_embeddings.py                                    (source fix for REVISE)
scripts/compute_per_group_ci_month3.py                              (~460 lines, new)
scripts/paired_delong_by_group.py                                   (~270 lines, new)

reports/month3/pooled_metrics_with_ci.csv                           (14 KB, 58 models)
reports/month3/per_group_metrics_with_ci.csv                        (124 KB, ≈500 rows)
reports/month3/paired_delong_superhybrid_vs_base.json               (82 KB, 114 pairs)
reports/month3/paired_sh_lr_vs_sh_rf_by_group.{csv,json}
reports/month3/paired_sh_lr_vs_deberta_cond_by_group.{csv,json}
reports/month3/paired_sh_rf_vs_threeprobs_by_group.{csv,json}

results/month3/per_dataset_analysis_v2.json                         (v2 probe re-run)
results/month3/layer_atlas.json                                     (patched, 64 cells)
results/month3/layer_atlas.json.bak_empty_cells                     (backup of pre-patch)
```

---

## 6. Ready for Phase 2

Phase 1 closes cleanly. All prerequisites for Phase 2 are in place.
The ULTRA_HYBRID build will:

1. Train LR + RF + XGB meta-stackers directly against
   `data/hybrid_table.parquet` (6344 rows × 366 cols) — already
   joined, no feature-extraction step needed at the Phase 2 boundary.
2. Re-use the Optuna study in `data/optuna_hybrid_v1_clean.db` as the
   seed search space for the XGB hyperparameter tuning, rather than
   re-running the whole Optuna loop.
3. Compare each ULTRA_HYBRID variant against SH_LR/SH_RF baselines
   via `scripts/paired_delong_by_group.py` on the newly-produced OOF
   `.npz` outputs. The paired-test machinery from Phase 1.3 is the
   exact tool the H1/H3 evaluation will use.
4. Maintain leakage-safety: every base-model prob feeding the stacker
   is already an OOF prob from its own 5-fold CV; the meta-learner
   gets a fresh outer 5-fold.

No outstanding blockers. Phase 2 can start immediately.
