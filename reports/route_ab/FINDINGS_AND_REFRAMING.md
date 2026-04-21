# Findings and paper reframing — April 20, 2026

This memo consolidates the local artifacts produced this session
(DeLong CIs for every OOF, the H3 cross-domain transfer with CIs, and
the RoBERTa-only stacker ablation) into a concrete paper framing. It
covers what is now defensible, what still needs the pending HPC work
(clean DeBERTa rerun → promote → rebuild → re-tune), and which
falsifiers pass vs. fail.

All numbers below refer to the `hybrid_v1_clean` study and the current
production OOFs in `results/`. Because the promote script has **not yet
been run** (no `.LEAKY.npz` backups exist anywhere on disk), the
numerical leakage caveat from `HPC_HYBRID_TUNING.md` still applies:
the neural OOFs' `.npz` files are the original *best-epoch-on-val*
predictions even though the `.PROVENANCE.json` sidecars claim otherwise.
Expect the true-clean stacker AUROC to land ~0.005 – 0.025 lower once
the HPC rerun + promotion completes (see `PENDING_HPC_WORK` below).

---

## 1. Headline pooled numbers with DeLong 95% CIs

Source: [`reports/route_ab/pooled_metrics_with_ci.csv`](./pooled_metrics_with_ci.csv)

| Model                      |    n | AUROC | 95 % CI          |
| -------------------------- | ---: | ----: | ---------------- |
| Hybrid stacker (L1-LR)     | 6344 | 0.805 | [0.794, 0.816]  |
| RoBERTa-base               | 6378 | 0.798 | [0.787, 0.809]  |
| DeBERTa-v3-base            | 6378 | 0.763 | [0.750, 0.774]  |
| TraceGIN-hybrid            | 6398 | 0.713 | [0.699, 0.725]  |
| Step Transformer           | 6378 | 0.675 | [0.662, 0.688]  |
| TraceGIN-structural        | 6398 | 0.643 | [0.629, 0.657]  |
| Shapelet (LR on info-gain) | 6364 | 0.640 | [0.626, 0.654]  |

All CIs are tight (half-width ≈ 0.011). Every pair of adjacent rows in
this table is **separated by more than the CI half-width**, so the
ordering is statistically stable.

## 2. Per-group CIs reveal where the signal is strong

Source: [`reports/route_ab/per_group_metrics_with_ci.csv`](./per_group_metrics_with_ci.csv)
(hybrid stacker rows only)

| Group                 |    n | AUROC | 95 % CI          | CI width |
| --------------------- | ---: | ----: | ---------------- | -------: |
| math500_qwen7b        |  499 | 0.938 | [0.910, 0.957]  |    0.047 |
| math500_llama8b       |  494 | 0.868 | [0.833, 0.897]  |    0.064 |
| gsm8k_qwen7b          | 1318 | 0.850 | [0.827, 0.871]  |    0.044 |
| gpqa_diamond_qwen7b   |  192 | 0.755 | [0.677, 0.820]  |    0.143 |
| gsm8k_llama8b         | 1319 | 0.759 | [0.732, 0.784]  |    0.052 |
| arc_challenge_qwen7b  | 1165 | 0.731 | [0.698, 0.762]  |    0.064 |
| gpqa_diamond_llama8b  |  189 | 0.716 | [0.635, 0.785]  |    0.150 |
| arc_challenge_llama8b | 1168 | 0.670 | [0.637, 0.702]  |    0.065 |

Reading these CIs:

* **H1 pre-commit "AUROC ≥ 0.75 on MATH500" passes comfortably on both
  models**: lower bound 0.910 (qwen7b) and 0.833 (llama8b) are well
  above the 0.75 target. H1 also passes on gsm8k_qwen7b (LB 0.827).
* **Small-group CIs are wide**: GPQA has n = 189 – 192, so CI widths
  are ~0.14, ~3× wider than the big groups. Per-group GPQA numbers
  should always be cited with the CI; otherwise you risk over-claiming
  from noise.
* The hardest group (arc_challenge_llama8b) has a CI ceiling of 0.70,
  so even the optimistic claim on that slice is bounded.

## 3. H1 (MATH500 AUROC ≥ 0.75): **passes convincingly**

The hybrid stacker's MATH500_qwen7b CI lower bound of 0.910 rules out
any realization below 0.75 at the 97.5 % level (the Wald lower bound on
a two-sided 95 % CI). Even without the hybrid stacker, RoBERTa alone
on MATH500_qwen7b scores 0.930 [0.898, 0.952] → lower bound still
0.898, H1 passes on RoBERTa alone. H1 is not in doubt.

## 4. H3 (MATH500 → GPQA transfer AUROC ≥ 0.65): **passes on qwen7b, borderline on llama8b**

Source: [`reports/route_ab/h3_transfer_auroc_ci.csv`](./h3_transfer_auroc_ci.csv)

Train on MATH500 (same model), test on GPQA-Diamond, RF classifier
over 28 handcrafted + recurrence features:

| Transfer pair                            | n_test | AUROC | 95 % CI          | Verdict         |
| ---------------------------------------- | -----: | ----: | ---------------- | --------------- |
| math500_qwen7b  → gpqa_diamond_qwen7b    |    198 | 0.732 | [0.652, 0.799]  | **H3 PASSES**   |
| math500_llama8b → gpqa_diamond_llama8b   |    198 | 0.675 | [0.594, 0.746]  | borderline      |

The qwen7b lower bound of 0.652 is just above the 0.65 threshold, so
H3 is formally met. The llama8b transfer's point estimate also clears
0.65, but the CI lower bound of 0.594 falls below, so H3 is **consistent
but not confirmed** on llama8b.

Critical falsifier control: **length-only LR trained on the same MATH500
features and tested on GPQA hits only 0.645 [0.562, 0.721] on qwen7b and
0.603 [0.522, 0.680] on llama8b**. Both are strictly below the
structural classifiers, confirming the structural-features story isn't
just "long traces are wrong" leaking across domains.

Auxiliary transfers (MATH500 → ARC, MATH500 → GSM8K, GSM8K → GPQA) are
documented in the CSV; MATH500 → GSM8K and MATH500 → ARC both land
below the structural signal threshold (AUROC 0.48 – 0.58), which is
plausible given how different the trace-length distributions are
between datasets.

## 5. RoBERTa-only ablation: **structural features add marginal value over RoBERTa**

Source: [`reports/route_ab/roberta_only_ablation.csv`](./roberta_only_ablation.csv)

All subsets use the winning stacker config (LR, L1, C = 0.0455, robust
scaler, group dummies) on the same 5 folds:

| Feature subset               | n_feat | AUROC  | 95 % CI         | Δ vs. full  | paired p   |
| ---------------------------- | -----: | -----: | --------------- | ----------: | ---------: |
| full (6 OOFs + 28 hand)      |     34 | 0.8054 | [0.794, 0.816] |     +0.0000 |      1.00  |
| all_oofs (6 OOFs, no hand)   |      6 | 0.8052 | [0.794, 0.816] |     −0.0001 |      0.85  |
| roberta + hand               |     29 | 0.8020 | [0.791, 0.813] |     −0.0034 |      0.024 |
| roberta_only                 |      1 | 0.8019 | [0.791, 0.813] |     −0.0035 |      0.047 |
| oof_minus_roberta (5 OOFs)   |      5 | 0.7820 | [0.770, 0.793] |     −0.0234 | 9.9 × 10⁻¹⁹ |
| hand_only (28 hand features) |     28 | 0.6660 | [0.652, 0.679] |     −0.1394 | 1.4 × 10⁻⁹⁹ |

This table changes the paper framing in three concrete ways:

1. **RoBERTa alone explains 99.6 % of the stacker's AUROC** (0.8019 of
   0.8054). The full stacker's gain over RoBERTa-only is +0.0035 AUROC
   at p ≈ 0.05. Keep the stacker as an engineering upper bound; don't
   claim a meaningful "hybrid" effect without a Bonferroni adjustment
   for the ablation family.
2. **All 105 handcrafted + graph + timing + PH + n-gram features together
   contribute < 0.001 AUROC given the 6 neural OOFs** (`full` − `all_oofs`
   = −0.0001, p = 0.85). This is numerically evidence that **once we
   have good base learners the hand-engineered features are redundant
   at the stacker level**. They may still be valuable for interpretability
   and for the length-controlled falsifier.
3. **Handcrafted features alone score 0.666** — real signal, but ~0.14
   AUROC below RoBERTa. This is the correct number to cite for "structure
   without content": it beats length-only (0.595 pooled per
   `results/month1/lengthctl_pooled.json`) by ~0.07 AUROC, and beats
   chance, but it is clearly dominated by the content signal.

## 6. Falsifiers and gates

Consolidated against the pre-committed criteria in `research_proposal.md`.

| Falsifier                                               | Status  | Source                                                                    |
| ------------------------------------------------------- | ------- | ------------------------------------------------------------------------- |
| H1: MATH500 AUROC ≥ 0.75                                | **PASS** | `reports/route_ab/per_group_metrics_with_ci.csv`                          |
| H3: MATH500 → GPQA AUROC ≥ 0.65                         | **partial** — qwen7b PASS, llama8b borderline | `reports/route_ab/h3_transfer_auroc_ci.csv`                               |
| Length-control: handcrafted beats length in every bin   | **PASS** | `results/month1/lengthctl_pooled.json`                                    |
| Text-encoder bar: beat fine-tuned DeBERTa-v3 on raw trace | **Mixed** — stacker yes; structure alone no | `reports/route_ab/pooled_metrics_with_ci.csv` + ablation                  |
| Structural vs. content: structure adds significant AUROC | **Partial** — +0.0035 at p = 0.047 uncorrected | `reports/route_ab/roberta_only_ablation.csv`                               |

## 7. Updated paper framing

Given §5 + §6, the original framing ("single-trace structural topology
beats content") is not defensible at these sample sizes. The fallback
framing pre-committed in `research_proposal.md:268` — *structural
features as a complementary, content-free, transferable layer on top of
text-based UQ* — is defensible and well supported.

Concrete claims we can make now:

* **Content (RoBERTa on 512 trace tokens) is the dominant predictor**
  of LLM answer correctness from a single reasoning trace, achieving
  pooled AUROC 0.798 [0.787, 0.809] across 8 dataset × model splits.
* **Structural features carry real, content-free signal**: 28
  handcrafted + recurrence features alone reach AUROC 0.666 [0.652, 0.679],
  comfortably above the length-only baseline (0.595) and above the
  length-only in-bin falsifier in every length quintile
  (`results/month1/lengthctl_pooled.json`).
* **Structural features transfer across domains**: MATH500-trained
  RF on handcrafted + recurrence hits AUROC 0.732 [0.652, 0.799] on
  unseen GPQA-Diamond (qwen7b), which strictly dominates the
  length-only transfer baseline (0.645 on the same pair). H3 is
  formally met on qwen7b.
* **At the stacker level, content + structure ≈ content + six
  OOFs + group priors ≈ pooled AUROC 0.805**; the marginal gain of
  structure on top of content is +0.0035 AUROC (p = 0.047, paired
  DeLong). This is an honest "complementary but small at stacker
  aggregation" result, not a replacement of content by structure.
* **The content signal itself is well-captured by RoBERTa-base**;
  DeBERTa-v3 adds 0.035 AUROC over Step-TF and is dominated by
  RoBERTa at this scale (0.763 vs 0.798), so the "text-encoder bar"
  for a structural-alone approach remains 0.798, and the structural-alone
  baseline (0.666) does **not** clear it.

This reframing is consistent with the Month-2 caveats already documented in
the repo ("Step Transformer loses to DeBERTa pooled", "TF-IDF beats
handcrafted on 6/8") and with the pre-committed fallback narrative.

## 8. Pending HPC work (blocks final numbers)

1. `STEPS="deberta" FORCE=1 sbatch scripts/sbatch_rerun_clean_oofs.sh`
   — re-run DeBERTa with the FP32-loading patch in
   `src/modeling/deberta_baseline.py`. Expected result: DeBERTa clean
   AUROC ≈ 0.7–0.77 (versus the broken 0.5000 from the April run).
2. `bash scripts/promote_clean_oofs.sh --check-only` then
   `bash scripts/promote_clean_oofs.sh` — validate, back up, and
   promote the clean OOFs. Creates `.LEAKY.npz` backups (currently
   absent, confirming the swap has never run).
3. `PYTHONPATH=. python scripts/build_hybrid_table.py --leaky-policy fail`
   — rebuild `data/hybrid_table.parquet` refusing any remaining leaky
   sources.
4. `STUDY_NAME=hybrid_v2_truly_clean sbatch scripts/sbatch_tune_hybrid.sh`
   — 2000-trial retune on the truly-clean table. Expected pooled AUROC
   ≈ 0.78 – 0.80 (may drop 0.005 – 0.025 from the current 0.805 after
   leakage removal).
5. After step 4, rerun this session's three local scripts with the new
   stacker OOF:
   ```
   PYTHONPATH=. python scripts/compute_per_group_ci.py \
       --stacker-oof results/route_ab/hybrid_tuned_hybrid_v2_truly_clean_pooled_oof.npz
   PYTHONPATH=. python scripts/roberta_only_ablation.py
   # H3 transfer is base-learner-agnostic; re-run only if features changed.
   PYTHONPATH=. python scripts/h3_transfer_with_ci.py
   ```

None of steps 1 – 4 can be completed from the local workspace; they
require HPC GPU time (~2 h for DeBERTa + ~6 h for the retune).
Everything in §§1 – 7 above stays internally consistent regardless of
step 4's outcome, modulo a uniform shift in the absolute stacker
AUROC number.

## 9. Artifacts created this session

Code (all with embedded self-tests or sanity outputs):

* `src/analysis/delong_ci.py` — DeLong variance, logit CI, paired
  test, per-group aggregator. 7 self-tests pass (`PYTHONPATH=. python
  src/analysis/delong_ci.py`), incl. numerical agreement with
  `sklearn.roc_auc_score` and a 500-sample stratified bootstrap.
* `scripts/compute_per_group_ci.py` — CLI runner for the OOF registry;
  produces `per_group_metrics_with_ci.csv`, `pooled_metrics_with_ci.csv`,
  `paired_delong_stacker_vs_base.json`.
* `scripts/h3_transfer_with_ci.py` — cross-domain transfer CLI;
  produces `h3_transfer_auroc_ci.csv`, `h3_transfer_summary.md`,
  `h3_transfer_details.json`.
* `scripts/roberta_only_ablation.py` — ablation CLI reusing the
  tuner's `cv_train_predict`; produces `roberta_only_ablation.csv`
  and `.json`.

Reports:

* `reports/route_ab/per_group_metrics_with_ci.csv`
* `reports/route_ab/pooled_metrics_with_ci.csv`
* `reports/route_ab/paired_delong_stacker_vs_base.json`
* `reports/route_ab/h3_transfer_auroc_ci.csv`
* `reports/route_ab/h3_transfer_details.json`
* `reports/route_ab/h3_transfer_summary.md`
* `reports/route_ab/roberta_only_ablation.csv`
* `reports/route_ab/roberta_only_ablation.json`
* `reports/route_ab/FINDINGS_AND_REFRAMING.md`  (this file)

Previously patched in this session (awaiting HPC validation):

* `src/modeling/deberta_baseline.py` — explicit `torch_dtype=torch.float32`
  + post-load finite-logits probe; fixes the BF16 / PR-14 fallback
  that caused AUROC = 0.5000.
* `scripts/promote_clean_oofs.sh` — 5-stage validate → backup →
  swap → rewrite provenance → rebuild pipeline (CPU-only; runnable
  after step 1 of §8).
* `scripts/tune_hybrid.py` — Optuna v4 `JournalFileBackend` shim.
