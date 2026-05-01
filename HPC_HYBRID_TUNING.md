# HPC Hybrid Tuning — Leakage Fix + Optuna Walkthrough

End-to-end runbook for the 2026-04-20 leakage patch and the subsequent
hyperparameter search for the hybrid meta-stacker. Covers everything from
"find the leak" to "publish a clean posterior" in one place.

**This is the companion to:**
- [PLAN_ROUTE_AB.md](PLAN_ROUTE_AB.md) — original A+B1 plan.
- [HPC_WALKTHROUGH_ROUTE_AB.md](HPC_WALKTHROUGH_ROUTE_AB.md) — Route A+B1 execution runbook that produced the OOFs this doc tunes over.
- [scripts/write_oof_provenance.py](scripts/write_oof_provenance.py) — sidecar writer.
- [scripts/build_hybrid_table.py](scripts/build_hybrid_table.py) — parquet builder.
- [scripts/tune_hybrid.py](scripts/tune_hybrid.py) — Optuna driver.
- [scripts/sbatch_rerun_clean_oofs.sh](scripts/sbatch_rerun_clean_oofs.sh) — HPC rerun (clean base OOFs).
- [scripts/sbatch_tune_hybrid.sh](scripts/sbatch_tune_hybrid.sh) — HPC batch runner for the Optuna study (tune + refit in one job).

**Scope of this doc:**

| Stage | Location | Runtime | Output |
|---|---|---|---|
| 0. Identify leakage | Local | — | diagnosis |
| 1. Patch base-model code | Local | 10 min | edits to `src/modeling/*.py` |
| 2. Flag tainted OOFs | Local | seconds | 6 `.PROVENANCE.json` sidecars |
| 3. Build hybrid table | Local | 30 s | `data/hybrid_table.parquet` |
| 4. Smoke tune (tainted) | Local | 2 min | 25-trial CSV + best JSON |
| 5. Full tune (tainted) | Local | ~2 h | 500-trial Optuna DB |
| 6. Rerun clean OOFs | HPC (SLURM, GPU) | 6-7 h | 5 `_clean_oof.npz` |
| 7. Promote + rebuild | Local | minutes | clean parquet |
| 8. Full retune (clean) | HPC (SLURM, CPU) | 1-6 h | final reportable study |

> **Notation:** `$REPO` means the repo root (`/home/marky/LLM-Reasoning-Trace-Topology` on laptop/WSL; `/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology` on Bouchet). All Python invocations assume `PYTHONPATH=$PWD` and `$PWD == $REPO`.

---

## Section 1 — Leakage context

### What was leaking

The five neural base-model OOFs were produced by training loops that
**selected the best epoch per fold using the held-out fold's AUROC**.
Each fold's OOF prediction came from whatever epoch achieved the highest
`val_AUROC`, where `val` was the same held-out fold used as OOF.

Pseudocode of the buggy pattern:

```python
for epoch in range(n_epochs):
    train_one_epoch()
    val_probs, val_labels = predict(fold_holdout)
    val_auroc = roc_auc_score(val_labels, val_probs)
    if val_auroc > best_auroc:
        best_auroc = val_auroc
        best_probs = val_probs.copy()   # ← leaks epoch-selection signal
        best_labels = val_labels.copy()
return best_labels, best_probs           # ← used as OOF
```

This is **soft test-set leakage**: the OOF doesn't see individual labels,
but every epoch's "should I keep training" decision was implicitly made
using labels of the exact examples whose predictions ended up in the OOF.
Over 3-10 epochs, that inflates pooled AUROC by roughly 0.01–0.025
(bounded by the number of epochs you had to pick from, the per-fold
label variance, and the base model's inherent calibration noise).

### Which files carried the leak

```
src/modeling/deberta_baseline.py   (powers BOTH RoBERTa and DeBERTa OOFs via --model)
src/modeling/step_transformer.py
src/modeling/trace_gnn.py          (DOUBLE LEAK — best-epoch + early-stop on val)
src/modeling/behavior_seq_lm.py    (present but not currently feeding the hybrid)
src/modeling/trace_mlm_encoder.py  (present but not currently feeding the hybrid)
```

`trace_gnn.py` was worse because it also used `val_AUROC` plateau as the
early-stopping criterion — double-dipping that compounds the first leak.

**Clean by construction** (unaffected):

```
src/modeling/shapelet_eval.py       (no epoch concept at all; info-gain mining + LR inside each fold)
scripts/build_route_a_features.py   (deterministic feature extractors)
src/modeling/hybrid_route_ab.py     (the meta-stacker — trained on OOFs, but does its own clean CV)
```

### What the patch does

Replace "pick best epoch by val_AUROC" with "use the **last** epoch's
predictions unconditionally". This removes the selection signal entirely
and gives you a leakage-safe OOF at the cost of maybe 1-3% of raw
accuracy (last epoch is usually within noise of best epoch anyway if
training has converged).

The marker string `LEAKAGE-SAFE OOF PROTOCOL` is present in all five
patched files — `scripts/sbatch_rerun_clean_oofs.sh` greps for it as a
precondition check. If you revert a file, the job aborts before spinning
up the GPU.

### Expected AUROC deltas post-rerun

| OOF | Tainted pooled AUROC | Expected clean | Delta |
|---|---|---|---|
| RoBERTa | ~0.80 | ~0.78 | −0.01 to −0.02 |
| DeBERTa | ~0.81 | ~0.79 | −0.01 to −0.02 |
| Step Transformer | ~0.70 | ~0.68 | −0.01 to −0.02 |
| TraceGIN-structural | ~0.72 | ~0.70 | −0.01 to −0.025 |
| TraceGIN-hybrid | ~0.74 | ~0.72 | −0.01 to −0.025 |
| Shapelet | clean already | — | — |

GIN deltas are at the top of the range because of the double leak (best
epoch + early stop). If clean AUROCs come out **higher** than tainted,
something went wrong — dig in before promoting.

### Provenance sidecars

Each tainted OOF has a sibling `*.PROVENANCE.json` written by
[scripts/write_oof_provenance.py](scripts/write_oof_provenance.py). The
build step (`build_hybrid_table.py`) reads these, embeds the provenance
into the parquet's META, and the tuner (`tune_hybrid.py`) respects a
`--leaky-policy` flag (`allow` / `warn` / `fail`).

Inspect one:

```bash
cat $REPO/results/month2/deberta_pooled_oof.npz.PROVENANCE.json
```

---

## Section 2 — Quickstart on laptop / WSL (tainted OOFs)

You can run the full tuning loop *right now* against the existing OOFs
without waiting for the HPC rerun. The results are exploration-only
(AUROC inflated), but they validate the pipeline, the Optuna study
infrastructure, and the param-space design.

### Step 2.1 — Write provenance sidecars (one-time)

```bash
cd $REPO
PYTHONPATH=. python scripts/write_oof_provenance.py
```

Expected output:

```
WROTE results/roberta_pooled_oof.npz.PROVENANCE.json
WROTE results/month2/deberta_pooled_oof.npz.PROVENANCE.json
WROTE results/step_transformer_pooled_oof.npz.PROVENANCE.json
WROTE results/route_ab/trace_gnn_hybrid_pooled_oof.npz.PROVENANCE.json
WROTE results/route_ab/trace_gnn_structural_pooled_oof.npz.PROVENANCE.json
WROTE results/route_ab/shapelet_oof.npz.PROVENANCE.json (clean)
Total: wrote 6 sidecars; 0 missing.
```

Dry-run first to preview: `--dry-run`.

### Step 2.2 — Build the hybrid table

```bash
PYTHONPATH=. python scripts/build_hybrid_table.py --leaky-policy warn
```

Expected output (actual values observed on the current data):

```
Final table: 6344 rows x 366 cols
  key cols  : ['group', 'item_id', 'label', 'fold']
  oof cols  : 6
  feat cols : 356
  per-group :
    arc_challenge_llama8b        1168
    arc_challenge_qwen7b         1165
    gpqa_diamond_llama8b          189
    gpqa_diamond_qwen7b           192
    gsm8k_llama8b                1319
    gsm8k_qwen7b                 1318
    math500_llama8b               494
    math500_qwen7b                499
⚠  WARNING: one or more base OOFs are LEAKY (best-epoch-on-val).
```

Outputs:
- `data/hybrid_table.parquet` (~3.8 MB)
- `data/hybrid_table.META.json` (full diagnostics + fold disagreement)

Fold disagreement to verify:

```bash
python -c "import json; m=json.load(open('data/hybrid_table.META.json'));\
print(json.dumps(m['fold_disagreement_vs_canonical'], indent=2))"
```

Expect RoBERTa/DeBERTa/StepTF at 100%, GIN-hyb/GIN-str at 87.6%. This is
fine — the canonical fold (RoBERTa's) is what the meta-CV will use, and
each base's own OOF is still leave-this-item-out under that base's own
scheme.

### Step 2.3 — Smoke tune (25 trials, ~2 min)

```bash
PYTHONPATH=. python scripts/tune_hybrid.py \
    --n-trials 25 \
    --study-name hybrid_smoke \
    --storage "journal:///data/optuna_hybrid_smoke.log"
```

> **Why `journal://`, not `sqlite://`?** On WSL / `\\wsl.localhost` UNC
> paths, SQLite WAL file locks fail (`database is locked`). Optuna's
> `JournalFileStorage` with `JournalFileOpenLock` is a flat append-only
> log that uses open-based file locks — works on every filesystem we
> care about. The `tune_hybrid.py` `_resolve_storage()` helper
> recognizes the `journal:///` URL and sets up the open-lock
> automatically. On HPC / native Linux, use `sqlite:///path.db` instead
> (faster, queryable).

Expected tail of stdout (actual observed values):

```
[INFO]   trial 0022  fam=xgb   sub=oof+hand          sc=standard  dum=True   iso=False  pooled=0.8050  weighted=0.7753  n_feat=34  t=0.9s
[INFO] ✓ Wrote results/route_ab/hybrid_tuning_trials.csv  (25 rows)
[INFO] ✓ Wrote results/route_ab/hybrid_tuned_best.json
[INFO] BEST: weighted_AUROC=0.7753  pooled_AUROC=0.8050
```

`oof+hand + XGBoost + standard scaler + group dummies` was the smoke
winner — no isotonic, only 34 features, gets weighted 0.7753. Baseline
LR stacker on all 6 OOFs pooled gives 0.8016 / weighted 0.7761, so the
tuner is already matching baseline after 25 trials.

### Step 2.4 — Full tune (~500 trials, ~2 h on laptop)

```bash
PYTHONPATH=. python scripts/tune_hybrid.py \
    --n-trials 500 \
    --study-name hybrid_v1_leaky \
    --storage "journal:///data/optuna_hybrid_v1_leaky.log" \
    --leaky-policy warn
```

This is safe to interrupt: Optuna resumes from the journal on next
invocation with the same `--study-name`. You can also run it in a tmux
session and periodically check trial count:

```bash
python -c "import optuna; \
s=optuna.load_study(study_name='hybrid_v1_leaky', \
 storage=optuna.storages.JournalStorage(\
   optuna.storages.JournalFileStorage('data/optuna_hybrid_v1_leaky.log', \
     lock_obj=optuna.storages.journal.JournalFileOpenLock('data/optuna_hybrid_v1_leaky.log')))); \
print(f'n_trials={len(s.trials)}  best={s.best_value:.4f}')"
```

### Step 2.5 — Refit best to generate OOF + per-group breakdown

```bash
PYTHONPATH=. python scripts/tune_hybrid.py --refit \
    --study-name hybrid_v1_leaky \
    --storage "journal:///data/optuna_hybrid_v1_leaky.log" \
    --refit-oof       results/route_ab/hybrid_tuned_leaky_pooled_oof.npz \
    --refit-metrics   results/route_ab/hybrid_tuned_leaky_pooled.json \
    --refit-per-group reports/route_ab/hybrid_tuned_leaky_per_group.csv
```

Outputs the refit's pooled+weighted AUROC, per-group AUROC table,
OOF `.npz` compatible with downstream analysis.

**Do not quote these leaky numbers in the paper.** They carry the
inflation. The right sequence of actions is: run this to sanity-check
the pipeline, then move to Section 3 for the clean rerun.

---

## Section 3 — HPC rerun (clean OOFs)

### Step 3.1 — Sync patched source to HPC

The patches need to reach the HPC working copy. Two options:

**Option A — git (preferred, keeps history):**

```bash
# On laptop
cd $REPO
git add src/modeling/deberta_baseline.py \
        src/modeling/step_transformer.py \
        src/modeling/trace_gnn.py \
        src/modeling/behavior_seq_lm.py \
        src/modeling/trace_mlm_encoder.py \
        src/modeling/shapelet_eval.py \
        scripts/write_oof_provenance.py \
        scripts/build_hybrid_table.py \
        scripts/tune_hybrid.py \
        scripts/sbatch_rerun_clean_oofs.sh \
        HPC_HYBRID_TUNING.md
git commit -m "Leakage fix (last-epoch OOF) + hybrid tuning infra"
git push origin update
```

```bash
# On HPC
ssh cpsc4770_ym466@bouchet.ycrc.yale.edu
cd /home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology
git fetch --all && git pull origin update

# Verify the patches arrived
grep -l "LEAKAGE-SAFE OOF PROTOCOL" src/modeling/*.py
# Should list all 5 patched files.
```

**Option B — rsync:**

```bash
rsync -av --include='src/modeling/*.py' \
          --include='scripts/write_oof_provenance.py' \
          --include='scripts/build_hybrid_table.py' \
          --include='scripts/tune_hybrid.py' \
          --include='scripts/sbatch_rerun_clean_oofs.sh' \
          --include='HPC_HYBRID_TUNING.md' \
    $REPO/ \
    cpsc4770_ym466@bouchet.ycrc.yale.edu:/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology/
```

### Step 3.2 — Submit the rerun job

```bash
# On HPC login node
cd /home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology
mkdir -p logs                     # sbatch writes to logs/clean_oofs_<jobid>.{out,err}
sbatch scripts/sbatch_rerun_clean_oofs.sh
# → Submitted batch job <JOBID>
```

The script:
1. Activates `torch311` conda env (override with `CONDA_ENV=`).
2. Greps all 5 source files for the `LEAKAGE-SAFE OOF PROTOCOL` marker;
   aborts with exit 3 if any file reverted.
3. Runs each OOF producer conditionally via a `run_if_needed` helper
   (skips steps whose `*_clean.json` already exists unless `FORCE=1`).
4. Re-builds step embeddings if `data/step_embeddings/*.npz < 8`.
5. Re-builds structural graphs if `data/graphs/*_graph.npz < 8`.
6. Re-builds hybrid graphs (+MiniLM encoding, ~30 min extra) if
   `data/graphs/hybrid/*_graph.npz < 8`.
7. Runs all 5 OOF producers with last-epoch protocol, writing to
   `*_clean.json` / `*_clean_oof.npz` (tainted originals preserved).
8. Prints a tainted-vs-clean AUROC delta table at the end.

Resource request (in the header):

```
--partition=gpu_rtx6000
--gpus=1
--cpus-per-task=4
--mem=48G
--time=12:00:00
```

Runtime budget: ~6-7 h. 12 h wall clock is headroom.

### Step 3.3 — Subset reruns (optional)

You can rerun a subset of the 5 OOFs by passing `STEPS=`:

```bash
# Just DeBERTa + GIN-hybrid (the two with the largest expected delta):
STEPS="deberta gin_hybrid" sbatch scripts/sbatch_rerun_clean_oofs.sh

# Force-rerun even if the *_clean.json exists (e.g. after a code fix):
FORCE=1 STEPS="roberta" sbatch scripts/sbatch_rerun_clean_oofs.sh
```

Valid tokens: `roberta`, `deberta`, `step`, `gin_structural`, `gin_hybrid`.
The `STEPS` string is checked via substring match (`*"$name"*`), so
`"gin"` matches both GIN variants; `"gin_structural"` matches only one.

### Step 3.4 — Monitor + pull results back

```bash
# On HPC
squeue -u cpsc4770_ym466 -j <JOBID>
tail -f logs/clean_oofs_<JOBID>.out

# After completion, look at the tainted-vs-clean delta table the job prints:
tail -40 logs/clean_oofs_<JOBID>.out

# Expected shape (deltas should be negative ~0.01-0.025):
#   model         tainted_AUROC   clean_AUROC    delta
#   -------------------------------------------------------
#   RoBERTa       0.8000          0.7820         -0.0180
#   DeBERTa       0.8110          0.7930         -0.0180
#   StepTF        0.7020          0.6880         -0.0140
#   GIN-str       0.7210          0.6990         -0.0220
#   GIN-hyb       0.7420          0.7200         -0.0220
```

If any delta is **positive** (clean > tainted), something's off. Likely
causes:
- Random seed variance on a noisy fold (acceptable if small, < 0.005).
- A patch accidentally dropped an essential loss term — inspect the
  `*_clean.json` metrics vs the tainted one for per-fold breakdowns.
- Training instability (NaN gradient, LR too high after the seed
  change) — check for missing folds in the clean `.npz`.

Pull clean artifacts back to laptop:

```bash
# On laptop
rsync -av \
  cpsc4770_ym466@bouchet.ycrc.yale.edu:/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology/results/ \
  $REPO/results/
```

---

## Section 4 — Promote clean OOFs and retune

After the HPC job lands with acceptable deltas, promote the cleans over
the tainted originals (keeping backups), rebuild the hybrid table with
`--leaky-policy fail` (which now succeeds because nothing is leaky),
and run a final tune.

### Step 4.1 — Promote

The sbatch script's end-of-log prints the exact promotion block. Here
it is for convenience:

```bash
cd $REPO
for pair in \
    "results/roberta_pooled_clean_oof.npz:results/roberta_pooled_oof.npz" \
    "results/month2/deberta_pooled_clean_oof.npz:results/month2/deberta_pooled_oof.npz" \
    "results/step_transformer_pooled_clean_oof.npz:results/step_transformer_pooled_oof.npz" \
    "results/route_ab/trace_gnn_structural_pooled_clean_oof.npz:results/route_ab/trace_gnn_structural_pooled_oof.npz" \
    "results/route_ab/trace_gnn_hybrid_pooled_clean_oof.npz:results/route_ab/trace_gnn_hybrid_pooled_oof.npz"
do
    src="${pair%:*}"; dst="${pair#*:}"
    [ -f "$src" ] || { echo "SKIP $src (missing)"; continue; }
    cp -v "$dst" "${dst%.npz}.LEAKY.npz"              # backup tainted
    cp -v "$src" "$dst"                               # promote clean
    cp -v "$src.PROVENANCE.json" "$dst.PROVENANCE.json" 2>/dev/null || true
done
```

After this:
- `results/**/<name>.LEAKY.npz` — preserved tainted OOFs (safe to delete
  once results stabilize).
- `results/**/<name>.npz` — clean OOFs in the canonical location.
- `results/**/<name>.npz.PROVENANCE.json` — must now have
  `"leaky": false` and a `protocol` reflecting the last-epoch fix.

Regenerate provenance so the sidecars match the new files:

```bash
# Edit scripts/write_oof_provenance.py — move the 5 ex-tainted OOFs from
# TAINTED[] to CLEAN[] with protocol="clean_last_epoch" — then:
PYTHONPATH=. python scripts/write_oof_provenance.py
```

Alternatively, if you want sidecars to auto-update from the new `.npz`
sha256, you can regenerate fresh sidecars but they'll still say the
source was tainted unless the Python config is edited. Either way, the
next step validates correctness.

### Step 4.2 — Rebuild hybrid table (fail on any leak)

```bash
PYTHONPATH=. python scripts/build_hybrid_table.py --leaky-policy fail
```

If any sidecar still reports `leaky: true`, this raises `RuntimeError`
and refuses to write. That's the gate — if it passes, the parquet is
fully clean.

Inspect the new META to confirm:

```bash
python -c "import json; m=json.load(open('data/hybrid_table.META.json'));\
print('any_leaky =', m['any_leaky']);\
print('provenance:');\
print(json.dumps(m['provenance_summary'], indent=2))"
# any_leaky = False
# provenance: all "leaky": false
```

### Step 4.3 — Final Optuna study against clean OOFs

You have two paths. Pick the sbatch path unless you have a specific
reason to tune locally — the HPC job is faster (8 cores vs laptop's
typically-fewer-available cores), fully logged, and protected from
the liblinear hang that can brick a local run (see Section 6.1).

**Path A (recommended): sbatch on HPC.**

On HPC, after your parquet is in place (rsync from laptop or rebuild
with `REBUILD_PARQUET=1`):

```bash
cd /home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology

# Default config (sqlite storage, 500 trials, clean-OOF gate, refit enabled)
sbatch scripts/sbatch_tune_hybrid.sh
# → Submitted batch job <JOBID>

# Or with overrides:
sbatch --export=ALL,STUDY_NAME=hybrid_v1_clean,N_TRIALS=1000,TRIAL_TIMEOUT=300 \
       scripts/sbatch_tune_hybrid.sh
```

The script (detailed in Section 5.5 below):
1. Activates `torch311` conda env.
2. Verifies the parquet exists (or rebuilds if `REBUILD_PARQUET=1`).
3. Pins thread counts (`OMP_NUM_THREADS`, etc.) to match `--cpus-per-task`
   so xgboost/lightgbm don't oversubscribe.
4. Runs `scripts/tune_hybrid.py` with `--trial-timeout 180` and
   `--n-jobs 8` (matching 8 allocated cores).
5. On completion, refits the best trial and saves OOF + per-group CSV.
6. Prints a top-10 leaderboard + completed/pruned/failed counts.

Resume later (same study name, same storage) to add more trials:

```bash
sbatch --export=ALL,STUDY_NAME=hybrid_v1_clean,N_TRIALS=500,DO_REFIT=1 \
       scripts/sbatch_tune_hybrid.sh
```

Monitor:

```bash
squeue -u cpsc4770_ym466
tail -f logs/tune_hybrid_<JOBID>.out
# or peek at the study without touching it:
python -c "import optuna; s=optuna.load_study(\
  study_name='hybrid_v1_clean', \
  storage='sqlite:///data/optuna_hybrid_v1_clean.db'); \
print(f'n_trials={len(s.trials)} best={s.best_value:.4f}')"
```

Pull results back to laptop:

```bash
# From laptop
rsync -av cpsc4770_ym466@bouchet.ycrc.yale.edu:/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology/results/route_ab/hybrid_tuned_* \
    $REPO/results/route_ab/
rsync -av cpsc4770_ym466@bouchet.ycrc.yale.edu:/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology/reports/route_ab/ \
    $REPO/reports/route_ab/
```

**Path B: Local tuning (laptop / WSL).**

Only use this if the HPC queue is busy or you're doing exploratory work.
Must use `journal:///` storage (SQLite fails on `\\wsl.localhost` — see
Section 6.1 for details). **Also set `--trial-timeout`** to avoid the
liblinear hang that could lock up your session:

```bash
PYTHONPATH=. python scripts/tune_hybrid.py \
    --n-trials 500 \
    --study-name hybrid_v1_clean \
    --storage "journal:///data/optuna_hybrid_v1_clean.log" \
    --leaky-policy fail \
    --trial-timeout 180 \
    --n-jobs 1
```

> On Windows, the `SIGALRM`-based timeout is a no-op (there's no such
> signal on Windows). You'll see a single warning at startup. If a
> trial hangs, you'll need to `kill -9` the Python process from another
> shell (see 6.1). This is another reason to prefer HPC for real runs.

The `--leaky-policy fail` here is belt-and-suspenders: the parquet is
already clean, but the flag protects you from accidentally pointing at
a stale `.META.json`.

Refit, saving final artifacts:

```bash
PYTHONPATH=. python scripts/tune_hybrid.py --refit \
    --study-name hybrid_v1_clean \
    --storage "journal:///data/optuna_hybrid_v1_clean.log" \
    --refit-oof       results/route_ab/hybrid_tuned_clean_pooled_oof.npz \
    --refit-metrics   results/route_ab/hybrid_tuned_clean_pooled.json \
    --refit-per-group reports/route_ab/hybrid_tuned_clean_per_group.csv
```

**These are the numbers to report.** Include the tainted-vs-clean
comparison in your paper methods section:

```
We identified a soft leakage in the base-model training loops (best
epoch per fold selected by the held-out fold's AUROC). After patching
(last-epoch OOF protocol) and re-running on HPC, pooled AUROC on the
hybrid stack dropped from 0.XXXX to 0.YYYY (delta -0.0ZZZ), which we
attribute to removing the epoch-selection signal. All hyperparameter
tuning and final reporting use the clean OOFs; the tainted OOFs are
retained in results/**/<name>.LEAKY.npz for reproducibility of the
detection.
```

### Step 4.4 — Compare leaky vs clean studies side-by-side

Sanity-check the HP recommendations didn't shift:

```bash
python - <<'PY'
import json
leaky = json.load(open('results/route_ab/hybrid_tuned_leaky_pooled.json'))
clean = json.load(open('results/route_ab/hybrid_tuned_clean_pooled.json'))
print(f"leaky : pooled={leaky['pooled_auroc']:.4f}  weighted={leaky['weighted_auroc']:.4f}")
print(f"clean : pooled={clean['pooled_auroc']:.4f}  weighted={clean['weighted_auroc']:.4f}")
print(f"delta : pooled={clean['pooled_auroc']-leaky['pooled_auroc']:+.4f}  weighted={clean['weighted_auroc']-leaky['weighted_auroc']:+.4f}")
print()
print('Config diff:')
leaky_cfg = {k: v for k, v in leaky['config'].items() if not k.startswith(('xgb_', 'lgbm_', 'lr_'))}
clean_cfg = {k: v for k, v in clean['config'].items() if not k.startswith(('xgb_', 'lgbm_', 'lr_'))}
for k in set(leaky_cfg) | set(clean_cfg):
    if leaky_cfg.get(k) != clean_cfg.get(k):
        print(f"  {k:20s}: leaky={leaky_cfg.get(k)!r:20s}  clean={clean_cfg.get(k)!r}")
PY
```

If the family/subset/scaler choices shifted, the inflation was actually
leading the tuner astray — worth a line in the paper. If they didn't,
the inflation was a uniform level-shift that preserved ranking and you
can lean on that.

---

## Section 5 — Commands reference

### 5.1 `scripts/build_hybrid_table.py`

| Flag | Default | Notes |
|---|---|---|
| `--output` | `data/hybrid_table.parquet` | Output parquet path |
| `--meta-output` | `data/hybrid_table.META.json` | Companion diagnostics JSON |
| `--leaky-policy` | `warn` | `allow` / `warn` / `fail` |
| `--dry-run` |  | Build + log but do not write |
| `--log-level` | `INFO` | Python log level |

Row counts per group on the current (tainted) data:

```
arc_challenge_llama8b   1168
arc_challenge_qwen7b    1165
gpqa_diamond_llama8b     189
gpqa_diamond_qwen7b      192
gsm8k_llama8b           1319
gsm8k_qwen7b            1318
math500_llama8b          494
math500_qwen7b           499
TOTAL                   6344
```

After clean rerun these should be identical — the last-epoch patch
doesn't drop or add examples.

### 5.2 `scripts/tune_hybrid.py`

| Flag | Default | Notes |
|---|---|---|
| `--parquet` | `data/hybrid_table.parquet` | Input |
| `--meta` | `data/hybrid_table.META.json` | For `--leaky-policy` gate |
| `--n-trials` | 100 | Trials this session (Optuna resumes existing) |
| `--timeout` | None | Hard wall-time cap (seconds) |
| `--study-name` | `hybrid_tuning` | Persistent Optuna study name |
| `--storage` | None (memory) | See storage URL table below |
| `--trials-csv` | `results/route_ab/hybrid_tuning_trials.csv` | |
| `--best-json` | `results/route_ab/hybrid_tuned_best.json` | |
| `--seed` | 42 | TPE seed + per-fold model seed |
| `--leaky-policy` | `warn` | Same semantics as builder |
| `--trial-timeout` | 180 | Per-trial SIGALRM cap (seconds). POSIX only; Windows no-ops with a one-time warning. |
| `--n-jobs` | 1 | xgboost / lightgbm per-fit thread count. Set to `--cpus-per-task` when running trials sequentially (the default). |
| `--refit` |  | Load study, refit best, save OOF |
| `--refit-oof` | `results/route_ab/hybrid_tuned_pooled_oof.npz` | |
| `--refit-metrics` | `results/route_ab/hybrid_tuned_pooled.json` | |
| `--refit-per-group` | `reports/route_ab/hybrid_tuned_per_group.csv` | |
| `--log-level` | `INFO` | |

**Storage URL table:**

| URL | Use when | Backend | Notes |
|---|---|---|---|
| (omitted) | one-off, < 100 trials | in-memory | `--refit` unavailable |
| `sqlite:///path.db` | HPC / native Linux | SQLite RDB | Fast, queryable via `optuna-dashboard` |
| `journal:///path.log` | laptop / WSL / UNC | Append-only journal | Uses `JournalFileOpenLock`, works everywhere |
| `mysql+pymysql://...` | shared study | MySQL | For team work; needs server |
| `postgresql://...` | shared study | PostgreSQL | Same |

**Hyperparameter search space** (from `make_objective`):

| Axis | Choices |
|---|---|
| `family` | `lr`, `xgb`, `lgbm` |
| `feature_subset` | `oof_only`, `oof+hand`, `oof+hand+graph`, `oof+ph`, `oof+timing`, `oof+ng`, `all`, `all_minus_ng`, `oof_minus_roberta` |
| `scaler` | `standard`, `robust`, `none` |
| `add_group_dummies` | `False`, `True` |
| `isotonic` | `False`, `True` (fold-internal 3-fold calibration) |
| `lr_C` | 1e-3 … 1e2 (log) |
| `lr_penalty` | `l2`, `l1` |
| `lr_class_weight_balanced` | `False`, `True` |
| `xgb_n_estimators` | 100 … 800 |
| `xgb_max_depth` | 3 … 10 |
| `xgb_lr` | 1e-3 … 0.3 (log) |
| `xgb_subsample` | 0.5 … 1.0 |
| `xgb_colsample` | 0.3 … 1.0 |
| `xgb_min_child` | 1 … 10 |
| `xgb_lambda` | 1e-3 … 10 (log) |
| `xgb_alpha` | 1e-3 … 10 (log) |
| `xgb_gamma` | 0 … 5 |
| `lgbm_n_estimators` | 100 … 800 |
| `lgbm_max_depth` | −1 … 12 |
| `lgbm_num_leaves` | 15 … 255 |
| `lgbm_lr` | 1e-3 … 0.3 (log) |
| `lgbm_subsample` | 0.5 … 1.0 |
| `lgbm_colsample` | 0.3 … 1.0 |
| `lgbm_min_child` | 5 … 100 |
| `lgbm_lambda` | 1e-3 … 10 (log) |
| `lgbm_alpha` | 1e-3 … 10 (log) |

**Primary objective:** per-group weighted AUROC (Simpson-corrected).
**Secondary (logged only):** pooled AUROC.

### 5.3 `scripts/sbatch_tune_hybrid.sh`

Submits the Optuna study as a CPU-only SLURM job, tuning in Stage 1 then
refitting the best trial in Stage 2 of the same allocation.

| Env var | Default | Purpose |
|---|---|---|
| `CONDA_ENV` | `torch311` | Conda environment name |
| `STUDY_NAME` | `hybrid_v1_clean` | Persistent Optuna study name |
| `STORAGE` | `sqlite:///data/optuna_hybrid_v1_clean.db` | URL — use SQLite on HPC (queryable via `optuna-dashboard`); `journal:///...` only if the HPC scratch FS misbehaves |
| `N_TRIALS` | 500 | Trials THIS submission; resume is free via matching `STUDY_NAME` |
| `TRIAL_TIMEOUT` | 180 | Per-trial SIGALRM cap (seconds) |
| `LEAKY_POLICY` | `fail` | `fail` (clean OOFs expected) / `warn` / `allow` |
| `N_JOBS` | 8 | Threads per xgb/lgbm fit; matches `--cpus-per-task` |
| `REBUILD_PARQUET` | 0 | `1` = regenerate `hybrid_table.parquet` before tuning |
| `DO_REFIT` | 1 | `1` = refit best trial in Stage 2; `0` = tune only |
| `SEED` | 42 | TPE + per-fold model seed |
| `PROJECT_DIR` | Bouchet path | Override for a different HPC |

SLURM header (edit if your cluster uses different partitions):

```
--job-name=tune_hybrid
--partition=day               # CPU-only; no GPU needed
--cpus-per-task=8
--mem=32G
--time=12:00:00
```

Output artifacts (anchored by `STUDY_NAME` for clean parallel studies):
- `results/route_ab/hybrid_tuning_trials_${STUDY_NAME}.csv`
- `results/route_ab/hybrid_tuned_best_${STUDY_NAME}.json`
- `results/route_ab/hybrid_tuned_${STUDY_NAME}_pooled_oof.npz` (if `DO_REFIT=1`)
- `results/route_ab/hybrid_tuned_${STUDY_NAME}_pooled.json` (if `DO_REFIT=1`)
- `reports/route_ab/hybrid_tuned_${STUDY_NAME}_per_group.csv` (if `DO_REFIT=1`)
- `logs/tune_hybrid_${SLURM_JOB_ID}.{out,err}`

Common invocations:

```bash
# Default — 500 trials + refit, sqlite storage, clean-OOF gate
sbatch scripts/sbatch_tune_hybrid.sh

# Bigger search, longer per-trial cap (allows more XGB n_estimators range)
sbatch --export=ALL,N_TRIALS=2000,TRIAL_TIMEOUT=600 \
       scripts/sbatch_tune_hybrid.sh

# Add trials to an existing study (resume)
sbatch --export=ALL,N_TRIALS=500,STUDY_NAME=hybrid_v1_clean \
       scripts/sbatch_tune_hybrid.sh

# Tune only, refit manually later
sbatch --export=ALL,DO_REFIT=0 scripts/sbatch_tune_hybrid.sh

# Against tainted OOFs (exploration only — don't report these numbers)
sbatch --export=ALL,STUDY_NAME=hybrid_v1_leaky,STORAGE=sqlite:///data/optuna_hybrid_v1_leaky.db,LEAKY_POLICY=warn \
       scripts/sbatch_tune_hybrid.sh
```

### 5.4 `scripts/sbatch_rerun_clean_oofs.sh`

| Env var | Default | Purpose |
|---|---|---|
| `CONDA_ENV` | `torch311` | Conda environment name |
| `STEPS` | `roberta deberta step gin_structural gin_hybrid` | Subset of steps to run (substring match) |
| `FORCE` | `0` | `1` to re-run even if `*_clean.json` exists |
| `PROJECT_DIR` | Bouchet path | Override for a different HPC |

SLURM header (edit if your cluster uses different partitions):

```
--job-name=clean_oofs
--partition=gpu_rtx6000
--gpus=1
--cpus-per-task=4
--mem=48G
--time=12:00:00
```

### 5.5 `scripts/write_oof_provenance.py`

| Flag | Purpose |
|---|---|
| `--dry-run` | Preview sidecars without writing |

Two Python lists inside the file govern behavior:
- `TAINTED[]` — writes a sidecar with `"leaky": true` + rerun CLI.
- `CLEAN[]` — writes a sidecar with `"leaky": false`.

After you promote clean OOFs in Section 4.1, move the 5 entries from
`TAINTED[]` to `CLEAN[]` (or edit in place) and re-run.

---

## Section 6 — Troubleshooting

### 6.1 Tuning hangs + Ctrl-C does nothing (LR + liblinear)

**Symptom:** Log streams trials normally for a while, then stops after
a `ConvergenceWarning` from liblinear. The process is alive but not
producing output; Ctrl-C has no visible effect for minutes to hours.
Example from a real session (trial 48 started and never reported):

```
trial 0047  fam=lgbm  ...  pooled=0.7999  weighted=0.7706  t=1.1s
<liblinear convergence warning>
<stuck — no further trial log lines>
```

**Cause:** The tuner picked a Logistic-Regression trial with
`penalty=l1`, which forces sklearn to use the `liblinear` solver.
liblinear is a C extension that holds the GIL across its inner loop;
Python signal handlers (including `KeyboardInterrupt`) cannot fire
until the C call returns. On a pathological configuration (wide
feature subset, tight regularization, `class_weight='balanced'`), a
single fit can run for 10+ minutes per fold, ×5 folds, × possibly ×3
inner isotonic folds. Ctrl-C queues up but only delivers when
liblinear returns between iterations — which might never happen
before you give up.

**Fix (already in scripts/tune_hybrid.py as of 2026-04-20):**

1. **Per-trial timeout via SIGALRM.** `--trial-timeout SEC` (default 180)
   installs a SIGALRM handler that raises `TrialTimeout` after the cap.
   The trial is pruned with `pruned_reason=trial_timeout` in user_attrs
   and Optuna moves on.
2. **LR `max_iter` capped at 500** (was 3000). Bounds the worst-case
   single-fit time.
3. **Warning noise silenced** (`ConvergenceWarning`, "X does not have
   valid feature names", "Inconsistent values: penalty=l1 with
   l1_ratio=0.0"). These fire on every pathological trial and
   drown the real trial log otherwise.

To unstick a run that's already hung (the SIGALRM fix only protects
future trials; the currently-stuck fit has to finish or be force-killed):

```bash
# From another shell
pkill -9 -f tune_hybrid
# or find the PID first:
ps aux | grep tune_hybrid
kill -9 <PID>

# If launched via srun:
squeue -u $USER
scancel <JOBID>
```

**Your Optuna study is safe** — completed trials were persisted to
the storage URL after each one. Resuming with the same `--study-name`
and `--storage` picks up from the last completed trial.

**Windows caveat:** SIGALRM doesn't exist on Windows, so the timeout is
a no-op there with a one-time warning. For real multi-hour sweeps on a
Windows Python, either run under WSL (Linux, SIGALRM works) or submit
to HPC via `scripts/sbatch_tune_hybrid.sh`. A short Ctrl-C window is
still possible on Windows if liblinear happens to hit a yield point,
but it's not reliable.

### 6.2 `database is locked` on laptop

**Symptom:** `sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) database is locked`
when running `tune_hybrid.py` with `--storage sqlite:///data/foo.db` on WSL.

**Cause:** Windows Python accessing `\\wsl.localhost\Ubuntu\...` can't
hold SQLite WAL write locks. The UNC mount doesn't forward advisory
file locks reliably.

**Fix:** Use `--storage journal:///data/foo.log` (note: `.log` not `.db`).
The `_resolve_storage()` helper in `tune_hybrid.py` sets up
`JournalFileOpenLock`, which uses `open()` flock (O_EXCL-style advisory
lock), not symlinks. Works on WSL, UNC, NFS, SMB, anything.

### 6.3 `PermissionError [WinError 5]` when using journal storage

**Symptom:** First attempt at `journal:///path/to.log` raises
`PermissionError [WinError 5]` with trace in
`optuna.storages.journal._file.JournalFileSymlinkLock`.

**Cause:** The default `JournalFileStorage` uses symlinks for locking,
and Windows won't create symlinks without admin (`SeCreateSymbolicLinkPrivilege`).

**Fix:** We explicitly use `JournalFileOpenLock` in `_resolve_storage`.
If you see this error anyway, your optuna version is < 3.6 — upgrade:
`pip install -U "optuna>=3.6,<5.0"`.

### 6.4 GPU OOM on DeBERTa

**Symptom:** `torch.cuda.OutOfMemoryError` during DeBERTa rerun.

**Cause:** DeBERTa-v3-base at batch-size 8 needs ~40 GB on fp32.
RTX6000 has 48 GB so this should just fit; A100 40 GB is tight.

**Fixes (in order of preference):**
1. Drop batch size: edit `sbatch_rerun_clean_oofs.sh` line for DeBERTa,
   change `--batch-size 8` → `--batch-size 4`. Doubles wall time,
   halves memory.
2. Use fp16: add `--fp16` to the DeBERTa invocation if the model code
   supports it (check `src/modeling/deberta_baseline.py`).
3. Swap to A100 80 GB partition if available.

### 6.5 Missing step embeddings or graphs on HPC

**Symptom:** The rerun script logs
`Only 0 step embeddings found (<8); rebuilding...` and then
`FileNotFoundError: data/parsed/*_parsed.jsonl` inside
`build_step_embeddings.py`.

**Cause:** Parsed JSONLs weren't synced to HPC.

**Fix:** From laptop:

```bash
rsync -av $REPO/data/parsed/ \
  cpsc4770_ym466@bouchet.ycrc.yale.edu:/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology/data/parsed/
```

Then resubmit.

### 6.6 Clean AUROC unexpectedly higher than tainted

**Symptom:** Delta table shows `+` instead of `−`.

**Diagnostic:**

```bash
python - <<'PY'
import numpy as np
from sklearn.metrics import roc_auc_score
for name, t, c in [
    ("deberta", "results/month2/deberta_pooled_oof.npz",
                 "results/month2/deberta_pooled_clean_oof.npz")]:
    tz = np.load(t, allow_pickle=True)
    cz = np.load(c, allow_pickle=True)
    # Per-fold comparison
    for f in sorted(np.unique(tz["oof_fold"])):
        m = tz["oof_fold"] == f
        ta = roc_auc_score(tz["y_true"][m], tz["oof_prob"][m])
        ca = roc_auc_score(cz["y_true"][m], cz["oof_prob"][m])
        print(f"fold {f}: tainted={ta:.4f}  clean={ca:.4f}  delta={ca-ta:+.4f}")
PY
```

Causes:
- Single noisy fold (delta < 0.005 per-fold is seed variance). Ignore.
- Consistent positive delta across folds → your patch changed something
  beyond removing the selection. Diff the old vs new training loop.
- Schema mismatch: clean OOF has different `item_ids` order than
  tainted (shouldn't happen, but check
  `np.array_equal(tz['item_ids'], cz['item_ids'])`). If mismatched,
  re-sort both by item_id before comparing.

### 6.7 Optuna study sampler state divergence between sessions

**Symptom:** Resuming a study with `--study-name foo` after a gap keeps
suggesting the same bad hyperparameters.

**Cause:** `TPESampler(seed=42)` is deterministic but stateless across
process restarts — each resume starts fresh TPE exploration. On small
studies (<50 trials per session), this looks like "stuck".

**Fix:** Increase `--n-trials` per session to > 50, and increase
`--seed` for each resume to de-correlate sampler streams:

```bash
# Session 1
PYTHONPATH=. python scripts/tune_hybrid.py --n-trials 200 \
  --study-name hybrid_v1_clean --storage journal:///... --seed 42

# Session 2 (resumes, fresh TPE seed)
PYTHONPATH=. python scripts/tune_hybrid.py --n-trials 200 \
  --study-name hybrid_v1_clean --storage journal:///... --seed 101
```

The study persists trials from all seeds; TPE uses the accumulated
history regardless of per-session `--seed`.

### 6.8 `module load miniconda` not found on non-Bouchet HPC

**Symptom:** `conda activate $CONDA_ENV failed`, exit 2 from the sbatch
script.

**Cause:** Module-system name / path differs.

**Fix:** Edit the conda-activation block of `sbatch_rerun_clean_oofs.sh`:

```bash
# Generic pattern — adjust to your cluster:
source /path/to/miniconda/etc/profile.d/conda.sh
conda activate $CONDA_ENV
```

Or `module load anaconda3` / `module load python/3.11` depending on
your cluster's lmod setup. Bouchet paths are hard-coded in the SLURM
scripts under `scripts/sbatch_*.sh` — any cluster-specific adaptation
goes here.

---

## Appendix A — One-page cheatsheet

```bash
# 1. Local smoke (tainted OOFs, exploration) — always use --trial-timeout
cd $REPO
PYTHONPATH=. python scripts/write_oof_provenance.py
PYTHONPATH=. python scripts/build_hybrid_table.py --leaky-policy warn
PYTHONPATH=. python scripts/tune_hybrid.py --n-trials 25 \
    --study-name smoke --storage "journal:///data/optuna_smoke.log" \
    --trial-timeout 180 --n-jobs 1

# 2. HPC clean rerun (base OOFs)
ssh cpsc4770_ym466@bouchet.ycrc.yale.edu
cd /home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology
git pull origin update
sbatch scripts/sbatch_rerun_clean_oofs.sh
# wait ~6-7h, check logs/clean_oofs_<JOBID>.out

# 3. Promote + rebuild (back on laptop, after rsync'ing results/)
cd $REPO
rsync -av cpsc4770_ym466@bouchet.ycrc.yale.edu:/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology/results/ results/
# run promotion block from Section 4.1
PYTHONPATH=. python scripts/write_oof_provenance.py
PYTHONPATH=. python scripts/build_hybrid_table.py --leaky-policy fail

# 4. Final HP tune on HPC (CPU, no GPU, ~1-6h, fully logged)
rsync -av $REPO/data/hybrid_table.parquet $REPO/data/hybrid_table.META.json \
    cpsc4770_ym466@bouchet.ycrc.yale.edu:/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology/data/
ssh cpsc4770_ym466@bouchet.ycrc.yale.edu
cd /home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology
sbatch scripts/sbatch_tune_hybrid.sh
# default: 500 trials, sqlite storage, trial_timeout=180s, refit enabled,
# leaky-policy=fail. See Section 5.3 for overrides.

# 5. Pull artifacts back + report
# (on laptop)
rsync -av cpsc4770_ym466@bouchet.ycrc.yale.edu:/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology/results/route_ab/hybrid_tuned_* $REPO/results/route_ab/
rsync -av cpsc4770_ym466@bouchet.ycrc.yale.edu:/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology/reports/route_ab/ $REPO/reports/route_ab/
python -c "import json; m=json.load(open('results/route_ab/hybrid_tuned_hybrid_v1_clean_pooled.json')); \
print(f\"pooled={m['pooled_auroc']:.4f}  weighted={m['weighted_auroc']:.4f}\")"
```

---

## Appendix B — Mental model of the fix

The research question is "can structural features predict correctness?"
The leakage confounds that by letting the base models (whose OOFs feed
the hybrid) implicitly use held-out labels to pick epochs. Post-fix:

```
BEFORE:
  train → {predict@ep1, predict@ep2, predict@ep3}
  best = argmax_{e} auroc(fold_k_labels, predict@e[fold_k])   ← leak
  OOF[fold_k] = predict@best[fold_k]

AFTER:
  train → {predict@ep1, predict@ep2, predict@ep3}
  OOF[fold_k] = predict@ep3[fold_k]                            ← deterministic
```

The hybrid stacker is **already** leakage-safe by construction —
cross-fit stacking on base OOFs doesn't double-leak. The bug was
entirely inside the base models. That's why patching the base models
and re-running them (without changing anything about the hybrid
stacker) fully resolves the issue.

---

_Last updated: 2026-04-20. Patch date: 2026-04-20._
