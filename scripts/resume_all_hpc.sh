#!/bin/bash
# resume_all_hpc.sh
#
# Single-command idempotent resume script for Phase 0 + Phase 1 + Phase 2
# (PH features + cross-dataset StepTF transfer + honest structural-only hybrid).
#
# Runs every step that hasn't already produced its output file, and prints
# a final summary. Safe to re-run after interruption — every step checks if
# its output already exists and skips if so.
#
# Design choices:
#   - Uses absolute Python path from the torch311 conda env so no shell-state
#     dependence on conda activation.
#   - nohup-friendly: no interactive prompts, all logs written to files.
#   - GPU work serialised (to avoid OOM across concurrent models); CPU work
#     (hybrid, PRR, PH) runs at the end.
#
# Usage (from the HPC compute node, inside tmux for safety):
#   cd /home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology
#   nohup bash scripts/resume_all_hpc.sh > logs/resume_all.log 2>&1 &
#   tail -f logs/resume_all.log
#
# If a step fails, check the per-step log under logs/ and re-run the script;
# earlier successful steps are skipped automatically.

set -uo pipefail
# Intentionally NOT using -e — we want to continue past individual step
# failures so the summary at the end still runs.

# ---------- Config ----------
PROJECT_DIR="/home/cpsc4770_ym466/project_cpsc4770/cpsc4770_ym466/LLM-Reasoning-Trace-Topology"
PY="/home/cpsc4770_ym466/.conda/envs/torch311/bin/python"
PIP="/home/cpsc4770_ym466/.conda/envs/torch311/bin/pip"

DATASETS=(
    math500_qwen7b math500_llama8b
    gsm8k_qwen7b gsm8k_llama8b
    gpqa_diamond_qwen7b gpqa_diamond_llama8b
    arc_challenge_qwen7b arc_challenge_llama8b
)

# ---------- Housekeeping ----------
cd "$PROJECT_DIR"
mkdir -p logs results/month2_v2 results/month2_v2/steptf_transfer \
         data/step_embeddings data/features/v2 data/question_embeddings

export PYTHONPATH="$PROJECT_DIR"
export HF_HOME="$PROJECT_DIR/.cache/huggingface"
mkdir -p "$HF_HOME"

header() {
    local msg="$1"
    echo ""
    echo "=========================================================================="
    echo "=== $(date +%T)   $msg"
    echo "=========================================================================="
}

# ---------- Sanity ----------
header "Sanity check: python, torch, GPU"
"$PY" -c "
import torch, sys
print('python:', sys.executable)
print('torch :', torch.__version__, 'cuda=', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU   :', torch.cuda.get_device_name(0))
" || { echo "FATAL: torch311 env not usable"; exit 1; }

# ---------- Ensure ripser available (for PH features) ----------
if ! "$PY" -c "import ripser" 2>/dev/null; then
    header "Installing ripser (one-time, for PH features)"
    "$PIP" install --user ripser 2>&1 | tail -5
fi

# =============================================================================
# 1. Rebuild step embeddings if any are missing
# =============================================================================
header "Step 1: step embeddings"
need_rebuild=0
for ds in "${DATASETS[@]}"; do
    [ -f "data/step_embeddings/${ds}.npz" ] || { need_rebuild=1; break; }
done
if [ "$need_rebuild" -eq 1 ]; then
    echo "Some .npz missing — rebuilding all (wipe first to keep behavior-vocab fix consistent)"
    rm -f data/step_embeddings/*.npz
    "$PY" scripts/build_step_embeddings.py --all --batch-size 256 \
        > logs/step_embeddings_rebuild.log 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "FATAL: step_embeddings rebuild failed (rc=$rc); see logs/step_embeddings_rebuild.log"
        exit 11
    fi
else
    echo "All 8 .npz present, skipping rebuild."
fi

# Verify step_types contain non-PAD ordinals (the bug sanity check)
"$PY" -c "
import numpy as np, glob, sys
bad = []
for p in sorted(glob.glob('data/step_embeddings/*.npz')):
    z = np.load(p, allow_pickle=True)
    types = z['step_types']
    flat = np.concatenate([t for t in types if len(t)]) if len(types) else np.array([])
    uniq = sorted(set(int(x) for x in flat))
    print(f'  {p}: step_types unique = {uniq}')
    if uniq == [0] or len(uniq) < 3:
        bad.append(p)
sys.exit(1 if bad else 0)
" || { echo "FATAL: step_types look wrong (see above)"; exit 12; }

# =============================================================================
# 2. StepTF pooled
# =============================================================================
header "Step 2: StepTF pooled (5-fold, seed=42)"
OUT="results/month2_v2/step_transformer_pooled.json"
if [ -f "$OUT" ]; then
    echo "Already done, skip."
else
    "$PY" src/modeling/step_transformer.py \
        --npz-glob "data/step_embeddings/*.npz" \
        --output   "$OUT" \
        --epochs 15 --batch-size 32 --lr 3e-4 --seed 42 \
        > logs/steptf_pooled.log 2>&1
    rc=$?
    echo "StepTF pooled rc=$rc (log: logs/steptf_pooled.log)"
fi

# =============================================================================
# 3. StepTF per-dataset (8 runs)
# =============================================================================
header "Step 3: StepTF per-dataset (8 runs)"
for ds in "${DATASETS[@]}"; do
    OUT="results/month2_v2/step_transformer_${ds}.json"
    NPZ="data/step_embeddings/${ds}.npz"
    if [ -f "$OUT" ]; then
        echo "  -> $ds  (done, skip)"
        continue
    fi
    if [ ! -f "$NPZ" ]; then
        echo "  -> $ds  (npz missing, skip)"
        continue
    fi
    echo "  -> $ds  $(date +%T)"
    "$PY" src/modeling/step_transformer.py \
        --npz "$NPZ" \
        --output "$OUT" \
        --epochs 15 --batch-size 32 --lr 3e-4 --seed 42 \
        > "logs/steptf_${ds}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "     WARNING: $ds failed (rc=$rc); see logs/steptf_${ds}.log"
    fi
done

# =============================================================================
# 4. PH (persistent homology) features on step embeddings
# =============================================================================
header "Step 4: Persistent-homology features (CPU)"
# Check if any PH CSV is missing
need_ph=0
for ds in "${DATASETS[@]}"; do
    [ -f "data/features/v2/${ds}_features_ph.csv" ] || { need_ph=1; break; }
done
if [ "$need_ph" -eq 1 ]; then
    "$PY" src/features/topology_features.py --all \
        > logs/topology_features.log 2>&1
    rc=$?
    echo "PH features rc=$rc (log: logs/topology_features.log)"
else
    echo "All 8 PH CSVs present, skip."
fi

# =============================================================================
# 5. StepTF cross-dataset transfer (8x8 matrix)
# =============================================================================
header "Step 5: StepTF cross-dataset transfer (8x8)"
SUMMARY="results/month2_v2/steptf_transfer/steptf_transfer_summary.csv"
if [ -f "$SUMMARY" ]; then
    echo "Summary CSV exists, skip (re-run with --only <src> to regenerate subset)"
else
    "$PY" scripts/run_cross_dataset_steptf_transfer.py \
        --out-dir results/month2_v2/steptf_transfer \
        --skip-existing \
        > logs/steptf_transfer.log 2>&1
    rc=$?
    echo "cross-dataset transfer rc=$rc (log: logs/steptf_transfer.log)"
fi

# =============================================================================
# 6. Honest structural-only hybrid, 4 seeds
#    (if teammate confirms v1 DeBERTa OOF is clean, swap in --deberta-oof below)
# =============================================================================
header "Step 6: Structural-only hybrid x 4 seeds"
STEP_OOF="results/month2_v2/step_transformer_pooled_oof.npz"
if [ ! -f "$STEP_OOF" ]; then
    echo "SKIP hybrid: $STEP_OOF not found (Step 2 failed or StepTF didn't save OOF)"
else
    for SEED in 42 1 2 3; do
        OUT="results/month2_v2/hybrid_structural_seed${SEED}.json"
        if [ -f "$OUT" ]; then echo "  -> seed=$SEED (done, skip)"; continue; fi
        echo "  -> seed=$SEED  $(date +%T)"
        "$PY" src/modeling/hybrid.py \
            --step-oof      "$STEP_OOF" \
            --features-glob "data/features/*_features_rec.csv" \
            --output        "$OUT" \
            --clf all --seed "$SEED" \
            > "logs/hybrid_structural_seed${SEED}.log" 2>&1
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "     WARNING: hybrid seed=$SEED failed; see logs/hybrid_structural_seed${SEED}.log"
        fi
    done
fi

# =============================================================================
# 7. PRR backfill on all new OOF files
# =============================================================================
header "Step 7: PRR backfill"
"$PY" scripts/backfill_prr.py --glob "results/month2_v2/*_oof.npz" \
    > logs/prr_backfill.log 2>&1 || echo "PRR backfill had issues; see logs/prr_backfill.log"

# =============================================================================
# 8. Summary dump
# =============================================================================
header "Step 8: Final summary"

echo ""
echo "=== results/month2_v2/ inventory ==="
ls -la results/month2_v2/ 2>/dev/null
echo ""
ls -la results/month2_v2/steptf_transfer/ 2>/dev/null | head -10

echo ""
echo "=== Per-dataset StepTF AUROC ==="
"$PY" - <<'PYEOF'
import json, os, glob
rows = []
for p in sorted(glob.glob('results/month2_v2/step_transformer_*.json')):
    if '_oof' in p: continue
    try:
        d = json.load(open(p))
    except Exception as e:
        print(f"{p}: ERROR {e}"); continue
    s = d.get('summary') or d.get('overall') or d
    ds = os.path.basename(p).replace('step_transformer_', '').replace('.json', '')
    rows.append((ds, s.get('auroc_mean', float('nan')), s.get('auroc_std', 0)))
print(f"{'dataset':<32s} {'AUROC':>10s}  {'±std':>7s}")
print('-' * 52)
for ds, m, s in rows:
    print(f"{ds:<32s} {m:>10.4f}  {s:>7.4f}")
PYEOF

echo ""
echo "=== PH feature means by correctness (headline diagnostic) ==="
"$PY" - <<'PYEOF'
import pandas as pd, glob
paths = sorted(glob.glob('data/features/v2/*_features_ph.csv'))
if not paths:
    print("(no PH CSVs found yet)")
else:
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    feats = [c for c in df.columns if c.startswith('h0_') or c.startswith('h1_')]
    print(f"n={len(df)} (pooled over {len(paths)} datasets)")
    for f in feats:
        c = df.loc[df.is_correct == 1, f].mean()
        w = df.loc[df.is_correct == 0, f].mean()
        print(f"  {f:<26s}  correct={c:.3f}  incorrect={w:.3f}  Δ={c-w:+.3f}")
PYEOF

echo ""
echo "=== Structural hybrid seeds (n_samples must be 6378, NOT 25512) ==="
"$PY" - <<'PYEOF'
import json, glob
files = sorted(glob.glob('results/month2_v2/hybrid_structural_seed*.json'))
if not files:
    print("(no hybrid JSONs found)")
for f in files:
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f"{f}: ERROR {e}"); continue
    for v_name in ('step_only', 'handcrafted+rec', 'STRUCTURAL_FULL'):
        if v_name not in d['variants']: continue
        vd = d['variants'][v_name]
        n = vd['n_samples']
        for clf in ('lr', 'rf', 'xgb'):
            if clf not in vd['clfs']: continue
            s = vd['clfs'][clf]['summary']
            print(f"{f.split('/')[-1]:<40s} n={n:<5d} {v_name:<20s} "
                  f"{clf:<3s} AUROC={s['auroc_mean']:.4f}±{s['auroc_std']:.4f}")
PYEOF

echo ""
echo "=== StepTF cross-dataset AUROC matrix (rows=source, cols=target) ==="
"$PY" - <<'PYEOF'
import pandas as pd, os
p = 'results/month2_v2/steptf_transfer/steptf_transfer_summary.csv'
if not os.path.exists(p):
    print("(summary CSV missing)"); raise SystemExit(0)
df = pd.read_csv(p)
ds = ["math500_qwen7b","math500_llama8b","gsm8k_qwen7b","gsm8k_llama8b",
      "gpqa_diamond_qwen7b","gpqa_diamond_llama8b",
      "arc_challenge_qwen7b","arc_challenge_llama8b"]
try:
    pivot = df.pivot(index='source', columns='target', values='auroc').reindex(index=ds, columns=ds)
    print(pivot.round(4).to_string(na_rep='  .   '))
    off = df[df['source'] != df['target']]
    print(f"\nOff-diagonal: n={len(off)}  AUROC mean={off['auroc'].mean():.4f}  "
          f"min={off['auroc'].min():.4f}  max={off['auroc'].max():.4f}")
except Exception as e:
    print(f"pivot error: {e}")
PYEOF

echo ""
echo "=== PRR backfill summary ==="
for p in results/month2_v2/*_metrics_prr.json; do
    [ -f "$p" ] || continue
    "$PY" -c "
import json
d = json.load(open('$p'))
o = d['overall']
print(f\"  $(basename $p)  AUROC={o.get('auroc', 0):.4f}  PRR={o.get('prr', 0):+.4f}\")
"
done

echo ""
echo "$(date +%T)   DONE."
