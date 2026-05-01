#!/bin/bash
# check_hpc_state.sh
#
# Fast, read-only inventory of what's present vs. missing on the HPC project
# dir after losing results/month2_v2/. Run from the project root on Bouchet.
#
# Usage:
#   bash scripts/check_hpc_state.sh

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "== PWD: $PWD =="
echo ""

printf '%-55s %s\n' 'ASSET' 'STATUS'
printf '%-55s %s\n' '-----' '------'

# Tier 1 — must be present (irreplaceable without regenerating traces)
for f in \
    data/traces/math500_qwen7b_traces.jsonl \
    data/traces/math500_llama8b_traces.jsonl \
    data/traces/gsm8k_qwen7b_traces.jsonl \
    data/traces/gsm8k_llama8b_traces.jsonl \
    data/traces/gpqa_diamond_qwen7b_traces.jsonl \
    data/traces/gpqa_diamond_llama8b_traces.jsonl \
    data/traces/arc_challenge_qwen7b_traces.jsonl \
    data/traces/arc_challenge_llama8b_traces.jsonl ; do
    if [ -f "$f" ]; then printf '%-55s OK  (%d lines)\n' "$f" "$(wc -l < "$f")"
    else printf '%-55s %s\n' "$f" "MISSING (trace regen required, SKIP THIS STEP)"; fi
done

echo ""

# Tier 2 — needed for StepTF; easy to rebuild from traces + parser
for ds in math500_qwen7b math500_llama8b gsm8k_qwen7b gsm8k_llama8b \
          gpqa_diamond_qwen7b gpqa_diamond_llama8b \
          arc_challenge_qwen7b arc_challenge_llama8b ; do
    f="data/step_embeddings/${ds}.npz"
    if [ -f "$f" ]; then printf '%-55s OK  (%d bytes)\n' "$f" "$(stat -c '%s' "$f" 2>/dev/null || echo 0)"
    else printf '%-55s %s\n' "$f" "MISSING — will rebuild (~2 min GPU each)"; fi
done

echo ""

# Tier 3 — feature CSVs (needed for hybrid)
for ds in math500_qwen7b math500_llama8b gsm8k_qwen7b gsm8k_llama8b \
          gpqa_diamond_qwen7b gpqa_diamond_llama8b \
          arc_challenge_qwen7b arc_challenge_llama8b ; do
    f="data/features/${ds}_features_rec.csv"
    if [ -f "$f" ]; then printf '%-55s OK  (%d lines)\n' "$f" "$(wc -l < "$f")"
    else printf '%-55s %s\n' "$f" "MISSING — feature rebuild needed"; fi
done

echo ""

# Tier 4 — results/
echo "== results/ top-level =="
ls -la results/ 2>/dev/null | head -30 || echo "(missing)"
echo ""
echo "== results/month1/ =="
ls -la results/month1/ 2>/dev/null || echo "(missing)"
echo ""
echo "== results/month2/ (old, v1) =="
ls -la results/month2/ 2>/dev/null || echo "(missing)"
echo ""
echo "== results/month2_v2/ (the one we lost) =="
ls -la results/month2_v2/ 2>/dev/null || echo "(missing or empty)"
echo ""

# Tier 5 — baselines (from earlier local run; may or may not be on HPC)
echo "== per-dataset baselines =="
for letter in a b c d e f g ; do
    n=$(ls -1 results/baseline_${letter}_*.json 2>/dev/null | wc -l)
    printf '  baseline_%s_*.json:  %d files\n' "$letter" "$n"
done

echo ""
echo "== scripts that should exist =="
for s in \
    scripts/build_step_embeddings.py \
    scripts/backfill_prr.py \
    scripts/run_cross_dataset_steptf_transfer.py \
    scripts/run_cross_model_transfer.py \
    src/features/topology_features.py \
    src/modeling/hybrid.py \
    src/modeling/step_transformer.py \
    src/modeling/deberta_baseline.py \
    src/modeling/cv_utils.py ; do
    [ -f "$s" ] && echo "  OK   $s" || echo "  MISS $s"
done

echo ""
echo "Done. Read above to decide what resume_all_hpc.sh needs to rebuild."
