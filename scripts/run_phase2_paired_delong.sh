#!/usr/bin/env bash
# Phase 2 paired DeLong driver.
#
# For every ULTRA_HYBRID / Route-AB variant and every (lr, rf, xgb) classifier,
# run paired DeLong (group, item_id)-aligned against each of the Phase-1
# headline baselines:
#
#   SH_LR  : results/month3/superhybrid_SuperHybrid_LR_oof.npz        (pooled n=6378)
#   SH_RF  : results/month3/superhybrid_SuperHybrid_RF_oof.npz        (pooled n=6378)
#   DC     : results/month2/deberta_conditioned_pooled_oof.npz        (text bar)
#   RoB    : results/roberta_pooled_oof.npz                           (text bar)
#
# Outputs land in reports/month3/paired/ as paired_<tag>_by_group.{csv,json}.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OOF_DIR="${OOF_DIR:-results/month3/ultra_hybrid}"
REPORT_DIR="${REPORT_DIR:-reports/month3/paired}"
mkdir -p "$REPORT_DIR"

# Fully-qualified, stable names
declare -A BASELINES=(
  [sh_lr]="results/month3/superhybrid_SuperHybrid_LR_oof.npz"
  [sh_rf]="results/month3/superhybrid_SuperHybrid_RF_oof.npz"
  [dc]="results/month2/deberta_conditioned_pooled_oof.npz"
  [rob]="results/roberta_pooled_oof.npz"
)

# Variants we care about for paired tests.  We keep the short list short
# to avoid a combinatorial explosion — only the ULTRAs and the key
# Route-AB ancestors.
VARIANTS=(
  ULTRA_HYBRID_ALL
  ULTRA_HYBRID_CORE
  ULTRA_STRUCTURAL
  ULTRA_TEXT_ONLY
  ULTRA_ALL_minus_route_a
  ULTRA_ALL_minus_gnn
  ULTRA_ALL_minus_text
  ULTRA_ALL_minus_probe
  ULTRA_ALL_minus_shapelet
  ROUTE_AB_plus_deberta_plus_step
  ROUTE_AB_plus_deberta
  ROUTE_AB_TOTAL
  ROUTE_A_FULL
  ROUTE_A_FULL_plus_gnn_h
)

CLFS=(lr rf xgb)

for v in "${VARIANTS[@]}"; do
  for clf in "${CLFS[@]}"; do
    challenger="$OOF_DIR/ultrahybrid_${v}__${clf}_oof.npz"
    if [[ ! -f "$challenger" ]]; then
      echo "  [skip] missing challenger $challenger"
      continue
    fi
    for blab in "${!BASELINES[@]}"; do
      baseline="${BASELINES[$blab]}"
      if [[ ! -f "$baseline" ]]; then
        echo "  [skip] missing baseline $baseline"
        continue
      fi
      # Condense (variant, clf, baseline) -> tag
      # Slash the "ULTRA_" prefix to keep tags short
      short_v="$(echo "$v" | sed -e 's/ULTRA_/u_/;s/ROUTE_A_FULL/rAfull/;s/ROUTE_AB_TOTAL/rAB/;s/_plus_/+/g;s/_minus_/-/g')"
      tag="p2_${short_v}_${clf}__vs__${blab}"
      PYTHONPATH=. python scripts/paired_delong_by_group.py \
          --a "$challenger" \
          --b "$baseline" \
          --tag "$tag" \
          --out-dir "$REPORT_DIR" \
          > /dev/null 2>&1 || {
            echo "  [warn] paired failed: $tag"; continue;
          }
      auc_line="$(grep "^OVERALL" "$REPORT_DIR/paired_${tag}_by_group.csv" 2>/dev/null || true)"
      echo "  [done] $tag"
    done
  done
done

echo ""
echo "All pairwise tests written to $REPORT_DIR"
ls "$REPORT_DIR" | wc -l
