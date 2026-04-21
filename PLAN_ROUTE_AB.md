# Route A + B1 Implementation Plan

**Goal:** Revive the *original* research claim — structural features from a
single reasoning trace can meet the H1 bar (AUROC ≥ 0.75 on MATH500) **and**
clear the text-encoder falsifier (beat DeBERTa-v3-base fine-tuned on the raw
trace) without using content signals.

**Current diagnosis** (from results/):
- `handcrafted+rec` (30 feats): 0.626 / 0.666 AUROC on MATH500 qwen/llama —
  far below DeBERTa's 0.799 / 0.648.
- StepTF (MiniLM + behavior ordinals): 0.778 / 0.666 — DeBERTa-tied but still
  leaks MiniLM content. 5 AUROC points of its edge is content, not structure.
- `FULL_HYBRID` (30 handcrafted + StepTF + DeBERTa OOFs): 0.776 / 0.646 — text
  dominates; structure adds ~0.005 and loses on 4/8 combos.

**Strategic thesis.** The current structural features under-describe the
trace: they collapse 50–200 behavior episodes into 30 scalars, throw away
*order statistics* (motifs, timing, graph topology), and duplicate signals
DeBERTa already gets from the last 512 tokens. To beat DeBERTa we need
features DeBERTa **cannot** access: whole-trace graph topology, long-range
recurrence patterns, explicit behavior-sequence motifs, content-free coords.

We chase this with two routes in combination:
- **Route A** — five new content-free feature families (n-gram motifs,
  shapelets, graph descriptors, structural PH, inter-event timing).
- **Route B1** — a GNN over the behavior-episode graph, trained end-to-end
  with OOF probabilities that feed the hybrid stack.

All features are computed from `data/parsed/{dataset}_{model}_parsed.jsonl`,
which carries the legacy 7-class taxonomy (F/V/B/R/S/H/C). No re-parsing is
needed.

---

## 0. File-organization plan

```
src/
  features/
    ngram_features.py        # A1 — behavior n-gram motifs (local)
    graph_features.py        # A3 — trace-graph descriptors (local)
    timing_features.py       # A5 — inter-event timing (local)
    structural_ph_features.py# A4 — content-free persistent homology (HPC-friendly, CPU ok)
    shapelet_features.py     # A2 — fold-aware shapelet mining (HPC-recommended)
  modeling/
    trace_gnn.py             # B1 — GNN model + 5-fold OOF training (HPC/GPU)
scripts/
  build_route_a_features.py  # orchestrator: run all 5 feature extractors
  build_trace_graphs.py      # dump PyG .pt graphs for GNN
  sbatch_route_a.sh          # HPC batch (all Route-A features)
  sbatch_trace_gnn.sh        # HPC batch (GNN OOF training)
  run_route_ab_hybrid.sh     # orchestrator: full hybrid variants with Route-A+B1
data/
  features/                  # new CSVs land here: *_ngram.csv, *_graph.csv ...
  graphs/                    # per-dataset PyG graph artifacts (*.pt)
results/
  route_ab/                  # new metric JSONs + ablation tables
```

Every CSV is keyed by `(item_id, dataset)` where `dataset` = canonical
`dataset_model` (e.g., `math500_qwen7b`) so it merges cleanly with the
existing hybrid loader.

---

## 1. Feature families (Route A)

### A1 — Behavior n-gram features  [`src/features/ngram_features.py`]

**Hypothesis.** Specific short motifs predict correctness: e.g. `B→V→F`
("backtracked then verified then continued") should correlate with correct;
`V→V→V` with rumination (incorrect).

**Spec.**
- Read parsed JSONL → `behavior_sequence` (string over 7 chars: FVBRSHC).
- Features (raw counts + rate-normalized):
  - **Bigrams** (all 49): `ngram_2_FF, ngram_2_FV, ..., ngram_2_CC`. Count +
    `rate_2_XY = count / (L-1)`.
  - **Trigrams**: only the K=50 most frequent trigrams **over the whole
    corpus** (across all 8 combos), to bound the feature count. Same
    count + rate encoding.
  - **Position-weighted motifs**: for each of a curated list of 12 motifs
    (`BV`, `VB`, `BF`, `VF`, `FC`, `HB`, `RF`, `BVF`, `VBV`, `FVF`, `BVB`,
    `HHH`), compute a *position-weighted* count:
    `sum over occurrences of w_i`, with `w_i = (pos_i + 1) / L` —
    near-end occurrences count more (where failures usually crystallize).
  - **Rare-motif indicator**: `has_tri_XYZ` binary for K=20 "tail" motifs
    that appear in < 5% of traces corpus-wide.

- Total features: ~49 (bigram count) + 49 (bigram rate) + 50 (trigram count)
  + 50 (trigram rate) + 12 (pos-weighted) + 20 (rare indicators) ≈ 230.
- Writes `data/features/{dataset}_{model}_ngram.csv` with columns
  `item_id, dataset, <230 features>`.

**Compute.** Local-only (minutes). No parallelism required.

### A2 — Shapelet mining  [`src/features/shapelet_features.py`]

**Hypothesis.** A short (3–8 char) categorical subsequence exists that
maximally separates correct from incorrect traces — the "gold shapelet."
Shapelets are order-preserving and length-invariant, so they catch
qualitative patterns the n-gram table misses (e.g., "*first* backtrack
followed within 2 steps by a verify").

**Spec.**
- Candidate generation: all K-subsequences for K ∈ {3, 4, 5, 6, 7, 8}
  extracted from every training-fold trace. Dedup → candidate set C (~20k
  per fold for our corpus size).
- Distance: for each trace `t` and shapelet `s`, compute the *minimum
  Hamming distance* of `s` to any length-|s| window of `t`:
  `d(t, s) = min over i: hamming(t[i:i+|s|], s) / |s|`.
- Mining: for each candidate, compute an *information gain* score over the
  training set's label partition using the binary split
  `d(t, s) ≤ τ_s`, τ_s chosen per-shapelet by best-split search. Keep the
  **top-40** shapelets by info-gain.
- Per-item features: for the top-40 shapelets, emit `dist_<k>` (the distance
  `d(t, shapelet_k)`) → 40 numeric features per item.
- **Critical — leakage safety:** shapelet mining and τ selection must
  happen **inside each CV fold's training split**. Therefore we
  *cannot* emit a single shapelet CSV here. Instead we emit a **precomputed
  distance matrix** per item per candidate and let the fold-aware evaluator
  (extends `cv_utils.stratified_split`) pick shapelets inside each fold.
  We write:
    - `data/features/{dataset}_{model}_shapelet_distmat.npz`
      keys: `item_ids (N,)`, `candidates (M,)` (char strings),
      `distance (N, M, float32)`. M ~20k.
  Mining/selection lives in an evaluator in
  `src/modeling/shapelet_eval.py` (new).
- Baseline-C-plus will not use shapelets directly; instead a dedicated OOF
  predictor (`shapelet_oof`) produces OOF probs that the hybrid can consume.

**Compute.** Distance matrix dominates: O(N · M · L) = ~6400 · 20k · 50 ≈
6e9 ops per dataset. HPC-recommended (numba/torch vectorized); sbatch it.

### A3 — Trace-graph descriptors  [`src/features/graph_features.py`]

**Hypothesis (Minegishi et al., NeurIPS 2025 analogue).** The reasoning
trace forms a directed graph over episodes; its *topology* — diameter,
clustering, centralization, small-worldness — carries correctness signal.

**Graph construction.** Nodes = episodes (labeled by behavior type +
position-in-trace). Edges:
1. **Temporal**: `ep_i → ep_{i+1}` (sequential transitions).
2. **Referential**: `ep_i → ep_j` (j > i) iff `cos_sim(emb_i, emb_j) ≥ 0.7`
   AND `j - i ≥ 3`. Embeddings are the **recurrence-feature MiniLM
   embeddings already cached** (from `recurrence_features.py`). *Not*
   content-free, but A3 is for the unified hybrid, not the pure-structural
   claim. A content-free variant uses behavior-type agreement instead of
   cosine (see A4).

**Descriptors** (all on the weighted directed graph):
- Nodes, edges, edge density, avg in-degree, avg out-degree.
- Average clustering coefficient (undirected proj).
- Global efficiency (shortest-path-based).
- Pseudo-diameter (longest shortest path; approximate for speed).
- Betweenness centralization, eigenvector centralization (top-node
  dominance).
- Modularity (Louvain).
- Small-world σ = (C / C_rand) / (L / L_rand), with a bootstrap-rewired
  random graph.
- Largest SCC size / N.
- Cycle count ≥ 3 (length-bounded).
- Spectral radius of the adjacency matrix.
- Von Neumann graph entropy.

Total: 15 features.
Writes `data/features/{dataset}_{model}_graph.csv`.

**Compute.** NetworkX + optional `python-louvain`. Local; a few seconds per
trace → minutes per dataset.

### A4 — Pure-structural persistent homology  [`src/features/structural_ph_features.py`]

**Hypothesis.** `topology_features_v2.py` already computes PH, but from
**MiniLM embeddings** — content leaks in. Swap the point cloud for a 13-d
**content-free per-episode coordinate** and re-compute PH.

**Per-episode 13-d coord.**
- 7 one-hot components: `[is_F, is_V, is_B, is_R, is_S, is_H, is_C]`.
- Normalized position: `position / max(L-1, 1)`.
- Token-count z-score within the trace.
- Running behavior-diversity: `H(counts_so_far) / log(7)` (up to this step).
- Time-since-last-B, time-since-last-V, time-since-last-restart (episodes),
  normalized by L.

Compute PH (ripser, H₀ + H₁) + persistence images (4×4) exactly like
`topology_features_v2.py`, but on this new point cloud. 4 length-normalized
scalars + 2 × 16 = 32 PI cells + 7 summary stats = 43 features.
Writes `data/features/{dataset}_{model}_structural_ph.csv`.

**Compute.** ripser is O(n³) worst-case but fast in practice. Local is
viable, HPC is faster. Dispatch via the same sbatch as A2.

### A5 — Inter-event timing dynamics  [`src/features/timing_features.py`]

**Hypothesis.** Beyond *which* behaviors appear, *when* and *how
clustered* they are carries signal. A burst of verifications near the end
differs from a single mid-trace verification.

**Spec.** For each of the 7 behaviors b ∈ {F,V,B,R,S,H,C}:
- Arrival counts: number of b episodes.
- First-passage: position of first b occurrence (normalized by L; -1 if
  never).
- Last-occurrence: position of last b (normalized; -1 if never).
- Inter-event intervals: mean and std of gaps between consecutive b
  occurrences (in episodes; -1 if 0–1 occurrence).
- Burstiness coefficient B_b = (σ_ie - μ_ie) / (σ_ie + μ_ie).
  (Goh & Barabási 2008; 0 = Poisson, 1 = maximally bursty.)

Plus global dynamics:
- Global inter-episode interval mean/std (if episodes come with
  `token_count`, we use the cumulative-token timeline for "real time").
- Entropy of the *interval* distribution.
- Peak-to-average ratio of behavior-change events.

Total: 7 × 6 + 4 = 46 features.
Writes `data/features/{dataset}_{model}_timing.csv`.

**Compute.** Pure numpy. Local; seconds per dataset.

---

## 2. Trace-graph GNN (Route B1) [`src/modeling/trace_gnn.py`]

**Model.** PyTorch Geometric. Two-layer GIN (Xu et al. 2019) with hidden
dim 128, LayerNorm, ReLU, 0.2 dropout. Node features: [7-d one-hot behavior,
normalized position, z-scored token count, log1p(confidence)] = 10-d.
Optional: append MiniLM embeddings of episode text (dim 384) behind a flag
`--use_content_emb` so we can ablate content vs structure.

Readout: **attention pooling** (GlobalAttention with a 2-layer scoring
MLP) + **concat** with (mean, max) of node embeddings → 3×128 = 384-d
graph vector → 2-layer classification head → P(correct).

**Training protocol.**
- 5-fold stratified split over (item_id, dataset-model group), same seed 42.
- AdamW lr 3e-4, weight decay 1e-5. 30 epochs, early stop on val AUROC
  (patience 5). Batch size 32 (graphs, not nodes). Class-balanced sampler.
- For each fold, emit OOF probs for the held-out slice. Concatenate across
  folds → 1 prob per item.
- Two model variants: `trace_gnn_structural` (10-d nodes, no content) and
  `trace_gnn_hybrid` (+MiniLM nodes). Both saved as OOF NPZ.

**Output.** `results/route_ab/trace_gnn_structural_oof.npz` and
`results/route_ab/trace_gnn_hybrid_oof.npz` with keys
`{item_ids, y_true, oof_prob, groups}` — same schema as existing hybrid
OOFs.

**Compute.** GPU-required. Single A100 ~1hr per variant (8 datasets, 5
folds). HPC batch.

---

## 3. Hybrid integration [`src/modeling/hybrid_route_ab.py`]

Wraps `src/modeling/hybrid.py` pattern. New CSV loaders for
ngram/graph/timing/structural-ph feature CSVs + a GNN-OOF loader.
Variants to evaluate:

| Variant | Includes | Tests |
|---|---|---|
| `baselineC` | handcrafted-25 | baseline reproduction |
| `baselineC+recurrence` | handcrafted-25 + rec-5 | current strongest structural |
| `baselineC+ngram` | handcrafted + rec + A1 | n-gram boost |
| `baselineC+graph` | + A3 | graph-topology boost |
| `baselineC+timing` | + A5 | timing boost |
| `baselineC+structural_ph` | + A4 | PH pure-structural |
| `ROUTE_A_FULL` | handcrafted+rec + A1+A3+A4+A5 | all Route A |
| `ROUTE_A_FULL+shapelet` | + shapelet OOF prob | + A2 |
| `ROUTE_A_FULL+gnn_structural` | + GNN structural OOF | + B1 content-free |
| `ROUTE_A_FULL+gnn_hybrid` | + GNN hybrid OOF | + B1 with content |
| `ROUTE_AB_TOTAL` | ROUTE_A_FULL + shapelet + GNN_structural | final structural stack |
| `ROUTE_AB + deberta` | + DeBERTa OOF | complementary test (is structure additive to text?) |
| `ROUTE_AB + deberta + step` | full | upper bound |

**Shapelet OOF predictor** [`src/modeling/shapelet_eval.py`]: inside each
of 5 folds, mine shapelets + choose τ on train split, then compute the
distance features for test split, train a logistic-regression on *those 40
distances* (no other features), emit OOF probs. This keeps leakage out
while letting shapelets enter the stack as a scalar.

**Evaluation.** Same metrics as existing (AUROC, AUPRC, ECE, Acc@80,
Acc@90, PRR). Emit:
- `results/route_ab/route_ab_pooled.json` — all variants × {lr, rf, xgb}.
- `results/route_ab/per_dataset_summary.csv` — per (dataset, variant).
- `results/route_ab/falsifier_table.csv` — for each dataset:
  `DeBERTa vs ROUTE_AB_TOTAL`; passes if ROUTE_AB_TOTAL ≥ DeBERTa on ≥ 5/8.

---

## 4. HPC vs local split

| Step | Where | Why |
|---|---|---|
| A1 ngram | local | pure string counting, seconds |
| A3 graph | local | NetworkX, minutes |
| A5 timing | local | numpy, seconds |
| A4 structural PH | HPC (CPU) | ripser batched |
| A2 shapelet distmat | HPC (CPU, vectorized) | O(NML) dominant |
| B1 GNN training | HPC (GPU) | PyG, ~1hr/variant |
| Hybrid eval | local | 5-fold sklearn |

Two sbatch scripts:
- `scripts/sbatch_route_a.sh` — feature extraction (A2 distmat + A4 PH).
- `scripts/sbatch_trace_gnn.sh` — GNN OOF training (GPU).

---

## 5. Self-tests and validation

Each module gets a `if __name__ == "__main__":` smoke test that builds a
synthetic 20-trace mini-corpus and asserts:
- Output CSV has the expected schema and no NaNs.
- Row count = N items, columns include `item_id`, `dataset`.
- Smoke AUROC with LogisticRegression ≥ 0.5 (sanity that features
  aren't all-zero).

Integration test (`scripts/validate_route_ab.py`): runs all 5 feature
extractors on `data/parsed/math500_qwen7b_parsed.jsonl`, builds
`hybrid_route_ab.py` in `baselineC+ngram` mode, reports AUROC.

---

## 6. Execution order

1. A1 ngram (fast, de-risks pipeline).
2. A3 graph, A5 timing (independent).
3. Orchestrator + local feature run on all 8 combos.
4. Baseline-C-plus integration + smoke AUROC.
5. A4 structural PH, A2 shapelet distmat — HPC.
6. Shapelet fold-aware evaluator + OOF.
7. B1 GNN dataset builder + trainer + HPC run.
8. Hybrid wiring + final ablation table.
9. Falsifier comparison against DeBERTa.

This plan is the contract. When any step diverges (e.g., compute
blowing up, a feature degenerate), update this file and the TodoWrite list
before continuing.
