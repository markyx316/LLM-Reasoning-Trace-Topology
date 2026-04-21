"""
graph_features.py - Trace-graph topological descriptors.

Constructs a directed trace-graph per reasoning trace and computes 15
graph-level descriptors that capture the *topology* of the reasoning
process: density, clustering, diameter, centralization, modularity,
spectral radius, cycle counts, small-worldness, etc.

Analogue for text-space reasoning graphs of Minegishi et al. (NeurIPS
2025), but content-free (edges derived from structural coincidences, not
semantic similarity) so the claim remains a pure-structural one.

Graph construction (content-free mode, default):
    Nodes:  one per episode.
    Node attrs: behavior_type (0..6 ordinal), position (0..L-1),
                token_count (0..*), normalized_position (0..1).
    Edges:
      1) Temporal chain:  ep_i -> ep_{i+1}  (weight 1.0 by default)
      2) Behavior-recurrence:  ep_i -> ep_j  (j > i)  iff
         behavior[i] == behavior[j]  AND  j - i >= MIN_GAP (default 3)
         (weight 1 / (j - i), so closer recurrences dominate).

Descriptors (see _compute_descriptors):
  - n_nodes, n_edges, edge_density
  - avg_in_degree, avg_out_degree, max_in_degree, max_out_degree
  - avg_clustering (undirected projection)
  - approx_diameter (longest shortest path, approximate)
  - betweenness_centralization, eigenvector_centralization
  - modularity (Louvain on undirected projection; fallback to label propagation)
  - largest_scc_frac
  - n_simple_cycles_up_to_len_5
  - spectral_radius (largest |eigenvalue| of adjacency)
  - von_neumann_entropy
  - small_world_sigma (against a rewired random graph)

Output:
    data/features/{dataset}_{model}_graph.csv
        columns: item_id, dataset, is_correct, <15 graph descriptors>

Usage:
    PYTHONPATH=. python src/features/graph_features.py \\
        --parsed-glob "data/parsed/*_parsed.jsonl" --output-dir data/features/

    PYTHONPATH=. python src/features/graph_features.py \\
        --parsed data/parsed/math500_qwen7b_parsed.jsonl --output-dir data/features/
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import networkx as nx
except ImportError as e:
    raise RuntimeError("networkx is required for graph_features. "
                       "Install with: pip install networkx") from e


logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG
# =============================================================================

BEHAVIOR_CHARS: list[str] = ["F", "V", "B", "R", "S", "H", "C"]
BEHAVIOR_TO_IDX: dict[str, int] = {b: i for i, b in enumerate(BEHAVIOR_CHARS)}

# Minimum gap between two episodes for a behavior-recurrence edge to form.
# Smaller => denser graph (more short-range repetition captured).
MIN_RECURRENCE_GAP: int = 3

# Cap on simple-cycle enumeration length (graphs can have exponentially many).
MAX_CYCLE_LEN: int = 5

# Approx diameter sampling: cap on source nodes to avoid O(V^3) on long traces.
DIAMETER_SAMPLE_CAP: int = 30

# Size ceilings beyond which descriptors are replaced with neutral values
# rather than computed exactly. These kick in only on pathological long
# traces (typically GPQA / MATH500 with 300+ reasoning episodes) where
# all-pairs shortest-path and betweenness centrality would otherwise eat
# minutes per trace.
MAX_NODES_BETWEENNESS: int = 400      # betweenness_centrality is O(VE)
MAX_NODES_SMALL_WORLD: int = 300      # 3 rewires * all-pairs shortest paths
MAX_EDGES_SHORT_CYCLES: int = 3000    # simple_cycles is exp. in edge count
# Per-trace soft budget for compute_descriptors: if exceeded, log a warning.
SLOW_TRACE_LOG_SEC: float = 2.0

GRAPH_FEATURE_NAMES: list[str] = [
    "g_n_nodes", "g_n_edges", "g_edge_density",
    "g_avg_in_degree", "g_avg_out_degree",
    "g_avg_clustering",
    "g_approx_diameter",
    "g_betweenness_centralization", "g_eigenvector_centralization",
    "g_modularity",
    "g_largest_scc_frac",
    "g_cycles_leq5",
    "g_spectral_radius",
    "g_von_neumann_entropy",
    "g_small_world_sigma",
]


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_trace_graph(episodes: list[dict],
                      min_gap: int = MIN_RECURRENCE_GAP
                      ) -> nx.DiGraph:
    """
    Build a directed trace-graph from a list of episode dicts.

    Each episode must contain:  {"behavior": <char>, "position": <int>,
                                 "token_count": <int>}.
    """
    G: nx.DiGraph = nx.DiGraph()
    L = len(episodes)
    if L == 0:
        return G

    # Nodes
    for i, ep in enumerate(episodes):
        b = ep.get("behavior", "F")
        if b not in BEHAVIOR_TO_IDX:
            b = "F"
        G.add_node(i,
                   behavior=b,
                   behavior_idx=BEHAVIOR_TO_IDX[b],
                   position=i,
                   norm_position=i / max(L - 1, 1),
                   token_count=int(ep.get("token_count", 0)))

    # Temporal chain edges
    for i in range(L - 1):
        G.add_edge(i, i + 1, weight=1.0, kind="temporal")

    # Behavior-recurrence edges
    if L >= min_gap + 1:
        # For each behavior type, link pairs with gap >= min_gap.
        idx_by_beh: dict[str, list[int]] = {}
        for i, ep in enumerate(episodes):
            b = ep.get("behavior", "F")
            idx_by_beh.setdefault(b, []).append(i)
        for _, positions in idx_by_beh.items():
            for a_idx in range(len(positions)):
                for b_idx in range(a_idx + 1, len(positions)):
                    i, j = positions[a_idx], positions[b_idx]
                    if j - i < min_gap:
                        continue
                    # If a temporal edge already exists (shouldn't, gap >= 3), skip
                    if G.has_edge(i, j):
                        continue
                    G.add_edge(i, j,
                               weight=1.0 / (j - i),
                               kind="recurrence")
    return G


# =============================================================================
# DESCRIPTORS
# =============================================================================

def _approx_diameter(G: nx.DiGraph,
                     cap: int = DIAMETER_SAMPLE_CAP) -> float:
    """Approximate diameter on the underlying undirected graph via BFS from
    a uniform sample of nodes. Returns 0.0 if graph has < 2 nodes."""
    U = G.to_undirected()
    nodes = list(U.nodes())
    if len(nodes) < 2:
        return 0.0
    rng = np.random.default_rng(42)
    if len(nodes) > cap:
        sample = rng.choice(nodes, size=cap, replace=False)
    else:
        sample = nodes
    max_d = 0
    for s in sample:
        lengths = nx.single_source_shortest_path_length(U, s)
        if lengths:
            m = max(lengths.values())
            if m > max_d:
                max_d = m
    return float(max_d)


def _centralization(values: list[float]) -> float:
    """
    Freeman-style centralization:  sum(max - v_i)  /  theoretical_max.
    Theoretical max = (N-1) * (max possible value - min possible value);
    for centrality scores in [0, 1], max possible diff per node is 1, so
    we take (N - 1) as the normalizer. Returns 0.0 for N <= 1.
    """
    if not values or len(values) <= 1:
        return 0.0
    m = max(values)
    return float(sum(m - v for v in values) / (len(values) - 1))


def _safe_modularity(U: nx.Graph) -> float:
    """Louvain modularity on the undirected projection; fallback to
    greedy modularity if the community_louvain package isn't present."""
    try:
        # Prefer python-louvain for consistency with literature
        import community as community_louvain  # noqa: F401
        partition = community_louvain.best_partition(U, random_state=42)
        communities: dict[int, list] = {}
        for node, c in partition.items():
            communities.setdefault(c, []).append(node)
        return float(nx.algorithms.community.modularity(
            U, list(communities.values())))
    except Exception:
        try:
            comms = list(nx.algorithms.community.greedy_modularity_communities(U))
            if not comms:
                return 0.0
            return float(nx.algorithms.community.modularity(U, comms))
        except Exception:
            return 0.0


def _spectral_radius(G: nx.DiGraph) -> float:
    """Spectral radius of the *undirected projection*. Directed trace graphs
    are DAGs (time flows forward), so their directed adjacency always has
    all-zero eigenvalues — not informative. The undirected projection
    captures the "graph size" signal we actually want."""
    if G.number_of_nodes() == 0:
        return 0.0
    U = G.to_undirected()
    A = nx.adjacency_matrix(U, weight="weight").astype(float).todense()
    A = np.asarray(A)
    try:
        eigs = np.linalg.eigvalsh(A)  # symmetric -> eigvalsh faster/stable
        return float(np.max(np.abs(eigs)))
    except Exception:
        return 0.0


def _von_neumann_entropy(G: nx.Graph) -> float:
    """Von Neumann graph entropy = Shannon entropy of the normalized-Laplacian
    spectrum. Uses the undirected projection. 0 for trivial graphs."""
    U = G.to_undirected()
    n = U.number_of_nodes()
    if n < 2:
        return 0.0
    try:
        L = nx.normalized_laplacian_matrix(U).astype(float).toarray()
        eigs = np.linalg.eigvalsh(L)
        # Treat small/negative eigs as 0
        eigs = np.clip(eigs, 0.0, None)
        total = float(eigs.sum())
        if total <= 1e-12:
            return 0.0
        p = eigs / total
        p = p[p > 1e-12]
        return float(-np.sum(p * np.log(p)) / np.log(n))
    except Exception:
        return 0.0


def _small_world_sigma(G: nx.Graph, n_rewire_trials: int = 3) -> float:
    """
    Watts-Strogatz small-world sigma: (C / C_rand) / (L / L_rand).
    C = average clustering, L = average shortest path length. Compared to
    a degree-preserving random rewiring of the same graph. We compute on
    the largest connected component of the undirected projection.
    Returns 1.0 as a neutral value on failure.
    """
    U = G.to_undirected()
    if U.number_of_nodes() < 4 or U.number_of_edges() < 2:
        return 1.0
    # Long traces produce dense recurrence graphs where all-pairs shortest
    # path on the largest component dominates wall-time. Cap with a neutral
    # return (sigma = 1 corresponds to "not more clustered than random").
    if U.number_of_nodes() > MAX_NODES_SMALL_WORLD:
        return 1.0
    components = sorted(nx.connected_components(U), key=len, reverse=True)
    H = U.subgraph(components[0]).copy()
    if H.number_of_nodes() < 4:
        return 1.0
    try:
        C = nx.average_clustering(H)
        L_ = nx.average_shortest_path_length(H)
    except Exception:
        return 1.0
    if L_ <= 0:
        return 1.0

    # Random baseline: degree-preserving double-edge swap rewiring
    C_rand_vals, L_rand_vals = [], []
    for _ in range(n_rewire_trials):
        try:
            R = nx.double_edge_swap(H.copy(),
                                    nswap=max(H.number_of_edges(), 1),
                                    max_tries=max(H.number_of_edges() * 5, 5),
                                    seed=42)
            C_rand_vals.append(nx.average_clustering(R))
            L_rand_vals.append(nx.average_shortest_path_length(R))
        except Exception:
            continue
    if not L_rand_vals:
        return 1.0
    C_rand = float(np.mean(C_rand_vals)) if C_rand_vals else C
    L_rand = float(np.mean(L_rand_vals)) if L_rand_vals else L_
    if L_rand <= 0 or C_rand <= 0:
        return 1.0
    sigma = (C / C_rand) / (L_ / L_rand)
    return float(sigma)


def _count_short_cycles(G: nx.DiGraph, max_len: int = MAX_CYCLE_LEN) -> int:
    """Count simple directed cycles up to length max_len. Uses
    simple_cycles on a *bounded* copy: we truncate enumeration at
    max_len to avoid exponential blow-up on dense graphs."""
    # Hard edge-count ceiling: simple_cycles is exponential in edges even
    # with length_bound; on traces with 500+ recurrence edges it can run
    # for minutes. Return 0 with a floor, matching the semantic of "we
    # couldn't enumerate cycles cheaply."
    if G.number_of_edges() > MAX_EDGES_SHORT_CYCLES:
        return 0
    try:
        count = 0
        for cycle in nx.simple_cycles(G, length_bound=max_len):
            count += 1
            if count > 10_000:  # hard stop; graphs don't need exact counts
                break
        return count
    except TypeError:
        # Older networkx: simple_cycles doesn't take length_bound
        count = 0
        try:
            for cycle in nx.simple_cycles(G):
                if len(cycle) <= max_len:
                    count += 1
                if count > 10_000:
                    break
        except Exception:
            return 0
        return count
    except Exception:
        return 0


def compute_descriptors(G: nx.DiGraph) -> dict[str, float]:
    """Return the 15-dim descriptor dict for a single trace graph."""
    n = G.number_of_nodes()
    e = G.number_of_edges()
    if n == 0:
        return {k: 0.0 for k in GRAPH_FEATURE_NAMES}

    edge_density = float(e / (n * (n - 1))) if n > 1 else 0.0

    in_degs = [d for _, d in G.in_degree()]
    out_degs = [d for _, d in G.out_degree()]

    avg_in = float(np.mean(in_degs)) if in_degs else 0.0
    avg_out = float(np.mean(out_degs)) if out_degs else 0.0

    U = G.to_undirected()
    try:
        avg_clust = float(nx.average_clustering(U))
    except Exception:
        avg_clust = 0.0

    approx_diam = _approx_diameter(G)

    # Centrality -> centralization. Betweenness is O(V*E); for very long
    # traces with dense recurrence edges this dominates. Skip above cap.
    if n > MAX_NODES_BETWEENNESS:
        bc_cent = 0.0
    else:
        try:
            bc = nx.betweenness_centrality(U)
            bc_cent = _centralization(list(bc.values()))
        except Exception:
            bc_cent = 0.0

    try:
        # Convergence can fail on small/degenerate graphs
        ec = nx.eigenvector_centrality_numpy(U)
        ec_cent = _centralization(list(ec.values()))
    except Exception:
        ec_cent = 0.0

    modularity = _safe_modularity(U)

    # Largest SCC fraction
    if n > 0:
        largest_scc = max(nx.strongly_connected_components(G), key=len)
        largest_scc_frac = float(len(largest_scc) / n)
    else:
        largest_scc_frac = 0.0

    cycles = _count_short_cycles(G)

    spectral_radius = _spectral_radius(G)
    vne = _von_neumann_entropy(G)
    sw_sigma = _small_world_sigma(G)

    return {
        "g_n_nodes": float(n),
        "g_n_edges": float(e),
        "g_edge_density": edge_density,
        "g_avg_in_degree": avg_in,
        "g_avg_out_degree": avg_out,
        "g_avg_clustering": avg_clust,
        "g_approx_diameter": approx_diam,
        "g_betweenness_centralization": bc_cent,
        "g_eigenvector_centralization": ec_cent,
        "g_modularity": modularity,
        "g_largest_scc_frac": largest_scc_frac,
        "g_cycles_leq5": float(cycles),
        "g_spectral_radius": spectral_radius,
        "g_von_neumann_entropy": vne,
        "g_small_world_sigma": sw_sigma,
    }


# =============================================================================
# PER-FILE PIPELINE
# =============================================================================

def _group_name_from_path(parsed_path: str) -> str:
    base = os.path.basename(parsed_path)
    if base.endswith("_parsed.jsonl"):
        return base[: -len("_parsed.jsonl")]
    return base.replace(".jsonl", "")


def _iter_parsed_records(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_csv_for_file(parsed_path: str, output_dir: str,
                       heartbeat_every: int = 25) -> str:
    """Read one parsed JSONL, build a graph per trace, emit graph-feature CSV.

    Emits a heartbeat log line every `heartbeat_every` records so a stuck
    pathological trace is visible in the logs instead of silently
    consuming minutes of wall-time.
    """
    import time as _time
    group = _group_name_from_path(parsed_path)
    rows = []
    n_total = 0
    n_skipped = 0
    n_slow = 0
    t_group_start = _time.time()
    t_last_hb = t_group_start

    for rec in _iter_parsed_records(parsed_path):
        n_total += 1
        episodes = rec.get("episodes", [])
        if not episodes:
            n_skipped += 1
            continue
        t_rec = _time.time()
        G = build_trace_graph(episodes)
        feats = compute_descriptors(G)
        dt_rec = _time.time() - t_rec
        if dt_rec >= SLOW_TRACE_LOG_SEC:
            n_slow += 1
            logger.info(
                f"  [graph/{group}] slow trace item={rec.get('item_id', '?')} "
                f"L={len(episodes)} n_nodes={G.number_of_nodes()} "
                f"n_edges={G.number_of_edges()} took {dt_rec:.1f}s"
            )
        row = {
            "item_id": rec["item_id"],
            "dataset": group,
            "is_correct": int(rec.get("is_correct", False)),
            **feats,
        }
        rows.append(row)

        if heartbeat_every > 0 and (n_total % heartbeat_every == 0):
            now = _time.time()
            logger.info(
                f"  [graph/{group}] heartbeat n_done={n_total} "
                f"rows={len(rows)} skipped_empty={n_skipped} "
                f"slow(>{SLOW_TRACE_LOG_SEC:.0f}s)={n_slow} "
                f"elapsed={now - t_group_start:.1f}s "
                f"dt_batch={now - t_last_hb:.1f}s"
            )
            t_last_hb = now

    df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, f"{group}_graph.csv")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"  {group}: n={len(df)}/{n_total}  feat_cols={df.shape[1] - 3}  "
                f"skipped={n_skipped}  slow={n_slow}  -> {out_path}")
    return out_path


# =============================================================================
# DRIVER
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parsed", help="Single parsed JSONL file")
    ap.add_argument("--parsed-glob", default="data/parsed/*_parsed.jsonl")
    ap.add_argument("--output-dir", default="data/features/")
    ap.add_argument("--skip-pilot", action="store_true", default=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.parsed:
        paths = [args.parsed]
    else:
        paths = sorted(glob.glob(args.parsed_glob))
        if args.skip_pilot:
            paths = [p for p in paths
                     if not os.path.basename(p).startswith(("pilot_", "_"))
                     and "_sc" not in os.path.basename(p)]
    if not paths:
        logger.error(f"No parsed JSONL files found")
        sys.exit(1)

    for path in paths:
        logger.info(f"Processing {path}")
        try:
            build_csv_for_file(path, args.output_dir)
        except Exception as e:
            logger.exception(f"  Failed on {path}: {e}")


# =============================================================================
# SELF-TEST
# =============================================================================

def _run_self_test():
    print("Running graph_features self-test...")

    # Case 1: empty trace
    G = build_trace_graph([])
    feats = compute_descriptors(G)
    assert feats["g_n_nodes"] == 0
    assert all(isinstance(v, float) for v in feats.values())

    # Case 2: short single-chain
    eps = [{"behavior": "F", "position": i, "token_count": 5} for i in range(5)]
    G = build_trace_graph(eps)
    assert G.number_of_nodes() == 5
    feats = compute_descriptors(G)
    # All same behavior => many recurrence edges (gap>=3 => (0,3),(0,4),(1,4))
    assert G.number_of_edges() >= 4 + 3  # 4 temporal + 3 recurrence
    assert feats["g_n_nodes"] == 5
    assert feats["g_spectral_radius"] > 0

    # Case 3: diverse behaviors
    seq = ["F", "V", "F", "B", "V", "F", "C"]
    eps = [{"behavior": b, "position": i, "token_count": 5}
           for i, b in enumerate(seq)]
    G = build_trace_graph(eps)
    feats = compute_descriptors(G)
    assert feats["g_n_nodes"] == 7
    assert feats["g_edge_density"] > 0
    assert 0 <= feats["g_avg_clustering"] <= 1 + 1e-9
    assert feats["g_approx_diameter"] >= 1

    # Case 4: synthetic discriminability
    import random
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    rng = random.Random(0)
    rows = []
    for i in range(40):
        label = i % 2
        if label == 1:
            # "correct": diverse behaviors, no long cycles
            L = rng.randint(15, 25)
            seq = [rng.choice(["F", "V", "F", "F", "S"]) for _ in range(L)]
            seq[-1] = "C"
        else:
            # "incorrect": many backtracks / hesitations / cycles
            L = rng.randint(25, 40)
            seq = [rng.choice(["F", "B", "H", "V", "B", "H"]) for _ in range(L)]
        eps = [{"behavior": b, "position": k, "token_count": rng.randint(3, 15)}
               for k, b in enumerate(seq)]
        G = build_trace_graph(eps)
        feats = compute_descriptors(G)
        rows.append({"label": label, **feats})

    df = pd.DataFrame(rows)
    y = df["label"].to_numpy(dtype=int)
    X = df.drop(columns=["label"]).to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    lr = LogisticRegression(max_iter=2000).fit(StandardScaler().fit_transform(X), y)
    p = lr.predict_proba(StandardScaler().fit_transform(X))[:, 1]
    auroc = roc_auc_score(y, p)
    print(f"  Synthetic in-sample AUROC: {auroc:.4f}  (diagnostic)")
    assert auroc > 0.7, f"synthetic AUROC unexpectedly low: {auroc:.4f}"

    print("All graph_features tests passed.")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _run_self_test()
    else:
        main()
