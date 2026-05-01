#!/usr/bin/env python3
"""
build_route_a_features.py - Orchestrator that runs the Route-A feature
extractors over every non-pilot parsed JSONL.

By default runs the three *fast-local* families:
    A1  n-gram motifs           -> {group}_ngram.csv
    A3  trace-graph descriptors -> {group}_graph.csv
    A5  inter-event timing      -> {group}_timing.csv

Opt-in for the slower families:
    --with-ph          A4 structural PH  -> {group}_structural_ph.csv
    --with-shapelet    A2 shapelet distance matrix (HPC recommended)
                       -> {group}_shapelet_distmat.npz

A1 depends on a *corpus-wide* vocab manifest (top-K trigrams + rare motifs
across the whole corpus). That manifest is built once up-front and cached at
``data/features/ngram_vocab.json``; every per-dataset CSV then reads the same
manifest to guarantee a consistent feature schema.

Usage:
    # Minimal: A1 + A3 + A5 for every non-pilot parsed JSONL
    PYTHONPATH=. python scripts/build_route_a_features.py

    # Add structural PH (slower, but still CPU-only)
    PYTHONPATH=. python scripts/build_route_a_features.py --with-ph

    # Only re-run the families that failed or are out of date
    PYTHONPATH=. python scripts/build_route_a_features.py --families timing graph
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.ngram_features import (
    build_ngram_vocab, save_vocab, load_vocab,
    build_csv_for_file as build_ngram_csv,
)
from src.features.graph_features import (
    build_csv_for_file as build_graph_csv,
)
from src.features.timing_features import (
    build_csv_for_file as build_timing_csv,
)

logger = logging.getLogger(__name__)


ALL_FAMILIES = ["ngram", "graph", "timing", "structural_ph", "shapelet"]
DEFAULT_LOCAL = ["ngram", "graph", "timing"]


# =============================================================================
# HELPERS
# =============================================================================

def _group_name_from_path(parsed_path: str) -> str:
    base = os.path.basename(parsed_path)
    if base.endswith("_parsed.jsonl"):
        return base[: -len("_parsed.jsonl")]
    return base.replace(".jsonl", "")


def _discover_parsed(parsed_glob: str, skip_pilot: bool = True) -> list[str]:
    paths = sorted(glob.glob(parsed_glob))
    if skip_pilot:
        paths = [
            p for p in paths
            if not os.path.basename(p).startswith(("pilot_", "_"))
            and "_sc" not in os.path.basename(p)
        ]
    return paths


def _csv_exists(output_dir: str, group: str, family: str) -> bool:
    if family == "shapelet":
        return os.path.exists(
            os.path.join(output_dir, f"{group}_shapelet_distmat.npz")
        )
    suffix_map = {
        "ngram": "ngram.csv",
        "graph": "graph.csv",
        "timing": "timing.csv",
        "structural_ph": "structural_ph.csv",
    }
    return os.path.exists(
        os.path.join(output_dir, f"{group}_{suffix_map[family]}")
    )


# =============================================================================
# FAMILY RUNNERS
# =============================================================================

def run_ngram_family(parsed_paths: list[str], output_dir: str,
                     vocab_path: str, force: bool = False):
    if force or not os.path.exists(vocab_path):
        logger.info(f"Building corpus-wide n-gram vocab ({len(parsed_paths)} files)...")
        vocab = build_ngram_vocab(parsed_paths)
        save_vocab(vocab, vocab_path)
    vocab = load_vocab(vocab_path)
    for p in parsed_paths:
        group = _group_name_from_path(p)
        if not force and _csv_exists(output_dir, group, "ngram"):
            logger.info(f"  [ngram] skip {group}  (exists)")
            continue
        t0 = time.time()
        try:
            build_ngram_csv(p, output_dir, vocab)
            logger.info(f"  [ngram] {group}  {time.time()-t0:.1f}s")
        except Exception as e:
            logger.exception(f"  [ngram] FAILED on {group}: {e}")


def run_graph_family(parsed_paths: list[str], output_dir: str,
                     force: bool = False):
    for p in parsed_paths:
        group = _group_name_from_path(p)
        if not force and _csv_exists(output_dir, group, "graph"):
            logger.info(f"  [graph] skip {group}  (exists)")
            continue
        t0 = time.time()
        try:
            build_graph_csv(p, output_dir)
            logger.info(f"  [graph] {group}  {time.time()-t0:.1f}s")
        except Exception as e:
            logger.exception(f"  [graph] FAILED on {group}: {e}")


def run_timing_family(parsed_paths: list[str], output_dir: str,
                      force: bool = False):
    for p in parsed_paths:
        group = _group_name_from_path(p)
        if not force and _csv_exists(output_dir, group, "timing"):
            logger.info(f"  [timing] skip {group}  (exists)")
            continue
        t0 = time.time()
        try:
            build_timing_csv(p, output_dir)
            logger.info(f"  [timing] {group}  {time.time()-t0:.1f}s")
        except Exception as e:
            logger.exception(f"  [timing] FAILED on {group}: {e}")


def run_structural_ph_family(parsed_paths: list[str], output_dir: str,
                             force: bool = False):
    from src.features.structural_ph_features import (
        build_csv_for_file as build_ph_csv,
    )
    for p in parsed_paths:
        group = _group_name_from_path(p)
        if not force and _csv_exists(output_dir, group, "structural_ph"):
            logger.info(f"  [structural_ph] skip {group}  (exists)")
            continue
        t0 = time.time()
        try:
            build_ph_csv(p, output_dir)
            logger.info(f"  [structural_ph] {group}  {time.time()-t0:.1f}s")
        except Exception as e:
            logger.exception(f"  [structural_ph] FAILED on {group}: {e}")


def run_shapelet_family(parsed_paths: list[str], output_dir: str,
                        force: bool = False, max_candidates: int = 2000,
                        k_range: Iterable[int] = range(3, 9)):
    """A2 distance-matrix precomputation. O(N*M*L) — HPC-recommended.

    Emits data/features/{group}_shapelet_distmat.npz per dataset. A separate
    fold-aware evaluator (src/modeling/shapelet_eval.py) then ranks shapelets
    and trains an OOF classifier.
    """
    from src.features.shapelet_features import build_distmat_for_file
    # Flatten k_range (Iterable) to min/max; the downstream function takes
    # scalar k_min/k_max rather than the full iterable. We assert the range
    # is contiguous since build_candidate_pool uses `range(k_min, k_max+1)`.
    _k_list = sorted(set(int(k) for k in k_range))
    if _k_list != list(range(_k_list[0], _k_list[-1] + 1)):
        raise ValueError(f"k_range must be contiguous; got {_k_list}")
    k_min_val, k_max_val = _k_list[0], _k_list[-1]
    for p in parsed_paths:
        group = _group_name_from_path(p)
        if not force and _csv_exists(output_dir, group, "shapelet"):
            logger.info(f"  [shapelet] skip {group}  (exists)")
            continue
        t0 = time.time()
        try:
            build_distmat_for_file(
                p, output_dir,
                k_min=k_min_val,
                k_max=k_max_val,
                max_per_k=max_candidates,
            )
            logger.info(f"  [shapelet] {group}  {time.time()-t0:.1f}s")
        except Exception as e:
            logger.exception(f"  [shapelet] FAILED on {group}: {e}")


# =============================================================================
# DRIVER
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parsed-glob", default="data/parsed/*_parsed.jsonl")
    ap.add_argument("--output-dir", default="data/features/")
    ap.add_argument("--vocab-path", default="data/features/ngram_vocab.json")
    ap.add_argument("--families", nargs="+", default=None,
                    choices=ALL_FAMILIES,
                    help="Explicit families to run.")
    ap.add_argument("--with-ph", action="store_true",
                    help="Include structural PH (A4).")
    ap.add_argument("--with-shapelet", action="store_true",
                    help="Include shapelet distance-matrix (A2); HPC-recommended.")
    ap.add_argument("--force", action="store_true",
                    help="Rebuild even if output files already exist.")
    ap.add_argument("--skip-pilot", action="store_true", default=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    paths = _discover_parsed(args.parsed_glob, skip_pilot=args.skip_pilot)
    if not paths:
        logger.error(f"No parsed JSONL files found at {args.parsed_glob}")
        sys.exit(1)
    logger.info(f"Discovered {len(paths)} parsed files:")
    for p in paths:
        logger.info(f"  {_group_name_from_path(p)}  ({p})")

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine families to run
    if args.families is not None:
        families = args.families
    else:
        families = list(DEFAULT_LOCAL)
        if args.with_ph:
            families.append("structural_ph")
        if args.with_shapelet:
            families.append("shapelet")
    logger.info(f"Running families: {families}")

    t_start = time.time()

    if "ngram" in families:
        logger.info("\n>>> A1 n-gram motif features")
        run_ngram_family(paths, args.output_dir, args.vocab_path, force=args.force)

    if "graph" in families:
        logger.info("\n>>> A3 trace-graph descriptors")
        run_graph_family(paths, args.output_dir, force=args.force)

    if "timing" in families:
        logger.info("\n>>> A5 inter-event timing")
        run_timing_family(paths, args.output_dir, force=args.force)

    if "structural_ph" in families:
        logger.info("\n>>> A4 structural persistent homology")
        run_structural_ph_family(paths, args.output_dir, force=args.force)

    if "shapelet" in families:
        logger.info("\n>>> A2 shapelet distance matrix")
        run_shapelet_family(paths, args.output_dir, force=args.force)

    dt = time.time() - t_start
    logger.info(f"\nDone.  Total time: {dt:.1f}s  "
                f"({dt / max(len(paths), 1):.1f}s per dataset)")
    # Short summary of what now exists
    logger.info("Outputs in " + args.output_dir + ":")
    for p in paths:
        g = _group_name_from_path(p)
        have = []
        for fam in families:
            if _csv_exists(args.output_dir, g, fam):
                have.append(fam)
        logger.info(f"  {g}: {', '.join(have) if have else '(none)'}")


if __name__ == "__main__":
    main()
