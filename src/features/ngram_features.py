"""
ngram_features.py - Behavior n-gram motif features from parsed reasoning traces.

Reads the 7-class behavior sequence (F, V, B, R, S, H, C) from
data/parsed/{dataset}_{model}_parsed.jsonl and emits a CSV of n-gram
motif descriptors per trace.

Feature families:
    - Bigrams (49 = 7x7) — raw count and rate per trace.
    - Trigrams — raw count and rate for the K=50 most-frequent trigrams in
      the full corpus across all dataset/model combos.
    - Position-weighted motifs — for 12 curated motifs, a weighted count
      where later occurrences carry more mass: sum w_i with
      w_i = (pos_i + 1) / L.
    - Rare-motif indicators — K=20 tail trigrams that appear in < 5% of
      traces corpus-wide; binary presence indicator per trace.

Writes:
    data/features/{dataset}_{model}_ngram.csv
        columns: item_id, dataset, is_correct, <~230 n-gram features>
    data/features/ngram_vocab.json
        manifest of the trigram vocabulary (so runs are reproducible
        and the same feature columns appear across datasets).

Self-test (run this file directly) builds a 20-trace synthetic corpus,
exercises the full extractor, and asserts smoke AUROC >= 0.5.

Usage:
    PYTHONPATH=. python src/features/ngram_features.py \\
        --parsed-glob "data/parsed/*_parsed.jsonl" \\
        --output-dir  data/features/

    # Single dataset
    PYTHONPATH=. python src/features/ngram_features.py \\
        --parsed data/parsed/math500_qwen7b_parsed.jsonl \\
        --output-dir data/features/
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG
# =============================================================================

# 7-class taxonomy used by the parsed JSONL files
BEHAVIOR_CHARS: list[str] = ["F", "V", "B", "R", "S", "H", "C"]
BEHAVIOR_SET: set[str] = set(BEHAVIOR_CHARS)

# Top-K trigrams to track (across the full corpus)
TOP_TRIGRAMS_K: int = 50

# Rare-motif indicator: trigrams that appear in < RARE_DOC_FRAC of traces
RARE_DOC_FRAC: float = 0.05
RARE_TRIGRAMS_K: int = 20

# Curated motifs for position-weighted count. These are motifs with
# cognitive-science meaning:
#   BV   - backtrack then verify ("found error, confirming fix")
#   VB   - verify then backtrack ("verification revealed error")
#   BF   - backtrack then continue forward ("recovered")
#   VF   - verify then continue forward ("confirmed, moving on")
#   FC   - forward then conclude ("smooth termination")
#   HB   - hesitate then backtrack ("doubt preceded error")
#   RF   - restart then forward ("new approach engaged")
#   BVF  - backtrack, verify, forward ("error -> fix -> continue")
#   VBV  - verify-backtrack-verify sandwich ("re-verified the fix")
#   FVF  - forward-verify-forward ("mid-stream check")
#   BVB  - backtrack, verify, backtrack ("fix then new error" — bad)
#   HHH  - 3x hesitation in a row ("rumination burst" — bad)
POSITION_WEIGHTED_MOTIFS: list[str] = [
    "BV", "VB", "BF", "VF", "FC", "HB", "RF",
    "BVF", "VBV", "FVF", "BVB", "HHH",
]


# =============================================================================
# PER-TRACE EXTRACTION
# =============================================================================

def _ngrams(seq: str, n: int) -> list[str]:
    """Emit all n-grams in a string sequence, honoring its left-to-right order."""
    if len(seq) < n:
        return []
    return [seq[i:i + n] for i in range(len(seq) - n + 1)]


def _bigram_features(seq: str) -> dict[str, float]:
    """All 49 possible bigram counts + rates."""
    out: dict[str, float] = {}
    counts = Counter(_ngrams(seq, 2))
    L_minus_1 = max(len(seq) - 1, 1)
    for a in BEHAVIOR_CHARS:
        for b in BEHAVIOR_CHARS:
            g = a + b
            c = counts.get(g, 0)
            out[f"ng2_{g}_count"] = float(c)
            out[f"ng2_{g}_rate"] = c / L_minus_1
    return out


def _trigram_features(seq: str, trigram_vocab: list[str]) -> dict[str, float]:
    """Counts and rates for the fixed trigram vocabulary."""
    out: dict[str, float] = {}
    counts = Counter(_ngrams(seq, 3))
    L_minus_2 = max(len(seq) - 2, 1)
    for g in trigram_vocab:
        c = counts.get(g, 0)
        out[f"ng3_{g}_count"] = float(c)
        out[f"ng3_{g}_rate"] = c / L_minus_2
    return out


def _rare_motif_features(seq: str, rare_vocab: list[str]) -> dict[str, float]:
    """Binary indicator: does this trace contain rare trigram X?"""
    present = set(_ngrams(seq, 3))
    return {f"ng3_rare_{g}_present": float(g in present) for g in rare_vocab}


def _position_weighted_features(seq: str,
                                motifs: list[str] = POSITION_WEIGHTED_MOTIFS
                                ) -> dict[str, float]:
    """
    For each motif, sum w_i = (pos_i + 1) / L over its occurrence positions.
    Later occurrences carry more mass. This surfaces the heuristic "late
    backtracking is worse than early backtracking".
    """
    out: dict[str, float] = {}
    L = max(len(seq), 1)
    for m in motifs:
        k = len(m)
        if len(seq) < k:
            out[f"ng_pos_{m}"] = 0.0
            continue
        weighted = 0.0
        for i in range(len(seq) - k + 1):
            if seq[i:i + k] == m:
                weighted += (i + 1) / L
        out[f"ng_pos_{m}"] = weighted
    return out


def extract_ngram_features(behavior_sequence: str,
                           trigram_vocab: list[str],
                           rare_vocab: list[str]) -> dict[str, float]:
    """
    Extract all n-gram motif features for a single trace's behavior sequence.
    """
    seq = "".join(c for c in behavior_sequence if c in BEHAVIOR_SET)
    feats: dict[str, float] = {}
    feats.update(_bigram_features(seq))
    feats.update(_trigram_features(seq, trigram_vocab))
    feats.update(_rare_motif_features(seq, rare_vocab))
    feats.update(_position_weighted_features(seq))
    feats["behavior_seq_len"] = float(len(seq))
    return feats


# =============================================================================
# CORPUS-LEVEL VOCABULARY BUILDING
# =============================================================================

def _iter_parsed_records(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _collect_sequences(parsed_paths: list[str]) -> list[str]:
    """Read every parsed JSONL and pull the behavior_sequence string."""
    sequences: list[str] = []
    for p in parsed_paths:
        for rec in _iter_parsed_records(p):
            seq = rec.get("behavior_sequence", "")
            if isinstance(seq, list):
                seq = "".join(seq)
            seq = "".join(c for c in seq if c in BEHAVIOR_SET)
            if seq:
                sequences.append(seq)
    return sequences


def build_ngram_vocab(parsed_paths: list[str],
                      top_k_trigrams: int = TOP_TRIGRAMS_K,
                      rare_k: int = RARE_TRIGRAMS_K,
                      rare_doc_frac: float = RARE_DOC_FRAC
                      ) -> dict:
    """
    Scan the full corpus (all parsed JSONLs) and compute the trigram
    vocabulary used by downstream feature extractors.

    Returns a manifest dict:
        {
          "trigram_vocab":   [top-K trigrams, sorted by -freq],
          "rare_trigrams":   [K least-frequent trigrams present in < 5% of docs],
          "corpus_size":     N traces,
          "sources":         [filename, ...],
        }
    """
    sequences = _collect_sequences(parsed_paths)
    N = len(sequences)

    # Corpus-wide trigram frequency (total occurrences)
    total_freq: Counter = Counter()
    # Per-document presence (document frequency)
    doc_freq: Counter = Counter()
    for seq in sequences:
        trigrams = _ngrams(seq, 3)
        total_freq.update(trigrams)
        doc_freq.update(set(trigrams))

    # Top-K trigrams by total frequency
    top_trigrams = [g for g, _ in total_freq.most_common(top_k_trigrams)]

    # Rare trigrams: doc_freq < rare_doc_frac * N, sorted by ascending freq
    rare_cutoff = rare_doc_frac * N
    rare_candidates = [(g, f) for g, f in total_freq.items()
                       if doc_freq[g] < rare_cutoff and f > 0]
    rare_candidates.sort(key=lambda x: (x[1], x[0]))
    rare_trigrams = [g for g, _ in rare_candidates[:rare_k]]

    manifest = {
        "trigram_vocab": top_trigrams,
        "rare_trigrams": rare_trigrams,
        "corpus_size": N,
        "sources": [os.path.basename(p) for p in parsed_paths],
        "behavior_chars": BEHAVIOR_CHARS,
        "position_weighted_motifs": POSITION_WEIGHTED_MOTIFS,
    }
    logger.info(f"Vocabulary: {N} traces, {len(top_trigrams)} top trigrams, "
                f"{len(rare_trigrams)} rare trigrams")
    return manifest


def save_vocab(manifest: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_vocab(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# PER-FILE CSV BUILDER
# =============================================================================

def _group_name_from_path(parsed_path: str) -> str:
    """
    Extract canonical `dataset_model` key from a parsed JSONL filename:
        data/parsed/math500_qwen7b_parsed.jsonl -> math500_qwen7b
    """
    base = os.path.basename(parsed_path)
    if base.endswith("_parsed.jsonl"):
        return base[: -len("_parsed.jsonl")]
    return base.replace(".jsonl", "")


def build_csv_for_file(parsed_path: str,
                       output_dir: str,
                       vocab: dict) -> str:
    """
    Read one parsed JSONL and emit a per-dataset ngram feature CSV.

    Returns path to the CSV written.
    """
    group = _group_name_from_path(parsed_path)
    trigram_vocab = vocab["trigram_vocab"]
    rare_vocab = vocab["rare_trigrams"]

    rows = []
    skipped = 0
    for rec in _iter_parsed_records(parsed_path):
        seq = rec.get("behavior_sequence", "")
        if isinstance(seq, list):
            seq = "".join(seq)
        seq = "".join(c for c in seq if c in BEHAVIOR_SET)
        if not seq:
            skipped += 1
            continue
        feats = extract_ngram_features(seq, trigram_vocab, rare_vocab)
        row = {
            "item_id": rec["item_id"],
            "dataset": group,
            "is_correct": int(rec.get("is_correct", False)),
            **feats,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, f"{group}_ngram.csv")
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"  {group}: n={len(df)}  feat_cols={df.shape[1] - 3}  "
                f"skipped_empty={skipped}  -> {out_path}")
    return out_path


# =============================================================================
# DRIVER
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parsed", help="Single parsed JSONL to process")
    ap.add_argument("--parsed-glob",
                    default="data/parsed/*_parsed.jsonl",
                    help="Glob for parsed JSONLs (used to build the vocabulary "
                         "and for per-file extraction unless --parsed is given)")
    ap.add_argument("--output-dir", default="data/features/")
    ap.add_argument("--vocab-path",
                    default="data/features/ngram_vocab.json",
                    help="Path to write/load the corpus vocabulary manifest")
    ap.add_argument("--rebuild-vocab", action="store_true",
                    help="Force rebuild of the vocabulary JSON even if it "
                         "exists on disk")
    ap.add_argument("--skip-pilot", action="store_true", default=True,
                    help="Skip pilot_*, _*, *_sc*.jsonl files (matches the "
                         "rest of the pipeline's exclusions)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Determine vocab source = all non-pilot parsed files
    vocab_source_paths = sorted(glob.glob(args.parsed_glob))
    if args.skip_pilot:
        vocab_source_paths = [p for p in vocab_source_paths
                              if not os.path.basename(p).startswith(("pilot_", "_"))
                              and "_sc" not in os.path.basename(p)]

    if not vocab_source_paths:
        logger.error(f"No parsed JSONL files found matching {args.parsed_glob}")
        sys.exit(1)

    # Build or load vocabulary
    if args.rebuild_vocab or not os.path.exists(args.vocab_path):
        logger.info(f"Building ngram vocabulary from "
                    f"{len(vocab_source_paths)} parsed JSONLs...")
        vocab = build_ngram_vocab(vocab_source_paths)
        save_vocab(vocab, args.vocab_path)
        logger.info(f"Vocabulary saved to {args.vocab_path}")
    else:
        vocab = load_vocab(args.vocab_path)
        logger.info(f"Loaded vocabulary from {args.vocab_path} "
                    f"(corpus={vocab.get('corpus_size')})")

    # Determine input files to process
    if args.parsed:
        process_paths = [args.parsed]
    else:
        process_paths = vocab_source_paths

    # Extract per-file CSVs
    for path in process_paths:
        logger.info(f"Processing {path}")
        try:
            build_csv_for_file(path, args.output_dir, vocab)
        except Exception as e:
            logger.exception(f"  Failed on {path}: {e}")


# =============================================================================
# SELF-TEST
# =============================================================================

def _run_self_test():
    """
    Smoke test: synthesize 20 traces over the 7-class alphabet, run the
    extractor, and assert basic correctness properties.
    """
    print("Running ngram_features self-test...")
    import random
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rng = random.Random(0)

    # Build a synthetic corpus where "correct" traces feature a late BVF motif
    # and "incorrect" ones feature HHH rumination bursts.
    fake = []
    for i in range(20):
        label = i % 2
        L = rng.randint(25, 60)
        body = "".join(rng.choices(BEHAVIOR_CHARS, k=L))
        if label == 1:
            tail = "BVF"
        else:
            tail = "HHH"
        seq = body + tail
        fake.append({"item_id": f"synth_{i:04d}", "behavior_sequence": seq,
                     "is_correct": label})

    # Write a temp parsed JSONL
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        parsed_path = os.path.join(tmp, "synth_test_parsed.jsonl")
        with open(parsed_path, "w") as f:
            for r in fake:
                f.write(json.dumps(r) + "\n")

        vocab = build_ngram_vocab([parsed_path])
        out_path = build_csv_for_file(parsed_path, tmp, vocab)
        df = pd.read_csv(out_path)

        assert len(df) == 20, f"expected 20 rows got {len(df)}"
        assert "item_id" in df.columns and "dataset" in df.columns
        assert "is_correct" in df.columns
        bigram_cols = [c for c in df.columns if c.startswith("ng2_")]
        assert len(bigram_cols) == 7 * 7 * 2, (
            f"expected {7*7*2} bigram cols (49 count + 49 rate), got {len(bigram_cols)}")
        pos_cols = [c for c in df.columns if c.startswith("ng_pos_")]
        assert len(pos_cols) == len(POSITION_WEIGHTED_MOTIFS)
        # BVF should be positive for label=1 traces, HHH for label=0
        by_label = df.groupby("is_correct").mean(numeric_only=True)
        assert by_label.loc[1, "ng_pos_BVF"] > by_label.loc[0, "ng_pos_BVF"], \
            "BVF weighted count should be higher for label=1"
        assert by_label.loc[0, "ng_pos_HHH"] > by_label.loc[1, "ng_pos_HHH"], \
            "HHH weighted count should be higher for label=0"

        # Smoke AUROC
        y = df["is_correct"].to_numpy(dtype=int)
        X_cols = [c for c in df.columns
                  if c not in ("item_id", "dataset", "is_correct")]
        X = df[X_cols].to_numpy(dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        lr = LogisticRegression(max_iter=2000).fit(X, y)
        p = lr.predict_proba(X)[:, 1]
        auroc = roc_auc_score(y, p)
        print(f"  Synthetic AUROC (train==test, diagnostic only): {auroc:.4f}")
        assert auroc > 0.8, f"synthetic AUROC should be high, got {auroc:.4f}"

    print("All ngram_features tests passed.")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _run_self_test()
    else:
        main()
