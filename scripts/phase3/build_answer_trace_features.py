"""Phase 3 (T6) — Answer-trace semantic-consistency features.

For each trace, embed (1) each step and (2) the final answer, then
derive scalar features that describe how the trace converges onto the
answer. These features capture "trajectory convergence" dynamics that
no Phase-1/2 feature encoded.

Features produced (per trace):
    ans_step_max_cos     max cosine(answer, step_i) over steps
    ans_step_argmax_idx  normalized index of the argmax step (0..1)
    ans_step_final_cos   cosine(answer, last step)
    ans_step_first_cos   cosine(answer, first step)
    ans_step_slope       OLS slope of cos(answer, step_i) vs i
    ans_step_mean        mean cos over steps
    ans_step_std         std of cos over steps
    ans_step_drift       last - first
    ans_step_peak_pos    normalized position of peak similarity
    ans_step_n_steps     step count (info for regressions)
    ans_step_last5_mean  mean cos over last 5 steps
    ans_step_early_peak  1 if argmax is in first 30% of steps, else 0

Cheap: reuses existing `data/step_embeddings/*.npz` (MiniLM-384d),
embeds the answer text with the SAME MiniLM model, computes cosines.

Usage
-----
    PYTHONPATH=. python scripts/phase3/build_answer_trace_features.py \
        --traces-glob 'data/traces/*_traces.jsonl' \
        --step-emb-dir data/step_embeddings \
        --out-dir data/features

Produces `data/features/{dataset}_{model}_ans_trace.csv`.
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

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return v / n


def compute_features(step_embs: np.ndarray, answer_emb: np.ndarray) -> dict:
    """step_embs: shape (T, D); answer_emb: shape (D,)."""
    T = len(step_embs)
    if T == 0:
        return {k: np.nan for k in (
            "ans_step_max_cos", "ans_step_argmax_idx", "ans_step_final_cos",
            "ans_step_first_cos", "ans_step_slope", "ans_step_mean",
            "ans_step_std", "ans_step_drift", "ans_step_peak_pos",
            "ans_step_last5_mean", "ans_step_early_peak",
        )} | {"ans_step_n_steps": 0}

    s_n = _normalize(step_embs)
    a_n = _normalize(answer_emb[None, :])[0]
    cos = s_n @ a_n  # shape (T,)
    idx = np.arange(T)
    # OLS slope of cos vs idx
    if T > 1:
        m = np.polyfit(idx, cos, 1)[0]
    else:
        m = 0.0
    argmax = int(np.argmax(cos))
    last5 = cos[-5:].mean() if T >= 5 else cos.mean()
    return {
        "ans_step_max_cos":    float(cos.max()),
        "ans_step_argmax_idx": float(argmax / max(T - 1, 1)),
        "ans_step_final_cos":  float(cos[-1]),
        "ans_step_first_cos":  float(cos[0]),
        "ans_step_slope":      float(m),
        "ans_step_mean":       float(cos.mean()),
        "ans_step_std":        float(cos.std()),
        "ans_step_drift":      float(cos[-1] - cos[0]),
        "ans_step_peak_pos":   float(argmax / max(T - 1, 1)),
        "ans_step_last5_mean": float(last5),
        "ans_step_early_peak": int(argmax < 0.3 * T),
        "ans_step_n_steps":    T,
    }


def _load_miniLM():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit(
            "sentence-transformers not installed. In the torch311 env:\n"
            "  pip install sentence-transformers"
        )
    # Same model used to build the step embeddings
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--traces-glob", default="data/traces/*_traces.jsonl")
    ap.add_argument("--step-emb-dir", default="data/step_embeddings")
    ap.add_argument("--out-dir", default="data/features")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("answer_trace")

    step_dir = Path(args.step_emb_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Group -> {item_id -> step_embeddings matrix}
    step_emb_by_group: dict[str, dict[str, np.ndarray]] = {}
    for npz_path in sorted(step_dir.glob("*.npz")):
        name = npz_path.stem  # e.g. math500_qwen7b
        z = np.load(npz_path, allow_pickle=True)
        ids = np.asarray(z["item_ids"]).astype(str)
        embs = np.asarray(z["embeddings"], dtype=object)
        step_emb_by_group[name] = dict(zip(ids.tolist(), embs.tolist()))
        log.info("Loaded step embeddings %s  n=%d", name, len(ids))

    encoder = _load_miniLM()

    trace_files = sorted(glob.glob(args.traces_glob))
    for tf in trace_files:
        base = os.path.basename(tf).replace("_traces.jsonl", "")
        if any(x in base.lower() for x in ("pilot", "dryrun", "_sc", ".bak")):
            continue
        if base not in step_emb_by_group:
            log.warning("No step embeddings for %s — skipping", base)
            continue

        # Load traces + build answer list
        item_ids: list[str] = []
        answers: list[str] = []
        labels: list[int] = []
        with open(tf, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                iid = str(r.get("item_id", ""))
                if not iid:
                    continue
                item_ids.append(iid)
                answers.append((r.get("answer_text") or "")[:2000])
                labels.append(int(bool(r.get("is_correct", 0))))

        log.info("[%s] embedding %d answer strings", base, len(answers))
        ans_embs = encoder.encode(
            answers, batch_size=args.batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=False,
        )

        records = []
        n_missing = 0
        for iid, ans_vec, lbl in zip(item_ids, ans_embs, labels):
            step_mat = step_emb_by_group[base].get(iid)
            if step_mat is None:
                n_missing += 1
                feats = {k: np.nan for k in (
                    "ans_step_max_cos", "ans_step_argmax_idx", "ans_step_final_cos",
                    "ans_step_first_cos", "ans_step_slope", "ans_step_mean",
                    "ans_step_std", "ans_step_drift", "ans_step_peak_pos",
                    "ans_step_last5_mean", "ans_step_early_peak",
                )}
                feats["ans_step_n_steps"] = 0
            else:
                feats = compute_features(np.asarray(step_mat, dtype=np.float32), ans_vec)
            feats["item_id"] = iid
            feats["is_correct"] = int(lbl)
            records.append(feats)
        df = pd.DataFrame.from_records(records)
        # Reorder cols
        meta = ["item_id", "is_correct"]
        feat_cols = [c for c in df.columns if c not in meta]
        df = df[meta + feat_cols]

        out_csv = out_dir / f"{base}_ans_trace.csv"
        df.to_csv(out_csv, index=False)
        log.info("[%s] wrote %s  n=%d  missing=%d", base, out_csv, len(df), n_missing)


if __name__ == "__main__":
    main()
