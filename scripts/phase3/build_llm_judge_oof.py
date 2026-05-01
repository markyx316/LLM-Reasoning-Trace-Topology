"""Phase 3 (T1) — Build an LLM-as-judge OOF.

For every trace in the production dataset, ask an independent frontier LLM
(DeepSeek-V3-chat by default) to read the problem + trace + final answer
and emit a calibrated correctness probability. Save as an OOF `.npz` in
the project's standard contract.

Why this helps
--------------
The Phase-2 ablation showed text encoders + hidden-state probe saturate
the stack. We need an *orthogonal* signal. An independently trained
frontier LLM reading the trace end-to-end is orthogonal to DeBERTa /
RoBERTa by architecture and by training data. Literature on LLM-as-judge
for math verification reports AUROCs in 0.78-0.88 range.

This script
-----------
- Loads every `data/traces/{dataset}_{model}_traces.jsonl` under a
  configurable glob (default: production 8 datasets).
- Builds a judgment prompt per trace.
- Fans out via ThreadPoolExecutor (default 16 workers, rate-limited).
- Parses the judge's response: looks for `Probability:` followed by a
  float in [0, 1]; falls back to keyword matching on CORRECT/INCORRECT.
- Retries transient failures with exponential backoff.
- Writes `results/month3/llm_judge_{judge_model}_oof.npz` with the
  standard OOF schema: item_ids, groups, y_true, oof_prob, oof_fold,
  seed, n_splits.

Note on OOF semantics
---------------------
This is NOT a cross-validated OOF in the usual sense — the judge is
not trained on any of our labels. Every score is equally "out of
sample" because the judge never saw our data in training (modulo
pretraining leakage, which is a universal caveat). For compatibility
with the paired-DeLong machinery we still emit `oof_fold` (set to -1
uniformly) and `n_splits` (set to 1).

Judge-model options (in order of preference)
--------------------------------------------
1. DeepSeek-V3 chat (`deepseek-chat`) — cheap, fast, good.
2. Claude Sonnet via ANTHROPIC_API_KEY (if set).
3. GPT-4o-mini via OPENAI_API_KEY (if set).

All three use OpenAI-compatible endpoints except Claude which uses its
own. This script speaks OpenAI-compatible JSON by default; for Claude
use `--judge-backend anthropic`.

Usage
-----
    # Local, default DeepSeek-V3:
    PYTHONPATH=. python scripts/phase3/build_llm_judge_oof.py \
        --traces-glob 'data/traces/*_traces.jsonl' \
        --output results/month3/llm_judge_deepseek_v3_oof.npz \
        --judge-model deepseek-chat \
        --workers 16

    # HPC (via sbatch): same CLI, but inside scripts/sbatch_phase3_llm_judge.sh
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Light import — just reuse the retry logic stylewise
import urllib.request
import urllib.error


# =========================================================================
# Judge prompt
# =========================================================================

JUDGE_SYSTEM_PROMPT = (
    "You are an expert judge evaluating whether a large language model's "
    "chain-of-thought reasoning arrived at the correct final answer. "
    "You do NOT have access to the ground-truth answer. You must decide "
    "based ONLY on the reasoning trace and the final answer presented by "
    "the model. Be rigorous and calibrated: do not default to 0.5."
)


JUDGE_USER_TEMPLATE = """Problem:
{problem}

Model's reasoning trace:
{trace}

Model's final answer: {answer}

Task:
Judge whether the model's final answer is correct. Follow these rules:

1. Work through the problem briefly yourself to anchor your judgment.
2. Check whether the reasoning is internally consistent and whether the
   final answer follows from it.
3. Common failure modes to watch for: arithmetic errors, misread
   problem, self-contradictions, abandoned reasoning that was then
   re-used, early commitment to a wrong approach that was never revised.

Respond in EXACTLY this format, with nothing else before or after:

Judgment: <CORRECT | INCORRECT | UNCERTAIN>
Probability: <a decimal in [0, 1] — your calibrated probability that the final answer is correct>
Reason: <one short sentence>
"""


# =========================================================================
# Prompt construction
# =========================================================================

def _truncate_middle(text: str, max_chars: int) -> str:
    """Keep the head and tail of a long string; the middle of a reasoning
    trace is usually the least informative for correctness (the opening
    identifies the plan; the tail is where the answer crystallizes)."""
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head - 40
    return text[:head] + "\n\n[... middle truncated ...]\n\n" + text[-tail:]


def build_judgment_prompt(
    problem: str,
    trace: str,
    answer: str,
    max_trace_chars: int = 8000,
) -> str:
    """Build a single judgment prompt, truncating long traces from the
    middle to fit within reasonable token budgets."""
    trace = _truncate_middle(trace or "", max_trace_chars)
    return JUDGE_USER_TEMPLATE.format(
        problem=(problem or "").strip(),
        trace=trace.strip(),
        answer=(answer or "").strip(),
    )


# =========================================================================
# Judge client
# =========================================================================

class OpenAICompatJudge:
    """Minimal OpenAI-compatible chat client. Works for DeepSeek-V3
    (`deepseek-chat`), GPT-4o-mini, and any vLLM-served endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 384,
        timeout: float = 90.0,
        max_retries: int = 5,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

    def judge(self, problem: str, trace: str, answer: str) -> dict:
        url = f"{self.base_url}/v1/chat/completions"
        user_msg = build_judgment_prompt(problem, trace, answer)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    url, data=body, headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                choice = data.get("choices", [{}])[0].get("message", {})
                return {
                    "raw_text": (choice.get("content") or "").strip(),
                    "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": data.get("usage", {}).get(
                        "completion_tokens", 0
                    ),
                }
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503):
                    delay = min(2.0 * (2**attempt), 60.0)
                    time.sleep(delay)
                    continue
                # Non-retryable
                err_body = e.read().decode("utf-8", errors="replace")
                return {"error": f"HTTP {e.code}: {err_body[:300]}"}
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                delay = min(2.0 * (2**attempt), 60.0)
                time.sleep(delay)
                continue
            except Exception as e:  # pragma: no cover
                return {"error": f"{type(e).__name__}: {e}"}
        return {"error": "max retries exceeded"}


class AnthropicJudge:
    """Claude judge via the Anthropic messages API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-5",
        max_tokens: int = 384,
        timeout: float = 90.0,
        max_retries: int = 5,
    ):
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = "https://api.anthropic.com"

    def judge(self, problem: str, trace: str, answer: str) -> dict:
        url = f"{self.base_url}/v1/messages"
        user_msg = build_judgment_prompt(problem, trace, answer)
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": JUDGE_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_msg}],
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    url, data=body, headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                content_blocks = data.get("content", [])
                text = "\n".join(
                    b.get("text", "") for b in content_blocks if b.get("type") == "text"
                ).strip()
                usage = data.get("usage", {})
                return {
                    "raw_text": text,
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                }
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 529):
                    delay = min(2.0 * (2**attempt), 60.0)
                    time.sleep(delay)
                    continue
                err_body = e.read().decode("utf-8", errors="replace")
                return {"error": f"HTTP {e.code}: {err_body[:300]}"}
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(min(2.0 * (2**attempt), 60.0))
                continue
            except Exception as e:  # pragma: no cover
                return {"error": f"{type(e).__name__}: {e}"}
        return {"error": "max retries exceeded"}


# =========================================================================
# Response parsing
# =========================================================================

_PROB_RE = re.compile(
    r"""probability\s*[:=]\s*(?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)""",
    re.IGNORECASE,
)
_JUDGMENT_RE = re.compile(
    r"""judgment\s*[:=]\s*(?P<val>CORRECT|INCORRECT|UNCERTAIN)""",
    re.IGNORECASE,
)


def parse_judge_response(text: str) -> tuple[Optional[float], Optional[str]]:
    """Return (probability, judgment_label).

    probability is pulled from the `Probability:` line if present;
    otherwise falls back to categorical mapping:
      CORRECT -> 0.9, INCORRECT -> 0.1, UNCERTAIN -> 0.5.
    Returns (None, None) if neither can be parsed.
    """
    prob: Optional[float] = None
    m = _PROB_RE.search(text)
    if m:
        try:
            val = float(m.group("val"))
            if 0.0 <= val <= 1.0:
                prob = val
        except ValueError:
            prob = None

    judgment: Optional[str] = None
    m2 = _JUDGMENT_RE.search(text)
    if m2:
        judgment = m2.group("val").upper()

    if prob is None and judgment is not None:
        prob = {"CORRECT": 0.9, "INCORRECT": 0.1, "UNCERTAIN": 0.5}[judgment]

    return prob, judgment


# =========================================================================
# Driver
# =========================================================================

def load_traces(globspec: str) -> list[dict]:
    """Load each line of each file matching globspec into a list of dicts."""
    files = sorted(glob.glob(globspec))
    rows: list[dict] = []
    for path in files:
        # Infer dataset/model from file name, e.g., math500_qwen7b_traces.jsonl
        base = os.path.basename(path).replace("_traces.jsonl", "")
        # group = dataset_model (the project's standard group key)
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                # Filter out pilots / broken rows
                if "is_correct" not in r:
                    continue
                item_id = str(r.get("item_id", ""))
                if not item_id:
                    continue
                rows.append({
                    "group": base,
                    "item_id": item_id,
                    "problem": r.get("problem", ""),
                    "trace": r.get("reasoning_trace", ""),
                    "answer": r.get("answer_text", ""),
                    "y_true": int(bool(r.get("is_correct"))),
                })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--traces-glob", default="data/traces/*_traces.jsonl")
    ap.add_argument(
        "--exclude-pilots", action="store_true", default=True,
        help="Skip files matching pilot_*, dryrun_*, _sc*, or *.bak",
    )
    ap.add_argument(
        "--output", required=True,
        help="Path to save OOF npz, e.g. results/month3/llm_judge_deepseek_v3_oof.npz",
    )
    ap.add_argument(
        "--judge-backend", choices=["openai", "anthropic"], default="openai",
    )
    ap.add_argument("--judge-model", default="deepseek-chat")
    ap.add_argument(
        "--base-url", default=None,
        help="OpenAI-compat base URL. Default reads DEEPSEEK_BASE_URL "
             "(fallback https://api.deepseek.com) for openai backend.",
    )
    ap.add_argument(
        "--api-key-env", default=None,
        help="Env var containing the API key. Default DEEPSEEK_API_KEY "
             "for openai backend, ANTHROPIC_API_KEY for anthropic backend.",
    )
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-trace-chars", type=int, default=8000)
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Dev knob: only process first N traces.",
    )
    ap.add_argument(
        "--cache-path", default=None,
        help="Optional JSONL cache of (group, item_id, raw_text) so reruns "
             "resume cheaply. Default: <output>.raw.jsonl",
    )
    ap.add_argument(
        "--log-level", default="INFO",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger("llm_judge")

    # --- Load env from .env if present ---
    dotenv_path = _ROOT / ".env"
    if dotenv_path.exists():
        for ln in dotenv_path.read_text().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln:
                continue
            k, v = ln.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    # --- Build judge ---
    if args.judge_backend == "openai":
        key_env = args.api_key_env or "DEEPSEEK_API_KEY"
        api_key = os.environ.get(key_env, "")
        if not api_key:
            log.error("Missing %s in env. Set it in .env or export it.", key_env)
            sys.exit(2)
        base_url = (
            args.base_url
            or os.environ.get("DEEPSEEK_BASE_URL")
            or "https://api.deepseek.com"
        )
        judge = OpenAICompatJudge(
            api_key=api_key, base_url=base_url, model=args.judge_model,
        )
        log.info("Judge: OpenAI-compat  base=%s  model=%s", base_url, args.judge_model)
    else:
        key_env = args.api_key_env or "ANTHROPIC_API_KEY"
        api_key = os.environ.get(key_env, "")
        if not api_key:
            log.error("Missing %s in env.", key_env)
            sys.exit(2)
        judge = AnthropicJudge(api_key=api_key, model=args.judge_model)
        log.info("Judge: Anthropic  model=%s", args.judge_model)

    # --- Load traces ---
    rows = load_traces(args.traces_glob)
    if args.exclude_pilots:
        rows = [
            r for r in rows
            if not any(tag in r["group"].lower() for tag in ("pilot", "dryrun", "_sc"))
        ]
    if args.limit:
        rows = rows[: args.limit]
    log.info("Loaded %d traces from %s", len(rows), args.traces_glob)

    # --- Cache ---
    cache_path = Path(args.cache_path or (args.output + ".raw.jsonl"))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[tuple[str, str], str] = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            for ln in f:
                try:
                    j = json.loads(ln)
                    cache[(j["group"], j["item_id"])] = j["raw_text"]
                except json.JSONDecodeError:
                    continue
        log.info("Loaded %d cached judgments from %s", len(cache), cache_path)

    cache_lock = threading.Lock()

    def _task(r: dict) -> dict:
        key = (r["group"], r["item_id"])
        if key in cache:
            raw = cache[key]
        else:
            result = judge.judge(
                problem=r["problem"], trace=r["trace"], answer=r["answer"],
            )
            if "error" in result:
                return {**r, "raw_text": "", "prob": np.nan, "judgment": None,
                        "error": result["error"]}
            raw = result["raw_text"]
            with cache_lock:
                with open(cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "group": r["group"], "item_id": r["item_id"],
                        "raw_text": raw,
                    }) + "\n")
                cache[key] = raw
        prob, judgment = parse_judge_response(raw)
        return {**r, "raw_text": raw, "prob": prob, "judgment": judgment}

    # --- Dispatch ---
    out_rows: list[dict] = []
    t0 = time.time()
    n_done = 0
    n_parse_fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_task, r) for r in rows]
        for fut in as_completed(futures):
            res = fut.result()
            out_rows.append(res)
            n_done += 1
            if res.get("prob") is None:
                n_parse_fail += 1
            if n_done % 50 == 0:
                elapsed = time.time() - t0
                rate = n_done / max(elapsed, 1e-3)
                eta = (len(rows) - n_done) / max(rate, 1e-6)
                log.info(
                    "%d/%d  rate=%.1f/s  parse_fail=%d  ETA=%.0fs",
                    n_done, len(rows), rate, n_parse_fail, eta,
                )

    # --- Save OOF ---
    # Order deterministically
    out_rows.sort(key=lambda r: (r["group"], r["item_id"]))
    item_ids = np.array([r["item_id"] for r in out_rows], dtype=object)
    groups = np.array([r["group"] for r in out_rows], dtype=object)
    y_true = np.array([r["y_true"] for r in out_rows], dtype=np.int32)
    probs = np.array(
        [np.nan if r["prob"] is None else r["prob"] for r in out_rows],
        dtype=np.float64,
    )
    # Fill parse failures with 0.5 (uninformative)
    probs_filled = np.where(np.isnan(probs), 0.5, probs)

    # Standard project schema
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        item_ids=item_ids,
        groups=groups,
        y_true=y_true,
        oof_prob=probs_filled,
        oof_fold=np.full(len(out_rows), -1, dtype=np.int32),
        seed=np.array([0]),
        n_splits=np.array([1]),
    )
    log.info(
        "Wrote %s  n=%d  parse_fail=%d  mean_prob=%.3f",
        out_path, len(out_rows), n_parse_fail, probs_filled.mean(),
    )

    # Quick AUROC sanity
    try:
        from sklearn.metrics import roc_auc_score
        mask = ~np.isnan(probs)
        if mask.sum() > 20:
            auc = roc_auc_score(y_true[mask], probs[mask])
            log.info("Judge AUROC (parsed rows, n=%d): %.4f", mask.sum(), auc)
            # Per-group
            groups_arr = np.asarray([r["group"] for r in out_rows])
            for g in sorted(set(groups_arr.tolist())):
                gm = (groups_arr == g) & mask
                if gm.sum() >= 30:
                    y = y_true[gm]
                    p = probs[gm]
                    if len(set(y)) == 2:
                        log.info(
                            "  %-26s AUROC=%.4f  n=%d  pos=%.2f",
                            g, roc_auc_score(y, p), gm.sum(), y.mean(),
                        )
    except Exception as e:  # pragma: no cover
        log.warning("AUROC sanity check skipped: %s", e)


if __name__ == "__main__":
    main()
