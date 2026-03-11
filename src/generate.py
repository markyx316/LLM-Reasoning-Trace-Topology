"""
generate.py — Trace generation pipeline for reasoning models.

Supports three inference backends:
  --mode local       : HuggingFace transformers (needs GPU/MPS; slow on CPU)
  --mode hf-api      : HuggingFace Inference API (free, rate-limited)
  --mode deepseek-api: DeepSeek API (OpenAI-compatible; needs DEEPSEEK_API_KEY)

Outputs one JSONL file per dataset to data/traces/{dataset}.jsonl
Each line: {id, prompt, trace, answer, ground_truth, correct, tokens, dataset}

Usage examples:
  # 50-item pilot on MATH500 via HF API
  python src/generate.py --dataset math500 --mode hf-api --pilot 50

  # Full MATH500 run locally
  python src/generate.py --dataset math500 --mode local

  # Self-consistency samples (8 per item) for MATH500
  python src/generate.py --dataset math500 --mode hf-api --n-samples 8 --suffix sc8
"""

import argparse
import gzip
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
TRACES_DIR = DATA_DIR / "traces"
TRACES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

# ─── Dataset loaders ─────────────────────────────────────────────────────────

def load_dataset_items(dataset_name: str) -> list[dict]:
    """Return list of {id, prompt, ground_truth} dicts."""
    from datasets import load_dataset

    if dataset_name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        return [
            {
                "id": f"math500_{i}",
                "prompt": _math_prompt(row["problem"]),
                "ground_truth": row["answer"],
                "score_type": "math_exact",
            }
            for i, row in enumerate(ds)
        ]

    if dataset_name == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        return [
            {
                "id": f"gsm8k_{i}",
                "prompt": _math_prompt(row["question"]),
                "ground_truth": row["answer"].split("####")[-1].strip(),
                "score_type": "numeric_exact",
            }
            for i, row in enumerate(ds)
        ]

    if dataset_name == "aime24":
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        return [
            {
                "id": f"aime24_{i}",
                "prompt": _math_prompt(row["Problem"]),
                "ground_truth": str(row["Answer"]),
                "score_type": "numeric_exact",
            }
            for i, row in enumerate(ds)
        ]

    if dataset_name == "gpqa":
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        return [
            {
                "id": f"gpqa_{i}",
                "prompt": _mc_prompt(row["Question"], [
                    row["Correct Answer"],
                    row["Incorrect Answer 1"],
                    row["Incorrect Answer 2"],
                    row["Incorrect Answer 3"],
                ]),
                "ground_truth": "A",  # We always put correct answer first then shuffle
                "score_type": "mc_letter",
                "_correct_answer": row["Correct Answer"],
                "_choices": [
                    row["Correct Answer"],
                    row["Incorrect Answer 1"],
                    row["Incorrect Answer 2"],
                    row["Incorrect Answer 3"],
                ],
            }
            for i, row in enumerate(ds)
        ]

    if dataset_name == "arc":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        return [
            {
                "id": f"arc_{i}",
                "prompt": _mc_prompt(row["question"], row["choices"]["text"]),
                "ground_truth": row["answerKey"],
                "score_type": "mc_letter",
            }
            for i, row in enumerate(ds)
        ]

    raise ValueError(f"Unknown dataset: {dataset_name}")


def _math_prompt(problem: str) -> str:
    return (
        f"Solve the following math problem. Show your work step by step.\n\n"
        f"Problem: {problem}\n\n"
        f"Please provide your final answer inside \\boxed{{}}."
    )


def _mc_prompt(question: str, choices: list[str]) -> str:
    letters = "ABCD"
    choices_str = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices[:4]))
    return (
        f"Answer the following multiple choice question.\n\n"
        f"Question: {question}\n\n{choices_str}\n\n"
        f"Respond with a single letter (A, B, C, or D) as your final answer."
    )


# ─── Answer extraction & scoring ─────────────────────────────────────────────

def extract_answer(response_text: str, score_type: str) -> str:
    """Extract the final answer from model output (outside <think> block)."""
    # Strip think block to get only the final answer portion
    final = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

    if score_type in ("math_exact",):
        # Look for \boxed{...}
        m = re.search(r"\\boxed\{([^}]+)\}", final)
        if m:
            return m.group(1).strip()
        # Fallback: last number-like token
        nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", final)
        return nums[-1] if nums else ""

    if score_type == "numeric_exact":
        nums = re.findall(r"-?\d+(?:\.\d+)?", final)
        return nums[-1] if nums else ""

    if score_type == "mc_letter":
        m = re.search(r"\b([A-D])\b", final)
        return m.group(1) if m else ""

    return final.strip()


def score_answer(predicted: str, ground_truth: str, score_type: str) -> bool:
    pred = predicted.strip().lower()
    gt = ground_truth.strip().lower()

    if score_type == "mc_letter":
        return pred == gt

    # Normalize math expressions (basic)
    pred = _normalize_math(pred)
    gt = _normalize_math(gt)
    return pred == gt


def _normalize_math(s: str) -> str:
    s = s.strip()
    # Remove LaTeX formatting
    s = re.sub(r"\\(text|mathrm|mathbf)\{([^}]+)\}", r"\2", s)
    s = re.sub(r"[{}\\ ]", "", s)
    # Normalize fractions: try to convert simple fractions to decimals for comparison
    try:
        if "/" in s and s.count("/") == 1:
            num, den = s.split("/")
            return f"{float(num)/float(den):.6f}"
    except Exception:
        pass
    try:
        return f"{float(s):.6f}"
    except Exception:
        return s


# ─── Trace extraction ─────────────────────────────────────────────────────────

def extract_trace(response_text: str) -> str:
    """Extract content of <think>...</think> block."""
    m = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Some models use different delimiters; fall back to full response
    return response_text.strip()


# ─── Inference backends ───────────────────────────────────────────────────────

def generate_local(prompts: list[str], temperature: float, max_tokens: int) -> list[str]:
    """Local HF transformers inference. Uses CUDA > MPS > CPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[generate] Loading {MODEL_ID} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    outputs = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        outputs.append(text)
    return outputs


def generate_hf_api(prompts: list[str], temperature: float, max_tokens: int) -> list[str]:
    """HuggingFace Inference API (serverless, free tier, rate-limited)."""
    from huggingface_hub import InferenceClient

    hf_token = os.environ.get("HF_TOKEN")
    client = InferenceClient(model=MODEL_ID, token=hf_token)

    outputs = []
    for i, prompt in enumerate(prompts):
        while True:
            try:
                result = client.text_generation(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    return_full_text=False,
                )
                outputs.append(result)
                break
            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    print(f"  [rate limit] sleeping 10s... ({e})")
                    time.sleep(10)
                else:
                    print(f"  [error] item {i}: {e}")
                    outputs.append("")
                    break
    return outputs


def generate_deepseek_api(prompts: list[str], temperature: float, max_tokens: int) -> list[str]:
    """DeepSeek API (OpenAI-compatible). Prompts for API key interactively."""
    import getpass
    from openai import OpenAI

    api_key = os.environ.get("DEEPSEEK_API_KEY") or getpass.getpass("Enter DeepSeek API key: ")
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    outputs = []
    for prompt in prompts:
        while True:
            try:
                resp = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                # deepseek-reasoner puts the thinking in reasoning_content
                thinking = getattr(resp.choices[0].message, "reasoning_content", "") or ""
                answer = resp.choices[0].message.content or ""
                # Reconstruct as a unified response with <think> block
                full = f"<think>{thinking}</think>\n{answer}"
                outputs.append(full)
                break
            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    print(f"  [rate limit] sleeping 15s...")
                    time.sleep(15)
                else:
                    print(f"  [error]: {e}")
                    outputs.append("")
                    break
    return outputs


BACKENDS = {
    "local": generate_local,
    "hf-api": generate_hf_api,
    "deepseek-api": generate_deepseek_api,
}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["math500", "gsm8k", "aime24", "gpqa", "arc"])
    parser.add_argument("--mode", default="hf-api",
                        choices=["local", "hf-api", "deepseek-api"])
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--pilot", type=int, default=None,
                        help="Only process the first N items (for testing)")
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Number of independent samples per item (for self-consistency)")
    parser.add_argument("--suffix", default="",
                        help="Appended to output filename, e.g. 'sc8' -> math500_sc8.jsonl")
    parser.add_argument("--resume", action="store_true",
                        help="Skip items already in the output file")
    args = parser.parse_args()

    items = load_dataset_items(args.dataset)
    if args.pilot:
        items = items[: args.pilot]
        print(f"[pilot] Running {len(items)} items from {args.dataset}")

    suffix = f"_{args.suffix}" if args.suffix else ""
    out_path = TRACES_DIR / f"{args.dataset}{suffix}.jsonl"

    # Resume: load already-completed IDs
    done_ids = set()
    if args.resume and out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"[resume] {len(done_ids)} items already done, skipping.")

    backend = BACKENDS[args.mode]
    total_correct = 0

    with open(out_path, "a") as fout:
        for item in items:
            item_id = item["id"]
            if item_id in done_ids:
                continue

            prompts = [item["prompt"]] * args.n_samples
            responses = backend(prompts, args.temperature, args.max_tokens)

            for sample_idx, response in enumerate(responses):
                trace = extract_trace(response)
                predicted = extract_answer(response, item["score_type"])
                correct = score_answer(predicted, item["ground_truth"], item["score_type"])
                token_count = len(trace.split())  # rough word-level proxy

                record = {
                    "id": item_id if args.n_samples == 1 else f"{item_id}_s{sample_idx}",
                    "dataset": args.dataset,
                    "prompt": item["prompt"],
                    "trace": trace,
                    "answer": predicted,
                    "ground_truth": item["ground_truth"],
                    "correct": correct,
                    "tokens": token_count,
                    "sample_idx": sample_idx,
                }
                fout.write(json.dumps(record) + "\n")
                fout.flush()

                if sample_idx == 0:
                    total_correct += int(correct)

            if args.n_samples == 1:
                status = "CORRECT" if correct else "wrong"
                print(f"  {item_id}: pred={predicted!r} gt={item['ground_truth']!r} [{status}] ({token_count} tok)")

    n = len(items)
    print(f"\nDone. {total_correct}/{n} correct ({100*total_correct/n:.1f}%)")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
