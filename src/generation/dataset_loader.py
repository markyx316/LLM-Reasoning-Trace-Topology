"""
dataset_loader.py - Unified dataset loading and preparation.

Loads all evaluation datasets into a common format:
  {
      "item_id": str,          # Unique identifier
      "dataset": str,          # Dataset name
      "problem": str,          # The problem/question text
      "ground_truth": str,     # The correct answer
      "answer_type": str,      # Scoring method
      "answer_extraction": str, # Special extraction (e.g., 'gsm8k')
      "prompt": str,           # The formatted prompt to send to the model
      "metadata": dict,        # Dataset-specific metadata (difficulty, etc.)
  }

Design decisions:
  - We load from HuggingFace datasets when available, falling back to
    local files if network is unavailable.
  - Multiple-choice problems are formatted with labeled choices in the
    prompt so the model sees the full context.
  - Each dataset item gets a deterministic item_id for reproducibility
    and cross-referencing.
"""

import json
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# INDIVIDUAL DATASET LOADERS
# =============================================================================

def load_math500(cache_dir: str = "data/raw") -> list[dict]:
    """
    Load the MATH-500 benchmark (500 competition math problems).

    Source: HuggingFaceH4/MATH-500 (public, no gating)
    Fields: problem, answer, level (int 1-5), subject
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test",
                          cache_dir=cache_dir)
    except Exception as e:
        logger.error(f"Failed to load MATH500 from HuggingFace: {e}")
        raise

    items = []
    for i, row in enumerate(ds):
        # The MATH dataset has fields: 'problem', 'answer', 'level', 'type'
        # Some versions use 'solution' instead of 'answer'
        answer = row.get("answer", row.get("solution", ""))

        item = {
            "item_id": f"math500_{i:04d}",
            "dataset": "math500",
            "problem": row["problem"],
            "ground_truth": answer,
            "answer_type": "exact_match_math",
            "answer_extraction": None,
            "prompt": row["problem"],
            "metadata": {
                "level": row.get("level", "unknown"),
                "subject": row.get("type", row.get("subject", "unknown")),
            },
        }
        items.append(item)

    logger.info(f"Loaded {len(items)} items from MATH500")
    return items


def load_gsm8k(cache_dir: str = "data/raw") -> list[dict]:
    """
    Load GSM8K (Grade School Math 8K) test set.

    Source: openai/gsm8k
    Format: word problem → numeric answer (after #### marker)
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test",
                          cache_dir=cache_dir)
    except Exception as e:
        logger.error(f"Failed to load GSM8K from HuggingFace: {e}")
        raise

    items = []
    for i, row in enumerate(ds):
        item = {
            "item_id": f"gsm8k_{i:04d}",
            "dataset": "gsm8k",
            "problem": row["question"],
            "ground_truth": row["answer"],
            "answer_type": "exact_match_numeric",
            "answer_extraction": "gsm8k",
            "prompt": row["question"],
            "metadata": {},
        }
        items.append(item)

    logger.info(f"Loaded {len(items)} items from GSM8K")
    return items


def _format_mc_choices(choices_data: dict) -> str:
    """Format multiple-choice options for the prompt."""
    if isinstance(choices_data, dict):
        labels = choices_data.get("label", [])
        texts = choices_data.get("text", [])
    elif isinstance(choices_data, list):
        # Some datasets provide a flat list
        labels = [chr(65 + i) for i in range(len(choices_data))]
        texts = choices_data
    else:
        return str(choices_data)

    lines = []
    for label, text in zip(labels, texts):
        lines.append(f"({label}) {text}")
    return "\n".join(lines)


def load_gpqa_diamond(cache_dir: str = "data/raw") -> list[dict]:
    """
    Load GPQA Diamond (Graduate-level science questions).

    Source: Idavidrein/gpqa, gpqa_diamond config
    Format: question + 4 choices → correct answer letter

    IMPORTANT: This is a GATED dataset. Before first use:
      1. Go to https://huggingface.co/datasets/Idavidrein/gpqa
      2. Log in and accept the terms of use
      3. Create a HuggingFace token: https://huggingface.co/settings/tokens
      4. Run: huggingface-cli login
         OR set: export HF_TOKEN=hf_your_token_here

    Fields in the dataset:
      - Question: the question text
      - Correct Answer: the correct answer text
      - Incorrect Answer 1/2/3: three wrong answers
      - Subdomain: the science domain
    """
    import random

    try:
        from datasets import load_dataset
        # Try loading with token from environment
        hf_token = os.environ.get("HF_TOKEN", None)
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train",
                          cache_dir=cache_dir, token=hf_token)
    except Exception as e:
        error_msg = str(e)
        if "gated" in error_msg.lower() or "401" in error_msg or "authorization" in error_msg.lower():
            logger.error(
                "GPQA Diamond is a GATED dataset. To access it:\n"
                "  1. Go to https://huggingface.co/datasets/Idavidrein/gpqa\n"
                "  2. Log in and accept the terms of use\n"
                "  3. Run: huggingface-cli login\n"
                "     OR set: export HF_TOKEN=hf_your_token_here\n"
                "  4. Re-run this script."
            )
        else:
            logger.error(f"Failed to load GPQA Diamond: {e}")
        raise

    items = []
    rng = random.Random(42)  # Deterministic shuffle for reproducibility

    for i, row in enumerate(ds):
        question = row.get("Question", row.get("question", ""))

        # Build choices from the correct + incorrect answer fields
        correct_answer_text = row.get("Correct Answer", "")
        incorrect_1 = row.get("Incorrect Answer 1", "")
        incorrect_2 = row.get("Incorrect Answer 2", "")
        incorrect_3 = row.get("Incorrect Answer 3", "")

        # Assemble choices and shuffle them so the correct answer
        # isn't always in position A
        choices_with_labels = [
            (correct_answer_text, True),
            (incorrect_1, False),
            (incorrect_2, False),
            (incorrect_3, False),
        ]
        # Remove empty choices
        choices_with_labels = [(c, is_corr) for c, is_corr in choices_with_labels if c.strip()]

        rng.shuffle(choices_with_labels)

        # Determine the correct answer letter after shuffling
        correct_letter = None
        choices_list = []
        for j, (choice_text, is_correct) in enumerate(choices_with_labels):
            choices_list.append(choice_text)
            if is_correct:
                correct_letter = chr(65 + j)  # A, B, C, D

        if correct_letter is None:
            logger.warning(f"GPQA item {i}: could not determine correct answer letter")
            correct_letter = "A"  # Fallback

        # Format choices for the prompt
        labels = [chr(65 + j) for j in range(len(choices_list))]
        choices_text = "\n".join(f"({l}) {c}" for l, c in zip(labels, choices_list))

        prompt = (
            f"{question}\n\n"
            f"Choices:\n{choices_text}\n\n"
            f"Please reason step by step and provide your final answer "
            f"as a single letter (A, B, C, or D)."
        )

        item = {
            "item_id": f"gpqa_{i:04d}",
            "dataset": "gpqa_diamond",
            "problem": question,
            "ground_truth": correct_letter,
            "answer_type": "multiple_choice",
            "answer_extraction": None,
            "prompt": prompt,
            "metadata": {
                "subdomain": row.get("Subdomain", row.get("subdomain", "unknown")),
                "choices": choices_list,
                "correct_answer_text": correct_answer_text,
            },
        }
        items.append(item)

    logger.info(f"Loaded {len(items)} items from GPQA Diamond")
    return items


def load_arc_challenge(cache_dir: str = "data/raw") -> list[dict]:
    """
    Load ARC-Challenge (AI2 Reasoning Challenge, hard subset).

    Source: allenai/ai2_arc, ARC-Challenge config
    Format: science question + 3-5 choices → correct answer letter
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test",
                          cache_dir=cache_dir)
    except Exception as e:
        logger.error(f"Failed to load ARC-Challenge from HuggingFace: {e}")
        raise

    items = []
    for i, row in enumerate(ds):
        question = row["question"]
        choices_data = row["choices"]
        answer_key = row["answerKey"]

        # Format choices
        choices_text = _format_mc_choices(choices_data)

        # Normalize answer key: sometimes it's "1","2","3","4" instead of A,B,C,D
        if answer_key in ("1", "2", "3", "4"):
            answer_key = chr(64 + int(answer_key))  # 1->A, 2->B, etc.

        prompt = f"{question}\n\n{choices_text}\n\nPlease reason step by step and provide your final answer as a single letter."

        item = {
            "item_id": f"arc_{i:04d}",
            "dataset": "arc_challenge",
            "problem": question,
            "ground_truth": answer_key,
            "answer_type": "multiple_choice",
            "answer_extraction": None,
            "prompt": prompt,
            "metadata": {
                "choices": choices_data,
            },
        }
        items.append(item)

    logger.info(f"Loaded {len(items)} items from ARC-Challenge")
    return items


# =============================================================================
# UNIFIED LOADER
# =============================================================================

DATASET_LOADERS = {
    "math500": load_math500,
    "gsm8k": load_gsm8k,
    "gpqa_diamond": load_gpqa_diamond,
    "arc_challenge": load_arc_challenge,
}


def load_dataset_items(
    dataset_name: str,
    cache_dir: str = "data/raw",
    limit: Optional[int] = None,
) -> list[dict]:
    """
    Load a dataset by name.

    Args:
        dataset_name: One of 'math500', 'gsm8k', 'gpqa_diamond', 'arc_challenge'
        cache_dir: Directory for caching downloaded datasets.
        limit: If set, return only the first N items (for debugging).

    Returns:
        List of item dicts in unified format.
    """
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_LOADERS.keys())}"
        )

    loader = DATASET_LOADERS[dataset_name]
    items = loader(cache_dir=cache_dir)

    if limit is not None:
        items = items[:limit]
        logger.info(f"Limited to {len(items)} items (debug mode)")

    return items


def load_all_datasets(
    cache_dir: str = "data/raw",
    datasets: Optional[list[str]] = None,
) -> dict[str, list[dict]]:
    """
    Load all (or specified) datasets.

    Args:
        cache_dir: Directory for caching downloaded datasets.
        datasets: If specified, load only these datasets.

    Returns:
        Dict mapping dataset name → list of items.
    """
    if datasets is None:
        datasets = list(DATASET_LOADERS.keys())

    all_data = {}
    for name in datasets:
        try:
            all_data[name] = load_dataset_items(name, cache_dir=cache_dir)
        except Exception as e:
            logger.error(f"Failed to load dataset {name}: {e}")
            all_data[name] = []

    # Summary
    total = sum(len(v) for v in all_data.values())
    logger.info(f"Loaded {total} total items across {len(all_data)} datasets")
    for name, items in all_data.items():
        logger.info(f"  {name}: {len(items)} items")

    return all_data


def save_dataset_items(items: list[dict], output_path: str):
    """Save dataset items to a JSONL file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w') as f:
        for item in items:
            # Make metadata JSON-serializable
            serializable = {k: v for k, v in item.items()}
            f.write(json.dumps(serializable, ensure_ascii=False) + '\n')
    logger.info(f"Saved {len(items)} items to {output_path}")


def load_jsonl(path: str) -> list[dict]:
    """Load items from a JSONL file."""
    items = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# =============================================================================
# CLI / SELF-TEST
# =============================================================================

def run_loader_tests():
    """Quick validation: try loading each dataset and check format."""
    import sys
    print("Testing dataset loaders...")
    print("(This will download datasets from HuggingFace on first run)\n")

    results = {}
    for name in DATASET_LOADERS:
        try:
            items = load_dataset_items(name, cache_dir="data/raw", limit=5)
            # Validate format
            required_fields = [
                "item_id", "dataset", "problem", "ground_truth",
                "answer_type", "prompt", "metadata"
            ]
            for field in required_fields:
                assert field in items[0], f"Missing field: {field}"
            assert len(items[0]["problem"]) > 0, "Empty problem text"
            assert len(items[0]["prompt"]) > 0, "Empty prompt"
            results[name] = f"OK ({len(items)} items loaded)"
            print(f"  {name}: OK")
            print(f"    Sample problem: {items[0]['problem'][:80]}...")
            print(f"    Ground truth:   {items[0]['ground_truth'][:50]}")
            print(f"    Answer type:    {items[0]['answer_type']}")
            print()
        except Exception as e:
            results[name] = f"FAILED: {e}"
            print(f"  {name}: FAILED - {e}\n")

    print("\nSummary:")
    for name, result in results.items():
        print(f"  {name}: {result}")

    all_ok = all("OK" in r for r in results.values())
    if all_ok:
        print("\nAll dataset loaders working correctly.")
    else:
        print("\nWARNING: Some loaders failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_loader_tests()
