#!/usr/bin/env python3
"""
repair_traces.py - Fix reasoning trace extraction in existing trace files.

Problem: HuggingFace Transformers with skip_special_tokens=True strips
the opening <think> tag (it's a special token) but leaves </think>.
This causes extract_think_block() to fail, producing empty traces.

Fix: Re-extract reasoning_trace and answer_text from the full_response
field using the corrected extraction logic that handles missing <think>.

Usage:
    cd reasoning-trace-uq
    PYTHONPATH=. python3 scripts/repair_traces.py

    # Or repair a specific file:
    PYTHONPATH=. python3 scripts/repair_traces.py data/traces/math500_qwen7b_traces.jsonl

This modifies trace files IN PLACE (creates .bak backups first).
After repair, re-run parsing and feature extraction:
    ./scripts/run_parsing.sh all
"""

import json
import os
import re
import sys
import shutil
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_think_block(full_text: str) -> tuple[str, str]:
    """
    Extract reasoning trace and final answer, handling three formats:
      1. <think>TRACE</think>ANSWER  (API output)
      2. TRACE</think>ANSWER         (HF output — <think> stripped)
      3. ANSWER                       (no tags)
    """
    # Case 1: Both tags
    m = re.search(r'<think>(.*?)</think>', full_text, re.DOTALL)
    if m:
        return m.group(1).strip(), full_text[m.end():].strip()

    # Case 2: Only closing tag
    close_idx = full_text.find('</think>')
    if close_idx >= 0:
        trace = full_text[:close_idx].strip()
        answer = full_text[close_idx + len('</think>'):].strip()
        return trace, answer

    # Case 3: No tags
    return "", full_text.strip()


def repair_file(filepath: str) -> dict:
    """
    Repair a single trace JSONL file.

    Returns stats dict with counts of fixed/skipped/error records.
    """
    # Create backup
    backup_path = filepath + ".bak"
    if not os.path.exists(backup_path):
        shutil.copy2(filepath, backup_path)
        logger.info(f"  Backup: {backup_path}")

    # Read all records
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    stats = {"total": len(records), "fixed": 0, "already_ok": 0, "no_response": 0, "errors": 0}

    for record in records:
        # Skip error records
        if "error" in record and "full_response" not in record:
            stats["errors"] += 1
            continue

        full_response = record.get("full_response", "")
        if not full_response:
            stats["no_response"] += 1
            continue

        # Check if trace is already populated
        existing_trace = record.get("reasoning_trace", "")
        if existing_trace and len(existing_trace) > 10:
            stats["already_ok"] += 1
            continue

        # Re-extract from full_response
        reasoning_trace, answer_text = extract_think_block(full_response)

        if reasoning_trace:
            record["reasoning_trace"] = reasoning_trace
            record["answer_text"] = answer_text
            record["trace_length"] = len(reasoning_trace)
            record["trace_token_count"] = len(reasoning_trace.split())

            # Re-score if we have the necessary fields
            if "ground_truth" in record and "answer_type" in record:
                try:
                    from src.generation.scoring import score_answer
                    score_result = score_answer(
                        model_answer_text=answer_text,
                        ground_truth=record["ground_truth"],
                        answer_type=record["answer_type"],
                        answer_extraction=record.get("answer_extraction"),
                    )
                    record["is_correct"] = score_result["is_correct"]
                    record["extracted_answer"] = score_result["extracted_answer"]
                    record["comparison_method"] = score_result["comparison_method"]
                except Exception as e:
                    logger.debug(f"  Re-scoring failed for {record.get('item_id')}: {e}")

            stats["fixed"] += 1
        else:
            stats["no_response"] += 1

    # Write repaired records back
    with open(filepath, 'w') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    return stats


def main():
    # Determine which files to repair
    if len(sys.argv) > 1:
        # Specific files provided
        files = [f for f in sys.argv[1:] if f.endswith('.jsonl')]
    else:
        # Auto-discover all trace files
        traces_dir = "data/traces"
        if not os.path.isdir(traces_dir):
            logger.error(f"Traces directory not found: {traces_dir}")
            logger.error("Run this script from the project root directory.")
            sys.exit(1)

        files = sorted([
            os.path.join(traces_dir, f)
            for f in os.listdir(traces_dir)
            if f.endswith('.jsonl')
        ])

    if not files:
        logger.error("No .jsonl files found to repair.")
        sys.exit(1)

    logger.info(f"Repairing {len(files)} trace files...")
    logger.info("")

    total_stats = {"total": 0, "fixed": 0, "already_ok": 0, "no_response": 0, "errors": 0}

    for filepath in files:
        filename = os.path.basename(filepath)
        logger.info(f"Processing: {filename}")

        stats = repair_file(filepath)

        for key in total_stats:
            total_stats[key] += stats[key]

        logger.info(f"  {stats['fixed']} fixed, {stats['already_ok']} already OK, "
                     f"{stats['no_response']} no trace, {stats['errors']} errors "
                     f"(of {stats['total']} total)")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("REPAIR SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total records:  {total_stats['total']}")
    logger.info(f"  Fixed:          {total_stats['fixed']}")
    logger.info(f"  Already OK:     {total_stats['already_ok']}")
    logger.info(f"  No trace found: {total_stats['no_response']}")
    logger.info(f"  Error records:  {total_stats['errors']}")
    logger.info("")

    if total_stats['fixed'] > 0:
        logger.info("Traces repaired! Now re-run parsing and feature extraction:")
        logger.info("")
        logger.info("  # Delete old parsed/feature files first")
        logger.info("  rm -f data/parsed/*_parsed.jsonl data/features/*_features.csv")
        logger.info("")
        logger.info("  # Re-run the full pipeline")
        logger.info("  ./scripts/run_parsing.sh all")
    else:
        logger.info("No repairs needed — all traces already have content.")


if __name__ == "__main__":
    main()
