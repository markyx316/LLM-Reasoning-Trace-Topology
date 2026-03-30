"""
parser_evaluation.py - Parser accuracy evaluation.

Provides tools for:
  1. Creating annotation templates for manual labeling
  2. Computing inter-annotator agreement (Cohen's Kappa)
  3. Per-class precision/recall/F1 analysis
  4. Confusion matrix visualization

Workflow:
  1. Run `create_annotation_template()` on a sample of traces
  2. Manually label each sentence in the generated CSV
  3. Run `evaluate_parser_accuracy()` to compare against parser output

Target: κ ≥ 0.65 (substantial agreement)
"""

import csv
import json
import os
import logging
from collections import Counter
from typing import Optional

from src.parsing.taxonomy import BehaviorType, sequence_to_string
from src.parsing.behavior_classifier import parse_trace

logger = logging.getLogger(__name__)


# =============================================================================
# ANNOTATION TEMPLATE CREATION
# =============================================================================

def create_annotation_template(
    traces_path: str,
    output_csv: str,
    n_samples: int = 100,
    per_dataset: int = 20,
    seed: int = 42,
    stratify_by_correctness: bool = True,
):
    """
    Create a CSV template for manual annotation of parser output.

    Generates a CSV with columns:
      - item_id: unique identifier
      - dataset: source dataset
      - is_correct: whether the model's answer was correct
      - sentence_idx: position in the trace
      - sentence_text: the sentence to annotate
      - parser_label: the parser's classification (for comparison)
      - manual_label: EMPTY — fill this in manually
      - notes: EMPTY — for annotator comments

    Args:
        traces_path: Path to JSONL file with generated traces.
        output_csv: Path to write the annotation CSV.
        n_samples: Total number of traces to sample.
        per_dataset: Max samples per dataset (if multiple datasets in file).
        seed: Random seed for sampling.
        stratify_by_correctness: If True, sample equal numbers of
            correct and incorrect traces per dataset.
    """
    import random
    random.seed(seed)

    # Load traces
    traces = []
    with open(traces_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                if record.get("reasoning_trace"):
                    traces.append(record)

    # Group by dataset
    by_dataset = {}
    for t in traces:
        ds = t.get("dataset", "unknown")
        by_dataset.setdefault(ds, []).append(t)

    # Sample
    sampled = []
    for ds, ds_traces in by_dataset.items():
        n = min(per_dataset, len(ds_traces))

        if stratify_by_correctness:
            correct = [t for t in ds_traces if t.get("is_correct", False)]
            incorrect = [t for t in ds_traces if not t.get("is_correct", False)]
            n_correct = min(n // 2, len(correct))
            n_incorrect = min(n - n_correct, len(incorrect))

            sample = (random.sample(correct, n_correct) +
                      random.sample(incorrect, n_incorrect))
        else:
            sample = random.sample(ds_traces, n)

        sampled.extend(sample)

    # Limit to n_samples total
    if len(sampled) > n_samples:
        sampled = random.sample(sampled, n_samples)

    logger.info(f"Sampled {len(sampled)} traces for annotation")

    # Parse each trace and create annotation rows
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    rows = []
    for trace_data in sampled:
        episodes = parse_trace(trace_data["reasoning_trace"])
        for ep in episodes:
            rows.append({
                "item_id": trace_data["item_id"],
                "dataset": trace_data.get("dataset", ""),
                "is_correct": trace_data.get("is_correct", ""),
                "sentence_idx": ep.position,
                "sentence_text": ep.text[:500],  # Truncate very long sentences
                "parser_label": ep.behavior.value,
                "parser_confidence": f"{ep.confidence:.2f}",
                "manual_label": "",   # TO BE FILLED BY ANNOTATOR
                "notes": "",          # ANNOTATOR COMMENTS
            })

    # Write CSV
    fieldnames = [
        "item_id", "dataset", "is_correct", "sentence_idx",
        "sentence_text", "parser_label", "parser_confidence",
        "manual_label", "notes"
    ]

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Annotation template written: {output_csv}")
    logger.info(f"  {len(sampled)} traces → {len(rows)} sentences to annotate")
    logger.info(f"  Valid labels: {', '.join(b.value + '=' + b.display_name for b in BehaviorType)}")


# =============================================================================
# AGREEMENT COMPUTATION
# =============================================================================

def evaluate_parser_accuracy(
    annotation_csv: str,
    output_report: Optional[str] = None,
) -> dict:
    """
    Evaluate parser accuracy against manual annotations.

    Reads the completed annotation CSV (with manual_label filled in)
    and computes agreement metrics.

    Args:
        annotation_csv: Path to the completed annotation CSV.
        output_report: If set, write a detailed report to this path.

    Returns:
        dict with:
          - kappa: Cohen's Kappa score
          - accuracy: Raw agreement percentage
          - per_class: dict of {behavior: {precision, recall, f1}}
          - confusion_matrix: 2D dict
          - n_annotated: number of annotated sentences
    """
    # Read annotations
    parser_labels = []
    manual_labels = []
    skipped = 0

    with open(annotation_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            manual = row.get("manual_label", "").strip().upper()
            parser = row.get("parser_label", "").strip().upper()

            if not manual:
                skipped += 1
                continue

            # Validate labels
            valid_labels = {b.value for b in BehaviorType}
            if manual not in valid_labels:
                logger.warning(f"Invalid manual label '{manual}' for "
                               f"{row.get('item_id', '?')}, sentence {row.get('sentence_idx', '?')}")
                skipped += 1
                continue
            if parser not in valid_labels:
                skipped += 1
                continue

            parser_labels.append(parser)
            manual_labels.append(manual)

    if not parser_labels:
        logger.error("No valid annotations found!")
        return {"error": "No valid annotations"}

    n = len(parser_labels)
    logger.info(f"Evaluating {n} annotations ({skipped} skipped)")

    # --- Raw accuracy ---
    matches = sum(p == m for p, m in zip(parser_labels, manual_labels))
    accuracy = matches / n

    # --- Cohen's Kappa ---
    kappa = _compute_cohens_kappa(parser_labels, manual_labels)

    # --- Per-class metrics ---
    all_labels = sorted(set(parser_labels + manual_labels))
    per_class = {}
    for label in all_labels:
        tp = sum(1 for p, m in zip(parser_labels, manual_labels) if p == label and m == label)
        fp = sum(1 for p, m in zip(parser_labels, manual_labels) if p == label and m != label)
        fn = sum(1 for p, m in zip(parser_labels, manual_labels) if p != label and m == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        bt = BehaviorType(label)
        per_class[label] = {
            "name": bt.display_name,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": sum(1 for m in manual_labels if m == label),
        }

    # --- Confusion matrix ---
    confusion = {}
    for label_true in all_labels:
        confusion[label_true] = {}
        for label_pred in all_labels:
            confusion[label_true][label_pred] = sum(
                1 for p, m in zip(parser_labels, manual_labels)
                if m == label_true and p == label_pred
            )

    results = {
        "kappa": round(kappa, 3),
        "accuracy": round(accuracy, 3),
        "per_class": per_class,
        "confusion_matrix": confusion,
        "n_annotated": n,
        "n_skipped": skipped,
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"Parser Evaluation Results")
    print(f"{'='*60}")
    print(f"  Annotated sentences:  {n}")
    print(f"  Raw accuracy:         {accuracy:.1%}")
    print(f"  Cohen's Kappa:        {kappa:.3f}")
    print(f"    (κ ≥ 0.65 = substantial agreement)")
    print()
    print(f"  Per-class metrics:")
    print(f"  {'Behavior':25s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Support':>8s}")
    print(f"  {'-'*55}")
    for label in all_labels:
        m = per_class[label]
        print(f"  {m['name']:25s} {m['precision']:6.3f} {m['recall']:6.3f} "
              f"{m['f1']:6.3f} {m['support']:8d}")

    # Write report
    if output_report:
        os.makedirs(os.path.dirname(output_report) or ".", exist_ok=True)
        with open(output_report, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Report saved: {output_report}")

    return results


def _compute_cohens_kappa(labels1: list[str], labels2: list[str]) -> float:
    """
    Compute Cohen's Kappa for inter-annotator agreement.

    Kappa accounts for chance agreement, unlike raw accuracy.
    Interpretation:
      κ < 0.20 : slight agreement
      0.21-0.40: fair agreement
      0.41-0.60: moderate agreement
      0.61-0.80: substantial agreement
      0.81-1.00: almost perfect agreement
    """
    n = len(labels1)
    if n == 0:
        return 0.0

    all_labels = sorted(set(labels1 + labels2))

    # Observed agreement
    po = sum(l1 == l2 for l1, l2 in zip(labels1, labels2)) / n

    # Expected agreement (by chance)
    pe = 0.0
    for label in all_labels:
        count1 = sum(1 for l in labels1 if l == label)
        count2 = sum(1 for l in labels2 if l == label)
        pe += (count1 / n) * (count2 / n)

    # Kappa
    if pe == 1.0:
        return 1.0  # Perfect agreement
    return (po - pe) / (1.0 - pe)


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parser evaluation tools")
    subparsers = parser.add_subparsers(dest="command")

    # Create template
    create_cmd = subparsers.add_parser("create", help="Create annotation template")
    create_cmd.add_argument("--traces", required=True, help="Input JSONL traces")
    create_cmd.add_argument("--output", required=True, help="Output CSV path")
    create_cmd.add_argument("--n-samples", type=int, default=100)
    create_cmd.add_argument("--per-dataset", type=int, default=20)

    # Evaluate
    eval_cmd = subparsers.add_parser("evaluate", help="Evaluate parser accuracy")
    eval_cmd.add_argument("--annotations", required=True, help="Completed annotation CSV")
    eval_cmd.add_argument("--report", default=None, help="Output report JSON path")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.command == "create":
        create_annotation_template(
            args.traces, args.output,
            n_samples=args.n_samples,
            per_dataset=args.per_dataset,
        )
    elif args.command == "evaluate":
        evaluate_parser_accuracy(args.annotations, output_report=args.report)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
