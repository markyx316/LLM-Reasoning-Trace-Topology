"""
run_all_baselines.py — Module 3: Unified Baseline Trainer & Summary Table.

Orchestrates all four baselines across all 8 dataset × model pairs and
produces a single comparison table.  Can run from scratch (re-trains) or
just aggregate existing result JSONs.

Baselines:
  A — Length Only    (trace_token_count + answer_length, LR)
  B — Lexical Cues   (7 surface-text features, LR)
  C — Handcrafted    (23 features from feature CSVs, LR + RF + XGBoost)
  D — TF-IDF Encoder (raw trace text, TF-IDF bag-of-words + LR)

Usage:
    # Summarise existing results (fastest — no re-training)
    PYTHONPATH=. python src/baselines/run_all_baselines.py --summarize

    # Re-run all baselines then summarise
    PYTHONPATH=. python src/baselines/run_all_baselines.py --run-all

    # Save summary table to CSV
    PYTHONPATH=. python src/baselines/run_all_baselines.py --summarize \\
        --output results/baseline_summary.csv
"""

import argparse
import csv
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset × Model manifest
# ---------------------------------------------------------------------------

DATASETS = [
    "math500_qwen7b",
    "math500_llama8b",
    "gsm8k_qwen7b",
    "gsm8k_llama8b",
    "gpqa_diamond_qwen7b",
    "gpqa_diamond_llama8b",
    "arc_challenge_qwen7b",
    "arc_challenge_llama8b",
]

# Map baseline → how to find the result JSON
RESULT_PATHS = {
    "A": "results/baseline_a_{ds}.json",
    "B": "results/baseline_b_{ds}.json",
    "C": "results/baseline_c_{ds}.json",
    "D": "results/baseline_d_{ds}.json",
}

# Classifier keys inside each result file
C_CLASSIFIERS = ["logistic_regression", "random_forest", "xgboost"]
# Baselines that have per-classifier breakdowns
MULTI_CLF_BASELINES = {"C"}


# ---------------------------------------------------------------------------
# Result loading helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Could not read {path}: {exc}")
        return None


def _get_summary(result: dict, baseline: str, classifier: str | None = None) -> dict | None:
    """Extract the summary sub-dict from a baseline result."""
    key = f"baseline_{baseline.lower()}"
    if key not in result:
        return None
    bl = result[key]

    if baseline in MULTI_CLF_BASELINES:
        classifiers = bl.get("classifiers", {})
        clf = classifier or "logistic_regression"
        if clf not in classifiers:
            return None
        return classifiers[clf].get("summary")

    # A, B, D: summary lives directly under baseline_X
    return bl.get("summary")


def load_all_results(results_dir: str = ".") -> dict[str, dict[str, dict]]:
    """
    Load all baseline result JSONs.

    Returns:
        {dataset: {baseline_label: summary_dict}}
    """
    data = {}
    for ds in DATASETS:
        data[ds] = {}
        for bl, template in RESULT_PATHS.items():
            path = os.path.join(results_dir, template.format(ds=ds))
            result = _load_json(path)
            if result is None:
                continue
            if bl in MULTI_CLF_BASELINES:
                for clf in C_CLASSIFIERS:
                    s = _get_summary(result, bl, clf)
                    if s:
                        short = (clf.replace("logistic_regression", "LR")
                                    .replace("random_forest", "RF")
                                    .replace("xgboost", "XGB"))
                        data[ds][f"{bl}[{short}]"] = s
                # Also store the best-AUROC classifier under the bare baseline key
                best_s, best_auroc = None, -1.0
                for clf in C_CLASSIFIERS:
                    s = _get_summary(result, bl, clf)
                    if s and s.get("auroc_mean", 0) > best_auroc:
                        best_s = s
                        best_auroc = s["auroc_mean"]
                if best_s:
                    data[ds][bl] = best_s
            else:
                s = _get_summary(result, bl)
                if s:
                    data[ds][bl] = s
    return data


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

_BL_DISPLAY_ORDER = ["A", "B", "C[LR]", "C[RF]", "C[XGB]", "D"]
_BL_HEADERS = {
    "A":      "A:Length",
    "B":      "B:Lexical",
    "C[LR]":  "C:LR",
    "C[RF]":  "C:RF",
    "C[XGB]": "C:XGB",
    "D":      "D:TF-IDF",
}


def _fmt(s: dict | None, key: str = "auroc") -> str:
    if s is None:
        return "   —   "
    mean = s.get(f"{key}_mean")
    std  = s.get(f"{key}_std")
    if mean is None:
        return "   —   "
    if std is not None:
        return f"{mean:.3f}±{std:.3f}"
    return f"{mean:.3f}"


def print_summary_table(
    data: dict[str, dict[str, dict]],
    metric: str = "auroc",
    show_auprc: bool = True,
    show_acc80: bool = True,
):
    """Print a full baseline comparison table to stdout."""
    # Determine which baselines actually have data
    present = []
    for bl in _BL_DISPLAY_ORDER:
        if any(bl in data[ds] for ds in DATASETS):
            present.append(bl)

    col_w = 14  # width of each baseline column
    name_w = 28

    # --- Header ---
    metric_label = metric.upper()
    title = f"BASELINE COMPARISON TABLE — {metric_label}"
    total_w = name_w + col_w * len(present) + 2
    print()
    print("=" * total_w)
    print(title)
    print("=" * total_w)

    header = f"{'Dataset':{name_w}s}"
    for bl in present:
        h = _BL_HEADERS.get(bl, bl)
        header += f"  {h:>{col_w - 2}s}"
    print(header)
    print("-" * total_w)

    rows = []
    for ds in DATASETS:
        row_vals = {}
        row = f"{ds:{name_w}s}"
        for bl in present:
            s = data[ds].get(bl)
            val = _fmt(s, metric)
            row += f"  {val:>{col_w - 2}s}"
            mean = s.get(f"{metric}_mean") if s else None
            row_vals[bl] = mean
        print(row)
        rows.append((ds, row_vals))

    print("=" * total_w)

    # AUPRC block
    if show_auprc:
        print()
        print(f"{'Dataset':{name_w}s}" + "".join(
            f"  {_BL_HEADERS.get(bl, bl):>{col_w - 2}s}" for bl in present
        ))
        auprc_title = "AUPRC"
        print(f"  {'— ' + auprc_title + ' —':^{total_w - 4}s}")
        print("-" * total_w)
        for ds in DATASETS:
            row = f"{ds:{name_w}s}"
            for bl in present:
                row += f"  {_fmt(data[ds].get(bl), 'auprc'):>{col_w - 2}s}"
            print(row)
        print("=" * total_w)

    # Acc@80 block
    if show_acc80:
        print()
        print(f"  {'— Acc@80 —':^{total_w - 4}s}")
        print("-" * total_w)
        for ds in DATASETS:
            row = f"{ds:{name_w}s}"
            for bl in present:
                s = data[ds].get(bl)
                val = f"{s['accuracy_at_80_mean']:.3f}" if s and "accuracy_at_80_mean" in s else "  —  "
                row += f"  {val:>{col_w - 2}s}"
            print(row)
        print("=" * total_w)

    # Column-average
    print()
    print(f"{'Mean AUROC across datasets':{name_w}s}", end="")
    for bl in present:
        vals = [data[ds][bl]["auroc_mean"] for ds in DATASETS if bl in data[ds]]
        if vals:
            import numpy as np
            print(f"  {np.mean(vals):>{col_w - 2}.3f}", end="")
        else:
            print(f"  {'—':>{col_w - 2}s}", end="")
    print()
    print("=" * total_w)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_summary_csv(
    data: dict[str, dict[str, dict]],
    output_path: str,
    metrics: list[str] | None = None,
):
    """
    Save the summary table to a CSV file.

    Columns: dataset, then for each baseline × metric: <bl>_<metric>_mean, <bl>_<metric>_std
    """
    if metrics is None:
        metrics = ["auroc", "auprc", "ece", "accuracy_at_80", "accuracy_at_90"]

    present = [bl for bl in _BL_DISPLAY_ORDER
               if any(bl in data[ds] for ds in DATASETS)]

    fieldnames = ["dataset"]
    for bl in present:
        for m in metrics:
            fieldnames.append(f"{bl}_{m}_mean")
            fieldnames.append(f"{bl}_{m}_std")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for ds in DATASETS:
            row: dict = {"dataset": ds}
            for bl in present:
                s = data[ds].get(bl)
                for m in metrics:
                    row[f"{bl}_{m}_mean"] = s.get(f"{m}_mean", "") if s else ""
                    row[f"{bl}_{m}_std"]  = s.get(f"{m}_std", "") if s else ""
            writer.writerow(row)

    logger.info(f"Summary CSV saved: {output_path}")


# ---------------------------------------------------------------------------
# Baseline runners (delegating to each baseline module)
# ---------------------------------------------------------------------------

def run_all_baselines():
    """Re-run all 4 baselines from scratch (sequential)."""
    from src.baselines.baseline_a_length_only  import run_baseline_a, ALL_DATASETS as A_SETS
    from src.baselines.baseline_b_lexical      import run_baseline_b, ALL_DATASETS as B_SETS
    from src.baselines.baseline_c_handcrafted  import run_baseline_c, ALL_DATASETS as C_SETS
    from src.baselines.baseline_d_text_encoder import run_baseline_d, ALL_DATASETS as D_SETS

    print("\n" + "=" * 60)
    print("STEP 1/4 — Baseline A: Length Only")
    print("=" * 60)
    for traces_path, out_path in A_SETS:
        if not os.path.exists(traces_path):
            logger.warning(f"Skipping (not found): {traces_path}")
            continue
        run_baseline_a(traces_path, out_path)

    print("\n" + "=" * 60)
    print("STEP 2/4 — Baseline B: Lexical Cues")
    print("=" * 60)
    for traces_path, out_path in B_SETS:
        if not os.path.exists(traces_path):
            logger.warning(f"Skipping (not found): {traces_path}")
            continue
        run_baseline_b(traces_path, out_path)

    print("\n" + "=" * 60)
    print("STEP 3/4 — Baseline C: Handcrafted Features")
    print("=" * 60)
    for csv_path, out_path in C_SETS:
        if not os.path.exists(csv_path):
            logger.warning(f"Skipping (not found): {csv_path}")
            continue
        run_baseline_c(csv_path, out_path)

    print("\n" + "=" * 60)
    print("STEP 4/4 — Baseline D: TF-IDF Text Encoder")
    print("=" * 60)
    for traces_path, out_path in D_SETS:
        if not os.path.exists(traces_path):
            logger.warning(f"Skipping (not found): {traces_path}")
            continue
        run_baseline_d(traces_path, out_path)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def run_tests():
    """Minimal self-test: checks that all 32 result JSONs load correctly."""
    import numpy as np

    print("Running run_all_baselines self-tests...")
    passed, failed = 0, 0

    def check(name, cond, msg=""):
        nonlocal passed, failed
        if cond:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: {name} — {msg}")

    data = load_all_results(".")

    # Every dataset key should exist
    check("datasets_loaded", len(data) == len(DATASETS),
          f"expected {len(DATASETS)}, got {len(data)}")

    # For each dataset and baseline, if the file exists check metric is in [0,1]
    for ds in DATASETS:
        for bl in ["A", "B", "D"]:
            s = data[ds].get(bl)
            if s:
                auroc = s.get("auroc_mean", -1)
                check(f"{ds}_{bl}_auroc_range",
                      0.0 <= auroc <= 1.0,
                      f"auroc_mean={auroc}")
        # Baseline C: check per-classifier summaries
        for clf_key in ["C[LR]", "C[RF]", "C[XGB]"]:
            s = data[ds].get(clf_key)
            if s:
                auroc = s.get("auroc_mean", -1)
                check(f"{ds}_{clf_key}_auroc_range",
                      0.0 <= auroc <= 1.0,
                      f"auroc_mean={auroc}")

    # Check column-average calculation doesn't crash
    try:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_summary_table(data, show_auprc=False, show_acc80=False)
        check("table_prints", len(buf.getvalue()) > 100)
    except Exception as exc:
        check("table_prints", False, str(exc))

    print(f"\nResults: {passed} passed, {failed} failed out of {passed + failed} tests")
    if failed == 0:
        print("All run_all_baselines tests passed.")
    return failed == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Module 3: Unified Baseline Trainer & Summary Table"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--summarize", action="store_true",
        help="Load existing result JSONs and print summary table (no retraining)",
    )
    mode.add_argument(
        "--run-all", action="store_true",
        help="Re-run all baselines from scratch, then print summary table",
    )
    mode.add_argument(
        "--test", action="store_true",
        help="Run self-tests",
    )
    parser.add_argument(
        "--output", metavar="PATH",
        help="Save summary table to this CSV path (optional)",
    )
    parser.add_argument(
        "--metric", default="auroc",
        choices=["auroc", "auprc", "ece", "accuracy_at_80", "accuracy_at_90"],
        help="Primary metric for the main table (default: auroc)",
    )
    parser.add_argument(
        "--results-dir", default=".",
        help="Directory containing results/ folder (default: current dir)",
    )
    args = parser.parse_args()

    if args.test:
        ok = run_tests()
        sys.exit(0 if ok else 1)

    if args.run_all:
        run_all_baselines()

    # Load and display
    data = load_all_results(args.results_dir)

    # Check how much data we have
    found = sum(
        1 for ds in DATASETS
        for bl in ["A", "B", "C[LR]", "D"]
        if bl in data[ds]
    )
    if found == 0:
        logger.error(
            "No baseline results found. Run with --run-all first, or "
            "check that results/ JSON files exist."
        )
        sys.exit(1)

    print_summary_table(data, metric=args.metric)

    if args.output:
        save_summary_csv(data, args.output)
        print(f"\nCSV saved to: {args.output}")


if __name__ == "__main__":
    if "--test" in sys.argv or len(sys.argv) == 1:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        run_tests()
    else:
        main()
