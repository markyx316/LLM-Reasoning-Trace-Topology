"""
analysis.py - Feature importance, error analysis, and publication figures.

Generates all figures and analysis tables for the paper:
  1. SHAP feature importance analysis
  2. Error analysis (confident-wrong traces)
  3. Selective generation curves (the "money plot")
  4. Feature distribution box plots (correct vs incorrect)
  5. Main results table (AUROC across datasets × methods)
  6. Cross-domain transfer heatmap
  7. Annotated example traces

Usage:
    PYTHONPATH=. python src/analysis/analysis.py \
        --results results/math500_results.json \
        --output paper/figures/
"""

import json
import os
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Matplotlib config for publication-quality figures
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Publication style
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
})

# Color palette (colorblind-friendly)
COLORS = {
    "ours": "#2171B5",           # Blue
    "trace_length": "#6BAED6",   # Light blue
    "verbalized": "#FDAE6B",     # Orange
    "self_consistency": "#E6550D", # Dark orange
    "semantic_entropy": "#74C476", # Green
    "perplexity": "#9E9AC8",     # Purple
}

METHOD_DISPLAY_NAMES = {
    "logistic_regression": "Ours (LogReg)",
    "random_forest": "Ours (RF)",
    "xgboost": "Ours (XGBoost)",
    "trace_length": "Trace Length",
    "verbalized_confidence": "Verbalized Conf.",
    "self_consistency": "Self-Consistency (N=8)",
    "semantic_entropy": "Semantic Entropy (N=8)",
    "perplexity": "Perplexity",
}


# =============================================================================
# 1. SELECTIVE GENERATION CURVES
# =============================================================================

def plot_selective_generation(
    results: dict,
    output_path: str,
    title: str = "Selective Generation: Accuracy vs. Coverage",
):
    """
    Plot accuracy vs. coverage curves for all methods.

    This is the "money plot" — it shows the practical benefit of each
    UQ method: by abstaining on uncertain predictions, how much can
    you improve accuracy?

    Args:
        results: Results dict from train_and_evaluate.py (must contain
                 cross_validation and/or baseline predictions).
        output_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    cv_results = results.get("cross_validation", {})

    for method_name, method_result in cv_results.items():
        preds = method_result.get("predictions", {})
        if not preds:
            continue

        y_true = np.array(preds["y_true"])
        y_prob = np.array(preds["y_prob"])

        # Compute selective generation curve
        sorted_idx = np.argsort(y_prob)[::-1]
        y_sorted = y_true[sorted_idx]

        coverages = np.linspace(0.05, 1.0, 100)
        accuracies = []
        for cov in coverages:
            n_select = max(int(cov * len(y_true)), 1)
            accuracies.append(y_sorted[:n_select].mean())

        display_name = METHOD_DISPLAY_NAMES.get(method_name, method_name)
        color = COLORS.get("ours", "#2171B5")

        ax.plot(coverages, accuracies, label=display_name,
                linewidth=2, color=color)

    # Add base accuracy line
    base_acc = results.get("base_accuracy", 0.5)
    ax.axhline(y=base_acc, color="gray", linestyle="--", linewidth=1,
               label=f"Base accuracy ({base_acc:.1%})")

    ax.set_xlabel("Coverage (fraction of questions answered)")
    ax.set_ylabel("Accuracy on answered questions")
    ax.set_title(title)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Selective generation plot saved: {output_path}")


# =============================================================================
# 2. FEATURE IMPORTANCE
# =============================================================================

def plot_feature_importance(
    results: dict,
    output_path: str,
    classifier: str = "random_forest",
    top_n: int = 15,
):
    """
    Plot horizontal bar chart of feature importance.

    Uses SHAP values if available, otherwise falls back to
    model-native feature importance (Gini for RF, coefficients for LogReg).
    """
    cv_results = results.get("cross_validation", {})
    clf_result = cv_results.get(classifier, {})
    importance = clf_result.get("feature_importance", [])

    if not importance:
        logger.warning(f"No feature importance data for {classifier}")
        return

    # Take top N
    importance = importance[:top_n]

    fig, ax = plt.subplots(figsize=(8, 6))

    names = [fi["feature"] for fi in reversed(importance)]
    values = [fi["importance"] for fi in reversed(importance)]

    # Color by feature group
    from src.features.feature_pipeline import get_feature_groups
    groups = get_feature_groups()
    group_colors = {
        "group1_length": "#4292C6",     # Blue
        "group2_structural": "#E6550D", # Orange
        "group3_meta": "#41AB5D",       # Green
    }

    colors = []
    for name in names:
        color = "#999999"
        for gname, gfeatures in groups.items():
            if name in gfeatures:
                color = group_colors.get(gname, "#999999")
                break
        colors.append(color)

    bars = ax.barh(range(len(names)), values, color=colors, edgecolor="white")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Features ({classifier.replace('_', ' ').title()})")

    # Legend for feature groups
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4292C6", label="Length & Proportion"),
        Patch(facecolor="#E6550D", label="Structural / Topological"),
        Patch(facecolor="#41AB5D", label="Content-Free Meta"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Feature importance plot saved: {output_path}")


# =============================================================================
# 3. FEATURE DISTRIBUTIONS
# =============================================================================

def plot_feature_distributions(
    features_csv: str,
    output_path: str,
    features_to_plot: Optional[list[str]] = None,
):
    """
    Box plots showing feature distributions for correct vs incorrect traces.

    This figure visually demonstrates which features carry signal.
    """
    df = pd.read_csv(features_csv)

    if features_to_plot is None:
        # Default: the most informative features
        features_to_plot = [
            "total_tokens", "backtrack_count", "verification_count",
            "vf_ratio", "transition_entropy", "cycle_count",
            "bt_position_mean", "wait_ratio", "negation_count",
        ]

    # Filter to available features
    features_to_plot = [f for f in features_to_plot if f in df.columns]
    n_features = len(features_to_plot)

    if n_features == 0:
        logger.warning("No features to plot")
        return

    cols = min(3, n_features)
    rows = (n_features + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, feat in enumerate(features_to_plot):
        ax = axes[i]

        correct = df[df["is_correct"] == 1][feat].dropna()
        incorrect = df[df["is_correct"] == 0][feat].dropna()

        bp = ax.boxplot(
            [correct, incorrect],
            labels=["Correct", "Incorrect"],
            patch_artist=True,
            widths=0.6,
        )

        bp["boxes"][0].set_facecolor("#4292C6")
        bp["boxes"][1].set_facecolor("#E6550D")
        bp["boxes"][0].set_alpha(0.7)
        bp["boxes"][1].set_alpha(0.7)

        ax.set_title(feat.replace("_", " ").title(), fontsize=10)
        ax.tick_params(axis="x", labelsize=9)

    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Feature Distributions: Correct vs Incorrect Traces",
                 fontsize=13, fontweight="bold")

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Feature distribution plot saved: {output_path}")


# =============================================================================
# 4. ABLATION STUDY CHART
# =============================================================================

def plot_ablation(results: dict, output_path: str):
    """
    Bar chart showing AUROC contribution of each feature group.
    """
    ablation = results.get("ablation", [])
    if not ablation:
        logger.warning("No ablation data")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))

    names = []
    aurocs = []
    stds = []
    colors_list = []

    color_map = {
        "group1_only": "#4292C6",
        "group2_only": "#E6550D",
        "group3_only": "#41AB5D",
        "group2_plus_group3": "#756BB1",
        "all_features": "#2171B5",
    }

    display_map = {
        "group1_only": "Length &\nProportion",
        "group2_only": "Structural /\nTopological",
        "group3_only": "Content-Free\nMeta",
        "group2_plus_group3": "Structural +\nMeta",
        "all_features": "All\nFeatures",
    }

    for ab in ablation:
        names.append(display_map.get(ab["variant"], ab["variant"]))
        aurocs.append(ab["auroc_mean"])
        stds.append(ab["auroc_std"])
        colors_list.append(color_map.get(ab["variant"], "#999999"))

    x = range(len(names))
    bars = ax.bar(x, aurocs, yerr=stds, capsize=4,
                  color=colors_list, edgecolor="white", linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("AUROC")
    ax.set_title("Ablation Study: Feature Group Contributions")
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8,
               label="Random chance")
    ax.set_ylim(0.4, 1.0)

    # Add value labels on bars
    for bar, val, std in zip(bars, aurocs, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Ablation plot saved: {output_path}")


# =============================================================================
# 5. MAIN RESULTS TABLE (LaTeX)
# =============================================================================

def generate_results_latex(
    result_files: list[str],
    output_path: str,
):
    """
    Generate the main results table in LaTeX format.

    Columns: Method
    Rows: Dataset × Metric (AUROC, AUPRC)
    """
    all_results = {}
    for fpath in result_files:
        with open(fpath) as f:
            data = json.load(f)
        dataset = data.get("dataset", os.path.basename(fpath))
        all_results[dataset] = data

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Main results: AUROC for correctness prediction across datasets and methods.}")
    lines.append(r"\label{tab:main_results}")
    lines.append(r"\small")

    # Get all methods
    methods = set()
    for data in all_results.values():
        cv = data.get("cross_validation", {})
        methods.update(cv.keys())
    methods = sorted(methods)

    n_cols = len(methods) + 1
    lines.append(r"\begin{tabular}{l" + "c" * len(methods) + "}")
    lines.append(r"\toprule")

    # Header
    header = "Dataset"
    for m in methods:
        header += f" & {METHOD_DISPLAY_NAMES.get(m, m)}"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    # Data rows
    for dataset_name, data in all_results.items():
        cv = data.get("cross_validation", {})
        row = dataset_name.replace("_features.csv", "").replace("_", r"\_")

        for m in methods:
            if m in cv:
                s = cv[m].get("summary", {})
                auroc = s.get("auroc_mean", 0)
                std = s.get("auroc_std", 0)
                row += f" & {auroc:.3f}$\\pm${std:.3f}"
            else:
                row += " & --"

        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(latex)

    logger.info(f"LaTeX table saved: {output_path}")
    return latex


# =============================================================================
# 6. SINGLE-FEATURE AUROC RANKING
# =============================================================================

def plot_single_feature_aurocs(results: dict, output_path: str, top_n: int = 15):
    """
    Bar chart ranking individual features by their standalone AUROC.

    This shows the marginal discriminative power of each feature.
    """
    sf = results.get("single_feature_aurocs", [])
    if not sf:
        return

    sf = sf[:top_n]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    names = [s["feature"].replace("_", " ").title() for s in reversed(sf)]
    aurocs = [s["auroc"] for s in reversed(sf)]

    from src.features.feature_pipeline import get_feature_groups
    groups = get_feature_groups()
    group_colors = {
        "group1_length": "#4292C6",
        "group2_structural": "#E6550D",
        "group3_meta": "#41AB5D",
    }

    colors = []
    for s in reversed(sf):
        fname = s["feature"]
        color = "#999999"
        for gname, gfeatures in groups.items():
            if fname in gfeatures:
                color = group_colors.get(gname, "#999999")
                break
        colors.append(color)

    ax.barh(range(len(names)), aurocs, color=colors, edgecolor="white")
    ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8,
               label="Random (0.5)")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("AUROC (single feature)")
    ax.set_title(f"Top {top_n} Individual Features by AUROC")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4292C6", label="Length & Proportion"),
        Patch(facecolor="#E6550D", label="Structural / Topological"),
        Patch(facecolor="#41AB5D", label="Content-Free Meta"),
        plt.Line2D([0], [0], color="gray", linestyle="--", label="Random (0.5)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Single feature AUROC plot saved: {output_path}")


# =============================================================================
# 7. GENERATE ALL FIGURES
# =============================================================================

def generate_all_figures(
    results_path: str,
    features_path: Optional[str] = None,
    output_dir: str = "paper/figures",
):
    """
    Generate all publication figures from a results JSON file.

    Args:
        results_path: Path to results JSON from train_and_evaluate.py.
        features_path: Path to features CSV (for distribution plots).
        output_dir: Directory to save figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(results_path) as f:
        results = json.load(f)

    dataset = results.get("dataset", "unknown").replace("_features.csv", "")

    # 1. Selective generation curves
    plot_selective_generation(
        results,
        os.path.join(output_dir, f"{dataset}_selective_generation.png"),
        title=f"Selective Generation: {dataset}",
    )

    # 2. Feature importance
    plot_feature_importance(
        results,
        os.path.join(output_dir, f"{dataset}_feature_importance.png"),
    )

    # 3. Ablation study
    plot_ablation(
        results,
        os.path.join(output_dir, f"{dataset}_ablation.png"),
    )

    # 4. Single-feature AUROCs
    plot_single_feature_aurocs(
        results,
        os.path.join(output_dir, f"{dataset}_single_feature_aurocs.png"),
    )

    # 5. Feature distributions (requires features CSV)
    if features_path and os.path.exists(features_path):
        plot_feature_distributions(
            features_path,
            os.path.join(output_dir, f"{dataset}_feature_distributions.png"),
        )

    logger.info(f"\nAll figures generated in: {output_dir}")


# =============================================================================
# SELF-TEST
# =============================================================================

def run_analysis_tests():
    """Test analysis with synthetic results."""
    print("Running analysis self-tests...")
    import tempfile

    np.random.seed(42)

    # Create synthetic results matching the output format of train_and_evaluate.py
    n = 100
    y_true = np.random.binomial(1, 0.6, n)
    y_prob_rf = np.clip(y_true + np.random.randn(n) * 0.3, 0, 1)

    from src.features.feature_pipeline import get_feature_names
    feature_names = get_feature_names()

    synthetic_results = {
        "dataset": "synthetic_test",
        "n_samples": n,
        "n_correct": int(y_true.sum()),
        "n_incorrect": int((1 - y_true).sum()),
        "base_accuracy": float(y_true.mean()),
        "cross_validation": {
            "random_forest": {
                "summary": {
                    "auroc_mean": 0.78, "auroc_std": 0.05,
                    "auprc_mean": 0.82, "auprc_std": 0.04,
                    "ece_mean": 0.08, "ece_std": 0.02,
                    "accuracy_at_80_mean": 0.85,
                    "accuracy_at_90_mean": 0.80,
                },
                "predictions": {
                    "y_true": y_true.tolist(),
                    "y_prob": y_prob_rf.tolist(),
                    "indices": list(range(n)),
                },
                "feature_importance": [
                    {"feature": fn, "importance": float(np.random.rand())}
                    for fn in feature_names
                ],
            },
        },
        "ablation": [
            {"variant": "group1_only", "auroc_mean": 0.72, "auroc_std": 0.06, "n_features": 9},
            {"variant": "group2_only", "auroc_mean": 0.76, "auroc_std": 0.05, "n_features": 10},
            {"variant": "group3_only", "auroc_mean": 0.55, "auroc_std": 0.08, "n_features": 4},
            {"variant": "group2_plus_group3", "auroc_mean": 0.77, "auroc_std": 0.04, "n_features": 14},
            {"variant": "all_features", "auroc_mean": 0.78, "auroc_std": 0.05, "n_features": 23},
        ],
        "single_feature_aurocs": [
            {"feature": fn, "auroc": float(0.5 + np.random.rand() * 0.3)}
            for fn in feature_names
        ],
    }

    # Sort single feature aurocs
    synthetic_results["single_feature_aurocs"].sort(
        key=lambda x: x["auroc"], reverse=True
    )

    tests_passed = 0
    tests_failed = 0

    def check(name, condition, msg=""):
        nonlocal tests_passed, tests_failed
        if condition:
            tests_passed += 1
        else:
            tests_failed += 1
            print(f"  FAIL: {name}: {msg}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save synthetic results
        results_path = os.path.join(tmpdir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(synthetic_results, f)

        # Generate all figures
        fig_dir = os.path.join(tmpdir, "figures")
        try:
            generate_all_figures(results_path, output_dir=fig_dir)

            # Check figures were created
            expected_files = [
                "synthetic_test_selective_generation.png",
                "synthetic_test_feature_importance.png",
                "synthetic_test_ablation.png",
                "synthetic_test_single_feature_aurocs.png",
            ]

            for fname in expected_files:
                fpath = os.path.join(fig_dir, fname)
                exists = os.path.exists(fpath)
                check(f"figure_{fname}", exists, f"not created")
                if exists:
                    size = os.path.getsize(fpath)
                    check(f"figure_{fname}_size", size > 1000,
                          f"suspiciously small: {size} bytes")

        except Exception as e:
            check("figure_generation", False, f"exception: {e}")

        # Test LaTeX table generation
        try:
            latex_path = os.path.join(tmpdir, "table.tex")
            latex = generate_results_latex([results_path], latex_path)
            check("latex_generated", os.path.exists(latex_path))
            check("latex_has_tabular", r"\begin{tabular}" in latex)
            check("latex_has_auroc", "0.78" in latex)
        except Exception as e:
            check("latex_generation", False, f"exception: {e}")

    print(f"\nResults: {tests_passed} passed, {tests_failed} failed "
          f"out of {tests_passed + tests_failed} tests")
    if tests_failed == 0:
        print("All analysis tests passed.")
    return tests_failed == 0


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate analysis figures")
    parser.add_argument("--results", required=True, help="Results JSON path")
    parser.add_argument("--features", default=None, help="Features CSV path")
    parser.add_argument("--output", default="paper/figures", help="Output dir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    generate_all_figures(args.results, args.features, args.output)


if __name__ == "__main__":
    if "--results" in " ".join(os.sys.argv):
        main()
    else:
        logging.basicConfig(level=logging.INFO)
        run_analysis_tests()
