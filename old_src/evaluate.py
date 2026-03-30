"""
evaluate.py — Evaluation utilities and selective generation analysis.

Key outputs:
  - Accuracy vs. coverage curves (selective generation)
  - AUROC, AUPRC, ECE for all methods
  - Accuracy@80% and Accuracy@90% coverage
  - Figure generation (matplotlib)

Usage:
  python src/evaluate.py --features data/features/math500.csv \
                          --results data/results/math500_results.json \
                          --plot
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).parent.parent
FIGS_DIR = ROOT / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "g1_token_count", "g1_episode_count",
    "g1_prop_F", "g1_prop_V", "g1_prop_B", "g1_prop_R", "g1_prop_S", "g1_prop_H",
    "g2_backtrack_count", "g2_verification_count", "g2_restart_count",
    "g2_vf_ratio", "g2_backtrack_pos_mean", "g2_first_conclusion_pos",
    "g2_verification_cluster_coeff", "g2_longest_forward_run",
    "g2_transition_entropy", "g2_cycle_count",
    "g3_wait_ratio", "g3_question_mark_count", "g3_negation_count",
    "g3_fourgram_repetition_rate",
]


# ─── Selective generation ─────────────────────────────────────────────────────

def selective_generation_curve(
    y_true: np.ndarray,
    confidence_scores: np.ndarray,
    n_points: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (coverage_levels, accuracy_at_coverage).
    At each coverage level, keep only the top-k% most confident predictions
    and compute accuracy on that subset.
    """
    n = len(y_true)
    sorted_idx = np.argsort(confidence_scores)[::-1]  # descending confidence
    coverages = np.linspace(0.1, 1.0, n_points)
    accuracies = []

    for cov in coverages:
        k = max(1, int(cov * n))
        top_k_idx = sorted_idx[:k]
        acc = y_true[top_k_idx].mean()
        accuracies.append(acc)

    return coverages, np.array(accuracies)


def accuracy_at_coverage(
    y_true: np.ndarray,
    confidence_scores: np.ndarray,
    coverage: float,
) -> float:
    n = len(y_true)
    k = max(1, int(coverage * n))
    top_k_idx = np.argsort(confidence_scores)[::-1][:k]
    return float(y_true[top_k_idx].mean())


# ─── Train RF on full data, return OOB confidence scores ─────────────────────

def train_and_get_oof_scores(
    X: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """
    5-fold OOF predictions from RF, used for selective generation curves.
    Returns confidence scores (predicted P(correct)) for all samples.
    """
    from sklearn.model_selection import StratifiedKFold
    oof_scores = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, val_idx in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X[tr_idx], y[tr_idx])
        oof_scores[val_idx] = clf.predict_proba(X[val_idx])[:, 1]
    return oof_scores


# ─── Plot: selective generation curves ───────────────────────────────────────

def plot_selective_generation(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    dataset_name: str,
    save_path: str | None = None,
):
    """
    curves: {method_name: (coverages, accuracies)}
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    styles = {
        "Ours (RF)":             {"color": "#e41a1c", "lw": 2.5, "ls": "-"},
        "Ours (LR)":             {"color": "#ff7f00", "lw": 1.5, "ls": "--"},
        "Self-consistency":      {"color": "#377eb8", "lw": 2.0, "ls": "-"},
        "Semantic entropy":      {"color": "#4daf4a", "lw": 2.0, "ls": "-"},
        "Verbalized confidence": {"color": "#984ea3", "lw": 1.5, "ls": "-."},
        "Trace length":          {"color": "#a65628", "lw": 1.5, "ls": ":"},
        "Token perplexity":      {"color": "#f781bf", "lw": 1.5, "ls": ":"},
        "Random":                {"color": "#999999", "lw": 1.0, "ls": ":"},
    }

    for name, (cov, acc) in curves.items():
        style = styles.get(name, {"color": "gray", "lw": 1.5, "ls": "-"})
        ax.plot(cov * 100, acc * 100, label=name, **style)

    ax.set_xlabel("Coverage (%)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Selective Generation — {dataset_name}", fontsize=13)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(10, 100)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--results", default=None, help="JSON results from train.py")
    parser.add_argument("--plot", action="store_true", help="Generate and save figures")
    args = parser.parse_args()

    df = pd.read_csv(args.features).dropna()
    dataset_name = Path(args.features).stem
    X = df[FEATURE_COLS].values
    y = df["correct"].values

    print(f"Dataset: {dataset_name} | n={len(df)} | accuracy={y.mean():.1%}")

    # OOF scores from RF
    print("Computing OOF confidence scores (RF)...")
    oof_rf = train_and_get_oof_scores(X, y)

    # Trace length baseline scores
    max_tok = df["g1_token_count"].quantile(0.99)
    trace_len_scores = 1.0 - (df["g1_token_count"].values / max_tok).clip(0, 1)

    print("\n--- Selective generation summary ---")
    for name, scores in [("Ours (RF)", oof_rf), ("Trace length", trace_len_scores)]:
        a80 = accuracy_at_coverage(y, scores, 0.80)
        a90 = accuracy_at_coverage(y, scores, 0.90)
        auroc = roc_auc_score(y, scores)
        print(f"  {name}: AUROC={auroc:.4f}, Acc@80%={a80:.1%}, Acc@90%={a90:.1%}")

    if args.plot:
        curves = {
            "Ours (RF)":  selective_generation_curve(y, oof_rf),
            "Trace length": selective_generation_curve(y, trace_len_scores),
            "Random": selective_generation_curve(y, np.random.rand(len(y))),
        }
        save_path = str(FIGS_DIR / f"selective_gen_{dataset_name}.pdf")
        plot_selective_generation(curves, dataset_name, save_path)


if __name__ == "__main__":
    main()
