"""
train_and_evaluate.py - Model training, evaluation, and comparison pipeline.

This module handles the complete experimental pipeline:

  1. Train classifiers (LogReg, RF, XGBoost) via stratified 5-fold CV
  2. Compute evaluation metrics (AUROC, AUPRC, ECE)
  3. Compare against all baselines
  4. Run ablation studies (feature group contributions)
  5. Generate selective generation curves
  6. Run cross-domain transfer experiments

Usage:
    PYTHONPATH=. python src/modeling/train_and_evaluate.py \
        --features data/features/math500_features.csv \
        --output results/math500_results.json

    # Cross-domain transfer:
    PYTHONPATH=. python src/modeling/train_and_evaluate.py \
        --train-features data/features/math500_features.csv \
        --test-features data/features/gpqa_features.csv \
        --output results/transfer_math_to_gpqa.json
"""

import argparse
import json
import os
import logging
from collections import OrderedDict
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, roc_curve, precision_recall_curve
)

logger = logging.getLogger(__name__)

# Suppress sklearn convergence warnings for cleaner output
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not installed, skipping XGB classifier")


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error.

    Measures how well predicted probabilities match empirical accuracy.
    Perfect calibration = ECE of 0.

    Args:
        y_true: Binary ground truth labels.
        y_prob: Predicted probabilities of positive class.
        n_bins: Number of bins for calibration.

    Returns:
        ECE score in [0, 1].
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        n_in_bin = mask.sum()

        if n_in_bin == 0:
            continue

        bin_accuracy = y_true[mask].mean()
        bin_confidence = y_prob[mask].mean()
        ece += (n_in_bin / len(y_true)) * abs(bin_accuracy - bin_confidence)

    return float(ece)


def compute_selective_generation(
    y_true: np.ndarray,
    confidence: np.ndarray,
    n_points: int = 100,
) -> dict:
    """
    Compute selective generation (accuracy vs. coverage) metrics.

    Sort items by confidence, then compute accuracy at each coverage level.
    The best UQ method achieves highest accuracy at every coverage level.

    Returns:
        dict with coverage array, accuracy array, and summary metrics.
    """
    # Sort by confidence (descending)
    sorted_idx = np.argsort(confidence)[::-1]
    y_sorted = y_true[sorted_idx]

    coverages = np.linspace(0.05, 1.0, n_points)
    accuracies = []

    for cov in coverages:
        n_select = max(int(cov * len(y_true)), 1)
        acc = y_sorted[:n_select].mean()
        accuracies.append(float(acc))

    # Summary metrics
    acc_at_80 = y_sorted[:max(int(0.8 * len(y_true)), 1)].mean()
    acc_at_90 = y_sorted[:max(int(0.9 * len(y_true)), 1)].mean()

    # Area under accuracy-coverage curve
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    au_acc_cov = float(_trapz(accuracies, coverages))

    return {
        "coverages": [float(c) for c in coverages],
        "accuracies": accuracies,
        "accuracy_at_80": float(acc_at_80),
        "accuracy_at_90": float(acc_at_90),
        "au_acc_cov": au_acc_cov,
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    method_name: str = "",
) -> dict:
    """
    Compute all evaluation metrics for a set of predictions.

    Args:
        y_true: Binary ground truth labels (1=correct, 0=incorrect).
        y_prob: Predicted probabilities / confidence scores.
        method_name: Name for logging.

    Returns:
        dict with all metrics.
    """
    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        logger.warning(f"{method_name}: Only one class present, metrics may be unreliable")
        return {
            "method": method_name,
            "auroc": 0.5,
            "auprc": float(y_true.mean()),
            "ece": 0.0,
            "accuracy": float(y_true.mean()),
            "f1": 0.0,
            "n_samples": len(y_true),
        }

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    ece = compute_ece(y_true, y_prob)

    # Binary accuracy at threshold 0.5
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Selective generation
    sel_gen = compute_selective_generation(y_true, y_prob)

    return {
        "method": method_name,
        "auroc": float(auroc),
        "auprc": float(auprc),
        "ece": float(ece),
        "accuracy": float(acc),
        "f1": float(f1),
        "accuracy_at_80": sel_gen["accuracy_at_80"],
        "accuracy_at_90": sel_gen["accuracy_at_90"],
        "au_acc_cov": sel_gen["au_acc_cov"],
        "n_samples": len(y_true),
        "n_correct": int(y_true.sum()),
        "n_incorrect": int((1 - y_true).sum()),
        "base_accuracy": float(y_true.mean()),
    }


# =============================================================================
# CLASSIFIER DEFINITIONS
# =============================================================================

def get_classifiers() -> dict:
    """Return dict of classifier name → (class, params)."""
    classifiers = OrderedDict()

    classifiers["logistic_regression"] = (
        LogisticRegression,
        {"C": 1.0, "solver": "lbfgs",
         "max_iter": 1000, "class_weight": "balanced", "random_state": 42}
    )

    classifiers["random_forest"] = (
        RandomForestClassifier,
        {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 5,
         "class_weight": "balanced", "random_state": 42}
    )

    if HAS_XGBOOST:
        classifiers["xgboost"] = (
            XGBClassifier,
            {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1,
             "random_state": 42, "eval_metric": "logloss",
             "use_label_encoder": False}
        )

    return classifiers


# =============================================================================
# CROSS-VALIDATION TRAINING
# =============================================================================

def train_cv(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    classifier_name: str = "random_forest",
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """
    Train a classifier using stratified K-fold cross-validation.

    Returns:
        dict with per-fold metrics, aggregated metrics, and fold predictions.
    """
    classifiers = get_classifiers()
    if classifier_name not in classifiers:
        raise ValueError(f"Unknown classifier: {classifier_name}. "
                         f"Available: {list(classifiers.keys())}")

    cls_class, cls_params = classifiers[classifier_name]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_metrics = []
    all_y_true = []
    all_y_prob = []
    all_indices = []

    # For feature importance (from last fold)
    last_model = None
    last_scaler = None

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Handle XGBoost scale_pos_weight
        params = cls_params.copy()
        if classifier_name == "xgboost":
            n_pos = y_train.sum()
            n_neg = len(y_train) - n_pos
            if n_pos > 0:
                params["scale_pos_weight"] = n_neg / n_pos

        # Train
        model = cls_class(**params)
        model.fit(X_train_scaled, y_train)

        # Predict
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        # Evaluate
        fold_result = evaluate_predictions(y_test, y_prob, f"fold_{fold_idx}")
        fold_metrics.append(fold_result)

        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob)
        all_indices.extend(test_idx)

        last_model = model
        last_scaler = scaler

    # Aggregate metrics
    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)

    agg = evaluate_predictions(all_y_true, all_y_prob, classifier_name)

    # Per-metric mean ± std across folds
    metric_keys = ["auroc", "auprc", "ece", "accuracy_at_80", "accuracy_at_90"]
    summary = {}
    for key in metric_keys:
        vals = [fm[key] for fm in fold_metrics]
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))

    # Feature importance
    importance = extract_feature_importance(last_model, feature_names, classifier_name)

    return {
        "classifier": classifier_name,
        "n_folds": n_splits,
        "aggregated_metrics": agg,
        "summary": summary,
        "fold_metrics": fold_metrics,
        "feature_importance": importance,
        "predictions": {
            "indices": [int(i) for i in all_indices],
            "y_true": [int(y) for y in all_y_true],
            "y_prob": [float(p) for p in all_y_prob],
        },
        "last_model": last_model,
        "last_scaler": last_scaler,
    }


def extract_feature_importance(
    model, feature_names: list[str], classifier_name: str
) -> list[dict]:
    """Extract feature importance from a trained model."""
    importances = []

    if classifier_name == "logistic_regression":
        coefs = model.coef_[0]
        for name, coef in zip(feature_names, coefs):
            importances.append({"feature": name, "importance": float(abs(coef)),
                                "coefficient": float(coef)})
    elif classifier_name in ("random_forest", "xgboost"):
        fi = model.feature_importances_
        for name, imp in zip(feature_names, fi):
            importances.append({"feature": name, "importance": float(imp)})

    importances.sort(key=lambda x: x["importance"], reverse=True)
    return importances


# =============================================================================
# ABLATION STUDY
# =============================================================================

def run_ablation_study(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    classifier_name: str = "random_forest",
) -> list[dict]:
    """
    Run ablation study: train with different feature subsets.

    Tests each feature group independently and in combination.
    """
    from src.features.feature_pipeline import get_feature_groups
    groups = get_feature_groups()

    variants = [
        ("group1_only", ["group1_length"]),
        ("group2_only", ["group2_structural"]),
        ("group3_only", ["group3_meta"]),
        ("group2_plus_group3", ["group2_structural", "group3_meta"]),
        ("all_features", list(groups.keys())),
    ]

    results = []
    for variant_name, group_keys in variants:
        # Get feature indices for this variant
        selected_features = []
        for gk in group_keys:
            selected_features.extend(groups[gk])

        feature_indices = [
            i for i, name in enumerate(feature_names)
            if name in selected_features
        ]
        selected_names = [feature_names[i] for i in feature_indices]

        if not feature_indices:
            logger.warning(f"No features for variant {variant_name}")
            continue

        X_subset = X[:, feature_indices]

        result = train_cv(X_subset, y, selected_names, classifier_name)

        results.append({
            "variant": variant_name,
            "n_features": len(feature_indices),
            "feature_names": selected_names,
            "auroc_mean": result["summary"]["auroc_mean"],
            "auroc_std": result["summary"]["auroc_std"],
            "auprc_mean": result["summary"]["auprc_mean"],
            "auprc_std": result["summary"]["auprc_std"],
            "accuracy_at_80_mean": result["summary"]["accuracy_at_80_mean"],
        })

        logger.info(f"  {variant_name:25s}: AUROC={result['summary']['auroc_mean']:.3f} "
                     f"± {result['summary']['auroc_std']:.3f} "
                     f"({len(feature_indices)} features)")

    return results


# =============================================================================
# CROSS-DOMAIN TRANSFER
# =============================================================================

def run_transfer_experiment(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    feature_names: list[str],
    train_name: str = "source",
    test_name: str = "target",
    classifier_name: str = "random_forest",
) -> dict:
    """
    Train on one dataset, test on another (no retraining).

    This tests whether structural features of uncertain reasoning
    generalize across domains.
    """
    classifiers = get_classifiers()
    cls_class, cls_params = classifiers[classifier_name]

    # Standardize using TRAINING set statistics only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle class weights
    params = cls_params.copy()
    if classifier_name == "xgboost":
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        if n_pos > 0:
            params["scale_pos_weight"] = n_neg / n_pos

    # Train on source
    model = cls_class(**params)
    model.fit(X_train_scaled, y_train)

    # Test on target
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    metrics = evaluate_predictions(y_test, y_prob, f"transfer_{train_name}_to_{test_name}")
    importance = extract_feature_importance(model, feature_names, classifier_name)

    return {
        "train_dataset": train_name,
        "test_dataset": test_name,
        "classifier": classifier_name,
        "metrics": metrics,
        "feature_importance": importance,
        "train_size": len(y_train),
        "test_size": len(y_test),
        "train_accuracy": float(y_train.mean()),
        "test_accuracy": float(y_test.mean()),
    }


# =============================================================================
# MAIN EXPERIMENT RUNNER
# =============================================================================

def load_features(csv_path: str) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """Load feature CSV and return X, y, feature_names, full_df."""
    df = pd.read_csv(csv_path)

    from src.features.feature_pipeline import get_feature_names
    feature_names = get_feature_names()

    # Ensure all expected features are present
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        logger.warning(f"Missing features: {missing}")
        feature_names = [f for f in feature_names if f in df.columns]

    X = df[feature_names].values.astype(float)
    y = df["is_correct"].values.astype(int)

    # Handle NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, feature_names, df


def run_full_experiment(
    features_path: str,
    output_path: str,
    test_features_path: Optional[str] = None,
):
    """
    Run the complete experiment pipeline.

    If test_features_path is provided, also runs transfer experiment.
    """
    logger.info(f"Loading features from: {features_path}")
    X, y, feature_names, df = load_features(features_path)

    logger.info(f"  Shape: {X.shape}, Positive rate: {y.mean():.1%}")
    logger.info(f"  Features: {len(feature_names)}")

    results = {
        "dataset": os.path.basename(features_path),
        "n_samples": len(y),
        "n_correct": int(y.sum()),
        "n_incorrect": int((1 - y).sum()),
        "base_accuracy": float(y.mean()),
    }

    # --- Within-dataset cross-validation ---
    logger.info("\n--- Within-Dataset Cross-Validation ---")
    cv_results = {}
    for clf_name in get_classifiers():
        logger.info(f"\nTraining {clf_name}...")
        result = train_cv(X, y, feature_names, clf_name)
        cv_results[clf_name] = {
            k: v for k, v in result.items()
            if k not in ("last_model", "last_scaler")
        }
        s = result["summary"]
        logger.info(f"  AUROC: {s['auroc_mean']:.3f} ± {s['auroc_std']:.3f}")
        logger.info(f"  AUPRC: {s['auprc_mean']:.3f} ± {s['auprc_std']:.3f}")
        logger.info(f"  ECE:   {s['ece_mean']:.3f} ± {s['ece_std']:.3f}")
        logger.info(f"  Acc@80: {s['accuracy_at_80_mean']:.3f}")

    results["cross_validation"] = cv_results

    # --- Ablation study ---
    logger.info("\n--- Ablation Study ---")
    ablation = run_ablation_study(X, y, feature_names, "random_forest")
    results["ablation"] = ablation

    # --- Single-feature baselines ---
    logger.info("\n--- Single-Feature Baselines ---")
    single_feature_results = []
    for i, fname in enumerate(feature_names):
        # Use each feature as a single predictor
        # For features where higher = more uncertain, invert
        invert_features = {
            "total_tokens", "total_episodes", "backtrack_count",
            "restart_count", "transition_entropy", "cycle_count",
            "wait_ratio", "negation_count", "repetition_rate_4gram",
            "prop_backtrack", "prop_restart", "prop_hesitation",
            "v_clustering", "question_mark_count",
        }

        feature_vals = X[:, i]
        if fname in invert_features:
            feature_vals = -feature_vals  # Invert so higher = more confident

        # Normalize to [0, 1]
        fmin, fmax = feature_vals.min(), feature_vals.max()
        if fmax > fmin:
            feature_norm = (feature_vals - fmin) / (fmax - fmin)
        else:
            feature_norm = np.full_like(feature_vals, 0.5)

        if len(np.unique(y)) >= 2:
            auroc = roc_auc_score(y, feature_norm)
        else:
            auroc = 0.5

        single_feature_results.append({
            "feature": fname,
            "auroc": float(auroc),
        })

    single_feature_results.sort(key=lambda x: x["auroc"], reverse=True)
    results["single_feature_aurocs"] = single_feature_results

    logger.info("  Top 5 individual features:")
    for sf in single_feature_results[:5]:
        logger.info(f"    {sf['feature']:30s} AUROC={sf['auroc']:.3f}")

    # --- Transfer experiment ---
    if test_features_path:
        logger.info(f"\n--- Transfer Experiment ---")
        X_test, y_test, _, _ = load_features(test_features_path)
        transfer = run_transfer_experiment(
            X, y, X_test, y_test, feature_names,
            train_name=os.path.basename(features_path).replace("_features.csv", ""),
            test_name=os.path.basename(test_features_path).replace("_features.csv", ""),
        )
        results["transfer"] = {
            k: v for k, v in transfer.items()
            if not isinstance(v, (np.ndarray,))
        }
        logger.info(f"  Transfer AUROC: {transfer['metrics']['auroc']:.3f}")

    # --- Save results ---
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved: {output_path}")

    # --- Print summary table ---
    print_results_table(results)

    return results


def print_results_table(results: dict):
    """Print a formatted summary table of all results."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Dataset: {results['dataset']}")
    print(f"Samples: {results['n_samples']} "
          f"({results['n_correct']} correct, {results['n_incorrect']} incorrect)")
    print(f"Base accuracy: {results['base_accuracy']:.1%}")
    print()

    # Cross-validation results
    print(f"{'Method':25s} {'AUROC':>12s} {'AUPRC':>12s} {'ECE':>12s} {'Acc@80':>8s}")
    print(f"{'-'*70}")

    cv = results.get("cross_validation", {})
    for clf_name, clf_result in cv.items():
        s = clf_result["summary"]
        print(f"{clf_name:25s} "
              f"{s['auroc_mean']:.3f}±{s['auroc_std']:.3f}  "
              f"{s['auprc_mean']:.3f}±{s['auprc_std']:.3f}  "
              f"{s['ece_mean']:.3f}±{s['ece_std']:.3f}  "
              f"{s['accuracy_at_80_mean']:.3f}")

    # Ablation results
    print(f"\n{'Ablation':25s} {'AUROC':>12s} {'N Features':>10s}")
    print(f"{'-'*50}")
    for ab in results.get("ablation", []):
        print(f"{ab['variant']:25s} "
              f"{ab['auroc_mean']:.3f}±{ab['auroc_std']:.3f}  "
              f"{ab['n_features']:10d}")


# =============================================================================
# SELF-TEST
# =============================================================================

def run_modeling_tests():
    """Test the modeling pipeline with synthetic data."""
    print("Running modeling pipeline self-tests...")

    np.random.seed(42)

    # Create synthetic feature matrix (100 samples, 23 features)
    n_samples = 200
    n_features = 23

    # Simulate: correct answers have shorter traces and fewer backtracks
    y = np.random.binomial(1, 0.6, n_samples)  # 60% correct
    X = np.random.randn(n_samples, n_features)

    # Add signal: features 0 (total_tokens) and 9 (backtrack_count) correlate
    X[y == 1, 0] -= 1.0  # Correct → shorter traces
    X[y == 0, 9] += 1.5  # Incorrect → more backtracks

    from src.features.feature_pipeline import get_feature_names
    feature_names = get_feature_names()

    tests_passed = 0
    tests_failed = 0

    def check(name, condition, msg=""):
        nonlocal tests_passed, tests_failed
        if condition:
            tests_passed += 1
        else:
            tests_failed += 1
            print(f"  FAIL: {name}: {msg}")

    # Test 1: Cross-validation runs without error
    result = train_cv(X, y, feature_names, "logistic_regression", n_splits=3)
    check("cv_runs", result is not None)
    check("cv_has_metrics", "aggregated_metrics" in result)
    check("cv_auroc_range",
          0.0 <= result["summary"]["auroc_mean"] <= 1.0,
          f"AUROC={result['summary']['auroc_mean']}")
    check("cv_above_chance",
          result["summary"]["auroc_mean"] > 0.55,
          f"AUROC={result['summary']['auroc_mean']:.3f} should be > 0.55 with signal")

    # Test 2: Random forest
    rf_result = train_cv(X, y, feature_names, "random_forest", n_splits=3)
    check("rf_runs", rf_result is not None)
    check("rf_has_importance",
          len(rf_result["feature_importance"]) > 0)

    # Test 3: ECE computation
    ece = compute_ece(y, np.random.rand(n_samples))
    check("ece_range", 0 <= ece <= 1, f"ECE={ece}")

    # Test 4: Selective generation
    sel = compute_selective_generation(y, np.random.rand(n_samples))
    check("sel_has_coverages", len(sel["coverages"]) > 0)
    check("sel_has_accuracies", len(sel["accuracies"]) > 0)
    check("sel_acc_at_80", 0 <= sel["accuracy_at_80"] <= 1)

    # Test 5: Ablation study
    ablation = run_ablation_study(X, y, feature_names, "logistic_regression")
    check("ablation_runs", len(ablation) > 0)
    check("ablation_variants", len(ablation) >= 4, f"got {len(ablation)} variants")

    # Test 6: Transfer experiment
    X_test = np.random.randn(50, n_features)
    y_test = np.random.binomial(1, 0.5, 50)
    transfer = run_transfer_experiment(X, y, X_test, y_test, feature_names)
    check("transfer_runs", transfer is not None)
    check("transfer_has_metrics", "metrics" in transfer)

    # Test 7: Evaluate predictions
    metrics = evaluate_predictions(y, np.random.rand(n_samples), "test")
    check("metrics_has_auroc", "auroc" in metrics)
    check("metrics_has_auprc", "auprc" in metrics)
    check("metrics_has_ece", "ece" in metrics)
    check("metrics_has_sel_gen", "accuracy_at_80" in metrics)

    print(f"\nResults: {tests_passed} passed, {tests_failed} failed "
          f"out of {tests_passed + tests_failed} tests")
    if tests_failed == 0:
        print("All modeling tests passed.")
    return tests_failed == 0


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate UQ models")
    parser.add_argument("--features", required=True,
                        help="Path to feature CSV (for within-dataset experiments)")
    parser.add_argument("--test-features", default=None,
                        help="Path to test feature CSV (for transfer experiments)")
    parser.add_argument("--output", required=True,
                        help="Path to save results JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    run_full_experiment(
        features_path=args.features,
        output_path=args.output,
        test_features_path=args.test_features,
    )


if __name__ == "__main__":
    if "--features" in " ".join(os.sys.argv):
        main()
    else:
        logging.basicConfig(level=logging.INFO)
        run_modeling_tests()
