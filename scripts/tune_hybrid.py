#!/usr/bin/env python
"""
tune_hybrid.py — Optuna hyperparameter search for the hybrid meta-stacker.

Inputs
------
    data/hybrid_table.parquet   (built by scripts/build_hybrid_table.py)
    data/hybrid_table.META.json (provenance; read for leaky-policy gating)

What this searches over
-----------------------
Hyperparameters span six categorical / continuous axes:

1. **Meta-learner family** — {LogisticRegression, XGBClassifier, LGBMClassifier}.
   Each has its own hyperparameter sub-space, sampled only when that family
   is chosen.
2. **Feature subset** — which groups of columns the meta sees:
      'oof_only'       : just the 6 OOF probabilities
      'oof+hand'       : + handcrafted + recurrence (30 features)
      'oof+hand+graph' : + graph structure (15)
      'oof+ph'         : + persistent-homology (36)
      'oof+timing'     : + timing/IEI (46)
      'oof+ng'         : + n-gram (231)
      'all'            : everything (~360 features)
      'all_minus_ng'   : everything except n-gram (n-gram can dominate)
3. **Scaling / preprocessing** — StandardScaler / RobustScaler / None.
4. **Group dummies** — add 8 one-hot group indicators (captures per-dataset
   bias the meta-learner could exploit).
5. **Isotonic calibration** — wrap the meta in a fold-level isotonic
   regression (trained on out-of-meta-fold predictions).
6. **L2 / LR hyperparameters** — family-specific.

Objective
---------
   primary:   per-group weighted AUROC (Simpson-corrected), using the
              canonical fold from the parquet.
   secondary: pooled AUROC (logged, not optimized).

Why weighted, not pooled: pooled AUROC on 8 datasets of very different base
rates (e.g. gsm8k_qwen7b at 90% correct vs arc_challenge_llama8b at 40%)
can be Simpson-inflated. Per-group weighted AUROC is a more honest signal
of "does the meta generalize" vs "does the meta lean on base-rate
differences between groups".

Outputs
-------
   results/route_ab/hybrid_tuning_trials.csv
     One row per trial: params + primary/secondary metrics + wall time.

   results/route_ab/hybrid_tuned_best.json
     Best trial's config + metrics + feature subset + provenance status.

   (optional, --refit) :
     results/route_ab/hybrid_tuned_pooled_oof.npz
     results/route_ab/hybrid_tuned_pooled.json
     reports/route_ab/hybrid_tuned_per_group.csv

Usage
-----
    # smoke (fast sanity check)
    PYTHONPATH=. python scripts/tune_hybrid.py --n-trials 20

    # full study
    PYTHONPATH=. python scripts/tune_hybrid.py --n-trials 500 \\
        --study-name hybrid_v1 --storage sqlite:///optuna_hybrid.db

    # resume / add more trials
    PYTHONPATH=. python scripts/tune_hybrid.py --n-trials 200 \\
        --study-name hybrid_v1 --storage sqlite:///optuna_hybrid.db

    # refit best config on all folds + save OOF + per-group table
    PYTHONPATH=. python scripts/tune_hybrid.py --refit \\
        --study-name hybrid_v1 --storage sqlite:///optuna_hybrid.db
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Silence warning families we expect and have reviewed:
#   * 'penalty' deprecation (sklearn 1.8): we keep penalty= because liblinear
#     still requires it; re-evaluate on sklearn 1.10.
#   * 'Inconsistent values: penalty=l1 with l1_ratio=0.0': the very same
#     deprecation bleeding into runtime validation.
#   * 'X does not have valid feature names': LGBM/XGB warn when fit was with
#     a DataFrame (or with numpy — they remember either way) and predict was
#     with a bare array. Harmless in our numpy-throughout pipeline.
#   * ConvergenceWarning from liblinear/lbfgs: logged once per failed fit;
#     Optuna already accounts for this via TrialPruned on timeout.
# -----------------------------------------------------------------------------
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", message=".*'penalty' was deprecated.*",
                        category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Inconsistent values: penalty.*",
                        category=UserWarning)
warnings.filterwarnings("ignore",
                        message=".*X does not have valid feature names.*",
                        category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler, StandardScaler


# =============================================================================
# PER-TRIAL TIMEOUT HANDLING
# =============================================================================
# The tuner picks LR+l1 trials that invoke sklearn's `liblinear` solver. That
# is a C extension holding the GIL across its inner loop; Python signal
# handlers (including SIGINT / KeyboardInterrupt) cannot fire until the C
# call returns. On hard problems (wide feature subsets, tight regularization)
# a single fit can take many minutes, making the process feel "stuck" and
# un-Ctrl-C-able.
#
# SIGALRM, unlike SIGINT, is delivered to the C signal handler which raises
# our `TrialTimeout` once liblinear yields between iterations. This gives us
# a hard per-trial wall-clock cap. Pair it with max_iter=500 for LR to bound
# the worst case further.
# =============================================================================

class TrialTimeout(Exception):
    """Raised when a single trial exceeds its wall-clock budget."""


def _alarm_handler(signum, frame):  # noqa: ARG001
    raise TrialTimeout()


# SIGALRM is POSIX-only. On Windows (incl. the Windows-side Python that can
# drive \\wsl.localhost paths), the guard becomes a no-op and we log once.
# For real long runs, submit to HPC via scripts/sbatch_tune_hybrid.sh, which
# runs Linux where SIGALRM is active.
_HAS_SIGALRM = hasattr(signal, "SIGALRM")
_WIN_WARN_FIRED = False


def _arm_trial_alarm(seconds: int):
    """Install SIGALRM handler and arm the alarm. POSIX only.

    Returns the previously-installed handler so the caller can restore it.
    Returns None on platforms without SIGALRM (and logs a one-time warning).
    """
    global _WIN_WARN_FIRED
    if not _HAS_SIGALRM:
        if not _WIN_WARN_FIRED:
            logger.warning(
                "Per-trial timeout disabled: SIGALRM not available on this "
                "platform (Windows?). Trials may hang on pathological LR+l1 "
                "configs and cannot be Ctrl-C'd. Submit to HPC via "
                "scripts/sbatch_tune_hybrid.sh for a real run."
            )
            _WIN_WARN_FIRED = True
        return None
    old = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(int(seconds))
    return old


def _disarm_trial_alarm(old_handler):
    """Reset SIGALRM. Accepts the sentinel returned by _arm_trial_alarm."""
    if not _HAS_SIGALRM:
        return
    signal.alarm(0)
    if old_handler is not None:
        signal.signal(signal.SIGALRM, old_handler)

REPO = Path(__file__).resolve().parents[1]
logger = logging.getLogger(__name__)

# Lazy imports: xgboost, lightgbm, optuna — only used inside the main body.


# =============================================================================
# DATA LOADING
# =============================================================================

def load_hybrid_table(parquet_path: Path, meta_path: Path,
                      leaky_policy: str = "warn") -> tuple[pd.DataFrame, dict]:
    if not parquet_path.exists():
        raise FileNotFoundError(f"{parquet_path} — run build_hybrid_table.py first")
    df = pd.read_parquet(parquet_path)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    any_leaky = meta.get("any_leaky", False)
    if any_leaky:
        msg = ("hybrid_table.parquet contains OOFs with LEAKY provenance "
               "(best-epoch-on-val). Tuning AUROC will be inflated by "
               "~0.01-0.025.")
        if leaky_policy == "fail":
            raise RuntimeError(msg)
        elif leaky_policy == "warn":
            logger.warning("⚠  " + msg)
            logger.warning("⚠  Re-run base models with patched code before "
                           "reporting final numbers.")
    return df, meta


def column_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    """Map feature-subset name -> list of column names."""
    oof_cols = [c for c in df.columns if c.startswith("oof_")]
    hand_cols = [c for c in df.columns if c.startswith("hand_")]
    graph_cols = [c for c in df.columns if c.startswith("graph_")]
    timing_cols = [c for c in df.columns if c.startswith("timing_")]
    ph_cols = [c for c in df.columns if c.startswith("ph_")]
    ng_cols = [c for c in df.columns if c.startswith("ng_")]

    subsets = {
        "oof_only":        oof_cols,
        "oof+hand":        oof_cols + hand_cols,
        "oof+hand+graph":  oof_cols + hand_cols + graph_cols,
        "oof+ph":          oof_cols + ph_cols,
        "oof+timing":      oof_cols + timing_cols,
        "oof+ng":          oof_cols + ng_cols,
        "all":             oof_cols + hand_cols + graph_cols + timing_cols + ph_cols + ng_cols,
        "all_minus_ng":    oof_cols + hand_cols + graph_cols + timing_cols + ph_cols,
        # Without the strongest single base (RoBERTa) — for robustness check
        "oof_minus_roberta": [c for c in oof_cols if c != "oof_roberta"],
    }
    return subsets


# =============================================================================
# METRICS
# =============================================================================

def per_group_weighted_auroc(df: pd.DataFrame, preds: np.ndarray) -> float:
    """Sum of n_g * AUROC_g / N over groups, skipping degenerate groups."""
    total = 0.0
    total_n = 0
    for g in df["group"].unique():
        m = (df["group"] == g).values
        yg = df.loc[m, "label"].values
        pg = preds[m]
        if len(np.unique(yg)) < 2:
            continue
        total += float(len(yg)) * roc_auc_score(yg, pg)
        total_n += len(yg)
    return total / max(total_n, 1)


def pooled_auroc(df: pd.DataFrame, preds: np.ndarray) -> float:
    return float(roc_auc_score(df["label"].values, preds))


# =============================================================================
# META TRAINING PER FOLD
# =============================================================================

def _build_scaler(kind: str):
    if kind == "none":
        return None
    if kind == "standard":
        return StandardScaler()
    if kind == "robust":
        return RobustScaler()
    raise ValueError(kind)


def _build_model(family: str, params: dict[str, Any], seed: int = 42,
                  n_jobs: int = 1):
    if family == "lr":
        # max_iter=500 (was 3000) to bound liblinear's worst-case wall time.
        # Trials that don't converge here will pop a ConvergenceWarning
        # (silenced) and yield a usable-but-suboptimal fit. Better than
        # hanging for tens of minutes on a pathological (penalty=l1,
        # class_weight=balanced, wide feature subset) combo.
        return LogisticRegression(
            C=params["C"],
            penalty=params["penalty"],
            solver=("liblinear" if params["penalty"] == "l1" else "lbfgs"),
            max_iter=500,
            class_weight="balanced" if params.get("class_weight_balanced") else None,
            random_state=seed,
        )
    if family == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=params["min_child_weight"],
            reg_lambda=params["reg_lambda"],
            reg_alpha=params["reg_alpha"],
            gamma=params["gamma"],
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=n_jobs,  # per-fit thread count; sbatch --cpus-per-task controls
            verbosity=0,
        )
    if family == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            num_leaves=params["num_leaves"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_samples=params["min_child_samples"],
            reg_lambda=params["reg_lambda"],
            reg_alpha=params["reg_alpha"],
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=-1,
        )
    raise ValueError(family)


def _one_hot_groups(df: pd.DataFrame) -> np.ndarray:
    groups = sorted(df["group"].unique())
    idx = {g: i for i, g in enumerate(groups)}
    onehot = np.zeros((len(df), len(groups)), dtype=np.float32)
    for i, g in enumerate(df["group"].values):
        onehot[i, idx[g]] = 1.0
    return onehot


def cv_train_predict(df: pd.DataFrame,
                     feat_cols: list[str],
                     family: str,
                     model_params: dict,
                     scaler_kind: str = "standard",
                     add_group_dummies: bool = False,
                     isotonic: bool = False,
                     seed: int = 42,
                     n_jobs: int = 1,
                     ) -> np.ndarray:
    """Cross-fit: for each fold, train on (fold != k), predict on (fold == k).
    Returns the full-length OOF prediction vector."""
    X = df[feat_cols].values.astype(np.float32)
    if add_group_dummies:
        X = np.hstack([X, _one_hot_groups(df)])
    y = df["label"].values.astype(np.int32)
    fold = df["fold"].values.astype(np.int32)

    oof = np.zeros(len(df), dtype=np.float32)

    # Impute NaN with column median on train split
    for k in sorted(np.unique(fold)):
        tr = fold != k
        te = fold == k
        X_tr = X[tr].copy()
        X_te = X[te].copy()
        # NaN-median impute (per column, train-computed)
        med = np.nanmedian(X_tr, axis=0)
        col_nan = np.isnan(X_tr).any(axis=0)
        if col_nan.any():
            # Handle all-NaN cols by setting median to 0
            med = np.where(np.isnan(med), 0.0, med)
            for j in np.where(np.isnan(X_tr).any(axis=0))[0]:
                X_tr[np.isnan(X_tr[:, j]), j] = med[j]
            for j in np.where(np.isnan(X_te).any(axis=0))[0]:
                X_te[np.isnan(X_te[:, j]), j] = med[j]

        sc = _build_scaler(scaler_kind)
        if sc is not None:
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

        mdl = _build_model(family, model_params, seed=seed, n_jobs=n_jobs)
        mdl.fit(X_tr, y[tr])
        prob = mdl.predict_proba(X_te)[:, 1]

        # Optional fold-internal isotonic calibration using a further
        # split of the train set.
        if isotonic:
            from sklearn.model_selection import StratifiedKFold
            # inner 3-fold just for calibration curve fit on training set
            iskf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            inner_oof = np.zeros(len(X_tr), dtype=np.float32)
            for itr, ite in iskf.split(X_tr, y[tr]):
                m2 = _build_model(family, model_params, seed=seed, n_jobs=n_jobs)
                m2.fit(X_tr[itr], y[tr][itr])
                inner_oof[ite] = m2.predict_proba(X_tr[ite])[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(inner_oof, y[tr])
            prob = iso.transform(prob)

        oof[te] = prob

    return oof


# =============================================================================
# OPTUNA OBJECTIVE
# =============================================================================

def make_objective(df: pd.DataFrame, subsets: dict[str, list[str]],
                    seed: int = 42,
                    trial_timeout_sec: int = 180,
                    n_jobs: int = 1):
    def objective(trial):
        family = trial.suggest_categorical("family", ["lr", "xgb", "lgbm"])
        subset = trial.suggest_categorical("feature_subset",
                                           list(subsets.keys()))
        scaler_kind = trial.suggest_categorical(
            "scaler", ["standard", "robust", "none"])
        add_group_dummies = trial.suggest_categorical(
            "add_group_dummies", [False, True])
        isotonic = trial.suggest_categorical("isotonic", [False, True])

        if family == "lr":
            params = {
                "C": trial.suggest_float("lr_C", 1e-3, 1e2, log=True),
                "penalty": trial.suggest_categorical("lr_penalty", ["l2", "l1"]),
                "class_weight_balanced":
                    trial.suggest_categorical("lr_class_weight_balanced",
                                              [False, True]),
            }
        elif family == "xgb":
            params = {
                "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 800),
                "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
                "learning_rate": trial.suggest_float("xgb_lr", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("xgb_colsample", 0.3, 1.0),
                "min_child_weight": trial.suggest_int("xgb_min_child", 1, 10),
                "reg_lambda": trial.suggest_float("xgb_lambda", 1e-3, 10.0, log=True),
                "reg_alpha": trial.suggest_float("xgb_alpha", 1e-3, 10.0, log=True),
                "gamma": trial.suggest_float("xgb_gamma", 0.0, 5.0),
            }
        elif family == "lgbm":
            params = {
                "n_estimators": trial.suggest_int("lgbm_n_estimators", 100, 800),
                "max_depth": trial.suggest_int("lgbm_max_depth", -1, 12),
                "num_leaves": trial.suggest_int("lgbm_num_leaves", 15, 255),
                "learning_rate": trial.suggest_float("lgbm_lr", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("lgbm_subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("lgbm_colsample", 0.3, 1.0),
                "min_child_samples": trial.suggest_int("lgbm_min_child", 5, 100),
                "reg_lambda": trial.suggest_float("lgbm_lambda", 1e-3, 10.0, log=True),
                "reg_alpha": trial.suggest_float("lgbm_alpha", 1e-3, 10.0, log=True),
            }
        else:
            raise ValueError(family)

        # Penalize incompatible combos to save trials
        # (l1 on n_estimators/max_depth etc. is illegal only inside _build_model;
        #  here lr penalty l1 requires liblinear which is already handled.)

        feat_cols = subsets[subset]
        if not feat_cols:
            raise optuna.exceptions.TrialPruned()  # type: ignore[name-defined]

        # Arm the per-trial SIGALRM. This is the only signal that reliably
        # interrupts a hung liblinear/lbfgs inner loop — SIGINT cannot. On
        # Windows SIGALRM is absent, so the arm is a no-op (logged once).
        # We install and uninstall the handler per trial so that trials
        # outside the timeout region (e.g. metric computation) can still
        # be Ctrl-C'd normally.
        old_handler = _arm_trial_alarm(trial_timeout_sec)
        t0 = time.time()
        try:
            oof = cv_train_predict(
                df, feat_cols,
                family=family, model_params=params,
                scaler_kind=scaler_kind,
                add_group_dummies=add_group_dummies,
                isotonic=isotonic,
                seed=seed,
                n_jobs=n_jobs,
            )
        except TrialTimeout:
            elapsed = time.time() - t0
            _disarm_trial_alarm(old_handler)
            trial.set_user_attr("pruned_reason", "trial_timeout")
            trial.set_user_attr("elapsed_sec", elapsed)
            trial.set_user_attr("n_features", len(feat_cols))
            logger.warning(
                f"  trial {trial.number:04d}  TIMEOUT  "
                f"fam={family:4s}  sub={subset:16s}  "
                f"sc={scaler_kind:8s}  dum={add_group_dummies}  "
                f"iso={isotonic}  "
                f"n_feat={len(feat_cols)}  t={elapsed:.1f}s (cap={trial_timeout_sec}s)"
            )
            raise optuna.exceptions.TrialPruned()  # type: ignore[name-defined]
        finally:
            # Always disarm the alarm; restore prior handler.
            _disarm_trial_alarm(old_handler)

        pooled = pooled_auroc(df, oof)
        weighted = per_group_weighted_auroc(df, oof)
        elapsed = time.time() - t0

        trial.set_user_attr("pooled_auroc", pooled)
        trial.set_user_attr("weighted_auroc", weighted)
        trial.set_user_attr("elapsed_sec", elapsed)
        trial.set_user_attr("n_features", len(feat_cols))

        logger.info(f"  trial {trial.number:04d}  "
                    f"fam={family:4s}  sub={subset:16s}  "
                    f"sc={scaler_kind:8s}  dum={add_group_dummies}  "
                    f"iso={isotonic}  "
                    f"pooled={pooled:.4f}  weighted={weighted:.4f}  "
                    f"n_feat={len(feat_cols)}  t={elapsed:.1f}s")
        return weighted  # primary objective

    return objective


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet",
                    default="data/hybrid_table.parquet")
    ap.add_argument("--meta",
                    default="data/hybrid_table.META.json")
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--timeout", type=float, default=None,
                    help="Optional hard wall-time cap in seconds")
    ap.add_argument("--study-name", default="hybrid_tuning")
    ap.add_argument("--storage", default=None,
                    help=("Optuna storage URL. Accepts: "
                          "'sqlite:///path/to.db' (RDB — recommended on HPC "
                          "linux filesystems); "
                          "'journal:///path/to.log' (JournalStorage with "
                          "open-based file lock — works on WSL/UNC where "
                          "SQLite WAL locks fail); "
                          "None = in-memory (--refit unavailable)."))
    ap.add_argument("--trials-csv",
                    default="results/route_ab/hybrid_tuning_trials.csv")
    ap.add_argument("--best-json",
                    default="results/route_ab/hybrid_tuned_best.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--leaky-policy", choices=["allow", "warn", "fail"],
                    default="warn")
    ap.add_argument("--trial-timeout", type=int, default=180,
                    help=("Per-trial wall-clock cap in seconds (SIGALRM-based). "
                          "Trials exceeding this are pruned. Default 180s is "
                          "generous for non-pathological configs; raise to 600s "
                          "on slow hardware, drop to 60s for rapid sweeps."))
    ap.add_argument("--n-jobs", type=int, default=1,
                    help=("Threads per xgboost/lightgbm fit. Leave at 1 when "
                          "Optuna runs trials in parallel; bump to match "
                          "--cpus-per-task when running trials sequentially "
                          "(the default study.optimize() invocation)."))
    ap.add_argument("--refit", action="store_true",
                    help="Skip tuning; load study and refit best to save OOF.")
    ap.add_argument("--refit-oof",
                    default="results/route_ab/hybrid_tuned_pooled_oof.npz")
    ap.add_argument("--refit-metrics",
                    default="results/route_ab/hybrid_tuned_pooled.json")
    ap.add_argument("--refit-per-group",
                    default="reports/route_ab/hybrid_tuned_per_group.csv")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(levelname)s] %(message)s")

    global optuna
    import optuna  # noqa: F401  (import here so --refit without optuna is allowed if deferred)

    parquet = REPO / args.parquet
    meta = REPO / args.meta
    df, meta_dict = load_hybrid_table(parquet, meta,
                                       leaky_policy=args.leaky_policy)
    subsets = column_groups(df)
    logger.info(f"Loaded: {len(df)} rows, {df['group'].nunique()} groups, "
                f"label rate {df.label.mean():.3f}")
    logger.info(f"Feature subsets: " +
                ", ".join(f"{k}={len(v)}" for k, v in subsets.items()))

    if args.refit:
        return _refit_best(df, subsets, args)

    # Build study
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=0)
    storage = _resolve_storage(args.storage)
    if storage is not None:
        study = optuna.create_study(
            direction="maximize",
            study_name=args.study_name,
            storage=storage,
            load_if_exists=True,
            sampler=sampler, pruner=pruner,
        )
    else:
        study = optuna.create_study(
            direction="maximize", study_name=args.study_name,
            sampler=sampler, pruner=pruner,
        )

    logger.info(f"Optuna study: {args.study_name}  storage={args.storage or 'memory'}")
    logger.info(f"Starting {args.n_trials} trials (existing: {len(study.trials)})  "
                f"trial_timeout={args.trial_timeout}s  n_jobs={args.n_jobs}")
    study.optimize(
        make_objective(df, subsets, seed=args.seed,
                       trial_timeout_sec=args.trial_timeout,
                       n_jobs=args.n_jobs),
        n_trials=args.n_trials,
        timeout=args.timeout,
        gc_after_trial=True,
        # Let KeyboardInterrupt propagate out so Ctrl-C between trials works.
        catch=(),
    )

    # Dump trials to CSV
    os.makedirs(Path(REPO / args.trials_csv).parent, exist_ok=True)
    trials_df = study.trials_dataframe(
        attrs=("number", "value", "state", "params", "user_attrs"))
    trials_df.to_csv(REPO / args.trials_csv, index=False)
    logger.info(f"✓ Wrote {args.trials_csv}  ({len(trials_df)} rows)")

    # Best trial
    best = study.best_trial
    best_summary = {
        "study_name": args.study_name,
        "n_trials": len(study.trials),
        "best_trial_number": best.number,
        "best_weighted_auroc": best.value,
        "best_pooled_auroc": best.user_attrs.get("pooled_auroc"),
        "best_elapsed_sec": best.user_attrs.get("elapsed_sec"),
        "best_params": dict(best.params),
        "provenance": meta_dict.get("provenance_summary", {}),
        "any_leaky_in_table": meta_dict.get("any_leaky", False),
    }
    with open(REPO / args.best_json, "w") as f:
        json.dump(best_summary, f, indent=2, default=str)
    logger.info(f"✓ Wrote {args.best_json}")
    logger.info(f"BEST: weighted_AUROC={best.value:.4f}  "
                f"pooled_AUROC={best.user_attrs.get('pooled_auroc'):.4f}")


# =============================================================================
# REFIT MODE
# =============================================================================

def _resolve_storage(spec: str | None):
    """Map --storage URL to a concrete optuna storage object.

    Accepts:
      None                          -> None (in-memory)
      'sqlite:///path'              -> RDBStorage string URL (raw SQLite)
      'journal:///path/to.log'      -> JournalStorage with open-based lock
                                       (cross-filesystem safe; works on WSL)

    Strings other than 'journal:...' are passed through to optuna as-is, so
    MySQL/Postgres URLs work on HPC if the user prefers.
    """
    if spec is None:
        return None
    if spec.startswith("journal:///"):
        import optuna
        # The JournalFileOpenLock avoids symlinks — works on \\wsl.localhost
        # and networked filesystems without admin privileges.
        path = spec[len("journal:///"):]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        lock = optuna.storages.journal.JournalFileOpenLock(path)
        # Class was renamed in optuna v4.0: JournalFileStorage → JournalFileBackend
        # (moved into optuna.storages.journal). Try the new name first so we don't
        # emit a FutureWarning on modern installs; fall back for pre-4.0 environments.
        try:
            backend = optuna.storages.journal.JournalFileBackend(path, lock_obj=lock)
        except AttributeError:
            backend = optuna.storages.JournalFileStorage(path, lock_obj=lock)
        return optuna.storages.JournalStorage(backend)
    # Everything else: let optuna interpret the string (sqlite://, mysql://, ...)
    return spec


def _refit_best(df: pd.DataFrame, subsets: dict, args):
    import optuna
    if not args.storage:
        raise ValueError("--refit requires --storage to load a previous study")
    storage = _resolve_storage(args.storage)
    study = optuna.load_study(study_name=args.study_name, storage=storage)
    best = study.best_trial
    logger.info(f"Refitting best trial {best.number}: "
                f"weighted_AUROC={best.value:.4f}  params={dict(best.params)}")

    p = dict(best.params)
    family = p.pop("family")
    subset = p.pop("feature_subset")
    scaler_kind = p.pop("scaler")
    add_group_dummies = p.pop("add_group_dummies")
    isotonic = p.pop("isotonic")

    # Strip prefix from family-specific params
    model_params = {}
    prefix = family + "_"
    for k, v in p.items():
        if k.startswith(prefix):
            # Map back to the kwarg name _build_model expects
            short = k[len(prefix):]
            # Special: lr_C -> C, lr_penalty -> penalty
            if family == "lr":
                if short == "C": model_params["C"] = v
                elif short == "penalty": model_params["penalty"] = v
                elif short == "class_weight_balanced": model_params["class_weight_balanced"] = v
            elif family == "xgb":
                if short == "n_estimators": model_params["n_estimators"] = v
                elif short == "max_depth": model_params["max_depth"] = v
                elif short == "lr": model_params["learning_rate"] = v
                elif short == "subsample": model_params["subsample"] = v
                elif short == "colsample": model_params["colsample_bytree"] = v
                elif short == "min_child": model_params["min_child_weight"] = v
                elif short == "lambda": model_params["reg_lambda"] = v
                elif short == "alpha": model_params["reg_alpha"] = v
                elif short == "gamma": model_params["gamma"] = v
            elif family == "lgbm":
                if short == "n_estimators": model_params["n_estimators"] = v
                elif short == "max_depth": model_params["max_depth"] = v
                elif short == "num_leaves": model_params["num_leaves"] = v
                elif short == "lr": model_params["learning_rate"] = v
                elif short == "subsample": model_params["subsample"] = v
                elif short == "colsample": model_params["colsample_bytree"] = v
                elif short == "min_child": model_params["min_child_samples"] = v
                elif short == "lambda": model_params["reg_lambda"] = v
                elif short == "alpha": model_params["reg_alpha"] = v

    oof = cv_train_predict(
        df, subsets[subset], family=family, model_params=model_params,
        scaler_kind=scaler_kind, add_group_dummies=add_group_dummies,
        isotonic=isotonic, seed=args.seed, n_jobs=args.n_jobs,
    )
    pooled = pooled_auroc(df, oof)
    weighted = per_group_weighted_auroc(df, oof)
    logger.info(f"Refit: pooled={pooled:.4f}  weighted={weighted:.4f}")

    # Save OOF
    os.makedirs(Path(REPO / args.refit_oof).parent, exist_ok=True)
    np.savez_compressed(
        REPO / args.refit_oof,
        item_ids=df["item_id"].values.astype(object),
        groups=df["group"].values.astype(object),
        y_true=df["label"].values.astype(np.int32),
        oof_prob=oof.astype(np.float32),
        oof_fold=df["fold"].values.astype(np.int8),
        seed=np.int32(args.seed),
    )
    logger.info(f"✓ Wrote {args.refit_oof}")

    # Per-group table
    rows = []
    for g in sorted(df["group"].unique()):
        m = (df["group"] == g).values
        rows.append({
            "group": g, "n": int(m.sum()),
            "auroc": float(roc_auc_score(df.loc[m, "label"].values, oof[m])),
            "label_rate": float(df.loc[m, "label"].mean()),
        })
    pg = pd.DataFrame(rows)
    os.makedirs(Path(REPO / args.refit_per_group).parent, exist_ok=True)
    pg.to_csv(REPO / args.refit_per_group, index=False)
    logger.info(f"✓ Wrote {args.refit_per_group}")

    # Summary JSON
    summary = {
        "refit_of_study": args.study_name,
        "best_trial_number": best.number,
        "pooled_auroc": pooled,
        "weighted_auroc": weighted,
        "per_group": rows,
        "config": dict(best.params),
        "feature_subset_size": len(subsets[subset]),
    }
    with open(REPO / args.refit_metrics, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✓ Wrote {args.refit_metrics}")


if __name__ == "__main__":
    sys.exit(main() or 0)
