"""Track D: Fold-Isolated Class Imbalance Strategies, OOF Calibration, and Cost Thresholding (RQ4).

Strictly operates ONLY on the OOT Development Set (5-Fold CV).
Resampling (SMOTE / RandomUnderSampler) is applied STRICTLY inside each training fold,
never touching validation folds or the Locked OOT Test Set.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.data.load_data import get_project_root, load_config
from src.evaluation.metrics import compute_binary_metrics
from src.evaluation.tracker import ExperimentTracker
from src.models.registry import (
    build_fold_cache,
    build_model,
    load_dev_only_data,
    load_model_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _apply_fold_resampler(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    strategy: str,
    seed: int,
    n_threads: int,
) -> tuple[np.ndarray, np.ndarray]:
    if strategy in {"natural", "class_weight", "mild_weight_2.0"}:
        return X_tr, y_tr
    if strategy in {"smote_0.5", "smote_1.0"}:
        ratio = 0.5 if strategy == "smote_0.5" else 1.0
        # Standardize strictly within the training fold so raw-scale rupee features
        # do not dominate Euclidean k-NN distances inside SMOTE, then invert back.
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_tr)
        knn = NearestNeighbors(n_neighbors=5, n_jobs=n_threads)
        sampler = SMOTE(sampling_strategy=ratio, k_neighbors=knn, random_state=seed)
        X_res_scaled, y_res = sampler.fit_resample(X_scaled, y_tr)
        X_res = scaler.inverse_transform(X_res_scaled).astype(np.float32, copy=False)
        del X_scaled, X_res_scaled
        return X_res, y_res
    if strategy == "rus_0.5":
        sampler = RandomUnderSampler(sampling_strategy=0.5, random_state=seed)
        return sampler.fit_resample(X_tr, y_tr)
    raise ValueError(f"Unknown imbalance strategy '{strategy}'")


def _run_imbalance_for_cached_model(
    model_name: str,
    base_params: dict[str, Any],
    folds_cache: list[dict[str, Any]],
    strategies: list[str],
    y_dev: np.ndarray,
    cv_folds: np.ndarray,
    disbursed_amt: np.ndarray,
    tier: str,
    seed: int,
    n_threads: int,
    tracker: ExperimentTracker,
    track: str = "Track_D_ImbalanceCalibration",
    exp_prefix: str = "M4_D",
    oof_store: dict[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_dev = len(y_dev)

    for strat in strategies:
        base_strat_params = dict(base_params)
        if strat == "class_weight" and model_name != "xgboost":
            base_strat_params["class_weight"] = "balanced"
        elif strat == "mild_weight_2.0":
            base_strat_params["class_weight"] = None
            base_strat_params["scale_pos_weight"] = 2.0

        t0 = time.perf_counter()
        oof_probs = np.zeros(n_dev, dtype=np.float64)
        fold_aucs: list[float] = []

        for fd in folds_cache:
            params = dict(base_strat_params)
            if strat == "class_weight" and model_name == "xgboost":
                params["scale_pos_weight"] = float(np.sum(fd["y_tr"] == 0) / np.sum(fd["y_tr"] == 1))
            X_tr_res, y_tr_res = _apply_fold_resampler(fd["X_tr"], fd["y_tr"], strat, seed, n_threads)
            clf = build_model(model_name, params, random_state=seed, n_threads=n_threads)
            clf.fit(X_tr_res, y_tr_res)
            probs = clf.predict_proba(fd["X_va"])[:, 1]
            oof_probs[fd["va_idx"]] = probs
            fold_aucs.append(float(roc_auc_score(fd["y_va"], probs)))
            del X_tr_res, y_tr_res, clf
            gc.collect()

        dur = time.perf_counter() - t0
        if oof_store is not None:
            oof_store[f"{strat}__uncalibrated"] = oof_probs.copy()

        metrics = compute_binary_metrics(y_dev, oof_probs, threshold=0.5, loan_amounts=disbursed_amt)
        metrics["cv_roc_auc_mean"] = round(float(np.mean(fold_aucs)), 6)
        metrics["cv_roc_auc_std"] = round(float(np.std(fold_aucs, ddof=1)), 6)

        exp_id = f"{exp_prefix}_{model_name}_{strat}_uncalibrated"
        tracker.log_experiment(
            experiment_id=exp_id,
            track=track,
            model=model_name,
            feature_tier=tier,
            preprocessing=f"imbalance={strat}",
            dataset_split="oot_dev_5fold_cv",
            random_seed=seed,
            training_duration_sec=dur,
            status="SUCCESS",
            metrics=metrics,
            best_parameters={"imbalance_strategy": strat, **base_strat_params},
            notes=f"Fold-isolated {strat} on 5-Fold Dev CV",
        )
        rows.append(
            {
                "experiment_id": exp_id,
                "model": model_name,
                "imbalance_strategy": strat,
                "calibration": "uncalibrated",
                "duration_sec": round(dur, 2),
                **metrics,
            }
        )
        logger.info("  -> %s | %s | AUC=%.5f | PR-AUC=%.5f | Brier=%.5f ECE=%.5f (%.1fs)",
                    model_name, strat, rows[-1]["roc_auc"], rows[-1]["pr_auc"], rows[-1]["brier_score"], rows[-1]["ece"], dur)
    return rows


def main() -> int:
    root = get_project_root()
    base_cfg = load_config()
    model_cfg = load_model_config()
    seed = int(model_cfg["benchmark"]["random_seed"])
    n_threads = int(model_cfg["benchmark"]["n_threads"])

    results_dir = root / "experiments" / "results"
    best_params_path = results_dir / "m4_best_params.json"
    tuning_csv_path = results_dir / "m4_tuning_comparison.csv"
    if not best_params_path.exists() or not tuning_csv_path.exists():
        raise FileNotFoundError("Run tune_models.py before imbalance_calibration.py")

    best_params_doc = json.loads(best_params_path.read_text(encoding="utf-8"))
    tier = str(best_params_doc["feature_tier"])
    tune_df = pd.read_csv(tuning_csv_path).sort_values("tuned_cv_auc", ascending=False)

    cont_gbdt_ranked = [m for m in tune_df["model"].tolist() if m in {"lightgbm", "xgboost"}]
    best_cont_gbdt = cont_gbdt_ranked[0]

    logger.info("Track D targets: logistic_regression, best_gbdt=%s", best_cont_gbdt)

    dev_df, y_dev, cv_folds, _, disbursed_amt = load_dev_only_data(base_cfg)
    tracker = ExperimentTracker(results_dir)
    all_rows: list[dict[str, Any]] = []

    # 1. Logistic Regression (natural, class_weight, smote_0.5, smote_1.0, rus_0.5)
    logger.info("=== Track D1: Logistic Regression (Imbalance Strategies) ===")
    linear_cache = build_fold_cache(dev_df, y_dev, cv_folds, tier, "linear", True, base_cfg)
    all_rows.extend(
        _run_imbalance_for_cached_model(
            "logistic_regression",
            best_params_doc["models"]["logistic_regression"],
            linear_cache,
            ["natural", "class_weight", "smote_0.5", "smote_1.0", "rus_0.5"],
            y_dev, cv_folds, disbursed_amt, tier, seed, n_threads, tracker,
        )
    )
    del linear_cache
    gc.collect()

    # 2. Best GBDT (full comparison including fold-isolated SMOTE 0.5 & 1.0)
    logger.info("=== Track D2: %s (Imbalance Strategies incl. SMOTE) ===", best_cont_gbdt)
    tree_cache = build_fold_cache(dev_df, y_dev, cv_folds, tier, "tree", False, base_cfg)
    all_rows.extend(
        _run_imbalance_for_cached_model(
            best_cont_gbdt,
            best_params_doc["models"][best_cont_gbdt],
            tree_cache,
            ["natural", "class_weight", "smote_0.5", "smote_1.0", "rus_0.5"],
            y_dev, cv_folds, disbursed_amt, tier, seed, n_threads, tracker,
        )
    )
    del tree_cache
    gc.collect()

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(results_dir / "m4_imbalance_calibration.csv", index=False)
    logger.info("[OK] Saved m4_imbalance_calibration.csv (%d rows)", len(out_df))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
