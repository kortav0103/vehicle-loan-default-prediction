"""Track A (Baseline Benchmark) & Track B (Feature Tier Ablation) on Primary Dev Set.

Strictly operates ONLY on the OOT Development Set (<= 2018-10-22, N=182,501).
Locked OOT Test Set (>= 2018-10-23) is never loaded or touched.
"""

from __future__ import annotations

import gc
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.load_data import get_project_root, load_config
from src.evaluation.metrics import compute_binary_metrics
from src.evaluation.tracker import ExperimentTracker, save_environment_snapshot
from src.models.registry import (
    build_fold_cache,
    build_model,
    load_dev_only_data,
    load_model_config,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def evaluate_cv_from_cache(
    model_name: str,
    params: dict[str, Any],
    folds_cache: list[dict[str, Any]],
    n_total: int,
    y_dev: np.ndarray,
    disbursed_amt: np.ndarray,
    random_state: int,
    n_threads: int,
) -> tuple[dict[str, Any], np.ndarray, float]:
    """Train and evaluate a tabular model across precomputed leak-free CV folds."""
    oof_probs = np.zeros(n_total, dtype=np.float64)
    fold_aucs: list[float] = []
    t0 = time.perf_counter()

    for fold_data in folds_cache:
        clf = build_model(model_name, params, random_state=random_state, n_threads=n_threads)
        clf.fit(fold_data["X_tr"], fold_data["y_tr"])
        probs = clf.predict_proba(fold_data["X_va"])[:, 1]
        oof_probs[fold_data["va_idx"]] = probs
        fold_aucs.append(float(roc_auc_score(fold_data["y_va"], probs)))

    elapsed = time.perf_counter() - t0
    metrics = compute_binary_metrics(y_dev, oof_probs, threshold=0.5, loan_amounts=disbursed_amt)
    metrics["cv_roc_auc_mean"] = round(float(np.mean(fold_aucs)), 6)
    metrics["cv_roc_auc_std"] = round(float(np.std(fold_aucs, ddof=1)), 6)
    return metrics, oof_probs, elapsed


def evaluate_internal_temporal_blocks(
    dev_df: pd.DataFrame,
    y_dev: np.ndarray,
    temporal_blocks: np.ndarray,
    disbursed_amt: np.ndarray,
    tiers: tuple[str, ...],
    base_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run expanding-window temporal check INSIDE Dev Set (blocks 0..1->2, 0..2->3) for RQ2."""
    seed = int(model_cfg["benchmark"]["random_seed"])
    n_threads = int(model_cfg["benchmark"]["n_threads"])
    rows: list[dict[str, Any]] = []

    for tier in tiers:
        for val_block in (2, 3):
            tr_mask = (temporal_blocks >= 0) & (temporal_blocks < val_block)
            va_mask = temporal_blocks == val_block

            tr_idx = np.where(tr_mask)[0]
            va_idx = np.where(va_mask)[0]

            from src.features.preprocessing import build_preprocessing_pipeline

            pipe = build_preprocessing_pipeline(
                feature_tier=tier,
                model_family="tree",
                drop_redundant=False,
                winsor_quantiles=tuple(base_cfg["preprocessing"]["winsor_quantiles"]),
                te_smoothing=float(base_cfg["preprocessing"]["target_encoding_smoothing"]),
                te_cv_folds=int(base_cfg["preprocessing"]["target_encoding_cv_folds"]),
                random_state=seed,
            )
            X_tr = pipe.fit_transform(dev_df.iloc[tr_idx], y_dev[tr_idx]).astype(np.float32)
            X_va = pipe.transform(dev_df.iloc[va_idx]).astype(np.float32)

            t0 = time.perf_counter()
            lgb_clf = build_model("lightgbm", model_cfg["baselines"]["lightgbm"], random_state=seed, n_threads=n_threads)
            lgb_clf.fit(X_tr, y_dev[tr_idx])
            probs = lgb_clf.predict_proba(X_va)[:, 1]
            dur = time.perf_counter() - t0
            m = compute_binary_metrics(y_dev[va_idx], probs, loan_amounts=disbursed_amt[va_idx])
            rows.append(
                {
                    "split_protocol": f"dev_temporal_block_0..{val_block-1}_to_{val_block}",
                    "model": "lightgbm",
                    "feature_tier": tier,
                    "roc_auc": m["roc_auc"],
                    "pr_auc": m["pr_auc"],
                    "brier_score": m["brier_score"],
                    "ece": m["ece"],
                    "duration_sec": round(dur, 2),
                }
            )
            del X_tr, X_va, lgb_clf
            gc.collect()

    return rows


def main() -> int:
    root = get_project_root()
    base_cfg = load_config()
    model_cfg = load_model_config()
    seed = int(model_cfg["benchmark"]["random_seed"])
    n_threads = int(model_cfg["benchmark"]["n_threads"])
    default_tier = str(model_cfg["benchmark"]["default_feature_tier"])

    results_dir = root / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    save_environment_snapshot(results_dir, random_seed=seed)
    tracker = ExperimentTracker(results_dir)

    logger.info("Loading Primary OOT Development Set ONLY (Test Set remains isolated)...")
    dev_df, y_dev, cv_folds, temporal_blocks, disbursed_amt = load_dev_only_data(base_cfg)
    n_dev = len(dev_df)

    # -------------------------------------------------------------------------
    # TRACK A: BASELINE COMPARISON (6 core models on F2_safe, 5-Fold Stratified Dev CV)
    # -------------------------------------------------------------------------
    logger.info("=== TRACK A: Baseline Benchmark on feature_tier=%s ===", default_tier)
    baseline_rows: list[dict[str, Any]] = []

    linear_cache = build_fold_cache(dev_df, y_dev, cv_folds, default_tier, "linear", True, base_cfg)
    tree_cache = build_fold_cache(dev_df, y_dev, cv_folds, default_tier, "tree", False, base_cfg)

    model_specs = [
        ("logistic_regression", "linear", linear_cache, model_cfg["baselines"]["logistic_regression"], "Standard linear credit scorecard baseline (Winsor+SignedLog+RobustScaler+OHE+OOF_TE)"),
        ("mlp", "mlp", linear_cache, model_cfg["baselines"]["mlp"], "Tabular MLP baseline on scaled/encoded F2_safe features"),
        ("decision_tree", "tree", tree_cache, model_cfg["baselines"]["decision_tree"], "Controlled single decision tree baseline"),
        ("random_forest", "tree", tree_cache, model_cfg["baselines"]["random_forest"], "Bagging ensemble baseline"),
        ("xgboost", "tree", tree_cache, model_cfg["baselines"]["xgboost"], "XGBoost hist baseline with OOF Smoothed TE"),
        ("lightgbm", "tree", tree_cache, model_cfg["baselines"]["lightgbm"], "LightGBM leaf-wise baseline with OOF Smoothed TE"),
    ]

    for m_name, m_fam, cache_obj, params, note in model_specs:
        exp_id = f"M4_A_baseline_{m_name}_{default_tier}"
        try:
            logger.info("Running %s...", exp_id)
            metrics, _, dur = evaluate_cv_from_cache(
                m_name, params, cache_obj, n_dev, y_dev, disbursed_amt, seed, n_threads
            )
            tracker.log_experiment(
                experiment_id=exp_id,
                track="Track_A_Baseline",
                model=m_name,
                feature_tier=default_tier,
                preprocessing=f"{m_fam}_drop_redundant={m_fam in {'linear','mlp'}}",
                dataset_split="oot_dev_5fold_cv",
                random_seed=seed,
                training_duration_sec=dur,
                status="SUCCESS",
                metrics=metrics,
                best_parameters=params,
                notes=note,
            )
            baseline_rows.append({"experiment_id": exp_id, "model": m_name, "feature_tier": default_tier, "duration_sec": round(dur, 2), **metrics})
            logger.info("  -> %s | CV AUC=%.5f +- %.5f | PR-AUC=%.5f | Brier=%.5f | ECE=%.5f (%.1fs)",
                        m_name, metrics["cv_roc_auc_mean"], metrics["cv_roc_auc_std"], metrics["pr_auc"], metrics["brier_score"], metrics["ece"], dur)
        except Exception as exc:
            logger.exception("Experiment %s failed: %s", exp_id, exc)
            tracker.log_experiment(
                experiment_id=exp_id, track="Track_A_Baseline", model=m_name, feature_tier=default_tier,
                preprocessing=m_fam, dataset_split="oot_dev_5fold_cv", random_seed=seed,
                training_duration_sec=0.0, status="FAILED", metrics={}, best_parameters=params, notes=str(exc),
            )

    del linear_cache, tree_cache
    gc.collect()

    base_out_df = pd.DataFrame(baseline_rows).sort_values("cv_roc_auc_mean", ascending=False)
    base_out_df.to_csv(results_dir / "m4_baseline_comparison.csv", index=False)

    # -------------------------------------------------------------------------
    # TRACK B: FEATURE TIER ABLATION (F0 -> F1 -> F2_safe -> F2_plus_emp) FOR RQ2
    # -------------------------------------------------------------------------
    logger.info("=== TRACK B: Feature Tier Ablation (RQ2) ===")
    ablation_rows: list[dict[str, Any]] = []
    tiers = ("F0", "F1", "F2_safe", "F2_plus_emp")

    for tier in tiers:
        if tier == default_tier:
            # Reuse Track A results cleanly for F2_safe without redundant refit
            for row in baseline_rows:
                if row["model"] in {"logistic_regression", "xgboost", "lightgbm"}:
                    ablation_rows.append(
                        {
                            "split_protocol": "oot_dev_5fold_cv",
                            "model": row["model"],
                            "feature_tier": tier,
                            "roc_auc": row["cv_roc_auc_mean"],
                            "pr_auc": row["pr_auc"],
                            "brier_score": row["brier_score"],
                            "ece": row["ece"],
                            "duration_sec": row["duration_sec"],
                        }
                    )
            continue

        # Linear (LR)
        lin_c = build_fold_cache(dev_df, y_dev, cv_folds, tier, "linear", True, base_cfg)
        m_lr, _, dur_lr = evaluate_cv_from_cache(
            "logistic_regression", model_cfg["baselines"]["logistic_regression"], lin_c, n_dev, y_dev, disbursed_amt, seed, n_threads
        )
        eid_lr = f"M4_B_ablation_logistic_regression_{tier}"
        tracker.log_experiment(
            experiment_id=eid_lr, track="Track_B_FeatureAblation", model="logistic_regression",
            feature_tier=tier, preprocessing="linear_drop_redundant=True", dataset_split="oot_dev_5fold_cv",
            random_seed=seed, training_duration_sec=dur_lr, status="SUCCESS", metrics=m_lr,
            best_parameters=model_cfg["baselines"]["logistic_regression"], notes=f"Feature ablation tier {tier}",
        )
        ablation_rows.append({"split_protocol": "oot_dev_5fold_cv", "model": "logistic_regression", "feature_tier": tier, "roc_auc": m_lr["cv_roc_auc_mean"], "pr_auc": m_lr["pr_auc"], "brier_score": m_lr["brier_score"], "ece": m_lr["ece"], "duration_sec": round(dur_lr, 2)})
        del lin_c
        gc.collect()

        # Tree (XGBoost & LightGBM)
        tree_c = build_fold_cache(dev_df, y_dev, cv_folds, tier, "tree", False, base_cfg)
        for m_name in ("xgboost", "lightgbm"):
            m_tr, _, dur_tr = evaluate_cv_from_cache(
                m_name, model_cfg["baselines"][m_name], tree_c, n_dev, y_dev, disbursed_amt, seed, n_threads
            )
            eid_tr = f"M4_B_ablation_{m_name}_{tier}"
            tracker.log_experiment(
                experiment_id=eid_tr, track="Track_B_FeatureAblation", model=m_name,
                feature_tier=tier, preprocessing="tree_drop_redundant=False", dataset_split="oot_dev_5fold_cv",
                random_seed=seed, training_duration_sec=dur_tr, status="SUCCESS", metrics=m_tr,
                best_parameters=model_cfg["baselines"][m_name], notes=f"Feature ablation tier {tier}",
            )
            ablation_rows.append({"split_protocol": "oot_dev_5fold_cv", "model": m_name, "feature_tier": tier, "roc_auc": m_tr["cv_roc_auc_mean"], "pr_auc": m_tr["pr_auc"], "brier_score": m_tr["brier_score"], "ece": m_tr["ece"], "duration_sec": round(dur_tr, 2)})
        del tree_c
        gc.collect()

    # Internal Dev expanding-window temporal check across tiers (never touching Locked Test)
    logger.info("Running internal Dev expanding-window temporal check across tiers...")
    temp_rows = evaluate_internal_temporal_blocks(dev_df, y_dev, temporal_blocks, disbursed_amt, tiers, base_cfg, model_cfg)
    ablation_rows.extend(temp_rows)

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(results_dir / "m4_feature_ablation.csv", index=False)
    logger.info("[OK] Saved m4_baseline_comparison.csv and m4_feature_ablation.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
