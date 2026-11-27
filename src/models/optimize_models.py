"""Stage-2 Focused Hyperparameter Optimization, Forward Dev Temporal Stability, and Imbalance Verification.

Strictly operates ONLY on the Primary OOT Development Set (<= 2018-10-22, N=182,501).
Both Locked OOT Test Set and Locked Random Test Set remain 100% untouched.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from typing import Any

from src.data.load_data import get_project_root, load_config

os.environ.setdefault("MPLCONFIGDIR", str(get_project_root() / ".matplotlib_cache"))

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score

from src.evaluation.metrics import compute_binary_metrics
from src.evaluation.tracker import ExperimentTracker
from src.features.preprocessing import build_preprocessing_pipeline
from src.models.imbalance_calibration import _run_imbalance_for_cached_model
from src.models.registry import (
    build_fold_cache,
    build_model,
    load_dev_only_data,
    load_model_config,
)
from src.models.train_benchmark import evaluate_cv_from_cache

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _clip_int(val: Any, bounds: list[int]) -> int:
    return int(max(bounds[0], min(bounds[1], round(float(val)))))


def _clip_float(val: Any, bounds: list[float]) -> float:
    return float(max(bounds[0], min(bounds[1], float(val))))


def _log_trial_callback(model_name: str, every: int = 5):
    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if (trial.number + 1) % every == 0 or trial.number == 0:
            logger.info(
                "  [%s] trial %d finished | value=%.5f | best=%.5f",
                model_name,
                trial.number + 1,
                float(trial.value) if trial.value is not None else float("nan"),
                float(study.best_value),
            )

    return _cb


def _tune_lightgbm_stage2(
    tree_cache: list[dict[str, Any]],
    tune_cfg: dict[str, Any],
    warm_params: dict[str, Any] | None,
    seed: int,
    n_threads: int,
) -> dict[str, Any]:
    space = tune_cfg["search_space"]
    es_rounds = int(tune_cfg["early_stopping_rounds"])
    max_est = int(tune_cfg["max_estimators"])

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": max_est,
            "num_leaves": trial.suggest_int("num_leaves", space["num_leaves"][0], space["num_leaves"][1]),
            "max_depth": trial.suggest_int("max_depth", space["max_depth"][0], space["max_depth"][1]),
            "min_child_samples": trial.suggest_int("min_child_samples", space["min_child_samples"][0], space["min_child_samples"][1]),
            "min_split_gain": trial.suggest_float("min_split_gain", space["min_split_gain"][0], space["min_split_gain"][1]),
            "learning_rate": trial.suggest_float("learning_rate", space["learning_rate"][0], space["learning_rate"][1], log=True),
            "subsample": trial.suggest_float("subsample", space["subsample"][0], space["subsample"][1]),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", space["colsample_bytree"][0], space["colsample_bytree"][1]),
            "reg_alpha": trial.suggest_float("reg_alpha", space["reg_alpha"][0], space["reg_alpha"][1], log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", space["reg_lambda"][0], space["reg_lambda"][1], log=True),
            "verbose": -1,
        }
        aucs = []
        best_iters = []
        for fd in tree_cache:
            clf = build_model("lightgbm", params, seed, n_threads)
            clf.fit(
                fd["X_tr"],
                fd["y_tr"],
                eval_set=[(fd["X_va"], fd["y_va"])],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(es_rounds, verbose=False)],
            )
            probs = clf.predict_proba(fd["X_va"])[:, 1]
            aucs.append(float(roc_auc_score(fd["y_va"], probs)))
            best_iters.append(int(clf.best_iteration_ or max_est))
        trial.set_user_attr("mean_best_iter", int(round(float(np.mean(best_iters)))))
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed + 100))
    if warm_params:
        study.enqueue_trial(
            {
                "num_leaves": _clip_int(warm_params["num_leaves"], space["num_leaves"]),
                "max_depth": _clip_int(warm_params["max_depth"], space["max_depth"]),
                "min_child_samples": _clip_int(warm_params["min_child_samples"], space["min_child_samples"]),
                "min_split_gain": _clip_float(warm_params.get("min_split_gain", 0.0), space["min_split_gain"]),
                "learning_rate": _clip_float(warm_params["learning_rate"], space["learning_rate"]),
                "subsample": _clip_float(warm_params["subsample"], space["subsample"]),
                "colsample_bytree": _clip_float(warm_params["colsample_bytree"], space["colsample_bytree"]),
                "reg_alpha": _clip_float(warm_params["reg_alpha"], space["reg_alpha"]),
                "reg_lambda": _clip_float(warm_params["reg_lambda"], space["reg_lambda"]),
            }
        )
    study.optimize(objective, n_trials=int(tune_cfg["n_trials"]), callbacks=[_log_trial_callback("lightgbm", 5)])

    best_p = dict(study.best_params)
    best_p["n_estimators"] = int(study.best_trial.user_attrs.get("mean_best_iter", 700))
    best_p["subsample_freq"] = 1
    best_p["verbose"] = -1
    return best_p


def _tune_xgboost_stage2(
    tree_cache: list[dict[str, Any]],
    tune_cfg: dict[str, Any],
    warm_params: dict[str, Any] | None,
    seed: int,
    n_threads: int,
) -> dict[str, Any]:
    space = tune_cfg["search_space"]
    es_rounds = int(tune_cfg["early_stopping_rounds"])
    max_est = int(tune_cfg["max_estimators"])

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": max_est,
            "max_depth": trial.suggest_int("max_depth", space["max_depth"][0], space["max_depth"][1]),
            "min_child_weight": trial.suggest_float("min_child_weight", space["min_child_weight"][0], space["min_child_weight"][1], log=True),
            "learning_rate": trial.suggest_float("learning_rate", space["learning_rate"][0], space["learning_rate"][1], log=True),
            "subsample": trial.suggest_float("subsample", space["subsample"][0], space["subsample"][1]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", space["colsample_bytree"][0], space["colsample_bytree"][1]),
            "gamma": trial.suggest_float("gamma", space["gamma"][0], space["gamma"][1]),
            "reg_alpha": trial.suggest_float("reg_alpha", space["reg_alpha"][0], space["reg_alpha"][1], log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", space["reg_lambda"][0], space["reg_lambda"][1], log=True),
            "tree_method": "hist",
            "eval_metric": "auc",
            "early_stopping_rounds": es_rounds,
        }
        aucs = []
        best_iters = []
        for fd in tree_cache:
            clf = build_model("xgboost", params, seed, n_threads)
            clf.fit(fd["X_tr"], fd["y_tr"], eval_set=[(fd["X_va"], fd["y_va"])], verbose=False)
            probs = clf.predict_proba(fd["X_va"])[:, 1]
            aucs.append(float(roc_auc_score(fd["y_va"], probs)))
            best_iters.append(int(clf.best_iteration + 1 if clf.best_iteration is not None else max_est))
        trial.set_user_attr("mean_best_iter", int(round(float(np.mean(best_iters)))))
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed + 200))
    if warm_params:
        study.enqueue_trial(
            {
                "max_depth": _clip_int(warm_params["max_depth"], space["max_depth"]),
                "min_child_weight": _clip_float(warm_params["min_child_weight"], space["min_child_weight"]),
                "learning_rate": _clip_float(warm_params["learning_rate"], space["learning_rate"]),
                "subsample": _clip_float(warm_params["subsample"], space["subsample"]),
                "colsample_bytree": _clip_float(warm_params["colsample_bytree"], space["colsample_bytree"]),
                "gamma": _clip_float(warm_params["gamma"], space["gamma"]),
                "reg_alpha": _clip_float(warm_params["reg_alpha"], space["reg_alpha"]),
                "reg_lambda": _clip_float(warm_params["reg_lambda"], space["reg_lambda"]),
            }
        )
    study.optimize(objective, n_trials=int(tune_cfg["n_trials"]), callbacks=[_log_trial_callback("xgboost", 5)])

    best_p = dict(study.best_params)
    best_p["n_estimators"] = int(study.best_trial.user_attrs.get("mean_best_iter", 1000))
    best_p["tree_method"] = "hist"
    best_p["eval_metric"] = "logloss"
    return best_p


def _run_forward_temporal_check(
    dev_df: pd.DataFrame,
    y_dev: np.ndarray,
    temporal_blocks: np.ndarray,
    disbursed_amt: np.ndarray,
    model_specs: list[tuple[str, str, dict[str, Any], float]],
    base_cfg: dict[str, Any],
    tier: str,
    seed: int,
    n_threads: int,
) -> list[dict[str, Any]]:
    """Evaluate final candidates on internal Dev expanding temporal split (blocks 0..2 -> block 3)."""
    tr_idx = np.where((temporal_blocks >= 0) & (temporal_blocks <= 2))[0]
    va_idx = np.where(temporal_blocks == 3)[0]
    assert len(set(tr_idx) & set(va_idx)) == 0
    assert len(tr_idx) + len(va_idx) == len(dev_df)

    winsor_q = tuple(base_cfg["preprocessing"]["winsor_quantiles"])
    te_smooth = float(base_cfg["preprocessing"]["target_encoding_smoothing"])
    te_folds = int(base_cfg["preprocessing"]["target_encoding_cv_folds"])

    # Fit linear & tree preprocessors strictly on temporal blocks 0..2
    lin_pipe = build_preprocessing_pipeline(tier, "linear", True, winsor_q, te_smooth, te_folds, seed)
    X_tr_lin = lin_pipe.fit_transform(dev_df.iloc[tr_idx], y_dev[tr_idx]).astype(np.float32)
    X_va_lin = lin_pipe.transform(dev_df.iloc[va_idx]).astype(np.float32)

    tree_pipe = build_preprocessing_pipeline(tier, "tree", False, winsor_q, te_smooth, te_folds, seed)
    X_tr_tree = tree_pipe.fit_transform(dev_df.iloc[tr_idx], y_dev[tr_idx]).astype(np.float32)
    X_va_tree = tree_pipe.transform(dev_df.iloc[va_idx]).astype(np.float32)

    y_tr = y_dev[tr_idx]
    y_va = y_dev[va_idx]
    amt_va = disbursed_amt[va_idx]

    rows: list[dict[str, Any]] = []

    for label, m_name, params, cv_auc in model_specs:
        is_linear = m_name == "logistic_regression"
        X_tr = X_tr_lin if is_linear else X_tr_tree
        X_va = X_va_lin if is_linear else X_va_tree

        t0 = time.perf_counter()
        clf = build_model(m_name, params, seed, n_threads)
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_va)[:, 1]
        dur = time.perf_counter() - t0

        m = compute_binary_metrics(y_va, probs, threshold=0.5, loan_amounts=amt_va)
        rows.append(
            {
                "model": label,
                "split_protocol": "dev_temporal_block_0..2_to_3",
                "n_train": int(len(tr_idx)),
                "n_val": int(len(va_idx)),
                "val_default_rate": round(float(np.mean(y_va)), 6),
                "cv_roc_auc_mean": round(cv_auc, 6),
                "temporal_block3_roc_auc": m["roc_auc"],
                "auc_drop_cv_to_temp": round(cv_auc - m["roc_auc"], 6),
                "temporal_block3_pr_auc": m["pr_auc"],
                "temporal_block3_ks": m["ks_stat"],
                "temporal_block3_brier": m["brier_score"],
                "temporal_block3_ece": m["ece"],
                "temporal_block3_cost_5to1_bayes": m["expected_cost_5to1_bayes"],
                "fit_sec": round(dur, 2),
            }
        )
        del clf

    del X_tr_lin, X_va_lin, X_tr_tree, X_va_tree
    gc.collect()
    return rows


def generate_opt_stage2_figure(
    tune_df: pd.DataFrame,
    temp_df: pd.DataFrame,
    base_df: pd.DataFrame,
    s1_df: pd.DataFrame,
    fig_path: os.PathLike[str],
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "axes.titlesize": 11, "axes.titleweight": "bold"})

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))

    # Left Panel: Baseline -> Stage 1 -> Stage 2 progression for selected & reference models
    base_map = dict(zip(base_df["model"], base_df["cv_roc_auc_mean"]))
    s1_map = dict(zip(s1_df["model"], s1_df["tuned_cv_auc"]))
    s2_map = dict(zip(tune_df["model"], tune_df["stage2_cv_auc"]))
    s2_sd_map = dict(zip(tune_df["model"], tune_df["cv_roc_auc_std"]))

    models_order = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
    labels_pretty = ["Logistic Reg\n(C=0.1)", "Random Forest\n(Baseline Kept)", "XGBoost\n(Hist)", "LightGBM\n(Leaf-wise)"]
    x = np.arange(len(models_order))
    w = 0.25

    b_vals = [base_map.get(m, np.nan) for m in models_order]
    s1_vals = [s1_map.get(m, np.nan) for m in models_order]
    s2_vals = [s2_map.get(m, np.nan) for m in models_order]

    axes[0].bar(x - w, b_vals, width=w, label="Track A: Baseline", color="#718096", alpha=0.88)
    axes[0].bar(x, s1_vals, width=w, label="Track C: Stage-1 Tuning", color="#3182ce", alpha=0.88)
    axes[0].bar(
        x + w,
        s2_vals,
        width=w,
        yerr=[s2_sd_map.get(m, 0.0) if np.isfinite(s2_map.get(m, np.nan)) else 0.0 for m in models_order],
        capsize=3,
        label="Stage-2 Focused Optuna",
        color="#2f855a",
        alpha=0.92,
    )
    for i, m in enumerate(models_order):
        if np.isfinite(s2_vals[i]):
            axes[0].text(x[i] + w, s2_vals[i] + s2_sd_map.get(m, 0.0) + 0.0005, f"{s2_vals[i]:.4f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        elif m == "random_forest" and np.isfinite(b_vals[i]):
            axes[0].text(x[i] - w, b_vals[i] + 0.0005, f"{b_vals[i]:.4f}", ha="center", va="bottom", fontsize=7.5)
        elif m == "logistic_regression" and np.isfinite(s1_vals[i]):
            axes[0].text(x[i], s1_vals[i] + 0.0005, f"{s1_vals[i]:.4f}", ha="center", va="bottom", fontsize=7.5)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels_pretty, fontsize=8.5)
    axes[0].set_ylabel("5-Fold Dev CV ROC-AUC")
    axes[0].set_ylim(0.660, 0.684)
    axes[0].set_title("Focused Tuning Progression (5-Fold Dev CV, F2_safe)")
    axes[0].legend(loc="upper left", fontsize=8)

    # Right Panel: 5-Fold CV ROC-AUC vs Forward Dev Temporal Block 3 ROC-AUC & ECE
    t_plot = temp_df[~temp_df["model"].str.contains("blend")].copy()
    x2 = np.arange(len(t_plot))
    w2 = 0.34
    axes[1].bar(x2 - w2 / 2, t_plot["cv_roc_auc_mean"], width=w2, color="#2b6cb0", alpha=0.88, label="5-Fold Stratified Dev CV AUC")
    axes[1].bar(x2 + w2 / 2, t_plot["temporal_block3_roc_auc"], width=w2, color="#dd6b20", alpha=0.90, label="Dev Future Temporal Block 3 AUC")
    for i, (_, row) in enumerate(t_plot.iterrows()):
        axes[1].text(
            x2[i] + w2 / 2,
            row["temporal_block3_roc_auc"] + 0.0006,
            f"{row['temporal_block3_roc_auc']:.4f}\n(ECE {row['temporal_block3_ece']:.3f})",
            ha="center",
            va="bottom",
            fontsize=7.2,
        )
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([m.replace("_", "\n") for m in t_plot["model"]], fontsize=8.2)
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_ylim(0.640, 0.684)
    axes[1].set_title("CV vs. Chronological Dev Block 3 Stability (Drift & ECE)")
    axes[1].legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    root = get_project_root()
    base_cfg = load_config()
    model_cfg = load_model_config()
    seed = int(model_cfg["benchmark"]["random_seed"])
    n_threads = int(model_cfg["benchmark"]["n_threads"])
    tier = str(model_cfg["benchmark"]["default_feature_tier"])

    results_dir = root / "experiments" / "results"
    fig_dir = root / "experiments" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tracker = ExperimentTracker(results_dir)

    base_df = pd.read_csv(results_dir / "m4_baseline_comparison.csv")
    s1_df = pd.read_csv(results_dir / "m4_tuning_comparison.csv")
    s1_params = json.loads((results_dir / "m4_best_params.json").read_text(encoding="utf-8"))["models"]
    m4_oof = np.load(results_dir / "m4_best_oof_predictions.npz")

    base_auc_map = dict(zip(base_df["model"], base_df["cv_roc_auc_mean"]))
    s1_auc_map = dict(zip(s1_df["model"], s1_df["tuned_cv_auc"]))

    dev_df, y_dev, cv_folds, temporal_blocks, disbursed_amt = load_dev_only_data(base_cfg)
    assert np.array_equal(y_dev, m4_oof["y_dev"])
    assert np.array_equal(cv_folds, m4_oof["cv_folds"])
    n_dev = len(dev_df)

    logger.info("=== Building F2_safe tree fold cache (RAM-safe float32) ===")
    tree_cache = build_fold_cache(dev_df, y_dev, cv_folds, tier, "tree", False, base_cfg)

    opt_cfg = model_cfg["optimization_stage2"]
    best_params_out: dict[str, Any] = {
        "feature_tier": tier,
        "excluded_models_rationale": {
            "decision_tree": "High variance single tree (ROC-AUC 0.6389), dominated by RF & GBDTs.",
            "logistic_regression": "Saturated in Stage-1 GridSearch at C=0.1 (0.66529 vs C=1.0 0.66526); kept as interpretable linear anchor.",
            "mlp": "Comparable to LR (0.66583) at 4.5x compute cost and poorer interpretability; excluded from Stage-2 tuning.",
            "random_forest": "Regressed under Stage-1 Optuna tuning (0.66844 vs Baseline 0.66880); Baseline RF retained for reference.",
        },
        "models": {
            "logistic_regression": s1_params["logistic_regression"],
            "random_forest_baseline": model_cfg["baselines"]["random_forest"],
        },
    }

    tuning_rows: list[dict[str, Any]] = []
    oof_store: dict[str, np.ndarray] = {
        "y_dev": y_dev,
        "cv_folds": cv_folds,
        "temporal_blocks": temporal_blocks,
        "disbursed_amount": disbursed_amt,
        "logistic_regression": m4_oof["logistic_regression"],
    }

    # Recompute exact Baseline RandomForest OOF (since Stage-1 RF regressed -0.00037)
    logger.info("=== Recomputing Baseline RandomForest 5-Fold OOF (retained over regressed Stage-1) ===")
    m_rf, oof_rf, dur_rf = evaluate_cv_from_cache(
        "random_forest", model_cfg["baselines"]["random_forest"], tree_cache, n_dev, y_dev, disbursed_amt, seed, n_threads
    )
    oof_store["random_forest"] = oof_rf
    logger.info("  -> Retained Baseline RF OOF verified: CV AUC=%.5f (%.1fs)", m_rf["cv_roc_auc_mean"], dur_rf)

    # Tune Top-2 GBDT candidates: LightGBM and XGBoost
    tuners = [
        ("lightgbm", _tune_lightgbm_stage2, s1_params.get("lightgbm")),
        ("xgboost", _tune_xgboost_stage2, s1_params.get("xgboost")),
    ]

    for m_name, tuner_fn, warm_p in tuners:
        m_tune_cfg = opt_cfg[m_name]
        logger.info("=== Stage-2 Tuning %s (%s, %d trials, 5-Fold Dev CV) ===", m_name, m_tune_cfg["strategy"], m_tune_cfg["n_trials"])
        t0 = time.perf_counter()
        best_p = tuner_fn(tree_cache, m_tune_cfg, warm_p, seed, n_threads)
        tune_sec = time.perf_counter() - t0

        m_res, oof_arr, eval_sec = evaluate_cv_from_cache(m_name, best_p, tree_cache, n_dev, y_dev, disbursed_amt, seed, n_threads)
        oof_store[m_name] = oof_arr
        best_params_out["models"][m_name] = best_p

        b_auc = float(base_auc_map.get(m_name, m_res["cv_roc_auc_mean"]))
        s1_auc = float(s1_auc_map.get(m_name, b_auc))
        tracker.log_experiment(
            experiment_id=f"M5_OPT_stage2_{m_name}_{tier}",
            track="Track_E_Stage2Optimization",
            model=m_name,
            feature_tier=tier,
            preprocessing="tree_drop_redundant=False",
            dataset_split="oot_dev_5fold_cv",
            random_seed=seed,
            training_duration_sec=tune_sec + eval_sec,
            status="SUCCESS",
            metrics=m_res,
            best_parameters=best_p,
            notes=f"Stage-2 Optuna ({m_tune_cfg['n_trials']} trials, search={tune_sec:.1f}s, eval={eval_sec:.1f}s)",
        )
        tuning_rows.append(
            {
                "model": m_name,
                "stage": f"Stage2_{m_tune_cfg['strategy']} ({m_tune_cfg['n_trials']} trials)",
                "baseline_cv_auc": round(b_auc, 6),
                "stage1_cv_auc": round(s1_auc, 6),
                "stage2_cv_auc": m_res["cv_roc_auc_mean"],
                "delta_vs_baseline": round(m_res["cv_roc_auc_mean"] - b_auc, 6),
                "delta_vs_stage1": round(m_res["cv_roc_auc_mean"] - s1_auc, 6),
                "tuning_sec": round(tune_sec, 2),
                "eval_5fold_sec": round(eval_sec, 2),
                **m_res,
            }
        )
        logger.info(
            "  -> Stage-2 %s | CV AUC=%.5f +- %.5f (vs S1 %+.5f) | PR-AUC=%.5f | Brier=%.5f | ECE=%.5f (search=%.1fs, eval=%.1fs)",
            m_name,
            m_res["cv_roc_auc_mean"],
            m_res["cv_roc_auc_std"],
            m_res["cv_roc_auc_mean"] - s1_auc,
            m_res["pr_auc"],
            m_res["brier_score"],
            m_res["ece"],
            tune_sec,
            eval_sec,
        )

    tune_out_df = pd.DataFrame(tuning_rows).sort_values("stage2_cv_auc", ascending=False)
    tune_out_df.to_csv(results_dir / "m5_opt_tuning_results.csv", index=False)
    (results_dir / "m5_opt_best_params.json").write_text(json.dumps(best_params_out, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(results_dir / "m5_opt_oof_predictions.npz", **oof_store)
    logger.info("[OK] Saved m5_opt_tuning_results.csv, m5_opt_best_params.json, and m5_opt_oof_predictions.npz")

    # -------------------------------------------------------------------------
    # Stage-2 Imbalance & Mild Weighting Verification on Stage-2 LightGBM
    # -------------------------------------------------------------------------
    logger.info("=== Verifying Class Imbalance Strategies on Stage-2 LightGBM (incl. scale_pos_weight=2.0) ===")
    imb_oof_store: dict[str, np.ndarray] = {"y_dev": y_dev, "cv_folds": cv_folds, "disbursed_amount": disbursed_amt}
    imb_rows = _run_imbalance_for_cached_model(
        model_name="lightgbm",
        base_params=best_params_out["models"]["lightgbm"],
        folds_cache=tree_cache,
        strategies=["natural", "mild_weight_2.0", "class_weight", "rus_0.5", "smote_0.5", "smote_1.0"],
        y_dev=y_dev,
        cv_folds=cv_folds,
        disbursed_amt=disbursed_amt,
        tier=tier,
        seed=seed,
        n_threads=n_threads,
        tracker=tracker,
        track="Track_E_ImbalanceVerification",
        exp_prefix="M5_OPT_IMB",
        oof_store=imb_oof_store,
    )
    pd.DataFrame(imb_rows).to_csv(results_dir / "m5_imbalance_verification.csv", index=False)
    np.savez_compressed(results_dir / "m5_imbalance_oof_predictions.npz", **imb_oof_store)

    del tree_cache
    gc.collect()

    # -------------------------------------------------------------------------
    # Forward Dev Temporal Stability Check (blocks 0..2 -> block 3)
    # -------------------------------------------------------------------------
    logger.info("=== Running Internal Dev Forward Temporal Stability Check (blocks 0..2 -> block 3) ===")
    cv_lookup = dict(zip(tune_out_df["model"], tune_out_df["stage2_cv_auc"]))
    temp_specs = [
        ("logistic_regression_s1", "logistic_regression", s1_params["logistic_regression"], float(s1_auc_map["logistic_regression"])),
        ("random_forest_base", "random_forest", model_cfg["baselines"]["random_forest"], float(base_auc_map["random_forest"])),
        ("xgboost_stage2", "xgboost", best_params_out["models"]["xgboost"], float(cv_lookup["xgboost"])),
        ("lightgbm_stage2", "lightgbm", best_params_out["models"]["lightgbm"], float(cv_lookup["lightgbm"])),
    ]
    temp_rows = _run_forward_temporal_check(
        dev_df,
        y_dev,
        temporal_blocks,
        disbursed_amt,
        temp_specs,
        base_cfg,
        tier,
        seed,
        n_threads,
    )
    temp_df = pd.DataFrame(temp_rows)
    temp_df.to_csv(results_dir / "m5_opt_temporal_stability.csv", index=False)

    fig_path = fig_dir / "opt_02_stage2_tuning_and_temporal.png"
    generate_opt_stage2_figure(tune_out_df, temp_df, base_df, s1_df, fig_path)
    logger.info("[OK] Saved m5_opt_temporal_stability.csv, m5_imbalance_verification.csv, and %s", fig_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
