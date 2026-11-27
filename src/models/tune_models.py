"""Track C: Systematic Hyperparameter Tuning strictly on 5-Fold Primary Dev CV.

Tunes Logistic Regression (GridSearch), Random Forest (Optuna TPE), LightGBM (Optuna TPE),
and XGBoost (Optuna TPE), logging search spaces, budgets, computational costs, best parameters,
and full 5-Fold CV OOF predictions.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.load_data import get_project_root, load_config
from src.evaluation.metrics import compute_binary_metrics
from src.evaluation.tracker import ExperimentTracker
from src.models.registry import (
    build_fold_cache,
    build_model,
    load_dev_only_data,
    load_model_config,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _tune_logistic_regression(
    linear_cache: list[dict[str, Any]],
    c_grid: list[float],
    seed: int,
    n_threads: int,
) -> tuple[dict[str, Any], float]:
    best_c = float(c_grid[0])
    best_auc = -1.0
    for c_val in c_grid:
        aucs = []
        for fd in linear_cache:
            clf = build_model("logistic_regression", {"C": c_val, "max_iter": 1000, "solver": "lbfgs"}, seed, n_threads)
            clf.fit(fd["X_tr"], fd["y_tr"])
            probs = clf.predict_proba(fd["X_va"])[:, 1]
            aucs.append(float(roc_auc_score(fd["y_va"], probs)))
        mean_auc = float(np.mean(aucs))
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_c = float(c_val)
    return {"C": best_c, "max_iter": 1000, "solver": "lbfgs"}, best_auc


def _tune_random_forest(
    tree_cache: list[dict[str, Any]],
    tune_cfg: dict[str, Any],
    seed: int,
    n_threads: int,
) -> dict[str, Any]:
    space = tune_cfg["search_space"]
    sub_folds = tree_cache[: int(tune_cfg.get("cv_folds_tuning", 3))]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", space["n_estimators"]),
            "max_depth": trial.suggest_int("max_depth", space["max_depth"][0], space["max_depth"][1]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", space["min_samples_leaf"][0], space["min_samples_leaf"][1]),
            "max_features": trial.suggest_categorical("max_features", space["max_features"]),
        }
        aucs = []
        for fd in sub_folds:
            clf = build_model("random_forest", params, seed, n_threads)
            clf.fit(fd["X_tr"], fd["y_tr"])
            probs = clf.predict_proba(fd["X_va"])[:, 1]
            aucs.append(float(roc_auc_score(fd["y_va"], probs)))
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=int(tune_cfg["n_trials"]))
    return dict(study.best_params)


def _tune_lightgbm(
    tree_cache: list[dict[str, Any]],
    tune_cfg: dict[str, Any],
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

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=int(tune_cfg["n_trials"]))
    best_p = dict(study.best_params)
    best_p["n_estimators"] = int(study.best_trial.user_attrs.get("mean_best_iter", 400))
    best_p["subsample_freq"] = 1
    best_p["verbose"] = -1
    return best_p


def _tune_xgboost(
    tree_cache: list[dict[str, Any]],
    tune_cfg: dict[str, Any],
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
            "reg_alpha": trial.suggest_float("reg_alpha", space["reg_alpha"][0], space["reg_alpha"][1], log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", space["reg_lambda"][0], space["reg_lambda"][1], log=True),
            "gamma": trial.suggest_float("gamma", space["gamma"][0], space["gamma"][1]),
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

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=int(tune_cfg["n_trials"]))
    best_p = dict(study.best_params)
    best_p["n_estimators"] = int(study.best_trial.user_attrs.get("mean_best_iter", 400))
    best_p["tree_method"] = "hist"
    best_p["eval_metric"] = "logloss"
    return best_p


def main() -> int:
    root = get_project_root()
    base_cfg = load_config()
    model_cfg = load_model_config()
    seed = int(model_cfg["benchmark"]["random_seed"])
    n_threads = int(model_cfg["benchmark"]["n_threads"])
    tier = str(model_cfg["benchmark"]["default_feature_tier"])

    results_dir = root / "experiments" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    tracker = ExperimentTracker(results_dir)

    dev_df, y_dev, cv_folds, _, disbursed_amt = load_dev_only_data(base_cfg)
    n_dev = len(dev_df)

    # Load baseline results for before/after comparison
    base_csv = results_dir / "m4_baseline_comparison.csv"
    base_df = pd.read_csv(base_csv) if base_csv.exists() else pd.DataFrame()
    base_auc_map = dict(zip(base_df["model"], base_df["cv_roc_auc_mean"])) if not base_df.empty else {}

    best_params_all: dict[str, Any] = {"feature_tier": tier, "models": {}}
    tuning_rows: list[dict[str, Any]] = []
    oof_store: dict[str, np.ndarray] = {"y_dev": y_dev, "cv_folds": cv_folds, "disbursed_amount": disbursed_amt}

    # 1. Tune Logistic Regression
    logger.info("=== Tuning Logistic Regression (GridSearch C) ===")
    linear_cache = build_fold_cache(dev_df, y_dev, cv_folds, tier, "linear", True, base_cfg)
    t0 = time.perf_counter()
    lr_best, _ = _tune_logistic_regression(linear_cache, model_cfg["tuning"]["logistic_regression"]["C_grid"], seed, n_threads)
    tune_sec_lr = time.perf_counter() - t0

    # Full 5-fold evaluation
    from src.models.train_benchmark import evaluate_cv_from_cache

    m_lr, oof_lr, eval_sec_lr = evaluate_cv_from_cache("logistic_regression", lr_best, linear_cache, n_dev, y_dev, disbursed_amt, seed, n_threads)
    oof_store["logistic_regression"] = oof_lr
    best_params_all["models"]["logistic_regression"] = lr_best
    tracker.log_experiment(
        experiment_id=f"M4_C_tuned_logistic_regression_{tier}", track="Track_C_Tuning", model="logistic_regression",
        feature_tier=tier, preprocessing="linear_drop_redundant=True", dataset_split="oot_dev_5fold_cv",
        random_seed=seed, training_duration_sec=tune_sec_lr + eval_sec_lr, status="SUCCESS",
        metrics=m_lr, best_parameters=lr_best, notes=f"GridSearch over C (search={tune_sec_lr:.1f}s, eval={eval_sec_lr:.1f}s)",
    )
    tuning_rows.append({"model": "logistic_regression", "strategy": "GridSearch (6 C values)", "baseline_cv_auc": base_auc_map.get("logistic_regression"), "tuned_cv_auc": m_lr["cv_roc_auc_mean"], "delta_auc": round(m_lr["cv_roc_auc_mean"] - base_auc_map.get("logistic_regression", m_lr["cv_roc_auc_mean"]), 6), "tuning_sec": round(tune_sec_lr, 2), "eval_5fold_sec": round(eval_sec_lr, 2), **m_lr})
    logger.info("  -> Tuned LR: CV AUC=%.5f (C=%s)", m_lr["cv_roc_auc_mean"], lr_best["C"])
    del linear_cache
    gc.collect()

    # 2. Tune Random Forest, LightGBM, XGBoost
    tree_cache = build_fold_cache(dev_df, y_dev, cv_folds, tier, "tree", False, base_cfg)
    for m_name, tuner_fn in [
        ("random_forest", _tune_random_forest),
        ("lightgbm", _tune_lightgbm),
        ("xgboost", _tune_xgboost),
    ]:
        logger.info("=== Tuning %s (Optuna TPE, %d trials) ===", m_name, model_cfg["tuning"][m_name]["n_trials"])
        t0 = time.perf_counter()
        best_p = tuner_fn(tree_cache, model_cfg["tuning"][m_name], seed, n_threads)
        tune_sec = time.perf_counter() - t0
        m_res, oof_arr, eval_sec = evaluate_cv_from_cache(m_name, best_p, tree_cache, n_dev, y_dev, disbursed_amt, seed, n_threads)
        oof_store[m_name] = oof_arr
        best_params_all["models"][m_name] = best_p
        tracker.log_experiment(
            experiment_id=f"M4_C_tuned_{m_name}_{tier}", track="Track_C_Tuning", model=m_name,
            feature_tier=tier, preprocessing="tree_drop_redundant=False", dataset_split="oot_dev_5fold_cv",
            random_seed=seed, training_duration_sec=tune_sec + eval_sec, status="SUCCESS",
            metrics=m_res, best_parameters=best_p,
            notes=f"Optuna TPE ({model_cfg['tuning'][m_name]['n_trials']} trials, search={tune_sec:.1f}s, eval_5fold={eval_sec:.1f}s)",
        )
        tuning_rows.append({"model": m_name, "strategy": f"Optuna TPE ({model_cfg['tuning'][m_name]['n_trials']} trials)", "baseline_cv_auc": base_auc_map.get(m_name), "tuned_cv_auc": m_res["cv_roc_auc_mean"], "delta_auc": round(m_res["cv_roc_auc_mean"] - base_auc_map.get(m_name, m_res["cv_roc_auc_mean"]), 6), "tuning_sec": round(tune_sec, 2), "eval_5fold_sec": round(eval_sec, 2), **m_res})
        logger.info("  -> Tuned %s: CV AUC=%.5f | PR-AUC=%.5f | ECE=%.5f (search=%.1fs, eval=%.1fs)",
                    m_name, m_res["cv_roc_auc_mean"], m_res["pr_auc"], m_res["ece"], tune_sec, eval_sec)

    del tree_cache
    gc.collect()

    tune_df = pd.DataFrame(tuning_rows).sort_values("tuned_cv_auc", ascending=False)
    tune_df.to_csv(results_dir / "m4_tuning_comparison.csv", index=False)
    (results_dir / "m4_best_params.json").write_text(json.dumps(best_params_all, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(results_dir / "m4_best_oof_predictions.npz", **oof_store)
    logger.info("[OK] Saved m4_tuning_comparison.csv, m4_best_params.json, and m4_best_oof_predictions.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
