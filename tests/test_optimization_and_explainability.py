"""Unit tests for Stage-2 Optimization, Nested OOF Thresholding, and Exact TreeSHAP Explainability."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from src.data.load_data import get_project_root
from src.evaluation.threshold_analysis import _select_threshold_on_train_folds, evaluate_nested_oof_regimes
from src.explainability.analyze_explainability import assign_domain_group


def test_treeshap_additive_identity_exact() -> None:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(250, 8)).astype(np.float32)
    y = ((X[:, 0] + 0.6 * X[:, 1] - 0.4 * X[:, 2]) > 0.3).astype(np.int64)

    m_lgb = lgb.LGBMClassifier(n_estimators=25, learning_rate=0.05, num_leaves=15, random_state=42, verbose=-1)
    m_lgb.fit(X[:200], y[:200])
    lgb_contrib = m_lgb.predict(X[200:], pred_contrib=True)
    lgb_margin = m_lgb.predict(X[200:], raw_score=True)
    assert float(np.max(np.abs(lgb_contrib.sum(axis=1) - lgb_margin))) < 1e-5

    m_xgb = xgb.XGBClassifier(n_estimators=25, learning_rate=0.05, max_depth=4, tree_method="hist", random_state=42)
    m_xgb.fit(X[:200], y[:200])
    dm = xgb.DMatrix(X[200:])
    xgb_contrib = m_xgb.get_booster().predict(dm, pred_contribs=True)
    xgb_inter = m_xgb.get_booster().predict(dm, pred_interactions=True)
    xgb_margin = m_xgb.get_booster().predict(dm, output_margin=True)
    assert float(np.max(np.abs(xgb_contrib.sum(axis=1) - xgb_margin))) < 1e-4
    assert float(np.max(np.abs(xgb_inter.sum(axis=(1, 2)) - xgb_margin))) < 1e-4


def test_nested_oof_threshold_isolation() -> None:
    rng = np.random.default_rng(42)
    n = 500
    y = rng.binomial(1, 0.21, size=n).astype(np.int64)
    p = np.clip(0.15 + 0.25 * y + rng.normal(0.0, 0.08, size=n), 0.01, 0.99)
    folds = np.arange(n) % 5
    amts = rng.uniform(30000, 90000, size=n)
    grid = np.linspace(0.05, 0.65, 121)

    # Corrupting fold 4 labels must NEVER change threshold selected for fold 4 from folds 0..3
    t_orig = _select_threshold_on_train_folds("F1_Optimal", y[folds != 4], p[folds != 4], grid)
    y_mod = y.copy()
    y_mod[folds == 4] = 1 - y_mod[folds == 4]
    t_after = _select_threshold_on_train_folds("F1_Optimal", y_mod[folds != 4], p[folds != 4], grid)
    assert t_orig == t_after

    rows = evaluate_nested_oof_regimes("synth", y, p, folds, amts, ["Default_0.50", "F1_Optimal", "Capacity_Top20pct_Queue"])
    assert len(rows) == 3
    for r in rows:
        assert r["tn"] + r["fp"] + r["fn"] + r["tp"] == n
        assert 0.0 <= r["flag_rate"] <= 1.0


def test_m5_artifacts_exist_valid_and_dev_only() -> None:
    root = get_project_root()
    results_dir = root / "experiments" / "results"
    required_csvs = [
        "m5_opt_tuning_results.csv",
        "m5_opt_temporal_stability.csv",
        "m5_imbalance_verification.csv",
        "m5_threshold_sweep.csv",
        "m5_threshold_comparison.csv",
        "m5_explainability_feature_ranking.csv",
        "m5_shap_interactions.csv",
        "m5_dev_subgroup_audit.csv",
    ]
    for fname in required_csvs:
        path = results_dir / fname
        assert path.exists(), f"Missing deliverable CSV: {path}"
        df = pd.read_csv(path)
        assert not df.empty, f"Empty deliverable CSV: {path}"

    # Verify strict Dev-only counts & zero NaNs in core columns
    t_comp = pd.read_csv(results_dir / "m5_threshold_comparison.csv")
    assert (t_comp["n"] == 182501).all()
    assert (t_comp["tn"] + t_comp["fp"] + t_comp["fn"] + t_comp["tp"] == 182501).all()
    assert t_comp[["precision", "recall", "f1", "cost_5to1", "cost_amount_weighted"]].notna().all().all()

    opt_tune = pd.read_csv(results_dir / "m5_opt_tuning_results.csv")
    assert opt_tune[["stage2_cv_auc", "pr_auc", "brier_score", "ece"]].notna().all().all()

    temp_df = pd.read_csv(results_dir / "m5_opt_temporal_stability.csv")
    assert set(temp_df["split_protocol"].unique()) == {"dev_temporal_block_0..2_to_3"}
    assert temp_df[["temporal_block3_roc_auc", "temporal_block3_pr_auc", "temporal_block3_ece"]].notna().all().all()

    # Verify explainability exact TreeSHAP additivity & zero test-set contamination in experiment_log
    agree = json.loads((results_dir / "m5_explainability_agreement.json").read_text(encoding="utf-8"))
    assert agree["explain_split"] == "oot_dev_unseen_cv_fold_4"
    assert agree["lgb_treeshap_max_additivity_error"] < 1e-5
    assert agree["xgb_treeshap_max_additivity_error"] < 1e-4
    assert agree["xgb_interaction_max_additivity_error"] < 1e-4

    exp_log = pd.read_csv(results_dir / "experiment_log.csv")
    pre_test_log = exp_log[~exp_log["track"].isin(["Track_F_FinalTestEvaluation"])]
    assert not pre_test_log["dataset_split"].str.contains("test", case=False).any(), "Locked test leaked into pre-test experiment_log!"


def test_domain_group_assignment_for_credit_account_features() -> None:
    assert assign_domain_group("NEW.ACCTS.IN.LAST.SIX.MONTHS") == "Existing Credit Accounts & Balances"
    assert assign_domain_group("PRIMARY.INSTAL.AMT") == "Existing Credit Accounts & Balances"
    assert assign_domain_group("SEC.INSTAL.AMT") == "Existing Credit Accounts & Balances"
    assert assign_domain_group("age_at_disbursal") == "Borrower Demographics, Employment & KYC"
    assert assign_domain_group("supplier_id__oof_te") == "Operational Channel & Geography (Proxies)"
