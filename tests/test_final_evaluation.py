"""Unit tests for Final Locked-Test Outputs, Threshold Transfer, and Risk Deciles."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data.load_data import get_project_root
from src.evaluation.final_test_evaluation import _build_decile_risk_table, _run_threshold_transfer_oot
from src.evaluation.metrics import compute_binary_metrics, compute_expected_cost_fixed


def test_m6_final_deliverables_complete_and_valid() -> None:
    root = get_project_root()
    tables_dir = root / "reports" / "tables"

    required_csvs = [
        "m6_final_oot_comparison.csv",
        "m6_rq2_ablation_oot.csv",
        "m6_rq3_oot_vs_random.csv",
        "m6_final_threshold_transfer.csv",
        "m6_final_decile_table.csv",
        "m6_final_imbalance_oot.csv",
        "m6_final_subgroup_audit.csv",
    ]
    for fname in required_csvs:
        p = tables_dir / fname
        assert p.exists(), f"Missing final table: {p}"
        df = pd.read_csv(p)
        assert not df.empty, f"Empty final table: {p}"

    oot_comp = pd.read_csv(tables_dir / "m6_final_oot_comparison.csv")
    assert set(oot_comp["model"]) == {
        "lightgbm",
        "xgboost",
        "random_forest",
        "logistic_regression",
        "mlp",
        "decision_tree",
    }
    assert (oot_comp["n_test"] == 50653).all()
    assert oot_comp[["roc_auc", "pr_auc", "brier_score", "ece", "expected_cost_5to1_bayes"]].notna().all().all()

    dec = pd.read_csv(tables_dir / "m6_final_decile_table.csv")
    assert len(dec) == 10
    assert int(dec["n"].sum()) == 50653

    tt = pd.read_csv(tables_dir / "m6_final_threshold_transfer.csv")
    assert (tt["tn"] + tt["fp"] + tt["fn"] + tt["tp"] == 50653).all()

    xai_stab = json.loads((tables_dir / "m6_final_test_xai_stability.json").read_text(encoding="utf-8"))
    assert xai_stab["lgb_oot_treeshap_max_additivity_error"] < 1e-5

    cases = json.loads((tables_dir / "m6_final_test_cases.json").read_text(encoding="utf-8"))
    assert [c["case_type"] for c in cases] == ["TP", "TN", "FN", "FP"]

    # Verify m6_final_imbalance_oot.csv uses Dev-frozen thresholds (zero test-set threshold leakage)
    imb_oot = pd.read_csv(tables_dir / "m6_final_imbalance_oot.csv")
    assert "dev_frozen_cost5_thresh" in imb_oot.columns
    nat_row = imb_oot.loc[imb_oot["imbalance_strategy"] == "natural"].iloc[0]
    assert float(nat_row["expected_cost_5to1_emp_opt"]) > 0.70, (
        "Detected stale test-leaked cost_5to1_emp_opt (~0.6825) in m6_final_imbalance_oot.csv!"
    )


def test_frozen_threshold_evaluation_does_not_reoptimize_on_test() -> None:
    rng = np.random.default_rng(7)
    y_test = np.array([0] * 150 + [1] * 50, dtype=np.int64)
    probs = np.clip(0.15 + 0.35 * y_test + rng.normal(0, 0.1, size=200), 0.01, 0.99)
    amts = rng.uniform(40000, 90000, size=200)

    frozen_t = 0.22
    m = compute_binary_metrics(y_test, probs, threshold=0.5, loan_amounts=amts, opt_threshold=frozen_t)
    assert m["f1_opt_thresh"] == frozen_t
    expected_cost5_at_frozen = round(
        compute_expected_cost_fixed(y_test, (probs >= frozen_t).astype(np.int64), cost_fn=5.0, cost_fp=1.0), 6
    )
    assert m["expected_cost_5to1_emp_opt"] == expected_cost5_at_frozen


def test_decile_risk_table_and_threshold_transfer_logic() -> None:
    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1] * 10, dtype=np.int64)
    p = np.linspace(0.05, 0.95, 100)
    amts = np.full(100, 50000.0)

    dec_df = _build_decile_risk_table(y, p)
    assert len(dec_df) == 10
    assert dec_df["risk_decile"].tolist() == [f"D{i}" for i in range(1, 11)]
    assert int(dec_df["n"].sum()) == 100
    assert np.isclose(float(dec_df.loc[dec_df["risk_decile"] == "D1", "cum_default_capture_pct"].iloc[0]), 100.0)

    tt_df = _run_threshold_transfer_oot(
        y, amts, {"m_test": (p, {"RegimeA": (0.50, 0.48), "RegimeB": (0.20, 0.80)})}
    )
    assert len(tt_df) == 2
    assert (tt_df["tn"] + tt_df["fp"] + tt_df["fn"] + tt_df["tp"] == 100).all()
