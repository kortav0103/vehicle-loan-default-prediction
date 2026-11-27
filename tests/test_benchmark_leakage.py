"""Unit tests for Milestone 4 benchmark metrics, fold-isolated resampling, and Dev-only isolation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.data.splits import load_split_frames
from src.evaluation.metrics import (
    calibrate_oof_probabilities,
    compute_binary_metrics,
    compute_expected_cost_amount_weighted,
    compute_expected_cost_fixed,
)
from src.models.imbalance_calibration import _apply_fold_resampler
from src.models.registry import load_dev_only_data


def test_dev_only_loader_excludes_locked_test_completely() -> None:
    dev_df, y_dev, cv_folds, temp_blocks, disbursed_amt = load_dev_only_data()
    _, _, test_df, _, _ = load_split_frames("oot")

    assert len(dev_df) == 182501
    assert len(y_dev) == 182501
    assert len(disbursed_amt) == 182501
    assert set(cv_folds.tolist()) == {0, 1, 2, 3, 4}
    assert set(temp_blocks.tolist()) == {0, 1, 2, 3}
    assert len(set(dev_df["UniqueID"]) & set(test_df["UniqueID"])) == 0


def test_fold_isolated_resampling_preserves_validation_and_compatible() -> None:
    rng = np.random.default_rng(42)
    X_tr = rng.normal(size=(200, 8)).astype(np.float32)
    y_tr = np.array([0] * 160 + [1] * 40, dtype=np.int64)
    X_va = rng.normal(size=(50, 8)).astype(np.float32)
    y_va = np.array([0] * 40 + [1] * 10, dtype=np.int64)

    X_va_orig = X_va.copy()
    y_va_orig = y_va.copy()

    for strat in ("natural", "class_weight", "smote_0.5", "smote_1.0", "rus_0.5"):
        X_res, y_res = _apply_fold_resampler(X_tr, y_tr, strat, seed=42, n_threads=2)
        assert X_res.shape[1] == X_tr.shape[1]
        assert len(X_res) == len(y_res)
        if strat == "smote_1.0":
            assert int((y_res == 1).sum()) == int((y_res == 0).sum())
        elif strat == "smote_0.5":
            assert int((y_res == 1).sum()) == 80
        elif strat == "rus_0.5":
            assert int((y_res == 0).sum()) == 80

    # Validation data untouched
    assert np.array_equal(X_va, X_va_orig)
    assert np.array_equal(y_va, y_va_orig)


def test_evaluation_metrics_and_oof_calibration() -> None:
    rng = np.random.default_rng(42)
    n = 500
    y = rng.binomial(1, 0.21, size=n)
    raw_logits = np.where(y == 1, 0.7, -0.9) + rng.normal(0, 0.8, size=n)
    p = 1.0 / (1.0 + np.exp(-raw_logits))
    folds = np.arange(n) % 5
    amts = rng.uniform(30000, 90000, size=n)

    m = compute_binary_metrics(y, p, threshold=0.5, loan_amounts=amts)
    assert 0.5 < m["roc_auc"] <= 1.0
    assert abs(m["gini"] - (2 * m["roc_auc"] - 1)) < 1e-5
    assert 0.0 <= m["ks_stat"] <= 1.0
    assert 0.0 <= m["ece"] <= 1.0
    assert (m["tn_05"] + m["fp_05"] + m["fn_05"] + m["tp_05"]) == n

    c_fixed = compute_expected_cost_fixed(y, (p >= 0.5).astype(int), cost_fn=5.0, cost_fp=1.0)
    c_amt = compute_expected_cost_amount_weighted(y, (p >= 0.5).astype(int), amts)
    assert c_fixed >= 0.0 and c_amt >= 0.0

    p_platt = calibrate_oof_probabilities(y, p, folds, method="platt")
    p_iso = calibrate_oof_probabilities(y, p, folds, method="isotonic")
    assert p_platt.shape == p.shape and np.isfinite(p_platt).all()
    assert p_iso.shape == p.shape and np.isfinite(p_iso).all()
