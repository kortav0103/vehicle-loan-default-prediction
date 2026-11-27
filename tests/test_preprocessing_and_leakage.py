"""Unit tests verifying split integrity, feature engineering correctness, and zero data leakage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.load_data import (
    get_project_root,
    load_config,
    parse_disbursal_dates,
    parse_dob_century_safe,
)
from src.data.splits import load_split_frames
from src.features.engineering import LTFSFeatureEngineer, get_feature_rationale_table
from src.features.preprocessing import OOFSmoothedTargetEncoder, SignedLog1pTransformer, build_preprocessing_pipeline


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def test_raw_files_remain_immutable() -> None:
    root = get_project_root()
    meta = json.loads((root / "reports" / "audit" / "dataset_metadata.json").read_text(encoding="utf-8"))
    for entry in meta["raw_files"]:
        fp = root / entry["relative_path"]
        assert fp.exists()
        assert fp.stat().st_size == entry["size_bytes"]
        assert _sha256(fp) == entry["sha256"]


def test_split_integrity_and_zero_overlap() -> None:
    root = get_project_root()
    meta = json.loads((root / "data" / "interim" / "splits" / "split_metadata.json").read_text(encoding="utf-8"))
    assert meta["total_rows"] == 233154

    for split_name in ("oot_split", "random_split"):
        checks = meta[split_name]["integrity_checks"]
        assert checks["uid_overlap_train_val"] == 0
        assert checks["uid_overlap_train_test"] == 0
        assert checks["uid_overlap_val_test"] == 0
        assert checks["feature_row_overlap_dev_test"] == 0

    train_df, val_df, test_df, dev_df, split_df = load_split_frames("oot")
    assert len(train_df) + len(val_df) + len(test_df) == 233154
    assert len(train_df) + len(val_df) == len(dev_df)
    assert parse_disbursal_dates(dev_df["DisbursalDate"]).max() < parse_disbursal_dates(test_df["DisbursalDate"]).min()
    assert abs(train_df["loan_default"].mean() - val_df["loan_default"].mean()) < 1e-4


def test_dob_century_fix_and_signed_log1p() -> None:
    dob = pd.Series(["01-01-68", "15-11-91", "01-01-00"])
    disb_yr = pd.Series([2018, 2018, 2018])
    fixed = parse_dob_century_safe(dob, reference_years=disb_yr)
    assert fixed.dt.year.tolist() == [1968, 1991, 2000]

    s = SignedLog1pTransformer()
    arr = np.array([[-6678296.0], [0.0], [1000.0]])
    out = s.fit_transform(arr)
    assert np.isfinite(out).all()
    assert out[0, 0] < 0 and np.isclose(out[1, 0], 0.0) and out[2, 0] > 0


def test_oof_target_encoder_prevents_self_leakage() -> None:
    # Single-occurrence category 'RARE_1' with target=1 must NOT leak its own label into fit_transform
    # (>= 5 members per class so 5-fold StratifiedKFold emits zero warnings)
    X = pd.DataFrame(
        {
            "cat": [
                "A", "A", "A", "A", "A", "A", "A", "A",
                "B", "B", "B", "B",
                "RARE_1", "C", "C",
            ]
        }
    )
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0])

    enc = OOFSmoothedTargetEncoder(n_splits=5, smoothing=10.0, random_state=42)
    transformed = enc.fit_transform(X, y)

    # Row 12 is 'RARE_1' (only appears once, y=1). In its OOF fold, training folds have zero 'RARE_1',
    # so it maps strictly to the training-fold prior (<= 0.35), never pulled toward its own y=1!
    assert transformed[12, 0] <= 0.35

    # Calling .fit(X, y) followed by .transform(X) on the training data must emit a UserWarning
    # and still return the leak-free OOF encodings rather than in-sample category maps (ISS-05 fix)
    enc2 = OOFSmoothedTargetEncoder(n_splits=5, smoothing=10.0, random_state=42).fit(X, y)
    with pytest.warns(UserWarning, match="out-of-fold encodings"):
        transformed_via_fit_then_transform = enc2.transform(X)
    assert np.allclose(transformed, transformed_via_fit_then_transform)

    # Unseen cold-start category at transform time maps strictly to global_prior_
    X_unseen = pd.DataFrame({"cat": ["UNSEEN_XYZ"]})
    out_unseen = enc.transform(X_unseen)
    assert np.isclose(out_unseen[0, 0], float(np.mean(y)))


def test_feature_engineer_edge_cases_zero_asset_cost() -> None:
    train_df, _, _, _, _ = load_split_frames("oot")
    sample = train_df.iloc[:10].copy()
    sample.iloc[0, sample.columns.get_loc("asset_cost")] = 0
    sample.iloc[0, sample.columns.get_loc("disbursed_amount")] = 0
    sample.iloc[0, sample.columns.get_loc("PERFORM_CNS.SCORE")] = 0
    sample.iloc[0, sample.columns.get_loc("PERFORM_CNS.SCORE.DESCRIPTION")] = "UNKNOWN_FUTURE_CODE"

    for tier in ("F0", "F1", "F2_safe", "F2_plus_emp"):
        eng = LTFSFeatureEngineer(feature_tier=tier)
        out = eng.fit_transform(sample)
        for col in ("down_payment_ratio", "ltv_computed", "ltv_discrepancy", "credit_utilization_ratio"):
            if col in out.columns:
                assert np.isfinite(out[col].to_numpy(dtype=np.float64)).all()


def test_pipeline_no_test_contamination_and_all_tiers() -> None:
    train_df, val_df, _, _, _ = load_split_frames("oot")
    tr_sub = train_df.iloc[:4000].copy()
    va_sub = val_df.iloc[:1000].copy()
    y_tr = tr_sub["loan_default"].to_numpy()

    # Inject synthetic extreme/unseen row in a second copy of val to ensure transform(va_sub) is row-independent
    va_poisoned = va_sub.copy()
    va_poisoned.iloc[0, va_poisoned.columns.get_loc("disbursed_amount")] = 9_999_999_999
    va_poisoned.iloc[0, va_poisoned.columns.get_loc("supplier_id")] = -999999
    va_poisoned.iloc[0, va_poisoned.columns.get_loc("Employment.Type")] = "ALIEN_TYPE"

    for tier in ("F0", "F1", "F2_safe", "F2_plus_emp"):
        for family in ("linear", "tree"):
            pipe = build_preprocessing_pipeline(
                feature_tier=tier,
                model_family=family,
                drop_redundant=(family == "linear"),
                random_state=42,
            )
            X_tr_mat = pipe.fit_transform(tr_sub, y_tr)
            X_va_mat = pipe.transform(va_sub)
            X_va_poisoned_mat = pipe.transform(va_poisoned)

            assert X_tr_mat.shape[1] == X_va_mat.shape[1]
            assert np.isfinite(X_tr_mat).all()
            assert np.isfinite(X_va_mat).all()
            assert np.isfinite(X_va_poisoned_mat).all()
            # Rows 1..end of va_sub must be identical regardless of extreme row 0
            assert np.allclose(X_va_mat[1:], X_va_poisoned_mat[1:], equal_nan=True)

            out_names = pipe.named_steps["preprocessor"].get_feature_names_out()
            assert len(out_names) == X_tr_mat.shape[1]


def test_rationale_table_matches_f1_engineered_columns() -> None:
    train_df, _, _, _, _ = load_split_frames("oot")
    eng = LTFSFeatureEngineer(feature_tier="F1")
    out_df = eng.fit_transform(train_df.iloc[:100])
    rationale_df = get_feature_rationale_table()
    for feat in rationale_df["feature"]:
        assert feat in out_df.columns, f"Documented engineered feature '{feat}' missing in transformed DataFrame"
