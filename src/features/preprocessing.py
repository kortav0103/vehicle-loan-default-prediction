"""Fold-safe, stateful preprocessing transformers and ColumnTransformer builders.

Every stateful transformation (imputation, winsorization, scaling, one-hot encoding,
frequency encoding, and Out-of-Fold Bayesian smoothed target encoding) is strictly
fitted ONLY on training data to eliminate data leakage.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from src.features.engineering import LTFSFeatureEngineer, VALID_TIERS


class QuantileWinsorizer(BaseEstimator, TransformerMixin):
    """Clip numerical columns at quantile bounds learned solely from training data."""

    def __init__(self, lower_quantile: float = 0.005, upper_quantile: float = 0.995) -> None:
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: np.ndarray | pd.DataFrame, y: Any = None) -> "QuantileWinsorizer":
        arr = np.asarray(X, dtype=np.float64)
        self.lower_bounds_ = np.nanpercentile(arr, self.lower_quantile * 100.0, axis=0)
        self.upper_bounds_ = np.nanpercentile(arr, self.upper_quantile * 100.0, axis=0)
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        return np.clip(arr, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)


class SignedLog1pTransformer(BaseEstimator, TransformerMixin):
    """Apply sign(x) * log1p(|x|), safely handling both large positive tails and negative balances."""

    def fit(self, X: np.ndarray | pd.DataFrame, y: Any = None) -> "SignedLog1pTransformer":
        arr = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = arr.shape[1]
        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float64)
        return np.sign(arr) * np.log1p(np.abs(arr))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)


class TrainFrequencyEncoder(BaseEstimator, TransformerMixin):
    """Encode high-cardinality categories by their relative frequency in the training fold."""

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> "TrainFrequencyEncoder":
        df = pd.DataFrame(X).copy()
        self.feature_names_ = [str(c) for c in df.columns]
        self.freq_maps_: dict[str, dict[str, float]] = {}
        n_rows = float(len(df))
        for col in df.columns:
            vc = df[col].fillna("__MISSING__").astype(str).value_counts(dropna=False)
            self.freq_maps_[str(col)] = {k: float(v) / n_rows for k, v in vc.items()}
        self.n_features_in_ = df.shape[1]
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        df = pd.DataFrame(X)
        cols_out = []
        for idx, col in enumerate(df.columns):
            key = self.feature_names_[idx]
            fmap = self.freq_maps_[key]
            mapped = df[col].fillna("__MISSING__").astype(str).map(fmap).fillna(0.0).to_numpy(dtype=np.float64)
            cols_out.append(mapped)
        return np.column_stack(cols_out)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        base = list(input_features) if input_features is not None else self.feature_names_
        return np.asarray([f"{c}__freq" for c in base], dtype=object)


class OOFSmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """Out-of-Fold Bayesian smoothed target encoder preventing in-fold target leakage.

    During `fit_transform(X, y)`:
      - Uses StratifiedKFold inside X so row i's target never leaks into row i's encoding.
      - Also learns full-train category statistics for unseen validation/test calls via `.transform(X_new)`.
    Unseen cold-start categories map safely to `global_prior_`.
    """

    def __init__(self, n_splits: int = 5, smoothing: float = 50.0, random_state: int = 42) -> None:
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state

    @staticmethod
    def _compute_smoothed_map(
        series: pd.Series,
        y_arr: np.ndarray,
        smoothing: float,
        prior: float,
    ) -> dict[str, float]:
        tmp = pd.DataFrame({"cat": series.fillna("__MISSING__").astype(str), "y": y_arr})
        agg = tmp.groupby("cat", observed=False)["y"].agg(["count", "sum"])
        smoothed = (agg["sum"] + smoothing * prior) / (agg["count"] + smoothing)
        return smoothed.to_dict()

    @staticmethod
    def _fingerprint_df(df: pd.DataFrame) -> tuple[tuple[int, int], int]:
        str_df = pd.DataFrame({c: df[c].fillna("__MISSING__").astype(str) for c in df.columns})
        h = int(pd.util.hash_pandas_object(str_df, index=False).sum())
        return (df.shape, h)

    def _compute_oof_matrix(self, df: pd.DataFrame, y_arr: np.ndarray) -> np.ndarray:
        y_int = y_arr.astype(np.int64)
        out = np.zeros((len(df), df.shape[1]), dtype=np.float64)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        for tr_pos, va_pos in skf.split(df, y_int):
            fold_prior = float(np.mean(y_arr[tr_pos]))
            for j, col in enumerate(df.columns):
                tr_series = df[col].iloc[tr_pos]
                va_series = df[col].iloc[va_pos].fillna("__MISSING__").astype(str)
                cmap = self._compute_smoothed_map(tr_series, y_arr[tr_pos], self.smoothing, fold_prior)
                out[va_pos, j] = va_series.map(cmap).fillna(fold_prior).to_numpy(dtype=np.float64)
        return out

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any) -> "OOFSmoothedTargetEncoder":
        if y is None:
            raise ValueError("OOFSmoothedTargetEncoder requires target y during fit().")
        df = pd.DataFrame(X).copy()
        y_arr = np.asarray(y, dtype=np.float64)
        self.feature_names_ = [str(c) for c in df.columns]
        self.global_prior_ = float(np.mean(y_arr))
        self.category_maps_: dict[str, dict[str, float]] = {}
        for idx, col in enumerate(df.columns):
            key = self.feature_names_[idx]
            self.category_maps_[key] = self._compute_smoothed_map(
                df[col], y_arr, self.smoothing, self.global_prior_
            )
        self.n_features_in_ = df.shape[1]
        self._last_fit_fingerprint_ = self._fingerprint_df(df)
        self._last_fit_oof_ = self._compute_oof_matrix(df, y_arr)
        return self

    def fit_transform(self, X: pd.DataFrame | np.ndarray, y: Any = None, **fit_params: Any) -> np.ndarray:
        self.fit(X, y)
        return self._last_fit_oof_.copy()

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        df = pd.DataFrame(X)
        if getattr(self, "_last_fit_fingerprint_", None) == self._fingerprint_df(df):
            warnings.warn(
                "OOFSmoothedTargetEncoder.transform() was called on the exact training data passed to fit(); "
                "returning leak-free out-of-fold encodings instead of in-sample target maps.",
                UserWarning,
                stacklevel=2,
            )
            return self._last_fit_oof_.copy()
        cols_out = []
        for idx, col in enumerate(df.columns):
            key = self.feature_names_[idx]
            cmap = self.category_maps_[key]
            mapped = (
                df[col]
                .fillna("__MISSING__")
                .astype(str)
                .map(cmap)
                .fillna(self.global_prior_)
                .to_numpy(dtype=np.float64)
            )
            cols_out.append(mapped)
        return np.column_stack(cols_out)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        base = list(input_features) if input_features is not None else self.feature_names_
        return np.asarray([f"{c}__oof_te" for c in base], dtype=object)


class CollinearityDropper(BaseEstimator, TransformerMixin):
    """Optionally drop near-duplicate collinear features (e.g. for linear baselines)."""

    def __init__(self, columns_to_drop: tuple[str, ...] | list[str] = ()) -> None:
        self.columns_to_drop = tuple(columns_to_drop)

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CollinearityDropper":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.columns_to_drop if c in X.columns]
        return X.drop(columns=cols) if cols else X.copy()


def get_feature_groups(feature_tier: str = "F1", drop_redundant: bool = False) -> dict[str, list[str]]:
    """Return deterministic column groupings after LTFSFeatureEngineer(feature_tier)."""
    if feature_tier not in VALID_TIERS:
        raise ValueError(f"Unsupported feature_tier '{feature_tier}'")

    skewed_num = [
        "disbursed_amount",
        "asset_cost",
        "PRI.NO.OF.ACCTS",
        "PRI.ACTIVE.ACCTS",
        "PRI.OVERDUE.ACCTS",
        "PRI.CURRENT.BALANCE",
        "PRI.SANCTIONED.AMOUNT",
        "PRI.DISBURSED.AMOUNT",
        "SEC.NO.OF.ACCTS",
        "SEC.ACTIVE.ACCTS",
        "SEC.OVERDUE.ACCTS",
        "SEC.CURRENT.BALANCE",
        "SEC.SANCTIONED.AMOUNT",
        "SEC.DISBURSED.AMOUNT",
        "PRIMARY.INSTAL.AMT",
        "SEC.INSTAL.AMT",
        "NEW.ACCTS.IN.LAST.SIX.MONTHS",
        "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS",
        "NO.OF_INQUIRIES",
        "avg_acct_age_months",
        "credit_history_months",
    ]

    bounded_num = ["ltv", "age_at_disbursal"]
    binary_flags = ["Aadhar_flag", "PAN_flag", "VoterID_flag", "Driving_flag", "Passport_flag"]
    low_med_cat = ["Employment.Type", "State_ID", "manufacturer_id", "PERFORM_CNS.SCORE.DESCRIPTION"]
    high_card: list[str] = []

    if feature_tier == "F0":
        # Raw PERFORM_CNS.SCORE remains numeric in F0 before regime split
        bounded_num.append("PERFORM_CNS.SCORE")
    else:
        skewed_num.extend(
            [
                "estimated_down_payment",
                "total_accts",
                "total_active_accts",
                "total_overdue_accts",
                "total_current_balance",
                "total_sanctioned_amount",
                "total_disbursed_amount",
                "total_instal_amt",
                "history_span_gap_months",
            ]
        )
        bounded_num.extend(
            [
                "bureau_score_clean",
                "bureau_risk_ordinal",
                "down_payment_ratio",
                "ltv_computed",
                "ltv_discrepancy",
                "kyc_doc_count",
                "overdue_to_total_ratio",
                "active_to_total_ratio",
                "credit_utilization_ratio",
                "instalment_to_disbursed_ratio",
                "delinquent_6m_to_total_ratio",
                "inquiry_to_accts_ratio",
                "ltv_x_no_bureau",
                "age_at_first_credit_years",
            ]
        )
        binary_flags.extend(
            [
                "has_bureau_score",
                "has_sec_account",
                "flag_pri_overdue_gt_active",
                "flag_pri_bal_gt_sanctioned",
                "flag_neg_balance",
            ]
        )
        low_med_cat.append("bureau_unscored_group")
        # Note: drop raw PERFORM_CNS.SCORE from linear/F1 bounded_num because split into has_bureau_score + bureau_score_clean;
        # keep raw PERFORM_CNS.SCORE out when F1 is active by dropping or omitting it from ColumnTransformer.

    if feature_tier in {"F2_safe", "F2_plus_emp"}:
        high_card = ["branch_id", "supplier_id", "Current_pincode_ID"]
    if feature_tier == "F2_plus_emp":
        high_card.append("Employee_code_ID")

    if drop_redundant:
        redundant = {"PRI.DISBURSED.AMOUNT", "SEC.DISBURSED.AMOUNT", "total_disbursed_amount"}
        skewed_num = [c for c in skewed_num if c not in redundant]

    return {
        "skewed_num": skewed_num,
        "bounded_num": bounded_num,
        "binary_flags": binary_flags,
        "low_med_cat": low_med_cat,
        "high_card": high_card,
    }


def build_preprocessing_pipeline(
    feature_tier: str = "F1",
    model_family: str = "linear",
    drop_redundant: bool = False,
    winsor_quantiles: tuple[float, float] = (0.005, 0.995),
    te_smoothing: float = 50.0,
    te_cv_folds: int = 5,
    random_state: int = 42,
) -> Pipeline:
    """Build a complete end-to-end leak-free sklearn Pipeline for a given tier and model family.

    Parameters
    ----------
    feature_tier : {'F0', 'F1', 'F2_safe', 'F2_plus_emp'}
    model_family : {'linear', 'mlp', 'tree'}
        - 'linear' / 'mlp': median imputation + missingness indicators, quantile winsorization,
          signed_log1p, RobustScaler/StandardScaler, OHE, OOF Smoothed TE + Freq encoding.
        - 'tree': preserves native raw scales for numerical features (imputed only where needed,
          such as RandomForest compatibility), OHE for low/med categoricals, OOF Smoothed TE + Freq
          for high-cardinality IDs. (For CatBoost native mode, raw DataFrame after LTFSFeatureEngineer
          can be passed directly without OHE/TE).
    drop_redundant : bool
        If True, removes near-collinear PRI/SEC disbursed amounts (rho > 0.995 with sanctioned).
    """
    if model_family not in {"linear", "mlp", "tree"}:
        raise ValueError(f"Unsupported model_family '{model_family}'")

    groups = get_feature_groups(feature_tier=feature_tier, drop_redundant=drop_redundant)
    scale_features = model_family in {"linear", "mlp"}

    if scale_features:
        skewed_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("winsor", QuantileWinsorizer(winsor_quantiles[0], winsor_quantiles[1])),
                ("signed_log", SignedLog1pTransformer()),
                ("scaler", RobustScaler()),
            ]
        )
        bounded_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        skewed_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ]
        )
        bounded_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ]
        )

    binary_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)),
        ]
    )

    transformers: list[tuple[str, Any, list[str]]] = [
        ("num_skewed", skewed_pipe, groups["skewed_num"]),
        ("num_bounded", bounded_pipe, groups["bounded_num"]),
        ("binary", binary_pipe, groups["binary_flags"]),
        ("cat_low_med", cat_pipe, groups["low_med_cat"]),
    ]

    if groups["high_card"]:
        if scale_features:
            te_pipe = Pipeline(
                steps=[
                    ("oof_te", OOFSmoothedTargetEncoder(n_splits=te_cv_folds, smoothing=te_smoothing, random_state=random_state)),
                    ("scaler", StandardScaler()),
                ]
            )
            freq_pipe = Pipeline(
                steps=[
                    ("freq", TrainFrequencyEncoder()),
                    ("log1p", SignedLog1pTransformer()),
                    ("scaler", StandardScaler()),
                ]
            )
        else:
            te_pipe = Pipeline(
                steps=[
                    ("oof_te", OOFSmoothedTargetEncoder(n_splits=te_cv_folds, smoothing=te_smoothing, random_state=random_state)),
                ]
            )
            freq_pipe = Pipeline(
                steps=[
                    ("freq", TrainFrequencyEncoder()),
                ]
            )
        transformers.append(("hc_target_enc", te_pipe, groups["high_card"]))
        transformers.append(("hc_freq_enc", freq_pipe, groups["high_card"]))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    steps: list[tuple[str, Any]] = [
        ("feature_engineer", LTFSFeatureEngineer(feature_tier=feature_tier)),
    ]
    if drop_redundant:
        steps.append(
            ("collinearity_dropper", CollinearityDropper(["PRI.DISBURSED.AMOUNT", "SEC.DISBURSED.AMOUNT", "total_disbursed_amount"]))
        )
    steps.append(("preprocessor", preprocessor))

    return Pipeline(steps=steps)
