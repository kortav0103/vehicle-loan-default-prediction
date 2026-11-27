"""Model factory, leak-free Dev-only data loader, and memory-efficient fold preprocessing cache."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.data.load_data import get_project_root, load_config, parse_disbursal_dates
from src.data.splits import load_split_frames
from src.features.preprocessing import build_preprocessing_pipeline

logger = logging.getLogger(__name__)


def load_model_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load model benchmark and tuning configuration from YAML."""
    root = get_project_root()
    path = Path(config_path) if config_path is not None else root / "configs" / "models.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_dev_only_data(
    base_cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load strictly the Primary OOT Development Set (never exposing Locked Test Set).

    Returns (dev_df, y_dev, cv_folds_dev, temporal_blocks_dev, disbursed_amt_dev).
    """
    cfg = base_cfg or load_config()
    _, _, _, dev_df, split_df = load_split_frames("oot", cfg)

    dev_mask = split_df["is_dev"] == 1
    dev_split = split_df.loc[dev_mask].reset_index(drop=True)
    dev_df = dev_df.reset_index(drop=True)

    assert len(dev_df) == 182501, f"Expected 182,501 Dev rows, got {len(dev_df)}"
    assert (dev_df["UniqueID"].to_numpy() == dev_split["UniqueID"].to_numpy()).all()
    assert parse_disbursal_dates(dev_df["DisbursalDate"]).max() < pd.Timestamp(cfg["splits"]["oot_cutoff_date"])
    assert (dev_split["cv_fold"].to_numpy() >= 0).all()

    y_dev = dev_df["loan_default"].to_numpy(dtype=np.int64)
    cv_folds = dev_split["cv_fold"].to_numpy(dtype=np.int64)
    temporal_blocks = dev_split["temporal_val_block"].to_numpy(dtype=np.int64)
    disbursed_amt = dev_df["disbursed_amount"].to_numpy(dtype=np.float64)

    return dev_df, y_dev, cv_folds, temporal_blocks, disbursed_amt


def build_fold_cache(
    dev_df: pd.DataFrame,
    y_dev: np.ndarray,
    cv_folds: np.ndarray,
    feature_tier: str,
    model_family: str,
    drop_redundant: bool,
    base_cfg: dict[str, Any],
    cast_float32: bool = True,
) -> list[dict[str, Any]]:
    """Precompute leak-free per-fold train/val matrices as float32 to conserve RAM."""
    seed = int(base_cfg["project"]["random_seed"])
    winsor_q = tuple(base_cfg["preprocessing"]["winsor_quantiles"])
    te_smooth = float(base_cfg["preprocessing"]["target_encoding_smoothing"])
    te_folds = int(base_cfg["preprocessing"]["target_encoding_cv_folds"])

    unique_folds = sorted(int(f) for f in np.unique(cv_folds) if int(f) >= 0)
    folds_cache: list[dict[str, Any]] = []

    for f in unique_folds:
        tr_idx = np.where(cv_folds != f)[0]
        va_idx = np.where(cv_folds == f)[0]

        pipe = build_preprocessing_pipeline(
            feature_tier=feature_tier,
            model_family=model_family,
            drop_redundant=drop_redundant,
            winsor_quantiles=winsor_q,
            te_smoothing=te_smooth,
            te_cv_folds=te_folds,
            random_state=seed,
        )
        X_tr = pipe.fit_transform(dev_df.iloc[tr_idx], y_dev[tr_idx])
        X_va = pipe.transform(dev_df.iloc[va_idx])
        if cast_float32:
            X_tr = X_tr.astype(np.float32, copy=False)
            X_va = X_va.astype(np.float32, copy=False)

        folds_cache.append(
            {
                "fold": f,
                "tr_idx": tr_idx,
                "va_idx": va_idx,
                "X_tr": X_tr,
                "y_tr": y_dev[tr_idx],
                "X_va": X_va,
                "y_va": y_dev[va_idx],
            }
        )
    return folds_cache


def build_model(
    model_name: str,
    params: dict[str, Any] | None = None,
    random_state: int = 42,
    n_threads: int = 8,
) -> Any:
    """Instantiate a classifier from canonical name and parameter dictionary."""
    p = dict(params or {})

    if model_name == "logistic_regression":
        return LogisticRegression(
            C=float(p.get("C", 1.0)),
            max_iter=int(p.get("max_iter", 1000)),
            solver=p.get("solver", "lbfgs"),
            class_weight=p.get("class_weight", None),
            random_state=random_state,
        )

    if model_name == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=p.get("max_depth", 8),
            min_samples_leaf=int(p.get("min_samples_leaf", 50)),
            class_weight=p.get("class_weight", None),
            random_state=random_state,
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(p.get("n_estimators", 150)),
            max_depth=p.get("max_depth", 14),
            min_samples_leaf=int(p.get("min_samples_leaf", 25)),
            max_features=p.get("max_features", "sqrt"),
            class_weight=p.get("class_weight", None),
            n_jobs=n_threads,
            random_state=random_state,
        )

    if model_name == "xgboost":
        xgb_kwargs: dict[str, Any] = {
            "n_estimators": int(p.get("n_estimators", 400)),
            "learning_rate": float(p.get("learning_rate", 0.05)),
            "max_depth": int(p.get("max_depth", 6)),
            "min_child_weight": float(p.get("min_child_weight", 1.0)),
            "subsample": float(p.get("subsample", 0.8)),
            "colsample_bytree": float(p.get("colsample_bytree", 0.8)),
            "reg_alpha": float(p.get("reg_alpha", 0.0)),
            "reg_lambda": float(p.get("reg_lambda", 1.0)),
            "gamma": float(p.get("gamma", 0.0)),
            "scale_pos_weight": float(p.get("scale_pos_weight", 1.0)),
            "tree_method": p.get("tree_method", "hist"),
            "eval_metric": p.get("eval_metric", "logloss"),
            "n_jobs": n_threads,
            "random_state": random_state,
        }
        if "early_stopping_rounds" in p and p["early_stopping_rounds"] is not None:
            xgb_kwargs["early_stopping_rounds"] = int(p["early_stopping_rounds"])
        return XGBClassifier(**xgb_kwargs)

    if model_name == "lightgbm":
        lgb_kwargs: dict[str, Any] = {
            "n_estimators": int(p.get("n_estimators", 400)),
            "learning_rate": float(p.get("learning_rate", 0.05)),
            "num_leaves": int(p.get("num_leaves", 31)),
            "max_depth": int(p.get("max_depth", -1)),
            "min_child_samples": int(p.get("min_child_samples", 50)),
            "min_split_gain": float(p.get("min_split_gain", 0.0)),
            "subsample": float(p.get("subsample", 0.8)),
            "subsample_freq": int(p.get("subsample_freq", 1)),
            "colsample_bytree": float(p.get("colsample_bytree", 0.8)),
            "reg_alpha": float(p.get("reg_alpha", 0.0)),
            "reg_lambda": float(p.get("reg_lambda", 0.0)),
            "class_weight": p.get("class_weight", None),
            "n_jobs": n_threads,
            "random_state": random_state,
            "verbose": int(p.get("verbose", -1)),
        }
        if p.get("class_weight") is None and "scale_pos_weight" in p and p["scale_pos_weight"] is not None:
            lgb_kwargs["scale_pos_weight"] = float(p["scale_pos_weight"])
        return LGBMClassifier(**lgb_kwargs)

    if model_name == "mlp":
        hidden = tuple(int(x) for x in p.get("hidden_layer_sizes", (128, 64)))
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            activation=p.get("activation", "relu"),
            alpha=float(p.get("alpha", 0.001)),
            batch_size=int(p.get("batch_size", 512)),
            learning_rate_init=float(p.get("learning_rate_init", 0.001)),
            max_iter=int(p.get("max_iter", 35)),
            early_stopping=bool(p.get("early_stopping", True)),
            validation_fraction=float(p.get("validation_fraction", 0.1)),
            n_iter_no_change=int(p.get("n_iter_no_change", 6)),
            random_state=random_state,
        )

    raise ValueError(f"Unknown model_name '{model_name}'")
