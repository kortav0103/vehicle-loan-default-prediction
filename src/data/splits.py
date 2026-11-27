"""Deterministic, leak-free train/validation/test split generation and loader.

Generates and persists:
  - Primary Out-of-Time (OOT) split: Development <= 2018-10-22 vs Locked OOT Test >= 2018-10-23,
    plus stratified train/val (80/20 inside Dev) and 5-Fold Stratified CV indices.
  - Comparative Stratified Random 80/20 split (for RQ3 optimistic-bias quantification).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data.load_data import (
    get_project_root,
    load_config,
    load_raw_train,
    parse_disbursal_dates,
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _summarize_split(split_df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for label, mask in [
        ("train", split_df["split"] == "train"),
        ("val", split_df["split"] == "val"),
        ("dev_all", split_df["is_dev"] == 1),
        ("test_locked", split_df["split"] == "test"),
    ]:
        sub = split_df.loc[mask]
        summary[label] = {
            "n_rows": int(len(sub)),
            "pct_of_total": round(len(sub) / len(split_df) * 100, 4),
            "n_defaults": int(sub["loan_default"].sum()),
            "default_rate_pct": round(float(sub["loan_default"].mean() * 100), 4),
            "min_date": str(sub["disbursal_date"].min()),
            "max_date": str(sub["disbursal_date"].max()),
        }
    return summary


def _check_leakage_in_split(raw_df: pd.DataFrame, split_df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    id_col = cfg["schema"]["id_column"]
    target_col = cfg["schema"]["target_column"]
    feat_cols = [c for c in raw_df.columns if c not in {id_col, target_col}]

    dev_mask = (split_df["is_dev"] == 1).to_numpy()
    test_mask = (split_df["split"] == "test").to_numpy()
    train_mask = (split_df["split"] == "train").to_numpy()
    val_mask = (split_df["split"] == "val").to_numpy()

    uid_train = set(raw_df.loc[train_mask, id_col])
    uid_val = set(raw_df.loc[val_mask, id_col])
    uid_test = set(raw_df.loc[test_mask, id_col])

    # Check exact feature-row overlap across dev and test using deterministic row hashes
    row_hashes = pd.util.hash_pandas_object(raw_df[feat_cols], index=False)
    dev_hashes = set(row_hashes.loc[dev_mask].tolist())
    test_hashes = set(row_hashes.loc[test_mask].tolist())

    return {
        "uid_overlap_train_val": len(uid_train & uid_val),
        "uid_overlap_train_test": len(uid_train & uid_test),
        "uid_overlap_val_test": len(uid_val & uid_test),
        "feature_row_overlap_dev_test": len(dev_hashes & test_hashes),
    }


def build_and_save_splits(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate and persist deterministic OOT and Random splits."""
    cfg = config or load_config()
    root = get_project_root()
    splits_dir = root / cfg["paths"]["splits_dir"]
    splits_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["project"]["random_seed"])
    val_ratio = float(cfg["splits"]["val_size_within_dev"])
    rand_test_ratio = float(cfg["splits"]["random_test_size"])
    n_folds = int(cfg["splits"]["n_cv_folds"])
    oot_cutoff = pd.Timestamp(cfg["splits"]["oot_cutoff_date"])

    id_col = cfg["schema"]["id_column"]
    target_col = cfg["schema"]["target_column"]
    date_col = cfg["schema"]["disbursal_date_column"]

    raw_df = load_raw_train(cfg, verify_dims=True)
    disb_dt = parse_disbursal_dates(raw_df[date_col])
    assert disb_dt.notna().all(), "Found NaT values in DisbursalDate!"

    # -------------------------------------------------------------------------
    # 1. PRIMARY SPLIT: Out-of-Time (OOT) Temporal Hold-out
    # -------------------------------------------------------------------------
    is_oot_test = disb_dt >= oot_cutoff
    dev_indices = np.where(~is_oot_test.to_numpy())[0]
    test_indices = np.where(is_oot_test.to_numpy())[0]

    y_dev = raw_df.iloc[dev_indices][target_col].to_numpy()
    tr_idx_oot, val_idx_oot = train_test_split(
        dev_indices,
        test_size=val_ratio,
        stratify=y_dev,
        random_state=seed,
    )

    oot_split_col = np.empty(len(raw_df), dtype=object)
    oot_split_col[tr_idx_oot] = "train"
    oot_split_col[val_idx_oot] = "val"
    oot_split_col[test_indices] = "test"

    oot_cv_fold = np.full(len(raw_df), -1, dtype=np.int64)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_id, (_, fold_val_pos) in enumerate(skf.split(dev_indices, y_dev)):
        oot_cv_fold[dev_indices[fold_val_pos]] = fold_id

    # Expanding-window chronological blocks inside Dev (0: oldest 40%, 1: next 20%, 2: next 20%, 3: newest 20%)
    dev_dates_sorted = dev_indices[np.argsort(disb_dt.iloc[dev_indices].to_numpy(), kind="mergesort")]
    n_dev = len(dev_dates_sorted)
    b1 = int(n_dev * 0.40)
    b2 = int(n_dev * 0.60)
    b3 = int(n_dev * 0.80)
    temporal_block = np.full(len(raw_df), -1, dtype=np.int64)
    temporal_block[dev_dates_sorted[:b1]] = 0
    temporal_block[dev_dates_sorted[b1:b2]] = 1
    temporal_block[dev_dates_sorted[b2:b3]] = 2
    temporal_block[dev_dates_sorted[b3:]] = 3

    oot_df = pd.DataFrame(
        {
            "row_idx": np.arange(len(raw_df), dtype=np.int64),
            "UniqueID": raw_df[id_col].to_numpy(),
            "disbursal_date": disb_dt.dt.strftime("%Y-%m-%d"),
            "loan_default": raw_df[target_col].to_numpy(dtype=np.int64),
            "split": oot_split_col,
            "is_dev": (~is_oot_test).astype(np.int64),
            "cv_fold": oot_cv_fold,
            "temporal_val_block": temporal_block,
        }
    )

    assert pd.Timestamp(oot_df.loc[oot_df["is_dev"] == 1, "disbursal_date"].max()) < pd.Timestamp(
        oot_df.loc[oot_df["split"] == "test", "disbursal_date"].min()
    ), "OOT temporal boundary violated!"

    oot_path = splits_dir / "oot_split.csv"
    oot_df.to_csv(oot_path, index=False)

    # -------------------------------------------------------------------------
    # 2. COMPARATIVE SPLIT: Stratified Random 80/20 (for RQ3)
    # -------------------------------------------------------------------------
    all_idx = np.arange(len(raw_df), dtype=np.int64)
    y_all = raw_df[target_col].to_numpy(dtype=np.int64)
    rand_dev_idx, rand_test_idx = train_test_split(
        all_idx,
        test_size=rand_test_ratio,
        stratify=y_all,
        random_state=seed,
    )
    y_rand_dev = y_all[rand_dev_idx]
    rand_tr_idx, rand_val_idx = train_test_split(
        rand_dev_idx,
        test_size=val_ratio,
        stratify=y_rand_dev,
        random_state=seed,
    )

    rand_split_col = np.empty(len(raw_df), dtype=object)
    rand_split_col[rand_tr_idx] = "train"
    rand_split_col[rand_val_idx] = "val"
    rand_split_col[rand_test_idx] = "test"

    rand_is_dev = np.zeros(len(raw_df), dtype=np.int64)
    rand_is_dev[rand_dev_idx] = 1

    rand_cv_fold = np.full(len(raw_df), -1, dtype=np.int64)
    for fold_id, (_, fold_val_pos) in enumerate(skf.split(rand_dev_idx, y_rand_dev)):
        rand_cv_fold[rand_dev_idx[fold_val_pos]] = fold_id

    rand_df = pd.DataFrame(
        {
            "row_idx": all_idx,
            "UniqueID": raw_df[id_col].to_numpy(),
            "disbursal_date": disb_dt.dt.strftime("%Y-%m-%d"),
            "loan_default": y_all,
            "split": rand_split_col,
            "is_dev": rand_is_dev,
            "cv_fold": rand_cv_fold,
        }
    )

    rand_path = splits_dir / "random_split.csv"
    rand_df.to_csv(rand_path, index=False)

    metadata = {
        "random_seed": seed,
        "total_rows": int(len(raw_df)),
        "oot_split": {
            "file": oot_path.relative_to(root).as_posix(),
            "sha256": sha256_file(oot_path),
            "cutoff_date": cfg["splits"]["oot_cutoff_date"],
            "rationale": "Date-clean chronological hold-out separating 2018-08-01..2018-10-22 (Dev) from 2018-10-23..2018-10-31 (Locked OOT Test), matching competition OOT Nov-2018 nature.",
            "summary": _summarize_split(oot_df),
            "integrity_checks": _check_leakage_in_split(raw_df, oot_df, cfg),
        },
        "random_split": {
            "file": rand_path.relative_to(root).as_posix(),
            "sha256": sha256_file(rand_path),
            "test_size": rand_test_ratio,
            "rationale": "Comparative Stratified Random 80/20 hold-out split used strictly to quantify optimistic bias vs. OOT split (RQ3).",
            "summary": _summarize_split(rand_df),
            "integrity_checks": _check_leakage_in_split(raw_df, rand_df, cfg),
        },
    }

    meta_path = splits_dir / "split_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def load_split_frames(
    split_name: str = "oot",
    config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw train data joined with saved deterministic split indices.

    Returns (train_df, val_df, test_df, dev_df, split_df).
    """
    cfg = config or load_config()
    root = get_project_root()
    splits_dir = root / cfg["paths"]["splits_dir"]

    if split_name not in {"oot", "random"}:
        raise ValueError(f"Unsupported split_name='{split_name}'. Expected 'oot' or 'random'.")

    split_path = splits_dir / f"{split_name}_split.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}. Run `python -m src.data.splits` first.")

    raw_df = load_raw_train(cfg, verify_dims=True)
    split_df = pd.read_csv(split_path)
    assert len(split_df) == len(raw_df), "Split index length mismatch with raw train!"
    assert (split_df["UniqueID"].to_numpy() == raw_df[cfg["schema"]["id_column"]].to_numpy()).all(), (
        "UniqueID alignment mismatch between raw_df and split_df!"
    )

    train_mask = split_df["split"] == "train"
    val_mask = split_df["split"] == "val"
    test_mask = split_df["split"] == "test"
    dev_mask = split_df["is_dev"] == 1

    return (
        raw_df.loc[train_mask].copy(),
        raw_df.loc[val_mask].copy(),
        raw_df.loc[test_mask].copy(),
        raw_df.loc[dev_mask].copy(),
        split_df,
    )


if __name__ == "__main__":
    meta = build_and_save_splits()
    print(json.dumps(meta, indent=2, ensure_ascii=False))
