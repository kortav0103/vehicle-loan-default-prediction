"""Data loading and deterministic date/duration parsing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def get_project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parents[2]


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load YAML project configuration."""
    root = get_project_root()
    path = Path(config_path) if config_path is not None else root / "configs" / "base.yaml"
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_raw_train(config: dict[str, Any] | None = None, verify_dims: bool = True) -> pd.DataFrame:
    """Read raw labeled training CSV strictly read-only."""
    cfg = config or load_config()
    path = get_project_root() / cfg["paths"]["raw_train"]
    df = pd.read_csv(path, low_memory=False)
    if verify_dims:
        assert df.shape == (233154, 41), f"Unexpected raw train shape: {df.shape}"
    return df


def load_raw_test(config: dict[str, Any] | None = None, verify_dims: bool = True) -> pd.DataFrame:
    """Read raw Kaggle competition test CSV (unlabeled) strictly read-only."""
    cfg = config or load_config()
    path = get_project_root() / cfg["paths"]["raw_test"]
    df = pd.read_csv(path, low_memory=False)
    if verify_dims:
        assert df.shape == (112392, 40), f"Unexpected raw test shape: {df.shape}"
    return df


def parse_disbursal_dates(series: pd.Series) -> pd.Series:
    """Parse DisbursalDate strings ('dd-mm-yy') cleanly."""
    return pd.to_datetime(series, format="%d-%m-%y", errors="coerce")


def parse_dob_century_safe(
    dob_series: pd.Series,
    reference_years: pd.Series | int = 2018,
) -> pd.Series:
    """Parse 'dd-mm-yy' Date.of.Birth strings while fixing pandas 2-digit century rollover.

    In pandas default parsing, two-digit years <= '68 map to 20xx. Any parsed birth year
    exceeding the disbursement reference year is rolled back by 100 years into the 20th century.
    """
    parsed = pd.to_datetime(dob_series, format="%d-%m-%y", errors="coerce")
    if isinstance(reference_years, int):
        ref_yr = pd.Series(reference_years, index=parsed.index)
    else:
        ref_yr = pd.Series(reference_years, index=parsed.index)

    mask_future = parsed.dt.year > ref_yr
    if mask_future.any():
        parsed = parsed.mask(mask_future, parsed.loc[mask_future] - pd.DateOffset(years=100))
    return parsed


def parse_duration_months_series(series: pd.Series) -> pd.Series:
    """Convert bureau duration strings like '1yrs 11mon' into numeric total months."""
    extracted = (
        series.astype(str)
        .str.strip()
        .str.extract(r"^(?P<yrs>-?\d+)\s*yrs?\s*(?P<mons>-?\d+)\s*mon$", expand=True)
    )
    yrs = pd.to_numeric(extracted["yrs"], errors="coerce")
    mons = pd.to_numeric(extracted["mons"], errors="coerce")
    months = (yrs * 12.0 + mons).astype(np.float64)
    return months
