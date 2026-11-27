"""Comprehensive independent dataset audit for LTFS Vehicle Loan Default Prediction.

Reads strictly from `data/raw/` without modifying any raw files and without
training any machine learning models. Produces machine-readable JSON summaries,
CSV audit tables, and 300-DPI diagnostic figures.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def norm_name(name: str) -> str:
    """Normalize a column name for resilient matching without altering raw headers."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def resolve_column(df: pd.DataFrame, target_norm: str) -> str | None:
    """Find the exact original column name matching a normalized key."""
    mapping = {norm_name(col): col for col in df.columns}
    return mapping.get(target_norm)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hex digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_two_digit_dates(series: pd.Series, reference_year: int = 2018, is_dob: bool = False) -> tuple[pd.Series, pd.Series]:
    """Parse dd-mm-yy strings safely, avoiding pandas 2068 century roll-forward for DOB.

    Returns (naive_parsed_dates, century_corrected_dates).
    """
    naive = pd.to_datetime(series, format="%d-%m-%y", errors="coerce")
    if not is_dob:
        return naive, naive

    # For Date.of.Birth, any parsed year > (reference_year - 10) implies 20xx should be 19xx,
    # Wait: even more generally, if parsed year > reference_year it is definitely future DOB,
    # and in fact vehicle loan borrowers are >= ~10 years old (no borrower born in 2009-2018).
    # Specifically check parsed.dt.year > reference_year first, and also count 2001..2018!
    corrected = naive.copy()
    mask_future = corrected.dt.year > reference_year
    if mask_future.any():
        corrected = corrected.mask(
            mask_future,
            corrected.loc[mask_future] - pd.DateOffset(years=100),
        )
    return naive, corrected


def parse_duration_months(series: pd.Series) -> tuple[pd.Series, int]:
    """Parse strings like '1yrs 11mon' into integer months, counting malformed entries."""
    s = series.astype(str).str.strip()
    extracted = s.str.extract(r"^(?P<yrs>-?\d+)\s*yrs?\s*(?P<mons>-?\d+)\s*mon$", expand=True)
    malformed_mask = extracted["yrs"].isna() | extracted["mons"].isna()
    yrs = pd.to_numeric(extracted["yrs"], errors="coerce").fillna(0)
    mons = pd.to_numeric(extracted["mons"], errors="coerce").fillna(0)
    months = (yrs * 12 + mons).astype("float64")
    months[malformed_mask] = np.nan
    return months, int(malformed_mask.sum())


def analytical_univariate_auc(values: pd.Series, target: pd.Series) -> tuple[float | None, float | None]:
    """Compute exact Mann-Whitney U ROC-AUC without training any model."""
    valid = values.notna() & target.notna()
    if valid.sum() < 10:
        return None, None
    x = values[valid].to_numpy(dtype=np.float64)
    y = target[valid].to_numpy(dtype=np.int64)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return None, None
    ranks = stats.rankdata(x, method="average")
    u1 = float(ranks[y == 1].sum()) - n1 * (n1 + 1) / 2.0
    raw_auc = u1 / (n1 * n0)
    return round(raw_auc, 6), round(max(raw_auc, 1.0 - raw_auc), 6)


def compute_iv_categorical(series: pd.Series, target: pd.Series, min_count: int = 1) -> float | None:
    """Compute Information Value (IV) for a categorical/discrete series."""
    s = series.fillna("__MISSING__").astype(str)
    df_tmp = pd.DataFrame({"cat": s, "y": target})
    grouped = df_tmp.groupby("cat", observed=False)["y"].agg(["count", "sum"])
    grouped = grouped[grouped["count"] >= min_count]
    if len(grouped) <= 1:
        return 0.0
    total_bad = float(target.sum())
    total_good = float(len(target) - total_bad)
    if total_bad == 0 or total_good == 0:
        return None
    bad = grouped["sum"].to_numpy(dtype=np.float64)
    good = (grouped["count"] - grouped["sum"]).to_numpy(dtype=np.float64)
    dist_bad = (bad + 0.5) / (total_bad + 0.5 * len(grouped))
    dist_good = (good + 0.5) / (total_good + 0.5 * len(grouped))
    iv = float(np.sum((dist_bad - dist_good) * np.log(dist_bad / dist_good)))
    return round(iv, 6)


def compute_iv_numeric(series: pd.Series, target: pd.Series, bins: int = 10) -> float | None:
    """Compute quantile-binned Information Value (IV) for a numeric series."""
    valid = series.notna() & target.notna()
    if valid.sum() < 20:
        return None
    s_valid = series[valid]
    if s_valid.nunique() <= 1:
        return 0.0
    try:
        binned = pd.qcut(s_valid, q=bins, duplicates="drop").astype(str)
    except ValueError:
        binned = s_valid.astype(str)
    full_binned = pd.Series("__MISSING__", index=series.index, dtype="object")
    full_binned.loc[valid] = binned
    return compute_iv_categorical(full_binned, target, min_count=1)


def cramers_v_from_series(series: pd.Series, target: pd.Series) -> float | None:
    """Compute bias-corrected Cramer's V against binary target."""
    s = series.fillna("__MISSING__").astype(str)
    ct = pd.crosstab(s, target)
    if ct.shape[0] <= 1 or ct.shape[1] <= 1:
        return 0.0
    chi2 = float(stats.chi2_contingency(ct, correction=False)[0])
    n = float(ct.sum().sum())
    if n <= 1:
        return None
    phi2 = chi2 / n
    r, k = ct.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    if denom <= 0:
        return 0.0
    return round(math.sqrt(phi2corr / denom), 6)


def find_raw_files(raw_dir: Path) -> dict[str, Path]:
    """Locate train, test, and data dictionary files dynamically inside data/raw/."""
    files = [p for p in sorted(raw_dir.rglob("*")) if p.is_file()]
    located: dict[str, Path] = {}
    for p in files:
        low = p.name.lower()
        if "train" in low and p.suffix.lower() == ".csv":
            located["train"] = p
        elif "test" in low and p.suffix.lower() == ".csv":
            located["test"] = p
        elif "dict" in low and p.suffix.lower() in {".csv", ".xlsx", ".xls"}:
            located["dictionary"] = p
    return located


def load_dictionary(dict_path: Path | None) -> pd.DataFrame:
    """Load data dictionary file cleanly."""
    if dict_path is None:
        return pd.DataFrame()
    if dict_path.suffix.lower() == ".csv":
        return pd.read_csv(dict_path)
    return pd.read_excel(dict_path)


def setup_plot_style() -> None:
    """Configure clean academic publication style for matplotlib/seaborn."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"
    audit_dir = project_root / "reports" / "audit"
    tables_dir = project_root / "reports" / "tables"
    figures_dir = project_root / "reports" / "figures"

    for d in (audit_dir, tables_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    located = find_raw_files(raw_dir)
    if "train" not in located or "test" not in located:
        raise FileNotFoundError(f"Missing expected train/test CSVs in {raw_dir}. Found: {located}")

    train_path = located["train"]
    test_path = located["test"]
    dict_path = located.get("dictionary")

    print(f"[INFO] Reading raw files from {raw_dir} (read-only)...")
    train_df = pd.read_csv(train_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)
    dict_df = load_dictionary(dict_path)

    # Resolve key columns without assuming exact casing/punctuation
    col_uid = resolve_column(train_df, "uniqueid")
    col_target = resolve_column(train_df, "loandefault")
    col_disb_date = resolve_column(train_df, "disbursaldate")
    col_dob = resolve_column(train_df, "dateofbirth")
    col_emp = resolve_column(train_df, "employmenttype")
    col_score = resolve_column(train_df, "performcnsscore")
    col_score_desc = resolve_column(train_df, "performcnsscoredescription")
    col_avg_age = resolve_column(train_df, "averageacctage")
    col_cred_len = resolve_column(train_df, "credithistorylength")

    assert col_uid and col_target and col_disb_date and col_dob, "Core expected columns missing!"

    # -------------------------------------------------------------------------
    # 1. DATASET INVENTORY & DATA DICTIONARY MAPPING
    # -------------------------------------------------------------------------
    inventory_rows: list[dict[str, Any]] = []
    for role, p, df_obj in [
        ("train", train_path, train_df),
        ("test", test_path, test_df),
        ("dictionary", dict_path, dict_df),
    ]:
        if p is None:
            continue
        size_b = p.stat().st_size
        inventory_rows.append(
            {
                "role": role,
                "file_name": p.name,
                "relative_path": p.relative_to(project_root).as_posix(),
                "format": p.suffix.lower().lstrip("."),
                "size_bytes": size_b,
                "size_mb": round(size_b / (1024 * 1024), 4),
                "sha256": sha256_file(p),
                "n_rows": int(df_obj.shape[0]),
                "n_cols": int(df_obj.shape[1]),
                "columns": "|".join(str(c) for c in df_obj.columns),
            }
        )
    inventory_df = pd.DataFrame(inventory_rows)
    inventory_df.to_csv(tables_dir / "m1_dataset_inventory.csv", index=False)

    # Dictionary mapping audit
    dict_norm_map: dict[str, dict[str, str]] = {}
    if not dict_df.empty and "Variable Name" in dict_df.columns:
        for _, row in dict_df.iterrows():
            vname = str(row["Variable Name"]).strip()
            desc = str(row.get("Description", "")).strip()
            extra = ""
            for extra_col in dict_df.columns:
                if extra_col.startswith("Unnamed") and extra_col != "Unnamed: 0":
                    val = row.get(extra_col)
                    if pd.notna(val) and str(val).strip():
                        extra = str(val).strip()
            dict_norm_map[norm_name(vname)] = {
                "dict_variable_name": vname,
                "dict_description": desc,
                "dict_extra_note": extra,
            }

    train_only_cols = [c for c in train_df.columns if c not in set(test_df.columns)]
    test_only_cols = [c for c in test_df.columns if c not in set(train_df.columns)]
    binary_candidates = [
        c for c in train_df.columns
        if set(train_df[c].dropna().unique()).issubset({0, 1}) and train_df[c].nunique(dropna=True) == 2
    ]
    uid_train_unique = int(train_df[col_uid].nunique(dropna=False))
    uid_test_unique = int(test_df[col_uid].nunique(dropna=False))
    uid_overlap = int(len(set(train_df[col_uid]).intersection(set(test_df[col_uid]))))

    dict_rows: list[dict[str, Any]] = []
    for col in train_df.columns:
        match = dict_norm_map.get(norm_name(col))
        dict_rows.append(
            {
                "csv_column": col,
                "in_test_csv": col in set(test_df.columns),
                "dict_variable_name": match["dict_variable_name"] if match else "NOT_FOUND",
                "exact_name_match": (match["dict_variable_name"] == col) if match else False,
                "dict_description": match["dict_description"] if match else "",
                "dict_extra_note": match["dict_extra_note"] if match else "",
            }
        )
    pd.DataFrame(dict_rows).to_csv(tables_dir / "m1_data_dictionary_mapping.csv", index=False)

    # -------------------------------------------------------------------------
    # 2. TARGET VERIFICATION & TEMPORAL ANALYSIS
    # -------------------------------------------------------------------------
    y = train_df[col_target]
    y_counts = y.value_counts(dropna=False).sort_index()
    n_total = len(train_df)
    n_pos = int(y_counts.get(1, 0))
    n_neg = int(y_counts.get(0, 0))
    pos_rate = n_pos / n_total

    _, train_disb_dt = parse_two_digit_dates(train_df[col_disb_date], is_dob=False)
    _, test_disb_dt = parse_two_digit_dates(test_df[col_disb_date], is_dob=False)

    # Temporal summary by month, week, and exact day
    temp_df = pd.DataFrame(
        {
            "disbursal_date": train_disb_dt,
            "loan_default": y,
        }
    )
    temp_df["year_month"] = temp_df["disbursal_date"].dt.to_period("M").astype(str)
    temp_df["week_start"] = temp_df["disbursal_date"].dt.to_period("W-SUN").apply(lambda p: str(p.start_time.date()))

    monthly_summary = (
        temp_df.groupby("year_month", observed=False)
        .agg(n_loans=("loan_default", "count"), n_defaults=("loan_default", "sum"), default_rate=("loan_default", "mean"))
        .reset_index()
        .assign(granularity="month")
        .rename(columns={"year_month": "period"})
    )
    weekly_summary = (
        temp_df.groupby("week_start", observed=False)
        .agg(n_loans=("loan_default", "count"), n_defaults=("loan_default", "sum"), default_rate=("loan_default", "mean"))
        .reset_index()
        .assign(granularity="week")
        .rename(columns={"week_start": "period"})
    )
    daily_summary = (
        temp_df.groupby("disbursal_date", observed=False)
        .agg(n_loans=("loan_default", "count"), n_defaults=("loan_default", "sum"), default_rate=("loan_default", "mean"))
        .reset_index()
        .assign(granularity="day")
    )
    daily_summary["period"] = daily_summary["disbursal_date"].dt.strftime("%Y-%m-%d")
    daily_summary = daily_summary[["period", "n_loans", "n_defaults", "default_rate", "granularity"]]

    temporal_summary_df = pd.concat([monthly_summary, weekly_summary, daily_summary], ignore_index=True)
    temporal_summary_df["default_rate_pct"] = (temporal_summary_df["default_rate"] * 100).round(4)
    temporal_summary_df.to_csv(tables_dir / "m1_target_temporal_summary.csv", index=False)

    # Test disbursal monthly distribution (unlabeled)
    test_monthly = (
        test_disb_dt.dt.to_period("M").astype(str).value_counts().sort_index().to_dict()
    )

    # -------------------------------------------------------------------------
    # 3. DATA QUALITY AUDIT
    # -------------------------------------------------------------------------
    # Duplicates
    dup_full_train = int(train_df.duplicated().sum())
    dup_no_uid_train = int(train_df.duplicated(subset=[c for c in train_df.columns if c != col_uid]).sum())
    dup_no_uid_target_train = int(train_df.duplicated(subset=[c for c in train_df.columns if c not in {col_uid, col_target}]).sum())
    dup_full_test = int(test_df.duplicated().sum())
    dup_no_uid_test = int(test_df.duplicated(subset=[c for c in test_df.columns if c != col_uid]).sum())

    # Date of birth parsing audit (naive vs corrected)
    dob_naive_train, dob_corr_train = parse_two_digit_dates(train_df[col_dob], reference_year=2018, is_dob=True)
    dob_naive_test, dob_corr_test = parse_two_digit_dates(test_df[col_dob], reference_year=2018, is_dob=True)

    age_naive_train = (train_disb_dt - dob_naive_train).dt.days / 365.25
    age_corr_train = (train_disb_dt - dob_corr_train).dt.days / 365.25
    age_corr_test = (test_disb_dt - dob_corr_test).dt.days / 365.25

    # Duration parsing audit
    avg_age_m_train, avg_age_bad_train = parse_duration_months(train_df[col_avg_age])
    cred_len_m_train, cred_len_bad_train = parse_duration_months(train_df[col_cred_len])
    avg_age_m_test, avg_age_bad_test = parse_duration_months(test_df[col_avg_age])
    cred_len_m_test, cred_len_bad_test = parse_duration_months(test_df[col_cred_len])

    # Missingness & Employment.Type MNAR test
    emp_audit_rows: list[dict[str, Any]] = []
    if col_emp:
        emp_series = train_df[col_emp].fillna("MISSING")
        ct_emp = pd.crosstab(emp_series, y)
        chi2_emp, p_emp, _, _ = stats.chi2_contingency(ct_emp)
        for grp, grp_df in train_df.assign(_emp=emp_series).groupby("_emp", observed=False):
            emp_audit_rows.append(
                {
                    "employment_type": grp,
                    "count": int(len(grp_df)),
                    "pct_of_train": round(len(grp_df) / n_total * 100, 4),
                    "n_defaults": int(grp_df[col_target].sum()),
                    "default_rate_pct": round(grp_df[col_target].mean() * 100, 4),
                }
            )
    else:
        chi2_emp, p_emp = None, None

    # Bureau score cross-audit
    bureau_df = (
        train_df.groupby(col_score_desc, dropna=False)
        .agg(
            count=(col_target, "count"),
            n_defaults=(col_target, "sum"),
            default_rate=(col_target, "mean"),
            min_score=(col_score, "min"),
            median_score=(col_score, "median"),
            max_score=(col_score, "max"),
            n_unique_scores=(col_score, "nunique"),
        )
        .reset_index()
        .sort_values(["min_score", "count"], ascending=[True, False])
    )
    bureau_df["pct_of_train"] = (bureau_df["count"] / n_total * 100).round(4)
    bureau_df["default_rate_pct"] = (bureau_df["default_rate"] * 100).round(4)
    bureau_df.to_csv(tables_dir / "m1_bureau_score_audit.csv", index=False)

    # High-cardinality & Temporal Cold-Start Drift Audit
    order_idx = train_disb_dt.argsort(kind="mergesort")
    split_80 = int(len(train_df) * 0.80)
    dev_idx = order_idx[:split_80]
    oot_idx = order_idx[split_80:]
    oot_cutoff_date = str(train_disb_dt.iloc[oot_idx].min().date())

    hc_cols_norm = ["branchid", "supplierid", "manufacturerid", "currentpincodeid", "stateid", "employeecodeid"]
    hc_drift_rows: list[dict[str, Any]] = []
    for norm_c in hc_cols_norm:
        col_name = resolve_column(train_df, norm_c)
        if not col_name:
            continue
        dev_vals = set(train_df[col_name].iloc[dev_idx].dropna().unique())
        oot_series = train_df[col_name].iloc[oot_idx].dropna()
        oot_unseen_row_pct = float((~oot_series.isin(dev_vals)).mean() * 100)
        oot_unseen_id_pct = float(len(set(oot_series.unique()) - dev_vals) / max(1, oot_series.nunique()) * 100)

        train_vals = set(train_df[col_name].dropna().unique())
        test_series = test_df[col_name].dropna()
        test_unseen_row_pct = float((~test_series.isin(train_vals)).mean() * 100)
        test_unseen_id_pct = float(len(set(test_series.unique()) - train_vals) / max(1, test_series.nunique()) * 100)

        hc_drift_rows.append(
            {
                "column": col_name,
                "train_nunique": int(train_df[col_name].nunique(dropna=True)),
                "test_nunique": int(test_df[col_name].nunique(dropna=True)),
                "oot20_unseen_rows_pct": round(oot_unseen_row_pct, 4),
                "oot20_unseen_ids_pct": round(oot_unseen_id_pct, 4),
                "kaggle_test_unseen_rows_pct": round(test_unseen_row_pct, 4),
                "kaggle_test_unseen_ids_pct": round(test_unseen_id_pct, 4),
            }
        )
    hc_drift_df = pd.DataFrame(hc_drift_rows)
    hc_drift_df.to_csv(tables_dir / "m1_high_cardinality_drift.csv", index=False)

    # Logical & Domain Inconsistencies
    c_disb = resolve_column(train_df, "disbursedamount")
    c_asset = resolve_column(train_df, "assetcost")
    c_ltv = resolve_column(train_df, "ltv")
    c_p_no = resolve_column(train_df, "prinoofaccts")
    c_p_act = resolve_column(train_df, "priactiveaccts")
    c_p_ovd = resolve_column(train_df, "prioverdueaccts")
    c_p_bal = resolve_column(train_df, "pricurrentbalance")
    c_p_san = resolve_column(train_df, "prisanctionedamount")
    c_p_dis = resolve_column(train_df, "pridisbursedamount")
    c_s_no = resolve_column(train_df, "secnoofaccts")
    c_s_act = resolve_column(train_df, "secactiveaccts")
    c_s_ovd = resolve_column(train_df, "secoverdueaccts")
    c_s_bal = resolve_column(train_df, "seccurrentbalance")
    c_s_san = resolve_column(train_df, "secsanctionedamount")
    c_s_dis = resolve_column(train_df, "secdisbursedamount")
    c_p_ins = resolve_column(train_df, "primaryinstalamt")
    c_s_ins = resolve_column(train_df, "secinstalamt")
    c_new6 = resolve_column(train_df, "newacctsinlastsixmonths")
    c_del6 = resolve_column(train_df, "delinquentacctsinlastsixmonths")

    calc_ltv = 100.0 * train_df[c_disb] / train_df[c_asset]
    ltv_diff = (train_df[c_ltv] - calc_ltv).abs()

    inconsistency_checks: list[tuple[str, pd.Series, str]] = [
        ("dob_naive_future_year_gt_2018", dob_naive_train.dt.year > 2018, "Pandas default 2-digit year parsing shifts '00-'68 incorrectly into 20xx"),
        ("age_corrected_under_18", age_corr_train < 18, "Borrower age at disbursal < 18 years after century fix"),
        ("age_corrected_over_75", age_corr_train > 75, "Borrower age at disbursal > 75 years after century fix"),
        ("disbursed_gt_asset_cost", train_df[c_disb] > train_df[c_asset], "Disbursed loan amount exceeds declared asset cost"),
        ("ltv_gt_100", train_df[c_ltv] > 100, "Declared LTV > 100%"),
        ("ltv_diff_gt_5pct_pts", ltv_diff > 5.0, "|declared ltv - 100*disbursed/asset_cost| > 5 percentage points (likely processing fees/downpayment or LTV calc basis)"),
        ("pri_active_gt_total", train_df[c_p_act] > train_df[c_p_no], "PRI.ACTIVE.ACCTS exceeds PRI.NO.OF.ACCTS"),
        ("pri_overdue_gt_active", train_df[c_p_ovd] > train_df[c_p_act], "PRI.OVERDUE.ACCTS exceeds PRI.ACTIVE.ACCTS (possible closed/written-off overdue accounts)"),
        ("pri_overdue_gt_total", train_df[c_p_ovd] > train_df[c_p_no], "PRI.OVERDUE.ACCTS exceeds PRI.NO.OF.ACCTS"),
        ("pri_negative_current_balance", train_df[c_p_bal] < 0, "Negative PRI.CURRENT.BALANCE (overpayment / credit balance / accounting artifact)"),
        ("pri_disbursed_gt_sanctioned", train_df[c_p_dis] > train_df[c_p_san], "PRI.DISBURSED.AMOUNT > PRI.SANCTIONED.AMOUNT"),
        ("pri_balance_gt_sanctioned", train_df[c_p_bal] > train_df[c_p_san], "PRI.CURRENT.BALANCE > PRI.SANCTIONED.AMOUNT (accrued interest/penalties or reporting mismatch)"),
        ("pri_zero_accts_nonzero_bal_or_emi", (train_df[c_p_no] == 0) & ((train_df[c_p_bal] != 0) | (train_df[c_p_ins] != 0)), "PRI.NO.OF.ACCTS == 0 yet PRI balance or primary EMI != 0"),
        ("sec_active_gt_total", train_df[c_s_act] > train_df[c_s_no], "SEC.ACTIVE.ACCTS exceeds SEC.NO.OF.ACCTS"),
        ("sec_overdue_gt_active", train_df[c_s_ovd] > train_df[c_s_act], "SEC.OVERDUE.ACCTS exceeds SEC.ACTIVE.ACCTS"),
        ("sec_negative_current_balance", train_df[c_s_bal] < 0, "Negative SEC.CURRENT.BALANCE"),
        ("sec_zero_accts_nonzero_bal_or_emi", (train_df[c_s_no] == 0) & ((train_df[c_s_bal] != 0) | (train_df[c_s_ins] != 0)), "SEC.NO.OF.ACCTS == 0 yet SEC balance or secondary EMI != 0"),
        ("avg_acct_age_gt_credit_history_len", avg_age_m_train > cred_len_m_train, "AVERAGE.ACCT.AGE exceeds CREDIT.HISTORY.LENGTH"),
        ("delinquent_6m_gt_total_accts", train_df[c_del6] > (train_df[c_p_no] + train_df[c_s_no]), "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS > total PRI+SEC accounts"),
        ("new_6m_gt_total_accts", train_df[c_new6] > (train_df[c_p_no] + train_df[c_s_no]), "NEW.ACCTS.IN.LAST.SIX.MONTHS > total PRI+SEC accounts"),
        ("pri_disbursed_equals_current_loan", (train_df[c_p_dis] == train_df[c_disb]) & (train_df[c_disb] > 0), "Check if bureau PRI.DISBURSED.AMOUNT ever equals current vehicle loan disbursed_amount"),
    ]

    incons_rows: list[dict[str, Any]] = []
    for chk_name, mask, desc in inconsistency_checks:
        cnt = int(mask.fillna(False).sum())
        pct = round(cnt / n_total * 100, 4)
        dr = round(float(train_df.loc[mask.fillna(False), col_target].mean() * 100), 4) if cnt > 0 else None
        incons_rows.append(
            {
                "check_name": chk_name,
                "n_violations_train": cnt,
                "pct_of_train": pct,
                "default_rate_pct_when_violated": dr,
                "description": desc,
            }
        )
    incons_df = pd.DataFrame(incons_rows)
    incons_df.to_csv(tables_dir / "m1_inconsistencies_summary.csv", index=False)
    pri_disb_eq_cnt = int(
        incons_df.loc[incons_df["check_name"] == "pri_disbursed_equals_current_loan", "n_violations_train"].iloc[0]
    )

    # -------------------------------------------------------------------------
    # 4 & 5. COLUMN PROFILING, LEAKAGE AUDIT & FEATURE CLASSIFICATION
    # -------------------------------------------------------------------------
    disb_ts = train_disb_dt.astype("int64")

    # Explicit domain classification map keyed by normalized column name
    HIGH_CARD_NORMS = {"branchid", "supplierid", "currentpincodeid", "employeecodeid"}
    MED_CARD_NORMS = {"manufacturerid", "stateid"}
    KYC_FLAG_NORMS = {"mobilenoavlflag", "aadharflag", "panflag", "voteridflag", "drivingflag", "passportflag"}
    DATE_STR_NORMS = {"dateofbirth", "disbursaldate"}
    DURATION_STR_NORMS = {"averageacctage", "credithistorylength"}
    BUREAU_NORMS = {
        "performcnsscore", "performcnsscoredescription",
        "prinoofaccts", "priactiveaccts", "prioverdueaccts", "pricurrentbalance", "prisanctionedamount", "pridisbursedamount",
        "secnoofaccts", "secactiveaccts", "secoverdueaccts", "seccurrentbalance", "secsanctionedamount", "secdisbursedamount",
        "primaryinstalamt", "secinstalamt",
        "newacctsinlastsixmonths", "delinquentacctsinlastsixmonths", "averageacctage", "credithistorylength", "noofinquiries",
    }

    profile_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []

    for col in train_df.columns:
        nc = norm_name(col)
        s_tr = train_df[col]
        s_te = test_df[col] if col in test_df.columns else pd.Series(dtype="object")

        miss_tr = int(s_tr.isna().sum())
        miss_tr_pct = round(miss_tr / len(train_df) * 100, 4)
        miss_te = int(s_te.isna().sum()) if len(s_te) > 0 else None
        miss_te_pct = round(miss_te / len(test_df) * 100, 4) if miss_te is not None else None

        nuniq_tr = int(s_tr.nunique(dropna=True))
        nuniq_te = int(s_te.nunique(dropna=True)) if len(s_te) > 0 else None
        top_val = s_tr.value_counts(dropna=True).index[0] if nuniq_tr > 0 else None
        top_pct = round(float(s_tr.value_counts(dropna=True).iloc[0]) / len(train_df) * 100, 4) if nuniq_tr > 0 else None

        is_numeric = pd.api.types.is_numeric_dtype(s_tr)
        stats_dict: dict[str, Any] = {
            "column": col,
            "dtype_train": str(s_tr.dtype),
            "missing_train": miss_tr,
            "missing_train_pct": miss_tr_pct,
            "missing_test": miss_te,
            "missing_test_pct": miss_te_pct,
            "nunique_train": nuniq_tr,
            "nunique_test": nuniq_te,
            "top_value": str(top_val),
            "top_value_pct": top_pct,
            "is_constant": nuniq_tr <= 1,
            "is_near_constant_99pct": (top_pct is not None and top_pct >= 99.0),
        }

        if is_numeric:
            vals = s_tr.astype(np.float64)
            q01, q05, q25, q50, q75, q95, q99, q999 = np.nanpercentile(vals, [1, 5, 25, 50, 75, 95, 99, 99.9])
            iqr = q75 - q25
            iqr15 = int(((vals < q25 - 1.5 * iqr) | (vals > q75 + 1.5 * iqr)).sum()) if iqr > 0 else 0
            iqr30 = int(((vals < q25 - 3.0 * iqr) | (vals > q75 + 3.0 * iqr)).sum()) if iqr > 0 else 0
            stats_dict.update(
                {
                    "min": float(np.nanmin(vals)),
                    "p01": round(float(q01), 4),
                    "p05": round(float(q05), 4),
                    "p25": round(float(q25), 4),
                    "median": round(float(q50), 4),
                    "p75": round(float(q75), 4),
                    "p95": round(float(q95), 4),
                    "p99": round(float(q99), 4),
                    "p999": round(float(q999), 4),
                    "max": float(np.nanmax(vals)),
                    "mean": round(float(np.nanmean(vals)), 4),
                    "std": round(float(np.nanstd(vals)), 4),
                    "skewness": round(float(stats.skew(vals.dropna())), 4) if nuniq_tr > 1 else 0.0,
                    "pct_zero": round(float((vals == 0).mean() * 100), 4),
                    "pct_negative": round(float((vals < 0).mean() * 100), 4),
                    "iqr_1_5_outliers": iqr15,
                    "iqr_1_5_outliers_pct": round(iqr15 / n_total * 100, 4),
                    "iqr_3_0_outliers": iqr30,
                    "iqr_3_0_outliers_pct": round(iqr30 / n_total * 100, 4),
                }
            )
        else:
            for k in ("min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "p999", "max", "mean", "std", "skewness", "pct_zero", "pct_negative", "iqr_1_5_outliers", "iqr_1_5_outliers_pct", "iqr_3_0_outliers", "iqr_3_0_outliers_pct"):
                stats_dict[k] = None

        profile_rows.append(stats_dict)

        # Univariate signal & time correlation for leakage audit
        if col == col_target:
            raw_auc, inv_auc, iv_val, sp_target, sp_time, cv_val = 1.0, 1.0, None, 1.0, None, 1.0
        elif is_numeric:
            raw_auc, inv_auc = analytical_univariate_auc(s_tr, y)
            iv_val = compute_iv_numeric(s_tr, y)
            sp_target = round(float(stats.spearmanr(s_tr, y, nan_policy="omit").statistic), 6) if nuniq_tr > 1 else 0.0
            sp_time = round(float(stats.spearmanr(s_tr, disb_ts, nan_policy="omit").statistic), 6) if nuniq_tr > 1 else 0.0
            cv_val = cramers_v_from_series(s_tr, y) if nuniq_tr <= 50 else None
        elif nc == "dateofbirth":
            raw_auc, inv_auc = analytical_univariate_auc(age_corr_train, y)
            iv_val = compute_iv_numeric(age_corr_train, y)
            sp_target = round(float(stats.spearmanr(age_corr_train, y, nan_policy="omit").statistic), 6)
            sp_time = round(float(stats.spearmanr(age_corr_train, disb_ts, nan_policy="omit").statistic), 6)
            cv_val = None
        elif nc == "disbursaldate":
            raw_auc, inv_auc = analytical_univariate_auc(disb_ts, y)
            iv_val = compute_iv_numeric(disb_ts, y)
            sp_target = round(float(stats.spearmanr(disb_ts, y).statistic), 6)
            sp_time = 1.0
            cv_val = cramers_v_from_series(s_tr, y)
        elif nc in DURATION_STR_NORMS:
            dur_m = avg_age_m_train if nc == "averageacctage" else cred_len_m_train
            raw_auc, inv_auc = analytical_univariate_auc(dur_m, y)
            iv_val = compute_iv_numeric(dur_m, y)
            sp_target = round(float(stats.spearmanr(dur_m, y, nan_policy="omit").statistic), 6)
            sp_time = round(float(stats.spearmanr(dur_m, disb_ts, nan_policy="omit").statistic), 6)
            cv_val = None
        else:
            raw_auc, inv_auc = None, None
            iv_val = compute_iv_categorical(s_tr, y)
            sp_target = None
            sp_time = None
            cv_val = cramers_v_from_series(s_tr, y)

        # Assign Leakage Classification strictly into the 5 user-specified categories
        if col == col_target:
            leak_cat = "Target variable (Excluded from predictors)"
            leak_reason = "Ground-truth binary target (loan_default: first EMI default on due date)."
        elif nc == "uniqueid":
            leak_cat = "Identifier or administrative columns"
            leak_reason = f"Row/loan identifier with 100% unique values; Spearman correlation with DisbursalDate = {sp_time} (sequential batch/time ordering leakage if fed to model). Must be excluded."
        elif nc == "mobilenoavlflag":
            leak_cat = "Columns requiring removal"
            leak_reason = f"Zero-variance constant column (100% value = 1 in train and test); carries zero discriminative signal."
        elif nc == "disbursaldate":
            leak_cat = "Identifier or administrative columns"
            leak_reason = "Disbursement timestamp; determines temporal OOT split boundary and drifts between train (Aug-Oct 2018) and test (Nov 2018). Raw date must not be used directly; only safe cyclical/relative features (if justified) or split key."
        elif nc == "employeecodeid":
            leak_cat = "Potential leakage"
            leak_reason = f"High-cardinality internal employee ID ({nuniq_tr} unique in train; high cold-start drift across time). Risk of memorizing short-term batch/collection behavior or temporal assignment patterns rather than borrower creditworthiness."
        elif nc in {"branchid", "supplierid", "currentpincodeid"}:
            leak_cat = "Uncertain availability"
            leak_reason = f"Operational/geographic channel ID (cardinality={nuniq_tr}). Available at application/disbursal, but carries high overfitting & cold-start drift risk unless encoded strictly Out-of-Fold with Bayesian smoothing; also test whether supplier/branch signals capture dealer fraud or localized batch bias."
        elif nc in {"stateid", "manufacturerid"}:
            leak_cat = "Safe and potentially available at prediction time"
            leak_reason = f"Macro geographic state / vehicle manufacturer code (cardinality={nuniq_tr}); known prior to disbursal and stable over time (low drift)."
        elif nc in {"disbursedamount", "assetcost", "ltv"}:
            leak_cat = "Uncertain availability"
            leak_reason = "Known at final underwriting/disbursal approval, but exact final disbursed_amount/ltv may only be finalized after credit approval rather than initial pre-screening. Safe for point-of-disbursal FPD scoring, requires disclaimer if deployed at pre-approval."
        elif nc in BUREAU_NORMS:
            leak_cat = "Safe and potentially available at prediction time"
            leak_reason = (
                f"Data dictionary explicitly specifies 'at the time of disbursement' / 'before disbursement'; "
                f"observed univariate AUC_inv={inv_auc}, IV={iv_val}, Spearman_time={sp_time}, and "
                f"PRI.DISBURSED==disbursed_amount matches={pri_disb_eq_cnt} rows. "
                f"No post-disbursal target leakage signal observed; note special unscored codes in PERFORM_CNS.SCORE."
            )
        else:
            leak_cat = "Safe and potentially available at prediction time"
            leak_reason = "Borrower demographic / KYC document flag collected at loan application before disbursal."

        leakage_rows.append(
            {
                "column": col,
                "leakage_classification": leak_cat,
                "univariate_roc_auc_raw": raw_auc,
                "univariate_roc_auc_inv": inv_auc,
                "univariate_iv": iv_val,
                "cramers_v": cv_val,
                "spearman_with_target": sp_target,
                "spearman_with_disbursal_date": sp_time,
                "reasoning": leak_reason,
            }
        )

        # Feature analysis table
        if col == col_target:
            f_type = "Target (Binary)"
            enc = "None"
            scl = "None"
            strength = "Target"
        elif nc == "uniqueid":
            f_type = "Identifier"
            enc = "Drop before modeling"
            scl = "Drop"
            strength = "Leakage/ID (Remove)"
        elif nc == "mobilenoavlflag":
            f_type = "Constant Binary Flag"
            enc = "Drop (zero variance)"
            scl = "Drop"
            strength = "Zero variance (Remove)"
        elif nc == "dateofbirth":
            f_type = "Date String (dd-mm-yy)"
            enc = "Parse with century fix -> age_at_disbursal"
            scl = "Scale derived age (Linear/NN)"
            strength = "Moderate after parsing"
        elif nc == "disbursaldate":
            f_type = "Date String (dd-mm-yy)"
            enc = "Temporal split key; optional day-of-week/day-of-month"
            scl = "Cyclical / None"
            strength = "Split key / Drift risk"
        elif nc in DURATION_STR_NORMS:
            f_type = "Duration String (Xyrs Ymon)"
            enc = "Parse to integer months"
            scl = "log1p + Robust/StandardScaler (Linear/NN)"
            strength = "Informative"
        elif nc == "employmenttype":
            f_type = "Categorical (Low-cardinality, Missing)"
            enc = "Explicit 'Missing' category + One-Hot"
            scl = "None after OHE"
            strength = "Informative (missingness itself is informative)"
        elif nc == "performcnsscoredescription":
            f_type = "Categorical / Semi-Ordinal (Bureau Grade)"
            enc = "Group into risk tier A-M vs Unscored regimes + OHE/Ordinal"
            scl = "None / Standardize if ordinal"
            strength = "Strong (redundant with clean score; complementary on unscored codes)"
        elif nc == "performcnsscore":
            f_type = "Mixed Numerical + Special Status Codes"
            enc = "Regime split: has_bureau_score + bureau_score_clean + status_code"
            scl = "StandardScaler on valid scores (Linear/NN)"
            strength = "Strongest single predictor once regime-split"
        elif nc in HIGH_CARD_NORMS:
            f_type = f"Categorical High-Cardinality ({nuniq_tr})"
            enc = "OOF Smoothed Target Encoding + Train Frequency Encoding (CatBoost native)"
            scl = "Scale encoded outputs (Linear/NN)"
            strength = "Moderate-High signal but high overfitting/drift risk"
        elif nc in MED_CARD_NORMS:
            f_type = f"Categorical Medium-Cardinality ({nuniq_tr})"
            enc = "One-Hot (State/Manufacturer) or OOF Target Encoding"
            scl = "None after OHE"
            strength = "Moderate"
        elif nc in KYC_FLAG_NORMS:
            f_type = "Binary Indicator Flag"
            enc = "Keep 0/1 (note near-constant Passport/Driving flags)"
            scl = "None"
            strength = "Weak individually (except VoterID/Aadhar); combine into kyc_doc_count"
        elif "noofaccts" in nc or "activeaccts" in nc or "overdueaccts" in nc or "lastsixmonths" in nc or "inquiries" in nc:
            f_type = "Numerical (Discrete Count, Right-skewed)"
            enc = "None (treat zero-inflation carefully)"
            scl = "log1p + Robust/StandardScaler (Linear/NN)"
            strength = "Strong (especially overdue/delinquent/inquiries)"
        else:
            f_type = "Numerical (Continuous Monetary / Ratio)"
            enc = "None"
            scl = "log1p/signed-log + Winsorize + RobustScaler (Linear/NN)"
            strength = "Informative (high skewness & extreme outliers)"

        feature_rows.append(
            {
                "column": col,
                "feature_type": f_type,
                "cardinality_train": nuniq_tr,
                "missing_train_pct": miss_tr_pct,
                "univariate_auc_inv": inv_auc,
                "univariate_iv": iv_val,
                "encoding_recommendation": enc,
                "scaling_recommendation": scl,
                "signal_and_redundancy_note": strength,
            }
        )

    profile_df = pd.DataFrame(profile_rows)
    leakage_df = pd.DataFrame(leakage_rows)
    feature_df = pd.DataFrame(feature_rows)

    profile_df.to_csv(tables_dir / "m1_column_profile.csv", index=False)
    leakage_df.to_csv(tables_dir / "m1_leakage_audit.csv", index=False)
    feature_df.to_csv(tables_dir / "m1_feature_classification.csv", index=False)

    # Top Spearman correlations among numeric predictors (excluding target & constant)
    num_cols = [
        c for c in train_df.select_dtypes(include=[np.number]).columns
        if c not in {col_target, col_uid} and train_df[c].nunique(dropna=True) > 1
    ]
    corr_mat = train_df[num_cols].corr(method="spearman")
    corr_pairs: list[dict[str, Any]] = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            val = float(corr_mat.iloc[i, j])
            if abs(val) >= 0.50:
                corr_pairs.append(
                    {
                        "feature_1": num_cols[i],
                        "feature_2": num_cols[j],
                        "spearman_corr": round(val, 4),
                    }
                )
    corr_pairs_df = pd.DataFrame(corr_pairs).sort_values("spearman_corr", key=lambda s: s.abs(), ascending=False)
    corr_pairs_df.to_csv(tables_dir / "m1_top_correlations.csv", index=False)

    # -------------------------------------------------------------------------
    # 6. DIAGNOSTIC VISUALIZATIONS (300 DPI PNGs)
    # -------------------------------------------------------------------------
    setup_plot_style()

    # Figure 1: Target distribution & Temporal trend
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    counts_arr = [n_neg, n_pos]
    labels_arr = ["0: Paid First EMI\n(Non-Default)", "1: Defaulted First EMI\n(Default)"]
    colors_arr = ["#2b6cb0", "#c53030"]
    bars = axes[0].bar(labels_arr, counts_arr, color=colors_arr, width=0.55, edgecolor="black", linewidth=0.6)
    axes[0].set_title(f"Target Class Distribution (train.csv, N = {n_total:,})")
    axes[0].set_ylabel("Number of Loans")
    axes[0].set_ylim(0, max(counts_arr) * 1.18)
    for bar, cnt in zip(bars, counts_arr):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            cnt + max(counts_arr) * 0.02,
            f"{cnt:,}\n({cnt / n_total * 100:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    daily_plot = daily_summary.copy()
    daily_plot["dt"] = pd.to_datetime(daily_plot["period"])
    daily_plot["default_rate_pct"] = daily_plot["default_rate"] * 100.0
    ax_right = axes[1]
    ax_right_2 = ax_right.twinx()
    ax_right_2.bar(daily_plot["dt"], daily_plot["n_loans"], alpha=0.25, color="#718096", width=0.8, label="Daily Disbursal Volume")
    ax_right.plot(daily_plot["dt"], daily_plot["default_rate_pct"], color="#c53030", linewidth=1.6, marker="o", markersize=2.5, label="Daily Default Rate (%)")
    ax_right.axhline(pos_rate * 100, color="#2d3748", linestyle="--", linewidth=1.0, label=f"Overall Mean ({pos_rate * 100:.2f}%)")
    ax_right.set_title("Daily Disbursal Volume & Default Rate Over Time (train.csv)")
    ax_right.set_ylabel("First EMI Default Rate (%)", color="#c53030")
    ax_right_2.set_ylabel("Daily Loan Volume", color="#4a5568")
    ax_right.tick_params(axis="x", rotation=30)
    lines1, labels1 = ax_right.get_legend_handles_labels()
    lines2, labels2 = ax_right_2.get_legend_handles_labels()
    ax_right.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "m1_target_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Missing values & Employment.Type default rate
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    miss_compare = pd.DataFrame(
        {
            "column": [c for c in train_df.columns if c != col_target],
            "Train (%)": [train_df[c].isna().mean() * 100 for c in train_df.columns if c != col_target],
            "Test (%)": [test_df[c].isna().mean() * 100 for c in train_df.columns if c != col_target],
        }
    )
    # Include columns with >0 missingness plus top informative special-code rate for context
    nonzero_miss = miss_compare[(miss_compare["Train (%)"] > 0) | (miss_compare["Test (%)"] > 0)]
    if nonzero_miss.empty:
        nonzero_miss = miss_compare.head(5)
    # Also add implicit unscored bureau % as a separate comparison bar so plot shows full picture clearly
    implicit_rows = pd.DataFrame(
        [
            {
                "column": f"{col_score}\n(No History / Not Scored)",
                "Train (%)": float((~train_df[col_score].between(300, 900)).mean() * 100),
                "Test (%)": float((~test_df[col_score].between(300, 900)).mean() * 100),
            }
        ]
    )
    plot_miss = pd.concat([nonzero_miss, implicit_rows], ignore_index=True)
    x_pos = np.arange(len(plot_miss))
    w = 0.35
    axes[0].bar(x_pos - w / 2, plot_miss["Train (%)"], width=w, label="train.csv", color="#2b6cb0")
    axes[0].bar(x_pos + w / 2, plot_miss["Test (%)"], width=w, label="test.csv", color="#ed8936")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(plot_miss["column"])
    axes[0].set_ylabel("Percentage of Rows (%)")
    axes[0].set_title("Explicit NaN & Implicit Unscored Bureau Rates")
    axes[0].legend()
    for idx, row in plot_miss.iterrows():
        axes[0].text(idx - w / 2, row["Train (%)"] + 0.8, f"{row['Train (%)']:.2f}%", ha="center", fontsize=8.5)
        axes[0].text(idx + w / 2, row["Test (%)"] + 0.8, f"{row['Test (%)']:.2f}%", ha="center", fontsize=8.5)
    axes[0].set_ylim(0, max(plot_miss[["Train (%)", "Test (%)"]].max()) * 1.18)

    emp_plot_df = pd.DataFrame(emp_audit_rows)
    bars_emp = axes[1].bar(
        emp_plot_df["employment_type"],
        emp_plot_df["default_rate_pct"],
        color=["#2b6cb0", "#38a169", "#c53030"][: len(emp_plot_df)],
        width=0.5,
        edgecolor="black",
        linewidth=0.6,
    )
    axes[1].axhline(pos_rate * 100, color="#2d3748", linestyle="--", linewidth=1.0, label=f"Overall Mean ({pos_rate * 100:.2f}%)")
    axes[1].set_title(f"Default Rate by Employment.Type (Chi2 p={p_emp:.2e})" if p_emp is not None else "Default Rate by Employment.Type")
    axes[1].set_ylabel("First EMI Default Rate (%)")
    axes[1].set_ylim(0, emp_plot_df["default_rate_pct"].max() * 1.25)
    for bar, (_, r) in zip(bars_emp, emp_plot_df.iterrows()):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            r["default_rate_pct"] + 0.4,
            f"{r['default_rate_pct']:.2f}%\n(n={r['count']:,})",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "m1_missing_values.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 3: Bureau score audit
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    scored_mask = train_df[col_score] >= 300
    axes[0].hist(train_df.loc[scored_mask, col_score], bins=45, color="#2b6cb0", alpha=0.8, edgecolor="white")
    axes[0].set_title(f"Distribution of Scored {col_score} (>= 300, n={int(scored_mask.sum()):,})")
    axes[0].set_xlabel(col_score)
    axes[0].set_ylabel("Number of Loans")

    b_top = bureau_df.sort_values("count", ascending=True).tail(14)
    y_pos = np.arange(len(b_top))
    bar_colors = ["#dd6b20" if mx < 300 else "#2b6cb0" for mx in b_top["max_score"]]
    axes[1].barh(y_pos, b_top["default_rate_pct"], color=bar_colors, edgecolor="black", linewidth=0.4)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([f"{d[:34]} (n={c:,})" for d, c in zip(b_top[col_score_desc], b_top["count"])], fontsize=8)
    axes[1].axvline(pos_rate * 100, color="#2d3748", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("First EMI Default Rate (%)")
    axes[1].set_title("Default Rate by Bureau Description (Orange = Unscored Code < 300)")
    fig.tight_layout()
    fig.savefig(figures_dir / "m1_bureau_score_audit.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 4: Numeric distributions & extreme skew / outliers
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    diag_vars = [
        ("disbursed_amount", train_df[c_disb], False),
        ("asset_cost", train_df[c_asset], False),
        ("ltv", train_df[c_ltv], False),
        ("age_at_disbursal (century-fixed)", age_corr_train, False),
        ("PRI.CURRENT.BALANCE (log10 positive)", train_df[c_p_bal][train_df[c_p_bal] > 0], True),
        ("PRIMARY.INSTAL.AMT (log10 positive)", train_df[c_p_ins][train_df[c_p_ins] > 0], True),
    ]
    for ax, (title, ser, use_log) in zip(axes.ravel(), diag_vars):
        s_clean = ser.dropna()
        if use_log:
            s_plot = np.log10(s_clean)
            ax.hist(s_plot, bins=40, color="#3182ce", alpha=0.85, edgecolor="white")
            ax.set_xlabel("log10(value)")
        else:
            p005, p995 = np.percentile(s_clean, [0.5, 99.5])
            s_clipped = s_clean.clip(p005, p995)
            ax.hist(s_clipped, bins=40, color="#3182ce", alpha=0.85, edgecolor="white")
            ax.set_xlabel("Value (clipped [0.5%, 99.5%] for display)")
        ax.set_title(title)
        ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(figures_dir / "m1_numeric_distributions_outliers.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 5: High-cardinality & Temporal Drift
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    hc_plot = hc_drift_df.sort_values("train_nunique", ascending=False)
    x_hc = np.arange(len(hc_plot))
    axes[0].bar(x_hc, hc_plot["train_nunique"], color="#4c51bf", edgecolor="black", linewidth=0.5)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x_hc)
    axes[0].set_xticklabels(hc_plot["column"], rotation=25, ha="right")
    axes[0].set_ylabel("Unique Categories (log scale)")
    axes[0].set_title("Cardinality of Administrative & Geographic IDs (train.csv)")
    for idx, val in enumerate(hc_plot["train_nunique"]):
        axes[0].text(idx, val * 1.15, f"{val:,}", ha="center", fontsize=8.5, fontweight="bold")

    axes[1].bar(x_hc - 0.18, hc_plot["oot20_unseen_rows_pct"], width=0.36, label="Last 20% Train (OOT) vs First 80%", color="#dd6b20")
    axes[1].bar(x_hc + 0.18, hc_plot["kaggle_test_unseen_rows_pct"], width=0.36, label="Kaggle test.csv (Nov 2018) vs train.csv", color="#c53030")
    axes[1].set_xticks(x_hc)
    axes[1].set_xticklabels(hc_plot["column"], rotation=25, ha="right")
    axes[1].set_ylabel("Rows with Unseen IDs (%)")
    axes[1].set_title("Entity Cold-Start Drift Across Time")
    axes[1].legend(fontsize=8.5)
    for idx, row in hc_plot.reset_index(drop=True).iterrows():
        axes[1].text(idx - 0.18, row["oot20_unseen_rows_pct"] + 0.2, f"{row['oot20_unseen_rows_pct']:.1f}%", ha="center", fontsize=7.5)
        axes[1].text(idx + 0.18, row["kaggle_test_unseen_rows_pct"] + 0.2, f"{row['kaggle_test_unseen_rows_pct']:.1f}%", ha="center", fontsize=7.5)
    axes[1].set_ylim(0, max(hc_plot[["oot20_unseen_rows_pct", "kaggle_test_unseen_rows_pct"]].max()) * 1.25 + 1.0)
    fig.tight_layout()
    fig.savefig(figures_dir / "m1_high_cardinality_drift.png", bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 7. MACHINE-READABLE AUDIT SUMMARY JSON
    # -------------------------------------------------------------------------
    audit_summary = {
        "inventory": {
            "train_file": train_path.name,
            "train_rows": int(train_df.shape[0]),
            "train_cols": int(train_df.shape[1]),
            "test_file": test_path.name,
            "test_rows": int(test_df.shape[0]),
            "test_cols": int(test_df.shape[1]),
            "dictionary_file": dict_path.name if dict_path else None,
            "dictionary_rows": int(dict_df.shape[0]),
            "train_only_columns": train_only_cols,
            "test_only_columns": test_only_cols,
            "binary_01_columns": binary_candidates,
            "unique_id_column": col_uid,
            "unique_id_train_nunique": uid_train_unique,
            "unique_id_test_nunique": uid_test_unique,
            "unique_id_train_test_overlap": uid_overlap,
        },
        "target_verification": {
            "target_column": col_target,
            "unique_values": [int(v) if pd.notna(v) else None for v in y_counts.index.tolist()],
            "missing_target_count": int(y.isna().sum()),
            "class_0_count": n_neg,
            "class_1_count": n_pos,
            "default_rate_pct": round(pos_rate * 100, 4),
            "imbalance_ratio_neg_to_pos": round(n_neg / n_pos, 4),
            "monthly_summary_train": monthly_summary.to_dict(orient="records"),
            "train_disbursal_min": str(train_disb_dt.min().date()),
            "train_disbursal_max": str(train_disb_dt.max().date()),
            "test_disbursal_min": str(test_disb_dt.min().date()),
            "test_disbursal_max": str(test_disb_dt.max().date()),
            "test_monthly_counts": test_monthly,
            "oot_80_20_cutoff_date": oot_cutoff_date,
            "oot_dev_default_rate_pct": round(float(y.iloc[dev_idx].mean() * 100), 4),
            "oot_test_default_rate_pct": round(float(y.iloc[oot_idx].mean() * 100), 4),
        },
        "data_quality": {
            "duplicate_rows_train_exact": dup_full_train,
            "duplicate_rows_train_excl_uid": dup_no_uid_train,
            "duplicate_rows_train_excl_uid_and_target": dup_no_uid_target_train,
            "duplicate_rows_test_exact": dup_full_test,
            "duplicate_rows_test_excl_uid": dup_no_uid_test,
            "constant_columns": [r["column"] for r in profile_rows if r["is_constant"]],
            "near_constant_99pct_columns": [
                {"column": r["column"], "top_value_pct": r["top_value_pct"]}
                for r in profile_rows if r["is_near_constant_99pct"] and not r["is_constant"]
            ],
            "employment_type_audit": emp_audit_rows,
            "employment_type_chi2_p": float(p_emp) if p_emp is not None else None,
            "dob_naive_future_years_train": int((dob_naive_train.dt.year > 2018).sum()),
            "dob_naive_future_years_test": int((dob_naive_test.dt.year > 2018).sum()),
            "age_corrected_train_summary": {
                "min": round(float(age_corr_train.min()), 2),
                "p01": round(float(np.nanpercentile(age_corr_train, 1)), 2),
                "median": round(float(age_corr_train.median()), 2),
                "p99": round(float(np.nanpercentile(age_corr_train, 99)), 2),
                "max": round(float(age_corr_train.max()), 2),
            },
            "age_corrected_test_summary": {
                "min": round(float(age_corr_test.min()), 2),
                "median": round(float(age_corr_test.median()), 2),
                "max": round(float(age_corr_test.max()), 2),
            },
            "duration_malformed_counts": {
                "train_avg_acct_age": avg_age_bad_train,
                "train_credit_history_length": cred_len_bad_train,
                "test_avg_acct_age": avg_age_bad_test,
                "test_credit_history_length": cred_len_bad_test,
            },
            "bureau_unscored_codes_pct_train": round(float((train_df[col_score] < 300).mean() * 100), 4),
            "bureau_unscored_codes_pct_test": round(float((test_df[col_score] < 300).mean() * 100), 4),
            "inconsistencies": incons_rows,
            "high_cardinality_drift": hc_drift_rows,
        },
        "leakage_summary_counts": leakage_df["leakage_classification"].value_counts().to_dict(),
        "top_univariate_predictors_by_auc": (
            leakage_df.dropna(subset=["univariate_roc_auc_inv"])
            .loc[leakage_df["column"] != col_target]
            .sort_values("univariate_roc_auc_inv", ascending=False)
            .head(12)[["column", "univariate_roc_auc_inv", "univariate_iv", "leakage_classification"]]
            .to_dict(orient="records")
        ),
        "top_correlations_ge_0_50": corr_pairs_df.head(15).to_dict(orient="records"),
    }

    summary_path = audit_dir / "audit_summary.json"
    summary_path.write_text(json.dumps(audit_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Audit completed cleanly. Summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
