"""Deterministic, stateless feature engineering aligned with M1 audit findings.

Supports four nested feature tiers for RQ2 ablation experiments:
  - 'F0': Deterministic cleaned raw baseline (IDs/constants dropped, dates & durations parsed, missing employment preserved)
  - 'F1': F0 + 29 domain-driven, aggregation, ratio, and audited interaction/anomaly features
  - 'F2_safe': F1 + operational/geographic high-cardinality IDs (branch_id, supplier_id, Current_pincode_ID)
  - 'F2_plus_emp': F2_safe + Employee_code_ID (potential-leakage comparison arm)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.data.load_data import (
    parse_disbursal_dates,
    parse_dob_century_safe,
    parse_duration_months_series,
)

VALID_TIERS = ("F0", "F1", "F2_safe", "F2_plus_emp")

BUREAU_TIER_RISK_ORDER: dict[str, int] = {
    letter: idx for idx, letter in enumerate("ABCDEFGHIJKLM", start=1)
}


def _map_bureau_unscored_group(desc: pd.Series, score: pd.Series) -> pd.Series:
    """Group PERFORM_CNS.SCORE.DESCRIPTION into clean regime labels."""
    s = desc.fillna("MISSING").astype(str)
    out = np.where(
        score >= 300,
        "SCORED",
        np.where(
            s.str.contains("No Bureau History", case=False, na=False),
            "NO_HISTORY",
            np.where(
                s.str.contains("Guarantor", case=False, na=False),
                "GUARANTOR_ONLY",
                np.where(
                    s.str.contains("Inactive|36 months", case=False, na=False),
                    "INACTIVE_STALE",
                    "INSUFFICIENT_INFO",
                ),
            ),
        ),
    )
    return pd.Series(out, index=desc.index, dtype=object)


def _map_bureau_risk_ordinal(desc: pd.Series, score: pd.Series) -> pd.Series:
    """Map A-Very Low Risk..M-Very High Risk to 1..13; unscored codes (<300) to 0."""
    letters = desc.astype(str).str.strip().str[0].str.upper()
    mapped = letters.map(BUREAU_TIER_RISK_ORDER).fillna(0).astype(np.int64)
    mapped = mapped.where(score >= 300, 0)
    return mapped


def get_feature_rationale_table() -> pd.DataFrame:
    """Return structured documentation for all engineered features in F0/F1."""
    rows: list[dict[str, str]] = [
        # F0 deterministic parsing
        {"feature": "age_at_disbursal", "tier": "F0", "category": "Domain-driven", "formula": "(DisbursalDate - century_fixed(Date.of.Birth)) / 365.25", "rationale": "Fixes 8.52% 2-digit year century rollover ('00-'68 -> 20xx) discovered in M1 audit."},
        {"feature": "avg_acct_age_months", "tier": "F0", "category": "Domain-driven", "formula": "yrs*12 + mon from AVERAGE.ACCT.AGE", "rationale": "Converts 'Xyrs Ymon' string into numeric months cleanly (0 parse errors)."},
        {"feature": "credit_history_months", "tier": "F0", "category": "Domain-driven", "formula": "yrs*12 + mon from CREDIT.HISTORY.LENGTH", "rationale": "Converts credit history length string into numeric months."},
        # F1 Domain-driven
        {"feature": "has_bureau_score", "tier": "F1", "category": "Domain-driven", "formula": "1 if PERFORM_CNS.SCORE >= 300 else 0", "rationale": "Separates the 55.66% unscored/thin-file records from valid 300-890 bureau scores."},
        {"feature": "bureau_score_clean", "tier": "F1", "category": "Domain-driven", "formula": "PERFORM_CNS.SCORE if >= 300 else NaN", "rationale": "Prevents linear/MLP models from treating special status codes (0, 11-18) as ultra-low numeric scores."},
        {"feature": "bureau_unscored_group", "tier": "F1", "category": "Domain-driven", "formula": "Grouped unscored regime (SCORED/NO_HISTORY/GUARANTOR_ONLY/INACTIVE_STALE/INSUFFICIENT_INFO)", "rationale": "Preserves risk heterogeneity observed in M1 across unscored codes (18.4% vs 25.6% default)."},
        {"feature": "bureau_risk_ordinal", "tier": "F1", "category": "Domain-driven", "formula": "A->1 .. M->13 for scored; 0 for unscored", "rationale": "Ordinal risk tier encoding capturing the broad upward default progression across scored bureau grades A..M."},
        {"feature": "estimated_down_payment", "tier": "F1", "category": "Domain-driven", "formula": "asset_cost - disbursed_amount", "rationale": "Quantifies borrower initial equity buffer / skin-in-the-game."},
        {"feature": "down_payment_ratio", "tier": "F1", "category": "Domain-driven", "formula": "(asset_cost - disbursed_amount) / asset_cost", "rationale": "Scale-normalized equity share independent of vehicle segment price."},
        {"feature": "ltv_computed", "tier": "F1", "category": "Domain-driven", "formula": "100.0 * disbursed_amount / asset_cost", "rationale": "Raw ratio of disbursed amount to declared asset cost."},
        {"feature": "ltv_discrepancy", "tier": "F1", "category": "Domain-driven", "formula": "ltv - ltv_computed", "rationale": "Captures financing add-ons/fees/valuation gap (3.49% rows differ >5 pts in M1)."},
        # F1 Aggregation-based
        {"feature": "total_accts", "tier": "F1", "category": "Aggregation-based", "formula": "PRI.NO.OF.ACCTS + SEC.NO.OF.ACCTS", "rationale": "Consolidates primary and co-applicant/guarantor total loan count."},
        {"feature": "total_active_accts", "tier": "F1", "category": "Aggregation-based", "formula": "PRI.ACTIVE.ACCTS + SEC.ACTIVE.ACCTS", "rationale": "Total active debt obligations across primary and secondary roles."},
        {"feature": "total_overdue_accts", "tier": "F1", "category": "Aggregation-based", "formula": "PRI.OVERDUE.ACCTS + SEC.OVERDUE.ACCTS", "rationale": "Combined delinquent account count (secondary columns are 99.4% zero individually)."},
        {"feature": "total_current_balance", "tier": "F1", "category": "Aggregation-based", "formula": "PRI.CURRENT.BALANCE + SEC.CURRENT.BALANCE", "rationale": "Total outstanding principal exposure."},
        {"feature": "total_sanctioned_amount", "tier": "F1", "category": "Aggregation-based", "formula": "PRI.SANCTIONED.AMOUNT + SEC.SANCTIONED.AMOUNT", "rationale": "Total historical credit limits granted."},
        {"feature": "total_disbursed_amount", "tier": "F1", "category": "Aggregation-based", "formula": "PRI.DISBURSED.AMOUNT + SEC.DISBURSED.AMOUNT", "rationale": "Total historical credit disbursed."},
        {"feature": "total_instal_amt", "tier": "F1", "category": "Aggregation-based", "formula": "PRIMARY.INSTAL.AMT + SEC.INSTAL.AMT", "rationale": "Combined monthly EMI obligation prior to this vehicle loan."},
        {"feature": "has_sec_account", "tier": "F1", "category": "Aggregation-based", "formula": "1 if SEC.NO.OF.ACCTS > 0 else 0", "rationale": "Condenses sparse secondary-role presence (~2.5% non-zero) into a clean indicator."},
        {"feature": "kyc_doc_count", "tier": "F1", "category": "Aggregation-based", "formula": "Sum of Aadhar, PAN, VoterID, Driving, Passport flags", "rationale": "Aggregates identity verification breadth (excludes zero-variance MobileNo_Avl_Flag)."},
        # F1 Statistical / Ratio-based
        {"feature": "overdue_to_total_ratio", "tier": "F1", "category": "Statistical/Ratio", "formula": "total_overdue_accts / (total_accts + 1.0)", "rationale": "Smoothed historical default propensity per credit account."},
        {"feature": "active_to_total_ratio", "tier": "F1", "category": "Statistical/Ratio", "formula": "total_active_accts / (total_accts + 1.0)", "rationale": "Proportion of open vs. closed credit lines."},
        {"feature": "credit_utilization_ratio", "tier": "F1", "category": "Statistical/Ratio", "formula": "clip(total_current_balance / (total_sanctioned_amount + 1.0), -1.0, 5.0)", "rationale": "Leverage/repayment progress relative to sanctioned limit (clipped against extreme outliers)."},
        {"feature": "instalment_to_disbursed_ratio", "tier": "F1", "category": "Statistical/Ratio", "formula": "total_instal_amt / (disbursed_amount + 1.0)", "rationale": "Existing monthly debt service burden relative to new vehicle loan size."},
        {"feature": "delinquent_6m_to_total_ratio", "tier": "F1", "category": "Statistical/Ratio", "formula": "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS / (NEW.ACCTS.IN.LAST.SIX.MONTHS + 1.0)", "rationale": "Recent 6-month credit distress intensity."},
        {"feature": "inquiry_to_accts_ratio", "tier": "F1", "category": "Statistical/Ratio", "formula": "NO.OF_INQUIRIES / (total_accts + 1.0)", "rationale": "Credit-seeking hunger normalized by prior credit footprint."},
        {"feature": "history_span_gap_months", "tier": "F1", "category": "Statistical/Ratio", "formula": "credit_history_months - avg_acct_age_months", "rationale": "Measures credit seasoning dispersion / recent credit expansion."},
        # F1 Interaction & Audited Anomaly Flags
        {"feature": "flag_pri_overdue_gt_active", "tier": "F1", "category": "Interaction/Anomaly", "formula": "1 if PRI.OVERDUE.ACCTS > PRI.ACTIVE.ACCTS else 0", "rationale": "M1 audit found 2,607 rows with 27.46% default rate (closed/written-off overdue debt)."},
        {"feature": "flag_pri_bal_gt_sanctioned", "tier": "F1", "category": "Interaction/Anomaly", "formula": "1 if PRI.CURRENT.BALANCE > PRI.SANCTIONED.AMOUNT else 0", "rationale": "M1 audit found 7,593 rows with 24.25% default rate (accrued interest/limit overrun)."},
        {"feature": "flag_neg_balance", "tier": "F1", "category": "Interaction/Anomaly", "formula": "1 if (PRI.CURRENT.BALANCE < 0 or SEC.CURRENT.BALANCE < 0) else 0", "rationale": "M1 audit found negative balances carry lower default rate (14.51%)."},
        {"feature": "ltv_x_no_bureau", "tier": "F1", "category": "Interaction/Anomaly", "formula": "ltv * (1 - has_bureau_score)", "rationale": "Captures compounded risk of high leverage without verifiable credit bureau history."},
        {"feature": "age_at_first_credit_years", "tier": "F1", "category": "Interaction/Anomaly", "formula": "age_at_disbursal - credit_history_months / 12.0", "rationale": "Interaction between borrower age and credit bureau entry age."},
    ]
    return pd.DataFrame(rows)


class LTFSFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless, leak-free feature engineering transformer for LTFS vehicle loans."""

    def __init__(self, feature_tier: str = "F1") -> None:
        if feature_tier not in VALID_TIERS:
            raise ValueError(f"feature_tier must be one of {VALID_TIERS}, got '{feature_tier}'")
        self.feature_tier = feature_tier

    def fit(self, X: pd.DataFrame, y: Any = None) -> "LTFSFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Drop identifiers, constant columns, and target if accidentally present
        for drop_col in ("UniqueID", "MobileNo_Avl_Flag", "loan_default"):
            if drop_col in df.columns:
                df = df.drop(columns=[drop_col])

        # Parse dates safely
        disb_dt = parse_disbursal_dates(df["DisbursalDate"])
        dob_dt = parse_dob_century_safe(df["Date.of.Birth"], reference_years=disb_dt.dt.year)
        df["age_at_disbursal"] = ((disb_dt - dob_dt).dt.days / 365.25).astype(np.float64)

        # Parse credit duration strings into numeric months
        df["avg_acct_age_months"] = parse_duration_months_series(df["AVERAGE.ACCT.AGE"])
        df["credit_history_months"] = parse_duration_months_series(df["CREDIT.HISTORY.LENGTH"])

        # Normalize missing Employment.Type to explicit meaningful category
        df["Employment.Type"] = df["Employment.Type"].fillna("MISSING").astype(object)
        df["PERFORM_CNS.SCORE.DESCRIPTION"] = df["PERFORM_CNS.SCORE.DESCRIPTION"].fillna("MISSING").astype(object)

        # Drop raw string columns that have been deterministically parsed
        df = df.drop(columns=["Date.of.Birth", "DisbursalDate", "AVERAGE.ACCT.AGE", "CREDIT.HISTORY.LENGTH"])

        if self.feature_tier == "F0":
            for hc in ("branch_id", "supplier_id", "Current_pincode_ID", "Employee_code_ID"):
                if hc in df.columns:
                    df = df.drop(columns=[hc])
            return df

        # F1 engineered features
        score = df["PERFORM_CNS.SCORE"].astype(np.float64)
        desc = df["PERFORM_CNS.SCORE.DESCRIPTION"]
        has_score = (score >= 300).astype(np.int64)

        df["has_bureau_score"] = has_score
        df["bureau_score_clean"] = np.where(has_score == 1, score, np.nan)
        df["bureau_unscored_group"] = _map_bureau_unscored_group(desc, score)
        df["bureau_risk_ordinal"] = _map_bureau_risk_ordinal(desc, score)

        disb_amt = df["disbursed_amount"].astype(np.float64)
        asset = df["asset_cost"].astype(np.float64)
        ltv_decl = df["ltv"].astype(np.float64)

        df["estimated_down_payment"] = asset - disb_amt
        df["down_payment_ratio"] = np.where(asset > 0, (asset - disb_amt) / asset, 0.0)
        df["ltv_computed"] = np.where(asset > 0, 100.0 * disb_amt / asset, 0.0)
        df["ltv_discrepancy"] = ltv_decl - df["ltv_computed"]

        # Aggregation-based
        df["total_accts"] = df["PRI.NO.OF.ACCTS"] + df["SEC.NO.OF.ACCTS"]
        df["total_active_accts"] = df["PRI.ACTIVE.ACCTS"] + df["SEC.ACTIVE.ACCTS"]
        df["total_overdue_accts"] = df["PRI.OVERDUE.ACCTS"] + df["SEC.OVERDUE.ACCTS"]
        df["total_current_balance"] = df["PRI.CURRENT.BALANCE"] + df["SEC.CURRENT.BALANCE"]
        df["total_sanctioned_amount"] = df["PRI.SANCTIONED.AMOUNT"] + df["SEC.SANCTIONED.AMOUNT"]
        df["total_disbursed_amount"] = df["PRI.DISBURSED.AMOUNT"] + df["SEC.DISBURSED.AMOUNT"]
        df["total_instal_amt"] = df["PRIMARY.INSTAL.AMT"] + df["SEC.INSTAL.AMT"]
        df["has_sec_account"] = (df["SEC.NO.OF.ACCTS"] > 0).astype(np.int64)
        df["kyc_doc_count"] = (
            df["Aadhar_flag"]
            + df["PAN_flag"]
            + df["VoterID_flag"]
            + df["Driving_flag"]
            + df["Passport_flag"]
        ).astype(np.int64)

        # Statistical / ratio-based
        tot_accts_f = df["total_accts"].astype(np.float64)
        df["overdue_to_total_ratio"] = df["total_overdue_accts"].astype(np.float64) / (tot_accts_f + 1.0)
        df["active_to_total_ratio"] = df["total_active_accts"].astype(np.float64) / (tot_accts_f + 1.0)
        raw_util = df["total_current_balance"].astype(np.float64) / (df["total_sanctioned_amount"].astype(np.float64) + 1.0)
        df["credit_utilization_ratio"] = np.clip(raw_util, -1.0, 5.0)
        df["instalment_to_disbursed_ratio"] = df["total_instal_amt"].astype(np.float64) / (disb_amt + 1.0)
        df["delinquent_6m_to_total_ratio"] = (
            df["DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS"].astype(np.float64)
            / (df["NEW.ACCTS.IN.LAST.SIX.MONTHS"].astype(np.float64) + 1.0)
        )
        df["inquiry_to_accts_ratio"] = df["NO.OF_INQUIRIES"].astype(np.float64) / (tot_accts_f + 1.0)
        df["history_span_gap_months"] = df["credit_history_months"] - df["avg_acct_age_months"]

        # Interaction & M1-audited anomaly flags
        df["flag_pri_overdue_gt_active"] = (df["PRI.OVERDUE.ACCTS"] > df["PRI.ACTIVE.ACCTS"]).astype(np.int64)
        df["flag_pri_bal_gt_sanctioned"] = (df["PRI.CURRENT.BALANCE"] > df["PRI.SANCTIONED.AMOUNT"]).astype(np.int64)
        df["flag_neg_balance"] = (
            (df["PRI.CURRENT.BALANCE"] < 0) | (df["SEC.CURRENT.BALANCE"] < 0)
        ).astype(np.int64)
        df["ltv_x_no_bureau"] = ltv_decl * (1.0 - has_score)
        df["age_at_first_credit_years"] = df["age_at_disbursal"] - (df["credit_history_months"] / 12.0)

        # Tier filtering for high-cardinality operational/geographic IDs
        if self.feature_tier == "F1":
            for hc in ("branch_id", "supplier_id", "Current_pincode_ID", "Employee_code_ID"):
                if hc in df.columns:
                    df = df.drop(columns=[hc])
        elif self.feature_tier == "F2_safe":
            if "Employee_code_ID" in df.columns:
                df = df.drop(columns=["Employee_code_ID"])
        # F2_plus_emp retains all four high-cardinality columns

        return df
