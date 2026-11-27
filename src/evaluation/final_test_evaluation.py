"""Final Single-Use Locked Test Evaluation, Feature Ablation, OOT vs. Random & Test XAI.

Strictly freezes every preprocessor, model, and decision threshold ONLY on the
Development Set, then evaluates each Locked Test Set (`oot` N=50,653; `random` N=46,631) in
a single reproducible pass across the 6 core models.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from src.data.load_data import get_project_root, load_config, parse_disbursal_dates

os.environ.setdefault("MPLCONFIGDIR", str(get_project_root() / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit

from src.data.splits import load_split_frames
from src.evaluation.metrics import (
    compute_binary_metrics,
    compute_expected_cost_amount_weighted,
    compute_expected_cost_fixed,
    find_optimal_thresholds,
)
from src.evaluation.tracker import ExperimentTracker
from src.explainability.analyze_explainability import (
    _select_case_study_indices,
    _top_k_jaccard,
    assign_domain_group,
    canonical_feature_name,
)
from src.features.engineering import LTFSFeatureEngineer
from src.features.preprocessing import build_preprocessing_pipeline
from src.models.imbalance_calibration import _apply_fold_resampler
from src.models.registry import build_model, load_model_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _evaluate_frozen_predictions(
    model_label: str,
    y_test: np.ndarray,
    probs_test: np.ndarray,
    disbursed_test: np.ndarray,
    dev_opt_thresh: float,
    dev_cv_auc: float | None,
    fit_sec: float,
    predict_ms_per_1k: float,
    split_name: str,
    tier: str,
    preproc_label: str,
    params: dict[str, Any],
    seed: int,
    tracker: ExperimentTracker | None = None,
    exp_id: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    test_pos_rate = float(np.mean(y_test))
    m = compute_binary_metrics(
        y_test,
        probs_test,
        threshold=0.50,
        loan_amounts=disbursed_test,
        opt_threshold=dev_opt_thresh,
    )
    pr_lift = m["pr_auc"] / test_pos_rate if test_pos_rate > 0 else 0.0
    row = {
        "model": model_label,
        "feature_tier": tier,
        "split": split_name,
        "n_test": int(len(y_test)),
        "test_default_rate": round(test_pos_rate, 6),
        "dev_cv_roc_auc": round(float(dev_cv_auc), 6) if dev_cv_auc is not None else np.nan,
        "auc_shift_dev_to_test": round(float(m["roc_auc"] - dev_cv_auc), 6) if dev_cv_auc is not None else np.nan,
        "pr_auc_lift": round(pr_lift, 4),
        "fit_sec": round(float(fit_sec), 2),
        "predict_ms_per_1k": round(float(predict_ms_per_1k), 4),
        "dev_frozen_opt_thresh": round(float(dev_opt_thresh), 4),
        **m,
    }
    if tracker is not None and exp_id is not None:
        tracker.log_experiment(
            experiment_id=exp_id,
            track="Track_F_FinalTestEvaluation",
            model=model_label,
            feature_tier=tier,
            preprocessing=preproc_label,
            dataset_split=split_name,
            random_seed=seed,
            training_duration_sec=fit_sec,
            status="SUCCESS",
            metrics=m,
            best_parameters=params,
            notes=notes,
        )
    return row


def _build_rq3_row(
    tier: str,
    model_name: str,
    rand_metrics: dict[str, Any],
    rand_pos_rate: float,
    oot_lookup_f1: dict[str, dict[str, Any]],
    oot_lookup_f2: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    oot_ref = oot_lookup_f1[model_name] if tier == "F1" else oot_lookup_f2[model_name]
    oot_auc = float(oot_ref["oot_roc_auc"] if tier == "F1" else oot_ref["roc_auc"])
    oot_prl = float(oot_ref["oot_pr_auc_lift"] if tier == "F1" else oot_ref["pr_auc_lift"])
    oot_ece = float(oot_ref["oot_ece"] if tier == "F1" else oot_ref["ece"])
    oot_bri = float(oot_ref["oot_brier"] if tier == "F1" else oot_ref["brier_score"])
    rand_prl = float(rand_metrics["pr_auc"] / rand_pos_rate)
    return {
        "feature_tier": tier,
        "model": model_name,
        "rand_roc_auc": rand_metrics["roc_auc"],
        "oot_roc_auc": oot_auc,
        "optimistic_bias_auc": round(rand_metrics["roc_auc"] - oot_auc, 6),
        "rand_pr_auc": rand_metrics["pr_auc"],
        "rand_pr_lift": round(rand_prl, 4),
        "oot_pr_auc_lift": oot_prl,
        "pr_lift_bias": round(rand_prl - oot_prl, 4),
        "rand_ece": rand_metrics["ece"],
        "oot_ece": oot_ece,
        "ece_inflation_oot": round(oot_ece - rand_metrics["ece"], 6),
        "rand_brier": rand_metrics["brier_score"],
        "oot_brier": oot_bri,
    }


def _run_threshold_transfer_oot(
    y_test: np.ndarray,
    disbursed_test: np.ndarray,
    probs_map: dict[str, tuple[np.ndarray, dict[str, float]]],
) -> pd.DataFrame:
    """Apply strictly Dev-frozen thresholds onto Locked OOT Test predictions."""
    rows: list[dict[str, Any]] = []
    for m_label, (probs, thresh_dict) in probs_map.items():
        for regime, (t_val, dev_flag_rate) in thresh_dict.items():
            pred = (probs >= float(t_val)).astype(np.int64)
            tp = int(((y_test == 1) & (pred == 1)).sum())
            fp = int(((y_test == 0) & (pred == 1)).sum())
            fn = int(((y_test == 1) & (pred == 0)).sum())
            tn = int(((y_test == 0) & (pred == 0)).sum())
            n = len(y_test)
            flag_rate = (tp + fp) / n
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            rows.append(
                {
                    "model": m_label,
                    "regime": regime,
                    "dev_frozen_threshold": round(float(t_val), 5),
                    "dev_flag_rate": round(float(dev_flag_rate), 6),
                    "oot_flag_rate": round(flag_rate, 6),
                    "flag_rate_shift": round(flag_rate - float(dev_flag_rate), 6),
                    "approval_rate": round(1.0 - flag_rate, 6),
                    "accuracy": round((tp + tn) / n, 6),
                    "precision": round(prec, 6),
                    "recall": round(rec, 6),
                    "f1": round(f1, 6),
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                    "cost_3to1": round(compute_expected_cost_fixed(y_test, pred, 3.0, 1.0), 6),
                    "cost_5to1": round(compute_expected_cost_fixed(y_test, pred, 5.0, 1.0), 6),
                    "cost_10to1": round(compute_expected_cost_fixed(y_test, pred, 10.0, 1.0), 6),
                    "cost_amount_weighted": round(compute_expected_cost_amount_weighted(y_test, pred, disbursed_test), 6),
                }
            )
    return pd.DataFrame(rows)


def _build_decile_risk_table(y_test: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    """Create 10-tier risk banding decile table (D1 lowest risk -> D10 highest risk)."""
    order = np.argsort(probs, kind="mergesort")
    bins = np.array_split(order, 10)
    total_pos = float(np.sum(y_test))
    base_rate = float(np.mean(y_test))

    rows_desc: list[dict[str, Any]] = []
    cum_pos = 0
    for decile_idx in range(9, -1, -1):
        idx = bins[decile_idx]
        p_sub = probs[idx]
        y_sub = y_test[idx]
        pos_cnt = int(y_sub.sum())
        cum_pos += pos_cnt
        mean_pred = float(np.mean(p_sub))
        emp_rate = float(np.mean(y_sub))
        rows_desc.append(
            {
                "risk_decile": f"D{decile_idx + 1}",
                "n": int(len(idx)),
                "min_prob": round(float(np.min(p_sub)), 5),
                "max_prob": round(float(np.max(p_sub)), 5),
                "mean_pred_prob": round(mean_pred, 5),
                "actual_defaults": pos_cnt,
                "actual_default_rate": round(emp_rate, 5),
                "calibration_gap": round(emp_rate - mean_pred, 5),
                "lift_vs_portfolio": round(emp_rate / base_rate if base_rate > 0 else 0.0, 3),
                "cum_default_capture_pct": round(cum_pos / total_pos * 100.0 if total_pos > 0 else 0.0, 2),
            }
        )
    return pd.DataFrame(rows_desc[::-1])


def _run_oot_subgroup_audit(
    oot_test_df: pd.DataFrame,
    y_test: np.ndarray,
    disbursed_test: np.ndarray,
    probs_specs: list[tuple[str, np.ndarray, float]],
) -> pd.DataFrame:
    """Audit Locked OOT Test performance across Bureau, Employment, LTV, and Age bands."""
    eng_test = LTFSFeatureEngineer(feature_tier="F1").fit_transform(oot_test_df).reset_index(drop=True)
    age_vals = eng_test["age_at_disbursal"].to_numpy(dtype=np.float64)
    age_band = pd.cut(
        age_vals,
        bins=[0.0, 25.999, 35.999, 45.999, 120.0],
        labels=["Age_18_25", "Age_26_35", "Age_36_45", "Age_46_plus"],
    ).astype(str).to_numpy()
    bureau_slice = np.where(oot_test_df["PERFORM_CNS.SCORE"] >= 300, "Scored (>=300)", "Thin-file / Unscored (<300)")
    emp_slice = oot_test_df["Employment.Type"].fillna("MISSING").astype(str).to_numpy()
    ltv_q = pd.qcut(oot_test_df["ltv"], q=4, labels=["LTV_Q1_Lowest", "LTV_Q2", "LTV_Q3", "LTV_Q4_Highest"]).astype(str).to_numpy()

    slice_defs = [
        ("Bureau Regime", bureau_slice),
        ("Employment Type", emp_slice),
        ("LTV Quartile", ltv_q),
        ("Age Band", age_band),
    ]
    rows: list[dict[str, Any]] = []
    for dim_name, arr in slice_defs:
        for val in pd.unique(arr):
            mask = arr == val
            for m_label, p_arr, opt_t in probs_specs:
                m_sub = compute_binary_metrics(
                    y_test[mask],
                    p_arr[mask],
                    threshold=0.50,
                    loan_amounts=disbursed_test[mask],
                    opt_threshold=opt_t,
                )
                rows.append(
                    {
                        "slice_dimension": dim_name,
                        "slice_value": str(val),
                        "model": m_label,
                        "n": int(mask.sum()),
                        "default_rate": round(float(np.mean(y_test[mask])), 6),
                        "roc_auc": m_sub["roc_auc"],
                        "pr_auc": m_sub["pr_auc"],
                        "brier_score": m_sub["brier_score"],
                        "ece": m_sub["ece"],
                        "recall_at_opt": m_sub["recall_opt"],
                        "precision_at_opt": m_sub["precision_opt"],
                        "flag_rate_at_opt": round(float(np.mean(p_arr[mask] >= opt_t)), 6),
                        "expected_cost_5to1_bayes": m_sub["expected_cost_5to1_bayes"],
                    }
                )
    return pd.DataFrame(rows)


def _plot_final_04_decile_and_subgroups(
    fig_dir: Path,
    decile_df: pd.DataFrame,
    sub_df: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "axes.titlesize": 10.5, "axes.titleweight": "bold"})

    fig, axes = plt.subplots(1, 2, figsize=(15.4, 5.8))
    xd = np.arange(len(decile_df))
    w_d = 0.36
    axes[0].bar(xd - w_d / 2, decile_df["mean_pred_prob"] * 100.0, width=w_d, color="#3182ce", alpha=0.88, label="Predicted PD (%)")
    axes[0].bar(xd + w_d / 2, decile_df["actual_default_rate"] * 100.0, width=w_d, color="#dd6b20", alpha=0.90, label="Actual OOT FPD Rate (%)")
    ax_cum = axes[0].twinx()
    ax_cum.plot(xd, decile_df["cum_default_capture_pct"], color="#2f855a", marker="o", linewidth=2.0, label="Cum. Default Capture (%) [Top->Down]")
    ax_cum.grid(False)
    axes[0].set_ylim(0.0, 58.0)
    ax_cum.set_ylim(0.0, 128.0)
    axes[0].set_xticks(xd)
    axes[0].set_xticklabels(decile_df["risk_decile"])
    axes[0].set_ylabel("Default Probability / Rate (%)")
    ax_cum.set_ylabel("Cumulative Default Capture (%)", color="#2f855a")
    axes[0].set_title("A. 10-Decile Risk Calibration & Capture Gains (OOT LightGBM S2)")
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = ax_cum.get_legend_handles_labels()
    axes[0].legend(h0 + h1, l0 + l1, loc="upper left", fontsize=7.6)

    sub_lgb = sub_df[sub_df["model"] == "lightgbm_stage2"].copy()
    sub_lr = sub_df[sub_df["model"] == "logistic_regression_s1"].copy()
    merged_sub = pd.merge(
        sub_lgb[["slice_dimension", "slice_value", "n", "roc_auc"]],
        sub_lr[["slice_dimension", "slice_value", "roc_auc"]],
        on=["slice_dimension", "slice_value"],
        suffixes=("_lgb", "_lr"),
    )
    ys = np.arange(len(merged_sub))
    axes[1].barh(ys + 0.18, merged_sub["roc_auc_lgb"], height=0.35, color="#2b6cb0", alpha=0.90, label="LightGBM Stage-2")
    axes[1].barh(ys - 0.18, merged_sub["roc_auc_lr"], height=0.35, color="#c53030", alpha=0.82, label="LogisticReg (C=0.1)")
    for i, (_, r) in enumerate(merged_sub.iterrows()):
        axes[1].text(max(r["roc_auc_lgb"], r["roc_auc_lr"]) + 0.002, i, f"{r['roc_auc_lgb']:.4f} / {r['roc_auc_lr']:.4f}", va="center", fontsize=7.5)
    axes[1].set_yticks(ys)
    axes[1].set_yticklabels([f"{r['slice_value']} (n={r['n']:,})" for _, r in merged_sub.iterrows()], fontsize=8.0)
    axes[1].set_xlabel("Locked OOT Test ROC-AUC")
    axes[1].set_xlim(0.58, float(merged_sub["roc_auc_lgb"].max()) + 0.088)
    axes[1].set_title("B. Subgroup Slice Performance on Locked OOT Test")
    axes[1].legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_dir / "final_04_decile_lift_and_subgroups.png", bbox_inches="tight")
    plt.close(fig)


def _generate_all_final_figures(
    fig_dir: Path,
    y_oot: np.ndarray,
    oot_probs: dict[str, np.ndarray],
    oot_comp_df: pd.DataFrame,
    rq2_df: pd.DataFrame,
    rq3_df: pd.DataFrame,
    imb_oot_df: pd.DataFrame,
    decile_df: pd.DataFrame,
    sub_df: pd.DataFrame,
    xai_stab_df: pd.DataFrame,
    oot_cases: list[dict[str, Any]],
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "axes.titlesize": 10.5, "axes.titleweight": "bold"})

    oot_pos_rate = float(np.mean(y_oot))

    # 1. final_01_oot_roc_pr_curves.png
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.8))
    curve_specs = [
        ("lightgbm", "LightGBM (Tuned)", "#2b6cb0", "-"),
        ("xgboost", "XGBoost (Tuned)", "#dd6b20", "-."),
        ("random_forest", "RandomForest", "#805ad5", ":"),
        ("logistic_regression", "LogisticReg (C=0.1)", "#c53030", "-"),
        ("mlp", "MLP (128,64)", "#2f855a", "--"),
        ("decision_tree", "DecisionTree", "#718096", "-."),
    ]
    auc_lookup = dict(zip(oot_comp_df["model"], oot_comp_df["roc_auc"]))
    pr_lookup = dict(zip(oot_comp_df["model"], oot_comp_df["pr_auc"]))

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Random Chance (AUC=0.5000)")
    axes[1].axhline(oot_pos_rate, color="black", linestyle="--", linewidth=1.0, label=f"OOT Prior Baseline ({oot_pos_rate:.4f})")

    for key, lbl, col, ls in curve_specs:
        p = oot_probs[key]
        fpr, tpr, _ = roc_curve(y_oot, p)
        prec, rec, _ = precision_recall_curve(y_oot, p)
        axes[0].plot(fpr, tpr, linestyle=ls, color=col, linewidth=1.9, label=f"{lbl} (AUC={auc_lookup[key]:.4f})")
        axes[1].plot(rec, prec, linestyle=ls, color=col, linewidth=1.9, label=f"{lbl} (PR-AUC={pr_lookup[key]:.4f})")

    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].set_title("A. Locked OOT Test ROC Curves (N = 50,653)")
    axes[0].legend(loc="lower right", fontsize=8)

    axes[1].set_xlabel("Recall (Default Capture)")
    axes[1].set_ylabel("Precision (Default Rate Among Flagged)")
    axes[1].set_title("B. Locked OOT Test Precision-Recall Curves (Prior = 25.59%)")
    axes[1].set_ylim(0.18, 0.78)
    axes[1].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_dir / "final_01_oot_roc_pr_curves.png", bbox_inches="tight")
    plt.close(fig)

    # 2. final_02_oot_vs_random_rq3.png
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.6))
    sub_rq3 = rq3_df[rq3_df["feature_tier"] == "F2_safe"].copy()
    x3 = np.arange(len(sub_rq3))
    w3 = 0.35
    axes[0].bar(x3 - w3 / 2, sub_rq3["rand_roc_auc"], width=w3, color="#3182ce", alpha=0.88, label="Random 80/20 Hold-out")
    axes[0].bar(x3 + w3 / 2, sub_rq3["oot_roc_auc"], width=w3, color="#dd6b20", alpha=0.90, label="Locked OOT Test (>= Oct 23)")
    for i, (_, r) in enumerate(sub_rq3.iterrows()):
        axes[0].text(
            x3[i],
            max(r["rand_roc_auc"], r["oot_roc_auc"]) + 0.0012,
            f"Bias\n+{r['optimistic_bias_auc']:.4f}",
            ha="center",
            va="bottom",
            fontsize=7.3,
            color="#9b2c2c",
            fontweight="bold",
        )
    axes[0].set_xticks(x3)
    axes[0].set_xticklabels([m.replace("_", "\n") for m in sub_rq3["model"]], fontsize=8.0)
    axes[0].set_ylabel("ROC-AUC")
    ymin3 = float(min(sub_rq3["rand_roc_auc"].min(), sub_rq3["oot_roc_auc"].min())) - 0.015
    ymax3 = float(max(sub_rq3["rand_roc_auc"].max(), sub_rq3["oot_roc_auc"].max())) + 0.018
    axes[0].set_ylim(ymin3, ymax3)
    axes[0].set_title("A. Optimistic Bias: Stratified Random vs. OOT (F2_safe)")
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Perfect Calibration")
    bins = np.linspace(0.0, 1.0, 11)
    cal_curves = [
        ("lightgbm", "OOT LightGBM (Tuned)", "#dd6b20", "-o"),
        ("xgboost", "OOT XGBoost (Tuned)", "#3182ce", "-s"),
        ("random_forest", "OOT RandomForest", "#2f855a", "-^"),
        ("logistic_regression", "OOT LogisticRegression (C=0.1)", "#c53030", "--d"),
    ]
    ece_map = dict(zip(oot_comp_df["model"], oot_comp_df["ece"]))
    for key, lbl, col, fmt in cal_curves:
        p_c = oot_probs[key]
        xs, ys = [], []
        for bi in range(10):
            mask = (p_c >= bins[bi]) & (p_c <= bins[bi + 1] if bi == 9 else p_c < bins[bi + 1])
            if mask.sum() >= 100:
                xs.append(float(p_c[mask].mean()))
                ys.append(float(y_oot[mask].mean()))
        axes[1].plot(xs, ys, fmt, color=col, linewidth=1.8, markersize=4.5, label=f"{lbl} (ECE={ece_map[key]:.3f})")
    axes[1].set_xlabel("Mean Predicted Probability (Dev-Fitted)")
    axes[1].set_ylabel("Observed Default Rate on Locked OOT Test")
    axes[1].set_title("B. Prior-Shift Calibration Curve on OOT Test (20.6% -> 25.6%)")
    axes[1].set_xlim(0.0, 0.72)
    axes[1].set_ylim(0.0, 0.76)
    axes[1].legend(loc="upper left", fontsize=7.8)

    fig.tight_layout()
    fig.savefig(fig_dir / "final_02_oot_vs_random_rq3.png", bbox_inches="tight")
    plt.close(fig)

    # 3. final_03_rq2_ablation_oot_and_imbalance.png
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.4))
    tier_order = ["F0", "F1", "F2_safe", "F2_plus_emp"]
    for m_name, grp in rq2_df.groupby("model"):
        gs = grp.set_index("feature_tier").reindex(tier_order)
        axes[0].plot(tier_order, gs["oot_roc_auc"], marker="o", linewidth=2.1, label=f"{m_name} (OOT Test)")
    axes[0].set_xlabel("Feature Tier")
    axes[0].set_ylabel("Locked OOT Test ROC-AUC")
    axes[0].set_title("A. Feature Tier Ablation on Locked OOT Test (F0 -> F2_plus_emp)")
    axes[0].legend(fontsize=8.2)

    x_i = np.arange(len(imb_oot_df))
    ax_r = axes[1].twinx()
    b1 = axes[1].bar(x_i - 0.18, imb_oot_df["roc_auc"], width=0.35, color="#2b6cb0", alpha=0.88, label="OOT ROC-AUC (Left)")
    b2 = ax_r.bar(x_i + 0.18, imb_oot_df["ece"], width=0.35, color="#c53030", alpha=0.82, label="OOT Uncal ECE (Right)")
    axes[1].set_xticks(x_i)
    axes[1].set_xticklabels([s.replace("_", "\n") for s in imb_oot_df["imbalance_strategy"]], fontsize=8.0)
    axes[1].set_ylabel("Locked OOT Test ROC-AUC", color="#2b6cb0")
    ax_r.set_ylabel("Expected Calibration Error (ECE)", color="#c53030")
    ax_r.grid(False)
    axes[1].set_ylim(float(imb_oot_df["roc_auc"].min()) - 0.015, float(imb_oot_df["roc_auc"].max()) + 0.008)
    ax_r.set_ylim(0.0, float(imb_oot_df["ece"].max()) * 1.18)
    for bar, v in zip(b1, imb_oot_df["roc_auc"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.0008, f"{v:.4f}", ha="center", va="bottom", fontsize=7.2, color="#2b6cb0")
    axes[1].set_title("B. Class Imbalance Strategies on OOT Test (LightGBM Stage-2)")
    axes[1].legend(handles=[b1, b2], labels=["OOT ROC-AUC (Left)", "OOT ECE (Right)"], loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_dir / "final_03_rq2_ablation_oot_and_imbalance.png", bbox_inches="tight")
    plt.close(fig)

    # 4. final_04_decile_lift_and_subgroups.png
    _plot_final_04_decile_and_subgroups(fig_dir, decile_df, sub_df)

    # 5. final_05_test_xai_stability_and_errors.png
    fig = plt.figure(figsize=(15.5, 9.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.0, 1.0])
    ax_sc = fig.add_subplot(gs[:, 0])
    ax_sc.scatter(xai_stab_df["dev_mean_abs_shap"], xai_stab_df["oot_mean_abs_shap"], alpha=0.65, color="#2b6cb0", s=28)
    max_lim = float(max(xai_stab_df["dev_mean_abs_shap"].max(), xai_stab_df["oot_mean_abs_shap"].max())) * 1.24
    ax_sc.plot([0, max_lim], [0, max_lim], "r--", linewidth=1.0, label="y = x")
    top8 = xai_stab_df.head(8)
    offsets = [(4, 3), (4, -9), (4, 3), (4, -8), (4, 4), (4, -8), (4, 4), (4, -8)]
    for (_, r), (dx, dy) in zip(top8.iterrows(), offsets):
        ax_sc.annotate(r["feature"], (r["dev_mean_abs_shap"], r["oot_mean_abs_shap"]), textcoords="offset points", xytext=(dx, dy), fontsize=7.2)
    rho_val = spearmanr(xai_stab_df["dev_mean_abs_shap"], xai_stab_df["oot_mean_abs_shap"]).statistic
    ax_sc.set_xlim(0.0, max_lim)
    ax_sc.set_ylim(0.0, max_lim)
    ax_sc.set_xlabel("Dev Fold-4 Mean(|SHAP|)")
    ax_sc.set_ylabel("Locked OOT Test Mean(|SHAP|)")
    ax_sc.set_title(f"A. TreeSHAP Attribution Stability\n(Dev vs. OOT Test, Spearman rho={rho_val:.4f})")
    ax_sc.legend(loc="lower right", fontsize=8)

    case_axes = [
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]
    for ax_idx, (ax_c, case) in enumerate(zip(case_axes, oot_cases)):
        drivers = list(reversed(case["top_drivers"][:7]))
        lbls = [f"{d['feature']}={d['feature_value']:.3g}" for d in drivers]
        vals = [d["shap_value"] for d in drivers]
        cols = ["#c53030" if v > 0 else "#2b6cb0" for v in vals]
        yc = np.arange(len(lbls))
        ax_c.barh(yc, vals, color=cols, alpha=0.88, edgecolor="black", linewidth=0.4)
        ax_c.axvline(0.0, color="black", linewidth=0.8)
        m_abs = max(0.05, max(abs(v) for v in vals))
        for i, v in enumerate(vals):
            ax_c.text(v + (0.03 * m_abs * (1 if v >= 0 else -1)), i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=7.0)
        ax_c.set_yticks(yc)
        ax_c.set_yticklabels(lbls, fontsize=7.2)
        ax_c.set_xlim(-m_abs * 1.32, m_abs * 1.32)
        if ax_idx >= 2:
            ax_c.set_xlabel("SHAP Value (Log-Odds)", fontsize=8.5)
        ax_c.set_title(f"OOT {case['case_type']} (y={case['actual_y']}, p={case['pred_prob']:.3f})", fontsize=9.5)

    fig.tight_layout()
    fig.savefig(fig_dir / "final_05_test_xai_stability_and_errors.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    root = get_project_root()
    base_cfg = load_config()
    model_cfg = load_model_config()
    seed = int(model_cfg["benchmark"]["random_seed"])
    n_threads = int(model_cfg["benchmark"]["n_threads"])
    default_tier = str(model_cfg["benchmark"]["default_feature_tier"])

    results_dir = root / "experiments" / "results"
    tables_dir = root / "reports" / "tables"
    fig_dir = root / "reports" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tracker = ExperimentTracker(results_dir)
    best_params_doc = json.loads((results_dir / "m5_opt_best_params.json").read_text(encoding="utf-8"))
    best_params = best_params_doc["models"]
    baselines_cfg = model_cfg["baselines"]

    # Load Dev-frozen performance & threshold metadata strictly from pre-test results
    m4_base_df = pd.read_csv(results_dir / "m4_baseline_comparison.csv").set_index("model")
    m4_tune_df = pd.read_csv(results_dir / "m4_tuning_comparison.csv").set_index("model")
    m5_opt_df = pd.read_csv(results_dir / "m5_opt_tuning_results.csv").set_index("model")
    m5_thresh_df = pd.read_csv(results_dir / "m5_threshold_comparison.csv")

    dev_cv_auc_map = {
        "decision_tree": float(m4_base_df.loc["decision_tree", "cv_roc_auc_mean"]),
        "logistic_regression": float(m4_tune_df.loc["logistic_regression", "tuned_cv_auc"]),
        "mlp": float(m4_base_df.loc["mlp", "cv_roc_auc_mean"]),
        "random_forest": float(m4_base_df.loc["random_forest", "cv_roc_auc_mean"]),
        "xgboost": float(m5_opt_df.loc["xgboost", "stage2_cv_auc"]),
        "lightgbm": float(m5_opt_df.loc["lightgbm", "stage2_cv_auc"]),
    }
    dev_f1_thresh_map = {
        "decision_tree": float(m4_base_df.loc["decision_tree", "f1_opt_thresh"]),
        "logistic_regression": float(m5_thresh_df.loc[(m5_thresh_df["model"] == "logistic_regression_s1") & (m5_thresh_df["regime"] == "F1_Optimal"), "nested_thresh_mean"].iloc[0]),
        "mlp": float(m4_base_df.loc["mlp", "f1_opt_thresh"]),
        "random_forest": float(m4_base_df.loc["random_forest", "f1_opt_thresh"]),
        "xgboost": float(m5_thresh_df.loc[(m5_thresh_df["model"] == "xgboost_stage2_natural") & (m5_thresh_df["regime"] == "F1_Optimal"), "nested_thresh_mean"].iloc[0]),
        "lightgbm": float(m5_thresh_df.loc[(m5_thresh_df["model"] == "lightgbm_stage2_natural") & (m5_thresh_df["regime"] == "F1_Optimal"), "nested_thresh_mean"].iloc[0]),
    }

    # -------------------------------------------------------------------------
    # TRACK 1: SINGLE-USE LOCKED OOT TEST EVALUATION (N=50,653, 6 CORE MODELS)
    # -------------------------------------------------------------------------
    logger.info("=== TRACK 1: Single-Use Locked OOT Test Evaluation (N=50,653) ===")
    _, _, oot_test_df, oot_dev_df, _ = load_split_frames("oot", base_cfg)
    assert len(oot_dev_df) == 182501 and len(oot_test_df) == 50653
    assert len(set(oot_dev_df["UniqueID"]) & set(oot_test_df["UniqueID"])) == 0
    assert parse_disbursal_dates(oot_dev_df["DisbursalDate"]).max() < parse_disbursal_dates(oot_test_df["DisbursalDate"]).min()

    y_oot_dev = oot_dev_df["loan_default"].to_numpy(dtype=np.int64)
    y_oot_test = oot_test_df["loan_default"].to_numpy(dtype=np.int64)
    disb_oot_test = oot_test_df["disbursed_amount"].to_numpy(dtype=np.float64)

    winsor_q = tuple(base_cfg["preprocessing"]["winsor_quantiles"])
    te_smooth = float(base_cfg["preprocessing"]["target_encoding_smoothing"])
    te_folds = int(base_cfg["preprocessing"]["target_encoding_cv_folds"])

    oot_rows: list[dict[str, Any]] = []
    oot_probs: dict[str, np.ndarray] = {}

    # 1A. Linear/MLP branch on F2_safe
    lin_pipe = build_preprocessing_pipeline(default_tier, "linear", True, winsor_q, te_smooth, te_folds, seed)
    X_dev_lin = lin_pipe.fit_transform(oot_dev_df, y_oot_dev).astype(np.float32)
    X_test_lin = lin_pipe.transform(oot_test_df).astype(np.float32)

    lin_specs = [
        ("logistic_regression", best_params["logistic_regression"], "Frozen Tuned LogisticRegression (C=0.1)"),
        ("mlp", baselines_cfg["mlp"], "Frozen Tabular MLP (128,64) baseline"),
    ]
    for m_name, params, note in lin_specs:
        t0 = time.perf_counter()
        clf = build_model(m_name, params, seed, n_threads).fit(X_dev_lin, y_oot_dev)
        fit_sec = time.perf_counter() - t0
        tp0 = time.perf_counter()
        probs = clf.predict_proba(X_test_lin)[:, 1]
        pred_ms = (time.perf_counter() - tp0) * 1000.0 / (len(y_oot_test) / 1000.0)
        oot_probs[m_name] = probs
        oot_rows.append(
            _evaluate_frozen_predictions(
                m_name, y_oot_test, probs, disb_oot_test, dev_f1_thresh_map[m_name], dev_cv_auc_map[m_name],
                fit_sec, pred_ms, "oot_locked_test", default_tier, "linear_drop_redundant=True",
                params, seed, tracker, f"M6_FINAL_OOT_{m_name}", note,
            )
        )
        del clf
    del X_dev_lin, X_test_lin, lin_pipe
    gc.collect()

    # 1B. Tree branch on F2_safe
    tree_pipe = build_preprocessing_pipeline(default_tier, "tree", False, winsor_q, te_smooth, te_folds, seed)
    X_dev_tree = tree_pipe.fit_transform(oot_dev_df, y_oot_dev).astype(np.float32)
    X_test_tree = tree_pipe.transform(oot_test_df).astype(np.float32)
    tree_feat_names = [canonical_feature_name(c) for c in tree_pipe.named_steps["preprocessor"].get_feature_names_out()]

    tree_specs = [
        ("decision_tree", baselines_cfg["decision_tree"], "Controlled Decision Tree baseline"),
        ("random_forest", best_params["random_forest_baseline"], "Retained Baseline RandomForest"),
        ("xgboost", best_params["xgboost"], "Frozen Stage-2 XGBoost"),
        ("lightgbm", best_params["lightgbm"], "Frozen Stage-2 LightGBM (Natural Prior Champion)"),
    ]
    lgb_full_dev_clf = None
    for m_name, params, note in tree_specs:
        t0 = time.perf_counter()
        clf = build_model(m_name, params, seed, n_threads).fit(X_dev_tree, y_oot_dev)
        fit_sec = time.perf_counter() - t0
        tp0 = time.perf_counter()
        probs = clf.predict_proba(X_test_tree)[:, 1]
        pred_ms = (time.perf_counter() - tp0) * 1000.0 / (len(y_oot_test) / 1000.0)
        oot_probs[m_name] = probs
        oot_rows.append(
            _evaluate_frozen_predictions(
                m_name, y_oot_test, probs, disb_oot_test, dev_f1_thresh_map[m_name], dev_cv_auc_map[m_name],
                fit_sec, pred_ms, "oot_locked_test", default_tier, "tree_drop_redundant=False",
                params, seed, tracker, f"M6_FINAL_OOT_{m_name}", note,
            )
        )
        if m_name == "lightgbm":
            lgb_full_dev_clf = clf
        else:
            del clf

    # 1C. Immediate Locked-Test TreeSHAP & 4 Case Studies (while X_test_tree & lgb_full_dev_clf are in RAM)
    logger.info("=== Computing Locked OOT Test Exact TreeSHAP Stability & 4 Test Case Studies ===")
    sss_te = StratifiedShuffleSplit(n_splits=1, train_size=10000, random_state=seed)
    te_exp_idx, _ = next(sss_te.split(X_test_tree, y_oot_test))
    X_te_exp = X_test_tree[te_exp_idx]
    lgb_te_contrib = lgb_full_dev_clf.predict(X_te_exp, pred_contrib=True)
    lgb_te_margin = lgb_full_dev_clf.predict(X_te_exp, raw_score=True)
    te_shap_err = float(np.max(np.abs(lgb_te_contrib.sum(axis=1) - lgb_te_margin)))
    assert te_shap_err < 1e-5, f"OOT Test TreeSHAP additivity failed: {te_shap_err}"
    lgb_te_shap = lgb_te_contrib[:, :-1]
    oot_mean_abs = np.mean(np.abs(lgb_te_shap), axis=0)

    dev_xai_df = pd.read_csv(results_dir / "m5_explainability_feature_ranking.csv")
    dev_shap_map = dict(zip(dev_xai_df["feature"], dev_xai_df["lgb_mean_abs_shap"]))
    xai_stab_rows = []
    for j, fname in enumerate(tree_feat_names):
        xai_stab_rows.append(
            {
                "feature": fname,
                "domain_group": assign_domain_group(fname),
                "dev_mean_abs_shap": round(float(dev_shap_map.get(fname, 0.0)), 6),
                "oot_mean_abs_shap": round(float(oot_mean_abs[j]), 6),
            }
        )
    xai_stab_df = pd.DataFrame(xai_stab_rows).sort_values("oot_mean_abs_shap", ascending=False).reset_index(drop=True)
    stab_rho = float(spearmanr(xai_stab_df["dev_mean_abs_shap"], xai_stab_df["oot_mean_abs_shap"]).statistic)
    stab_jacc = _top_k_jaccard(
        xai_stab_df.set_index("feature")["dev_mean_abs_shap"],
        xai_stab_df.set_index("feature")["oot_mean_abs_shap"],
        15,
    )
    oot_group_shares = (
        xai_stab_df.groupby("domain_group", as_index=False)[["dev_mean_abs_shap", "oot_mean_abs_shap"]].sum()
    )
    oot_group_shares["dev_share_pct"] = (oot_group_shares["dev_mean_abs_shap"] / oot_group_shares["dev_mean_abs_shap"].sum() * 100.0).round(2)
    oot_group_shares["oot_share_pct"] = (oot_group_shares["oot_mean_abs_shap"] / oot_group_shares["oot_mean_abs_shap"].sum() * 100.0).round(2)

    # 4 OOT Test Case Studies at Dev-frozen t=0.205
    case_pos_map = _select_case_study_indices(y_oot_test, oot_probs["lightgbm"], thresh=dev_f1_thresh_map["lightgbm"])
    case_rel = [case_pos_map[k] for k in ("TP", "TN", "FN", "FP")]
    case_contribs = lgb_full_dev_clf.predict(X_test_tree[case_rel], pred_contrib=True)
    raw_audit_cols = [
        "UniqueID", "DisbursalDate", "disbursed_amount", "asset_cost", "ltv", "PERFORM_CNS.SCORE",
        "Employment.Type", "PRI.NO.OF.ACCTS", "PRI.ACTIVE.ACCTS", "PRI.OVERDUE.ACCTS",
        "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS", "NO.OF_INQUIRIES", "State_ID", "branch_id",
        "supplier_id", "Current_pincode_ID",
    ]
    oot_cases: list[dict[str, Any]] = []
    for idx_c, (c_type, rel_pos) in enumerate(zip(("TP", "TN", "FN", "FP"), case_rel)):
        phi = case_contribs[idx_c, :-1]
        base_val = float(case_contribs[idx_c, -1])
        top_j = np.argsort(-np.abs(phi))[:10]
        raw_row = oot_test_df.iloc[rel_pos]
        oot_cases.append(
            {
                "case_type": c_type,
                "oot_row_pos": int(rel_pos),
                "actual_y": int(y_oot_test[rel_pos]),
                "pred_prob": round(float(oot_probs["lightgbm"][rel_pos]), 6),
                "base_prob": round(float(expit(base_val)), 6),
                "raw_attributes": {c: (str(raw_row[c]) if isinstance(raw_row[c], str) or pd.isna(raw_row[c]) else float(raw_row[c])) for c in raw_audit_cols},
                "top_drivers": [
                    {
                        "feature": tree_feat_names[j],
                        "feature_value": round(float(X_test_tree[rel_pos, j]), 4),
                        "shap_value": round(float(phi[j]), 6),
                    }
                    for j in top_j
                ],
            }
        )
    (tables_dir / "m6_final_test_cases.json").write_text(json.dumps(oot_cases, indent=2, ensure_ascii=False), encoding="utf-8")

    # 1D. Immediate OOT Imbalance Check on F2_safe LightGBM (reuse X_dev_tree / X_test_tree)
    logger.info("=== Checking RQ4 Imbalance Strategies on Full OOT Dev -> Locked OOT Test ===")
    m5_imb_oof = dict(np.load(results_dir / "m5_imbalance_oof_predictions.npz"))
    dev_imb_y = m5_imb_oof["y_dev"]
    dev_imb_amt = m5_imb_oof["disbursed_amount"]
    imb_oot_rows: list[dict[str, Any]] = []
    lgb_fit_sec = next(r["fit_sec"] for r in oot_rows if r["model"] == "lightgbm")
    for strat in ("natural", "mild_weight_2.0", "class_weight", "rus_0.5", "smote_0.5", "smote_1.0"):
        # Freeze per-strategy F1 & empirical 5:1 cost thresholds strictly on Dev OOF (zero test leakage)
        strat_dev_p = m5_imb_oof[f"{strat}__uncalibrated"]
        dev_t_info = find_optimal_thresholds(dev_imb_y, strat_dev_p, loan_amounts=dev_imb_amt)
        strat_f1_t = dev_f1_thresh_map["lightgbm"] if strat == "natural" else float(dev_t_info["f1_opt_thresh"])
        strat_cost5_t = float(dev_t_info["empirical_cost_5to1_thresh"])
        strat_amt_t = float(dev_t_info["empirical_amount_weighted_thresh"])

        p_cfg = dict(best_params["lightgbm"])
        if strat == "class_weight":
            p_cfg["class_weight"] = "balanced"
        elif strat == "mild_weight_2.0":
            p_cfg["scale_pos_weight"] = 2.0
        elif strat == "natural":
            # Already evaluated cleanly in Track 1
            m_nat = compute_binary_metrics(
                y_oot_test,
                oot_probs["lightgbm"],
                0.50,
                disb_oot_test,
                opt_threshold=strat_f1_t,
                emp_cost_5to1_threshold=strat_cost5_t,
                emp_amount_weighted_threshold=strat_amt_t,
            )
            imb_oot_rows.append(
                {
                    "imbalance_strategy": strat,
                    "n_train_resampled": int(len(y_oot_dev)),
                    "fit_sec": lgb_fit_sec,
                    "dev_frozen_cost5_thresh": round(strat_cost5_t, 4),
                    **m_nat,
                }
            )
            continue

        t0 = time.perf_counter()
        X_res, y_res = _apply_fold_resampler(X_dev_tree, y_oot_dev, strat, seed, n_threads)
        clf_i = build_model("lightgbm", p_cfg, seed, n_threads).fit(X_res, y_res)
        dur_i = time.perf_counter() - t0
        p_i = clf_i.predict_proba(X_test_tree)[:, 1]
        m_i = compute_binary_metrics(
            y_oot_test,
            p_i,
            0.50,
            disb_oot_test,
            opt_threshold=strat_f1_t,
            emp_cost_5to1_threshold=strat_cost5_t,
            emp_amount_weighted_threshold=strat_amt_t,
        )
        imb_oot_rows.append(
            {
                "imbalance_strategy": strat,
                "n_train_resampled": int(len(y_res)),
                "fit_sec": round(dur_i, 2),
                "dev_frozen_cost5_thresh": round(strat_cost5_t, 4),
                **m_i,
            }
        )
        del X_res, y_res, clf_i
        gc.collect()
    imb_oot_df = pd.DataFrame(imb_oot_rows)
    imb_oot_df.to_csv(tables_dir / "m6_final_imbalance_oot.csv", index=False)

    del X_dev_tree, X_test_tree, tree_pipe, lgb_full_dev_clf
    gc.collect()

    # Save raw OOT test predictions
    np.savez_compressed(results_dir / "m6_final_oot_predictions.npz", y_oot_test=y_oot_test, disbursed_amount=disb_oot_test, **oot_probs)

    oot_comp_df = pd.DataFrame(oot_rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    oot_comp_df.to_csv(tables_dir / "m6_final_oot_comparison.csv", index=False)

    # -------------------------------------------------------------------------
    # TRACK 4A: FROZEN THRESHOLD TRANSFER, DECILE TABLE & SUBGROUP AUDIT
    # -------------------------------------------------------------------------
    logger.info("=== Evaluating Frozen Threshold Transfer, Decile Table, & OOT Subgroup Audit ===")
    thresh_lookup: dict[str, dict[str, tuple[float, float]]] = {}
    for m_key, m_col_lbl in [
        ("lightgbm", "lightgbm_stage2_natural"),
        ("xgboost", "xgboost_stage2_natural"),
    ]:
        sub_t = m5_thresh_df[m5_thresh_df["model"] == m_col_lbl].set_index("regime")
        thresh_lookup[m_key] = {
            reg: (float(sub_t.loc[reg, "nested_thresh_mean"]), float(sub_t.loc[reg, "flag_rate"]))
            for reg in (
                "Default_0.50",
                "Capacity_Top15pct_Queue",
                "Capacity_Top20pct_Queue",
                "Bayes_Cost_3to1",
                "F1_Optimal",
                "Bayes_Cost_5to1",
                "Bayes_Cost_10to1",
            )
        }
    thresh_transfer_df = _run_threshold_transfer_oot(
        y_oot_test,
        disb_oot_test,
        {k: (oot_probs[k], thresh_lookup[k]) for k in ("lightgbm", "xgboost")},
    )
    thresh_transfer_df.to_csv(tables_dir / "m6_final_threshold_transfer.csv", index=False)

    decile_df = _build_decile_risk_table(y_oot_test, oot_probs["lightgbm"])
    decile_df.to_csv(tables_dir / "m6_final_decile_table.csv", index=False)

    sub_df = _run_oot_subgroup_audit(
        oot_test_df,
        y_oot_test,
        disb_oot_test,
        [
            ("lightgbm_stage2", oot_probs["lightgbm"], dev_f1_thresh_map["lightgbm"]),
            ("xgboost_stage2", oot_probs["xgboost"], dev_f1_thresh_map["xgboost"]),
            ("logistic_regression_s1", oot_probs["logistic_regression"], dev_f1_thresh_map["logistic_regression"]),
        ],
    )
    sub_df.to_csv(tables_dir / "m6_final_subgroup_audit.csv", index=False)

    # Save XAI Stability + Cold-Start JSON
    oot_cold_start = {}
    for c in ("branch_id", "supplier_id", "Current_pincode_ID", "Employee_code_ID"):
        dev_set = set(oot_dev_df[c].astype(str))
        te_vals = oot_test_df[c].astype(str)
        oot_cold_start[c] = round(float((~te_vals.isin(dev_set)).mean() * 100.0), 3)

    xai_stab_doc = {
        "n_oot_shap_samples": int(len(X_te_exp)),
        "lgb_oot_treeshap_max_additivity_error": te_shap_err,
        "dev_vs_oot_spearman_rho": round(stab_rho, 4),
        "dev_vs_oot_top15_jaccard": round(stab_jacc, 4),
        "domain_group_shares": oot_group_shares.sort_values("oot_share_pct", ascending=False).to_dict(orient="records"),
        "oot_entity_cold_start_rate_pct": oot_cold_start,
        "top20_stability_features": xai_stab_df.head(20).to_dict(orient="records"),
    }
    (tables_dir / "m6_final_test_xai_stability.json").write_text(json.dumps(xai_stab_doc, indent=2, ensure_ascii=False), encoding="utf-8")

    # -------------------------------------------------------------------------
    # TRACK 2: RQ2 FINAL FEATURE ABLATION ON LOCKED OOT TEST (F0 -> F2_plus_emp)
    # -------------------------------------------------------------------------
    logger.info("=== TRACK 2: RQ2 Feature Ablation on Locked OOT Test (F0, F1, F2_safe, F2_plus_emp) ===")
    m4_abl_df = pd.read_csv(results_dir / "m4_feature_ablation.csv")
    dev_abl_cv = m4_abl_df[m4_abl_df["split_protocol"] == "oot_dev_5fold_cv"].set_index(["model", "feature_tier"])

    rq2_rows: list[dict[str, Any]] = []
    # Reuse F2_safe directly from Track 1 without redundant refit
    for m_name in ("logistic_regression", "lightgbm", "xgboost"):
        r_f2 = next(r for r in oot_rows if r["model"] == m_name)
        rq2_rows.append(
            {
                "model": m_name,
                "feature_tier": "F2_safe",
                "dev_cv_roc_auc": float(dev_abl_cv.loc[(m_name, "F2_safe"), "roc_auc"]),
                "oot_roc_auc": r_f2["roc_auc"],
                "oot_pr_auc": r_f2["pr_auc"],
                "oot_pr_auc_lift": r_f2["pr_auc_lift"],
                "oot_brier": r_f2["brier_score"],
                "oot_ece": r_f2["ece"],
                "fit_sec": r_f2["fit_sec"],
            }
        )

    for tier in ("F0", "F1", "F2_plus_emp"):
        # LR
        lp = build_preprocessing_pipeline(tier, "linear", True, winsor_q, te_smooth, te_folds, seed)
        X_tr_l = lp.fit_transform(oot_dev_df, y_oot_dev).astype(np.float32)
        X_te_l = lp.transform(oot_test_df).astype(np.float32)
        t0 = time.perf_counter()
        lr_c = build_model("logistic_regression", best_params["logistic_regression"], seed, n_threads).fit(X_tr_l, y_oot_dev)
        dur_l = time.perf_counter() - t0
        p_l = lr_c.predict_proba(X_te_l)[:, 1]
        m_l = compute_binary_metrics(y_oot_test, p_l, 0.50, disb_oot_test, opt_threshold=dev_f1_thresh_map["logistic_regression"])
        rq2_rows.append(
            {
                "model": "logistic_regression",
                "feature_tier": tier,
                "dev_cv_roc_auc": float(dev_abl_cv.loc[("logistic_regression", tier), "roc_auc"]),
                "oot_roc_auc": m_l["roc_auc"],
                "oot_pr_auc": m_l["pr_auc"],
                "oot_pr_auc_lift": round(m_l["pr_auc"] / float(np.mean(y_oot_test)), 4),
                "oot_brier": m_l["brier_score"],
                "oot_ece": m_l["ece"],
                "fit_sec": round(dur_l, 2),
            }
        )
        del X_tr_l, X_te_l, lp, lr_c

        # LightGBM & XGBoost
        tp = build_preprocessing_pipeline(tier, "tree", False, winsor_q, te_smooth, te_folds, seed)
        X_tr_t = tp.fit_transform(oot_dev_df, y_oot_dev).astype(np.float32)
        X_te_t = tp.transform(oot_test_df).astype(np.float32)
        for m_name in ("lightgbm", "xgboost"):
            t0 = time.perf_counter()
            clf_t = build_model(m_name, best_params[m_name], seed, n_threads).fit(X_tr_t, y_oot_dev)
            dur_t = time.perf_counter() - t0
            p_t = clf_t.predict_proba(X_te_t)[:, 1]
            m_t = compute_binary_metrics(y_oot_test, p_t, 0.50, disb_oot_test, opt_threshold=dev_f1_thresh_map[m_name])
            rq2_rows.append(
                {
                    "model": m_name,
                    "feature_tier": tier,
                    "dev_cv_roc_auc": float(dev_abl_cv.loc[(m_name, tier), "roc_auc"]),
                    "oot_roc_auc": m_t["roc_auc"],
                    "oot_pr_auc": m_t["pr_auc"],
                    "oot_pr_auc_lift": round(m_t["pr_auc"] / float(np.mean(y_oot_test)), 4),
                    "oot_brier": m_t["brier_score"],
                    "oot_ece": m_t["ece"],
                    "fit_sec": round(dur_t, 2),
                }
            )
            del clf_t
        del X_tr_t, X_te_t, tp
        gc.collect()

    rq2_df = pd.DataFrame(rq2_rows)
    rq2_df.to_csv(tables_dir / "m6_rq2_ablation_oot.csv", index=False)

    # Clear OOT dev/test frames before loading Random Split frames
    del oot_dev_df, oot_test_df
    gc.collect()

    # -------------------------------------------------------------------------
    # TRACK 3: RQ3 DUAL VALIDATION — OOT SPLIT vs. STRATIFIED RANDOM SPLIT
    # -------------------------------------------------------------------------
    logger.info("=== TRACK 3: RQ3 Comparative Evaluation on Locked Random Split (F1 & F2_safe) ===")
    _, _, rand_test_df, rand_dev_df, _ = load_split_frames("random", base_cfg)
    assert len(rand_dev_df) == 186523 and len(rand_test_df) == 46631
    assert len(set(rand_dev_df["UniqueID"]) & set(rand_test_df["UniqueID"])) == 0

    y_rand_dev = rand_dev_df["loan_default"].to_numpy(dtype=np.int64)
    y_rand_test = rand_test_df["loan_default"].to_numpy(dtype=np.int64)
    disb_rand_test = rand_test_df["disbursed_amount"].to_numpy(dtype=np.float64)
    rand_pos_rate = float(np.mean(y_rand_test))

    oot_lookup_f2 = {r["model"]: r for r in oot_rows}
    oot_lookup_f1 = rq2_df[rq2_df["feature_tier"] == "F1"].set_index("model").to_dict(orient="index")

    rq3_rows: list[dict[str, Any]] = []
    for tier in ("F1", "F2_safe"):
        # Linear (LR)
        lp_r = build_preprocessing_pipeline(tier, "linear", True, winsor_q, te_smooth, te_folds, seed)
        Xr_tr_l = lp_r.fit_transform(rand_dev_df, y_rand_dev).astype(np.float32)
        Xr_te_l = lp_r.transform(rand_test_df).astype(np.float32)
        lr_r = build_model("logistic_regression", best_params["logistic_regression"], seed, n_threads).fit(Xr_tr_l, y_rand_dev)
        p_r_lr = lr_r.predict_proba(Xr_te_l)[:, 1]
        m_r_lr = compute_binary_metrics(y_rand_test, p_r_lr, 0.50, disb_rand_test, opt_threshold=dev_f1_thresh_map["logistic_regression"])
        rq3_rows.append(_build_rq3_row(tier, "logistic_regression", m_r_lr, rand_pos_rate, oot_lookup_f1, oot_lookup_f2))
        del Xr_tr_l, Xr_te_l, lp_r, lr_r

        # Tree models
        tp_r = build_preprocessing_pipeline(tier, "tree", False, winsor_q, te_smooth, te_folds, seed)
        Xr_tr_t = tp_r.fit_transform(rand_dev_df, y_rand_dev).astype(np.float32)
        Xr_te_t = tp_r.transform(rand_test_df).astype(np.float32)
        tree_models_r = [("lightgbm", best_params["lightgbm"]), ("xgboost", best_params["xgboost"])]
        if tier == "F2_safe":
            tree_models_r.append(("random_forest", best_params["random_forest_baseline"]))
        for m_name, p_dict in tree_models_r:
            clf_r = build_model(m_name, p_dict, seed, n_threads).fit(Xr_tr_t, y_rand_dev)
            p_r = clf_r.predict_proba(Xr_te_t)[:, 1]
            m_r = compute_binary_metrics(y_rand_test, p_r, 0.50, disb_rand_test, opt_threshold=dev_f1_thresh_map[m_name])
            rq3_rows.append(_build_rq3_row(tier, m_name, m_r, rand_pos_rate, oot_lookup_f1, oot_lookup_f2))
            del clf_r
        del Xr_tr_t, Xr_te_t, tp_r
        gc.collect()

    rq3_df = pd.DataFrame(rq3_rows)
    rq3_df.to_csv(tables_dir / "m6_rq3_oot_vs_random.csv", index=False)

    # Generate all 5 Milestone 5 figures in reports/figures/
    logger.info("=== Generating 5 Publication-Grade Milestone 5 Figures in reports/figures/ ===")
    _generate_all_final_figures(
        fig_dir,
        y_oot_test,
        oot_probs,
        oot_comp_df,
        rq2_df,
        rq3_df,
        imb_oot_df,
        decile_df,
        sub_df,
        xai_stab_df,
        oot_cases,
    )
    logger.info("[OK] Completed Milestone 5 Final Single-Use Evaluation cleanly!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
