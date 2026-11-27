"""Threshold Optimization, Cost-Sensitive Trade-offs, and Reliability Diagnostics (Dev Only).

Strictly uses 5-Fold Out-of-Fold probabilities on the Primary OOT Development Set
(m5_opt_oof_predictions.npz & m5_imbalance_oof_predictions.npz). ZERO test-set access.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from src.data.load_data import get_project_root

os.environ.setdefault("MPLCONFIGDIR", str(get_project_root() / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.evaluation.metrics import (
    compute_ece,
    compute_expected_cost_amount_weighted,
    compute_expected_cost_fixed,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _metrics_at_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    disbursed_amt: np.ndarray,
) -> dict[str, Any]:
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()]
    n = len(y_true)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    flag_rate = (tp + fp) / n
    return {
        "n": int(n),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "approval_rate": round(1.0 - flag_rate, 6),
        "flag_rate": round(flag_rate, 6),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "specificity": round(float(spec), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "cost_3to1": round(compute_expected_cost_fixed(y_true, y_pred, cost_fn=3.0, cost_fp=1.0), 6),
        "cost_5to1": round(compute_expected_cost_fixed(y_true, y_pred, cost_fn=5.0, cost_fp=1.0), 6),
        "cost_10to1": round(compute_expected_cost_fixed(y_true, y_pred, cost_fn=10.0, cost_fp=1.0), 6),
        "cost_amount_weighted": round(compute_expected_cost_amount_weighted(y_true, y_pred, disbursed_amt), 6),
    }


def sweep_thresholds(
    model_label: str,
    y_true: np.ndarray,
    probs: np.ndarray,
    disbursed_amt: np.ndarray,
    grid: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for t in grid:
        pred = (probs >= float(t)).astype(np.int64)
        m = _metrics_at_threshold(y_true, pred, disbursed_amt)
        rows.append({"model": model_label, "threshold": round(float(t), 4), **m})
    return rows


def _select_threshold_on_train_folds(
    regime: str,
    y_tr: np.ndarray,
    p_tr: np.ndarray,
    grid: np.ndarray,
) -> float:
    if regime == "Default_0.50":
        return 0.50
    if regime == "Bayes_Cost_3to1":
        return 1.0 / 4.0
    if regime == "Bayes_Cost_5to1":
        return 1.0 / 6.0
    if regime == "Bayes_Cost_10to1":
        return 1.0 / 11.0
    if regime == "Capacity_Top20pct_Queue":
        return float(np.quantile(p_tr, 0.80))
    if regime == "Capacity_Top15pct_Queue":
        return float(np.quantile(p_tr, 0.85))

    pos_total = max(1, int(np.sum(y_tr == 1)))
    neg_total = max(1, int(np.sum(y_tr == 0)))
    n_tr = len(y_tr)

    best_score = -float("inf") if regime in {"F1_Optimal", "KS_YoudenJ_Optimal"} else float("inf")
    best_t = 0.50

    for t in grid:
        pred = p_tr >= float(t)
        tp = int((y_tr[pred] == 1).sum())
        fp = int((y_tr[pred] == 0).sum())
        fn = pos_total - tp
        if regime == "F1_Optimal":
            denom = 2 * tp + fp + fn
            val = (2.0 * tp / denom) if denom > 0 else 0.0
            if val > best_score:
                best_score, best_t = val, float(t)
        elif regime == "KS_YoudenJ_Optimal":
            val = (tp / pos_total) - (fp / neg_total)
            if val > best_score:
                best_score, best_t = val, float(t)
        elif regime == "Empirical_Cost_3to1":
            val = (3.0 * fn + fp) / n_tr
            if val < best_score:
                best_score, best_t = val, float(t)
        elif regime == "Empirical_Cost_5to1":
            val = (5.0 * fn + fp) / n_tr
            if val < best_score:
                best_score, best_t = val, float(t)
        else:
            raise ValueError(f"Unknown regime '{regime}'")

    return float(best_t)


def evaluate_nested_oof_regimes(
    model_label: str,
    y_dev: np.ndarray,
    probs: np.ndarray,
    cv_folds: np.ndarray,
    disbursed_amt: np.ndarray,
    regimes: list[str],
) -> list[dict[str, Any]]:
    """Select threshold strictly on K-1 folds and apply on held-out fold K."""
    grid = np.linspace(0.05, 0.65, 121)
    unique_folds = sorted(int(f) for f in np.unique(cv_folds) if int(f) >= 0)
    rows: list[dict[str, Any]] = []

    for regime in regimes:
        oof_pred = np.zeros(len(y_dev), dtype=np.int64)
        fold_thresholds: list[float] = []
        for f in unique_folds:
            tr_mask = cv_folds != f
            va_mask = cv_folds == f
            t_k = _select_threshold_on_train_folds(regime, y_dev[tr_mask], probs[tr_mask], grid)
            fold_thresholds.append(t_k)
            oof_pred[va_mask] = (probs[va_mask] >= t_k).astype(np.int64)

        m = _metrics_at_threshold(y_dev, oof_pred, disbursed_amt)
        rows.append(
            {
                "model": model_label,
                "regime": regime,
                "nested_thresh_mean": round(float(np.mean(fold_thresholds)), 5),
                "nested_thresh_std": round(float(np.std(fold_thresholds, ddof=1)), 5),
                **m,
            }
        )
    return rows


def generate_threshold_figure(
    sweep_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    imb_oof: dict[str, np.ndarray],
    fig_path: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "axes.titlesize": 10.5, "axes.titleweight": "bold"})

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.5))

    lgb_sweep = sweep_df[sweep_df["model"] == "lightgbm_stage2_natural"].copy()
    lgb_comp = comp_df[comp_df["model"] == "lightgbm_stage2_natural"].copy()
    thresh_map = dict(zip(lgb_comp["regime"], lgb_comp["nested_thresh_mean"]))

    # Panel A: Precision / Recall / F1 / Flag Rate vs Threshold
    ax = axes[0, 0]
    ax.plot(lgb_sweep["threshold"], lgb_sweep["precision"], label="Precision", linewidth=2.0, color="#2b6cb0")
    ax.plot(lgb_sweep["threshold"], lgb_sweep["recall"], label="Recall (Default Capture)", linewidth=2.0, color="#e53e3e")
    ax.plot(lgb_sweep["threshold"], lgb_sweep["f1"], label="F1-score", linewidth=2.0, color="#2f855a")
    ax.plot(lgb_sweep["threshold"], lgb_sweep["flag_rate"], label="Flagged Share (1 - Approval Rate)", linewidth=1.8, linestyle="--", color="#805ad5")

    markers = [
        ("Bayes 5:1 (0.167)", thresh_map["Bayes_Cost_5to1"], "#dd6b20"),
        (f"F1-Opt (~{thresh_map['F1_Optimal']:.3f})", thresh_map["F1_Optimal"], "#2f855a"),
        (f"Top-20% (~{thresh_map['Capacity_Top20pct_Queue']:.3f})", thresh_map["Capacity_Top20pct_Queue"], "#805ad5"),
        ("Default (0.500)", 0.50, "#4a5568"),
    ]
    for lbl, tv, col in markers:
        ax.axvline(tv, color=col, linestyle=":", linewidth=1.4, label=lbl)
    ax.set_xlabel("Decision Threshold t")
    ax.set_ylabel("Metric / Portfolio Fraction")
    ax.set_title("A. Precision-Recall-Flagging Trade-off (LightGBM Stage-2)")
    ax.set_xlim(0.05, 0.56)
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="upper right", fontsize=7.5, ncol=2)

    # Panel B: Asymmetric Expected Cost Curves & Distinct Cost Minima
    ax2 = axes[0, 1]
    ax2.plot(lgb_sweep["threshold"], lgb_sweep["cost_3to1"], label="Fixed Cost 3:1", color="#3182ce", linewidth=2.0)
    ax2.plot(lgb_sweep["threshold"], lgb_sweep["cost_5to1"], label="Fixed Cost 5:1", color="#dd6b20", linewidth=2.2)
    ax2.plot(lgb_sweep["threshold"], lgb_sweep["cost_10to1"], label="Fixed Cost 10:1", color="#c53030", linewidth=2.0)
    for col_name, color, dx, dy in [
        ("cost_3to1", "#3182ce", 6, -13),
        ("cost_5to1", "#dd6b20", 6, -13),
        ("cost_10to1", "#c53030", 6, 8),
    ]:
        idx_min = lgb_sweep[col_name].idxmin()
        t_min = float(lgb_sweep.loc[idx_min, "threshold"])
        c_min = float(lgb_sweep.loc[idx_min, col_name])
        ax2.scatter([t_min], [c_min], color=color, s=55, zorder=5, edgecolor="black", linewidth=0.6)
        ax2.annotate(
            f"t*={t_min:.2f}",
            (t_min, c_min),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=7.8,
            color=color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
        )

    ax2_r = ax2.twinx()
    ax2_r.plot(lgb_sweep["threshold"], lgb_sweep["cost_amount_weighted"] * 100.0, label="Amount-Weighted Loss (%)", color="#2f855a", linewidth=2.0, linestyle="-.")
    ax2_r.set_ylabel("Amount-Weighted Portfolio Loss (%)", color="#2f855a", fontsize=9)
    ax2_r.tick_params(axis="y", labelcolor="#2f855a")
    ax2_r.grid(False)

    ax2.set_xlabel("Decision Threshold t")
    ax2.set_ylabel("Expected Cost per Loan (FP=1 unit)")
    ax2.set_title("B. Asymmetric Cost Curves (Different Cost Profiles -> Different t*)")
    ax2.set_xlim(0.05, 0.56)
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7.5)

    # Panel C: Confusion Matrix Composition across 7 Regimes (LightGBM Stage-2)
    ax3 = axes[1, 0]
    order_reg = [
        ("Default_0.50", "Default\n(0.500)"),
        ("Capacity_Top15pct_Queue", f"Top-15%\n(~{thresh_map['Capacity_Top15pct_Queue']:.3f})"),
        ("Capacity_Top20pct_Queue", f"Top-20%\n(~{thresh_map['Capacity_Top20pct_Queue']:.3f})"),
        ("Bayes_Cost_3to1", "Cost 3:1\n(0.250)"),
        ("KS_YoudenJ_Optimal", f"KS/J-Opt\n(~{thresh_map['KS_YoudenJ_Optimal']:.3f})"),
        ("F1_Optimal", f"F1-Opt\n(~{thresh_map['F1_Optimal']:.3f})"),
        ("Bayes_Cost_5to1", "Cost 5:1\n(0.167)"),
        ("Bayes_Cost_10to1", "Cost 10:1\n(0.091)"),
    ]
    sub_c = lgb_comp.set_index("regime").reindex([r[0] for r in order_reg])
    n_tot = sub_c["n"].to_numpy(dtype=float)
    tp_pct = sub_c["tp"].to_numpy(dtype=float) / n_tot * 100.0
    fn_pct = sub_c["fn"].to_numpy(dtype=float) / n_tot * 100.0
    fp_pct = sub_c["fp"].to_numpy(dtype=float) / n_tot * 100.0
    tn_pct = sub_c["tn"].to_numpy(dtype=float) / n_tot * 100.0
    xc = np.arange(len(order_reg))
    ax3.bar(xc, tp_pct, label="TP (Caught Default)", color="#2f855a", alpha=0.92)
    ax3.bar(xc, fn_pct, bottom=tp_pct, label="FN (Missed Default)", color="#e53e3e", alpha=0.88)
    ax3.bar(xc, fp_pct, bottom=tp_pct + fn_pct, label="FP (False Alarm)", color="#ed8936", alpha=0.88)
    ax3.bar(xc, tn_pct, bottom=tp_pct + fn_pct + fp_pct, label="TN (Clean Approval)", color="#4299e1", alpha=0.85)
    ax3.set_xticks(xc)
    ax3.set_xticklabels([r[1] for r in order_reg], fontsize=7.8)
    ax3.set_ylabel("% of Dev Portfolio (N = 182,501)")
    ax3.set_title("C. Confusion Matrix Breakdown Across Nested OOF Regimes")
    ax3.set_ylim(0, 120)
    ax3.legend(loc="upper right", ncol=2, fontsize=7.5)

    # Panel D: Reliability Diagram (10 equal-width bins) on Stage-2 LightGBM
    ax4 = axes[1, 1]
    y_d = imb_oof["y_dev"]
    curves = [
        ("natural__uncalibrated", "Natural Prior", "#2b6cb0", "-o"),
        ("rus_0.5__uncalibrated", "RUS 0.5", "#2f855a", "-^"),
        ("class_weight__uncalibrated", "Class Weight Balanced", "#dd6b20", "--d"),
        ("smote_1.0__uncalibrated", "SMOTE 1.0", "#c53030", ":v"),
    ]
    ax4.plot([0, 1], [0, 1], "k--", linewidth=1.1, label="Perfect Calibration")
    bins = np.linspace(0.0, 1.0, 11)
    for key, base_lbl, col, fmt in curves:
        if key not in imb_oof:
            continue
        p_c = imb_oof[key]
        ece_val = compute_ece(y_d, p_c, n_bins=10)
        lbl = f"{base_lbl} (ECE={ece_val:.4f})"
        xs, ys = [], []
        for bi in range(10):
            mask = (p_c >= bins[bi]) & (p_c <= bins[bi + 1] if bi == 9 else p_c < bins[bi + 1])
            if mask.sum() >= 30:
                xs.append(float(p_c[mask].mean()))
                ys.append(float(y_d[mask].mean()))
        ax4.plot(xs, ys, fmt, label=lbl, color=col, linewidth=1.8, markersize=4.5)
    ax4.set_xlabel("Mean Predicted Probability")
    ax4.set_ylabel("Observed Empirical Default Rate")
    ax4.set_title("D. Calibration Reliability Curve (Natural vs. Weight / SMOTE)")
    ax4.set_xlim(0.0, 0.92)
    ax4.set_ylim(0.0, 0.72)
    ax4.legend(loc="upper left", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    root = get_project_root()
    results_dir = root / "experiments" / "results"
    fig_dir = root / "experiments" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    opt_oof = dict(np.load(results_dir / "m5_opt_oof_predictions.npz"))
    imb_oof = dict(np.load(results_dir / "m5_imbalance_oof_predictions.npz"))

    y_dev = opt_oof["y_dev"]
    cv_folds = opt_oof["cv_folds"]
    disbursed_amt = opt_oof["disbursed_amount"]
    assert len(y_dev) == 182501, f"Safety check failed: expected 182,501 Dev rows, got {len(y_dev)}"

    grid = np.round(np.linspace(0.05, 0.60, 111), 4)
    sweep_rows: list[dict[str, Any]] = []
    for label, key in [
        ("lightgbm_stage2_natural", "lightgbm"),
        ("xgboost_stage2_natural", "xgboost"),
        ("logistic_regression_s1", "logistic_regression"),
    ]:
        sweep_rows.extend(sweep_thresholds(label, y_dev, opt_oof[key], disbursed_amt, grid))

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(results_dir / "m5_threshold_sweep.csv", index=False)

    regimes = [
        "Default_0.50",
        "F1_Optimal",
        "KS_YoudenJ_Optimal",
        "Bayes_Cost_3to1",
        "Empirical_Cost_3to1",
        "Bayes_Cost_5to1",
        "Empirical_Cost_5to1",
        "Bayes_Cost_10to1",
        "Capacity_Top20pct_Queue",
        "Capacity_Top15pct_Queue",
    ]
    comp_rows: list[dict[str, Any]] = []
    candidates = [
        ("lightgbm_stage2_natural", opt_oof["lightgbm"]),
        ("xgboost_stage2_natural", opt_oof["xgboost"]),
        ("logistic_regression_s1", opt_oof["logistic_regression"]),
    ]
    for label, probs in candidates:
        comp_rows.extend(evaluate_nested_oof_regimes(label, y_dev, probs, cv_folds, disbursed_amt, regimes))

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(results_dir / "m5_threshold_comparison.csv", index=False)

    fig_path = fig_dir / "opt_01_threshold_tradeoffs.png"
    generate_threshold_figure(sweep_df, comp_df, imb_oof, fig_path)
    logger.info("[OK] Saved m5_threshold_sweep.csv (%d rows), m5_threshold_comparison.csv (%d rows), and %s",
                len(sweep_df), len(comp_df), fig_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
