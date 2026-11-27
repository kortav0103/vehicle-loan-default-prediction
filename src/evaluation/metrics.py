"""Evaluation metrics, out-of-fold probability calibration, and cost-sensitive thresholding."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) over equal-width probability bins."""
    y_t = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(len(y_t))
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        cnt = int(mask.sum())
        if cnt > 0:
            acc_bin = float(y_t[mask].mean())
            conf_bin = float(p[mask].mean())
            ece += (cnt / n) * abs(acc_bin - conf_bin)
    return float(ece)


def compute_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Kolmogorov-Smirnov (KS) statistic between default and non-default score distributions."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def compute_recall_at_top_k(y_true: np.ndarray, y_prob: np.ndarray, top_fraction: float) -> float:
    """Fraction of all actual defaults captured inside the highest-risk top_fraction slice."""
    n = len(y_true)
    k = max(1, int(round(n * top_fraction)))
    order = np.argsort(-np.asarray(y_prob, dtype=np.float64), kind="mergesort")
    total_pos = float(np.sum(y_true))
    if total_pos <= 0:
        return 0.0
    captured = float(np.sum(np.asarray(y_true)[order[:k]]))
    return captured / total_pos


def compute_expected_cost_fixed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
) -> float:
    """Mean asymmetric misclassification cost per loan (TP/TN cost = 0)."""
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)
    fn = int(((y_t == 1) & (y_p == 0)).sum())
    fp = int(((y_t == 0) & (y_p == 1)).sum())
    return float((cost_fn * fn + cost_fp * fp) / len(y_t))


def compute_expected_cost_amount_weighted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    loan_amounts: np.ndarray,
    lgd_rate: float = 0.50,
    margin_rate: float = 0.10,
) -> float:
    """Example-dependent normalized financial loss (inspired by Bahnsen et al., 2014).

    Missing a default loses `lgd_rate * disbursed_amount`; wrongly rejecting a good
    borrower forfeits `margin_rate * disbursed_amount`. Normalized by total portfolio disbursed_amount.
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)
    amt = np.asarray(loan_amounts, dtype=np.float64)
    total_amt = float(np.sum(amt))
    if total_amt <= 0:
        return 0.0
    fn_loss = float(np.sum(amt[(y_t == 1) & (y_p == 0)]) * lgd_rate)
    fp_loss = float(np.sum(amt[(y_t == 0) & (y_p == 1)]) * margin_rate)
    return (fn_loss + fp_loss) / total_amt


def find_optimal_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    loan_amounts: np.ndarray | None = None,
) -> dict[str, float]:
    """Identify F1-optimal, KS-optimal, and cost-optimal thresholds purely on OOF predictions."""
    y_t = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(y_prob, dtype=np.float64)
    grid = np.linspace(0.05, 0.90, 171)

    best_f1 = -1.0
    best_f1_t = 0.50
    best_j = -1.0
    best_j_t = 0.50
    best_cost_5to1 = float("inf")
    best_cost_5to1_t = 1.0 / 6.0

    pos_total = max(1, int(np.sum(y_t == 1)))
    neg_total = max(1, int(np.sum(y_t == 0)))

    for t in grid:
        pred = (p >= t).astype(np.int64)
        tp = int(((y_t == 1) & (pred == 1)).sum())
        fp = int(((y_t == 0) & (pred == 1)).sum())
        fn = pos_total - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / pos_total
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_f1_t = float(t)

        tpr = tp / pos_total
        fpr = fp / neg_total
        j_val = tpr - fpr
        if j_val > best_j:
            best_j = j_val
            best_j_t = float(t)

        cost5 = (5.0 * fn + 1.0 * fp) / len(y_t)
        if cost5 < best_cost_5to1:
            best_cost_5to1 = cost5
            best_cost_5to1_t = float(t)

    out = {
        "f1_opt_thresh": round(best_f1_t, 4),
        "ks_opt_thresh": round(best_j_t, 4),
        "empirical_cost_5to1_thresh": round(best_cost_5to1_t, 4),
        "bayes_cost_3to1_thresh": round(1.0 / 4.0, 4),
        "bayes_cost_5to1_thresh": round(1.0 / 6.0, 4),
        "bayes_cost_10to1_thresh": round(1.0 / 11.0, 4),
    }

    if loan_amounts is not None:
        best_amt_cost = float("inf")
        best_amt_t = 1.0 / 6.0
        for t in grid:
            pred = (p >= t).astype(np.int64)
            c_amt = compute_expected_cost_amount_weighted(y_t, pred, loan_amounts)
            if c_amt < best_amt_cost:
                best_amt_cost = c_amt
                best_amt_t = float(t)
        out["empirical_amount_weighted_thresh"] = round(best_amt_t, 4)

    return out


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    loan_amounts: np.ndarray | None = None,
    opt_threshold: float | None = None,
    emp_cost_5to1_threshold: float | None = None,
    emp_amount_weighted_threshold: float | None = None,
) -> dict[str, Any]:
    """Compute comprehensive discrimination, calibration, threshold, and cost metrics.

    When `opt_threshold` is provided (e.g., Dev-frozen threshold applied to Locked Test Set),
    no threshold search is performed on `y_true`; all empirical optimal cost metrics strictly
    use the frozen thresholds (`emp_cost_5to1_threshold` / `emp_amount_weighted_threshold`,
    defaulting to `opt_threshold`).
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-7, 1.0 - 1e-7)

    roc_auc = float(roc_auc_score(y_t, p))
    pr_auc = float(average_precision_score(y_t, p))
    ks_stat = compute_ks_statistic(y_t, p)
    brier = float(brier_score_loss(y_t, p))
    ll = float(log_loss(y_t, p))
    ece = compute_ece(y_t, p, n_bins=10)

    y_pred_05 = (p >= threshold).astype(np.int64)
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y_t, y_pred_05, labels=[0, 1]).ravel()]

    metrics: dict[str, Any] = {
        "roc_auc": round(roc_auc, 6),
        "gini": round(2.0 * roc_auc - 1.0, 6),
        "pr_auc": round(pr_auc, 6),
        "ks_stat": round(ks_stat, 6),
        "brier_score": round(brier, 6),
        "log_loss": round(ll, 6),
        "ece": round(ece, 6),
        "recall_at_top10pct": round(compute_recall_at_top_k(y_t, p, 0.10), 6),
        "recall_at_top20pct": round(compute_recall_at_top_k(y_t, p, 0.20), 6),
        "accuracy_05": round(float(accuracy_score(y_t, y_pred_05)), 6),
        "precision_05": round(float(precision_score(y_t, y_pred_05, zero_division=0)), 6),
        "recall_05": round(float(recall_score(y_t, y_pred_05, zero_division=0)), 6),
        "f1_05": round(float(f1_score(y_t, y_pred_05, zero_division=0)), 6),
        "tn_05": tn,
        "fp_05": fp,
        "fn_05": fn,
        "tp_05": tp,
    }

    if opt_threshold is not None:
        use_t = float(opt_threshold)
        t_emp_cost5 = float(emp_cost_5to1_threshold) if emp_cost_5to1_threshold is not None else use_t
        t_emp_amt = float(emp_amount_weighted_threshold) if emp_amount_weighted_threshold is not None else use_t
    else:
        thresh_info = find_optimal_thresholds(y_t, p, loan_amounts=loan_amounts)
        use_t = float(thresh_info["f1_opt_thresh"])
        t_emp_cost5 = (
            float(emp_cost_5to1_threshold)
            if emp_cost_5to1_threshold is not None
            else float(thresh_info["empirical_cost_5to1_thresh"])
        )
        t_emp_amt = (
            float(emp_amount_weighted_threshold)
            if emp_amount_weighted_threshold is not None
            else float(thresh_info.get("empirical_amount_weighted_thresh", 1.0 / 6.0))
        )

    y_pred_opt = (p >= use_t).astype(np.int64)
    tn_o, fp_o, fn_o, tp_o = [int(v) for v in confusion_matrix(y_t, y_pred_opt, labels=[0, 1]).ravel()]

    metrics.update(
        {
            "f1_opt_thresh": round(use_t, 4),
            "accuracy_opt": round(float(accuracy_score(y_t, y_pred_opt)), 6),
            "precision_opt": round(float(precision_score(y_t, y_pred_opt, zero_division=0)), 6),
            "recall_opt": round(float(recall_score(y_t, y_pred_opt, zero_division=0)), 6),
            "f1_opt": round(float(f1_score(y_t, y_pred_opt, zero_division=0)), 6),
            "tn_opt": tn_o,
            "fp_opt": fp_o,
            "fn_opt": fn_o,
            "tp_opt": tp_o,
            # Asymmetric Bayes threshold cost metrics (p* = 1 / (1 + ratio))
            "expected_cost_3to1_bayes": round(
                compute_expected_cost_fixed(y_t, (p >= 0.25).astype(np.int64), cost_fn=3.0, cost_fp=1.0), 6
            ),
            "expected_cost_5to1_bayes": round(
                compute_expected_cost_fixed(y_t, (p >= (1.0 / 6.0)).astype(np.int64), cost_fn=5.0, cost_fp=1.0), 6
            ),
            "expected_cost_10to1_bayes": round(
                compute_expected_cost_fixed(y_t, (p >= (1.0 / 11.0)).astype(np.int64), cost_fn=10.0, cost_fp=1.0), 6
            ),
            "expected_cost_5to1_emp_opt": round(
                compute_expected_cost_fixed(
                    y_t, (p >= t_emp_cost5).astype(np.int64), cost_fn=5.0, cost_fp=1.0
                ),
                6,
            ),
        }
    )

    if loan_amounts is not None:
        metrics["expected_cost_amount_weighted_bayes"] = round(
            compute_expected_cost_amount_weighted(y_t, (p >= (1.0 / 6.0)).astype(np.int64), loan_amounts), 6
        )
        metrics["expected_cost_amount_weighted_emp_opt"] = round(
            compute_expected_cost_amount_weighted(
                y_t, (p >= t_emp_amt).astype(np.int64), loan_amounts
            ),
            6,
        )

    return metrics


def calibrate_oof_probabilities(
    y_true: np.ndarray,
    oof_probs: np.ndarray,
    fold_ids: np.ndarray,
    method: str = "platt",
) -> np.ndarray:
    """Perform strict Out-of-Fold probability calibration (Platt sigmoid or Isotonic).

    Fits calibrator ONLY on K-1 folds of OOF predictions and transforms fold K,
    preventing overfitting of calibration curves.
    """
    if method not in {"platt", "isotonic"}:
        raise ValueError(f"Unsupported calibration method '{method}'")

    y_t = np.asarray(y_true, dtype=np.int64)
    p = np.clip(np.asarray(oof_probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    folds = np.asarray(fold_ids)
    unique_folds = sorted(int(f) for f in np.unique(folds) if int(f) >= 0)

    calibrated = np.zeros_like(p)
    logits = np.log(p / (1.0 - p))[:, None]

    for f in unique_folds:
        tr_m = folds != f
        va_m = folds == f
        if method == "platt":
            cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=500)
            cal.fit(logits[tr_m], y_t[tr_m])
            calibrated[va_m] = cal.predict_proba(logits[va_m])[:, 1]
        else:
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(p[tr_m], y_t[tr_m])
            calibrated[va_m] = cal.predict(p[va_m])

    return np.clip(calibrated, 1e-7, 1.0 - 1e-7)
