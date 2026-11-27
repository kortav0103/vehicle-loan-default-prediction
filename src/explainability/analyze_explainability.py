"""Multi-Method Model Explainability, Exact TreeSHAP Interactions, and Proxy Audit (Dev Only).

Trains clean Stage-2 LightGBM, Stage-2 XGBoost, and Tuned LogisticRegression strictly on
OOT Development folds (cv_fold != 4, n=146,001) and explains strictly unseen Dev validation
fold (cv_fold == 4, n=36,500). Locked Test Sets are NEVER accessed.
"""

from __future__ import annotations

import gc
import json
import logging
import os
from pathlib import Path
from typing import Any

from src.data.load_data import get_project_root, load_config

os.environ.setdefault("MPLCONFIGDIR", str(get_project_root() / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedShuffleSplit

from src.evaluation.metrics import compute_binary_metrics
from src.features.preprocessing import build_preprocessing_pipeline
from src.models.registry import build_model, load_dev_only_data, load_model_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def canonical_feature_name(raw_name: str) -> str:
    """Strip ColumnTransformer prefixes while keeping __oof_te / __freq suffixes intact."""
    for prefix in (
        "num_skewed__",
        "num_bounded__",
        "binary__",
        "cat_low_med__",
        "hc_target_enc__",
        "hc_freq_enc__",
    ):
        if raw_name.startswith(prefix):
            return raw_name[len(prefix) :]
    return raw_name


def assign_domain_group(canon_name: str) -> str:
    """Map canonical feature name to one of 6 semantic domain groups."""
    core = canon_name.replace("missingindicator_", "")
    if any(k in core for k in ("branch_id", "supplier_id", "Current_pincode_ID", "State_ID", "manufacturer_id")):
        return "Operational Channel & Geography (Proxies)"
    if any(k in core for k in ("OVERDUE", "DELINQUENT", "NO.OF_INQUIRIES", "overdue_", "delinquent_", "inquiry_")):
        return "Delinquency & Credit Inquiries"
    if any(k in core for k in ("PERFORM_CNS", "bureau_", "has_bureau_score")):
        return "Credit Bureau Score & Unscored Regime"
    if any(
        k in core
        for k in (
            "ltv",
            "disbursed_amount",
            "asset_cost",
            "down_payment",
        )
    ):
        return "Loan Terms & LTV"
    if any(
        k in core
        for k in (
            "PRI.",
            "PRIMARY.",
            "SEC.",
            "NEW.ACCTS",
            "total_",
            "has_sec_account",
            "avg_acct_age",
            "credit_history",
            "history_span_gap",
            "active_to_total",
            "credit_utilization",
            "instalment_to_disbursed",
            "flag_pri_bal",
            "flag_neg_balance",
        )
    ):
        return "Existing Credit Accounts & Balances"
    return "Borrower Demographics, Employment & KYC"


def _compute_spearman_directions(X: np.ndarray, shap_vals: np.ndarray) -> np.ndarray:
    dirs = np.zeros(X.shape[1], dtype=np.float64)
    for j in range(X.shape[1]):
        xj = X[:, j]
        sj = shap_vals[:, j]
        if np.std(xj) < 1e-12 or np.std(sj) < 1e-12:
            dirs[j] = 0.0
        else:
            rho = spearmanr(xj, sj).statistic
            dirs[j] = float(rho) if np.isfinite(rho) else 0.0
    return dirs


def _top_k_jaccard(a: pd.Series, b: pd.Series, k: int = 15) -> float:
    top_a = set(a.nlargest(k).index.tolist())
    top_b = set(b.nlargest(k).index.tolist())
    return float(len(top_a & top_b) / len(top_a | top_b))


def _select_case_study_indices(
    y_va: np.ndarray,
    probs_va: np.ndarray,
    thresh: float = 0.205,
) -> dict[str, int]:
    """Select 4 representative, clear validation-fold cases (TP, TN, FN, FP)."""
    # High-confidence True Positive
    tp_candidates = np.where((y_va == 1) & (probs_va >= thresh))[0]
    tp_target = 0.50
    tp_pos = int(tp_candidates[np.argmin(np.abs(probs_va[tp_candidates] - tp_target))])

    # Low-risk True Negative
    tn_candidates = np.where((y_va == 0) & (probs_va < thresh))[0]
    tn_target = 0.075
    tn_pos = int(tn_candidates[np.argmin(np.abs(probs_va[tn_candidates] - tn_target))])

    # Operationally dangerous False Negative (missed default predicted low risk)
    fn_candidates = np.where((y_va == 1) & (probs_va < thresh))[0]
    fn_target = 0.11
    fn_pos = int(fn_candidates[np.argmin(np.abs(probs_va[fn_candidates] - fn_target))])

    # Costly False Positive (flagged high risk, actually repaid)
    fp_candidates = np.where((y_va == 0) & (probs_va >= thresh))[0]
    fp_target = 0.42
    fp_pos = int(fp_candidates[np.argmin(np.abs(probs_va[fp_candidates] - fp_target))])

    return {"TP": tp_pos, "TN": tn_pos, "FN": fn_pos, "FP": fp_pos}


def _generate_shap_global_and_beeswarm(
    feat_df: pd.DataFrame,
    X_exp: np.ndarray,
    lgb_shap: np.ndarray,
    feat_names: list[str],
    fig_path: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "axes.titlesize": 10.5, "axes.titleweight": "bold"})

    top20 = feat_df.sort_values("lgb_mean_abs_shap", ascending=False).head(20).copy()
    top20_asc = top20.iloc[::-1].reset_index(drop=True)

    group_palette = {
        "Operational Channel & Geography (Proxies)": "#c53030",
        "Loan Terms & LTV": "#dd6b20",
        "Credit Bureau Score & Unscored Regime": "#2b6cb0",
        "Delinquency & Credit Inquiries": "#805ad5",
        "Existing Credit Accounts & Balances": "#2f855a",
        "Borrower Demographics, Employment & KYC": "#718096",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 8.2))

    # Left: Global mean(|SHAP|) colored by Semantic Domain Group
    y_pos = np.arange(len(top20_asc))
    bar_colors = [group_palette.get(g, "#4a5568") for g in top20_asc["domain_group"]]
    axes[0].barh(y_pos, top20_asc["lgb_mean_abs_shap"], color=bar_colors, alpha=0.90, edgecolor="black", linewidth=0.4)
    for i, v in enumerate(top20_asc["lgb_mean_abs_shap"]):
        axes[0].text(v + 0.0025, i, f"{v:.4f}", va="center", fontsize=7.8)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(top20_asc["feature"], fontsize=8.5)
    axes[0].set_xlabel("Mean(|SHAP value|) in Log-Odds (Stage-2 LightGBM)")
    axes[0].set_title("A. Top-20 Global Feature Importance (Colored by Domain Group)")
    axes[0].set_xlim(0.0, float(top20_asc["lgb_mean_abs_shap"].max()) * 1.22)

    handles = [plt.Rectangle((0, 0), 1, 1, color=col, alpha=0.9) for col in group_palette.values()]
    axes[0].legend(handles, list(group_palette.keys()), loc="lower right", fontsize=7.2, title="Domain Group", title_fontsize=7.5)

    # Right: Beeswarm summary scatter on subsample of 2,000 points for crisp visual density
    rng = np.random.default_rng(42)
    sub_idx = rng.choice(len(X_exp), size=min(2000, len(X_exp)), replace=False)
    name_to_idx = {nm: idx for idx, nm in enumerate(feat_names)}

    cmap = plt.get_cmap("coolwarm")
    for row_i, fname in enumerate(top20_asc["feature"]):
        j = name_to_idx[fname]
        s_vals = lgb_shap[sub_idx, j]
        x_vals = X_exp[sub_idx, j]
        lo, hi = np.quantile(x_vals, 0.02), np.quantile(x_vals, 0.98)
        if hi > lo:
            norm_x = np.clip((x_vals - lo) / (hi - lo), 0.0, 1.0)
        else:
            norm_x = np.full_like(x_vals, 0.5)
        jitter = rng.normal(0.0, 0.14, size=len(sub_idx))
        axes[1].scatter(s_vals, np.full(len(sub_idx), row_i) + jitter, c=norm_x, cmap=cmap, s=7, alpha=0.55, rasterized=True)

    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("SHAP Impact on Log-Odds of Default (<0 Lower Risk | >0 Higher Risk)")
    axes[1].set_title("B. Exact TreeSHAP Summary Beeswarm (Unseen Dev Fold 4, n=10k)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], pad=0.02, aspect=35)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["Low Value", "High Value"])
    cbar.set_label("Relative Feature Value (2%-98% clipped)", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def _generate_dependence_figure(
    X_exp: np.ndarray,
    lgb_shap: np.ndarray,
    feat_names: list[str],
    fig_path: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "axes.titlesize": 10.2, "axes.titleweight": "bold"})

    name_to_idx = {nm: idx for idx, nm in enumerate(feat_names)}
    rng = np.random.default_rng(42)
    idx_sub = rng.choice(len(X_exp), size=min(4000, len(X_exp)), replace=False)

    specs = [
        ("ltv", "has_bureau_score", "A. LTV (%) SHAP Impact (Colored by Has Bureau Score)", False),
        ("bureau_score_clean", "ltv", "B. Bureau Score Clean (>=300 Subset, Colored by LTV)", True),
        ("Current_pincode_ID__oof_te", "supplier_id__oof_te", "C. Pincode OOF-TE Proxy Impact (Colored by Supplier OOF-TE)", False),
        ("NO.OF_INQUIRIES", "ltv", "D. Recent Credit Inquiries Impact (Colored by LTV)", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.0))
    for ax, (x_name, c_name, title, scored_only) in zip(axes.ravel(), specs):
        jx = name_to_idx[x_name]
        jc = name_to_idx[c_name]
        mask = idx_sub.copy()
        if scored_only:
            hb_idx = name_to_idx["has_bureau_score"]
            mask = mask[X_exp[mask, hb_idx] > 0.5]
        xv = X_exp[mask, jx]
        yv = lgb_shap[mask, jx]
        cv = X_exp[mask, jc]
        if x_name == "NO.OF_INQUIRIES":
            keep = xv <= 8
            xv, yv, cv = xv[keep], yv[keep], cv[keep]
            xv = xv + rng.normal(0.0, 0.08, size=len(xv))

        sc = ax.scatter(xv, yv, c=cv, cmap="viridis", s=10, alpha=0.55, rasterized=True)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9, alpha=0.65)
        ax.set_xlabel(x_name)
        ax.set_ylabel("SHAP Value (Log-Odds)")
        ax.set_title(title)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.set_label(c_name, fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def _generate_agreement_figure(
    feat_df: pd.DataFrame,
    group_df: pd.DataFrame,
    fig_path: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "axes.titlesize": 10.5, "axes.titleweight": "bold"})

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))

    # Left: Normalized importance comparison across 5 methods for Top-15 features by LGBM SHAP
    top15 = feat_df.sort_values("lgb_mean_abs_shap", ascending=False).head(15).copy()
    cols_methods = [
        ("lgb_mean_abs_shap", "LGBM SHAP"),
        ("xgb_mean_abs_shap", "XGB SHAP"),
        ("lgb_perm_importance", "LGBM Perm AUC"),
        ("lgb_gain_norm", "LGBM Gain"),
        ("lr_abs_beta", "LR |beta|"),
    ]
    mat = pd.DataFrame(index=top15["feature"])
    for c, lbl in cols_methods:
        s = top15[c].astype(float)
        denom = float(np.nanmax(s)) if np.nanmax(s) > 0 else 1.0
        mat[lbl] = (s / denom).to_numpy()
    mat = mat.fillna(0.0)

    sns.heatmap(
        mat,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.4,
        ax=axes[0],
        cbar_kws={"label": "Normalized Relative Importance (Max = 1.0)"},
    )
    axes[0].set_title("A. Cross-Method Normalized Importance (Top-15 LGBM SHAP Features)")
    axes[0].set_ylabel("")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].tick_params(axis="y", labelsize=8.5)

    # Right: Domain Group Share of Total |SHAP| (LightGBM vs XGBoost)
    g_plot = group_df.sort_values("lgb_shap_share_pct", ascending=True)
    yg = np.arange(len(g_plot))
    h = 0.36
    axes[1].barh(yg + h / 2, g_plot["lgb_shap_share_pct"], height=h, label="LightGBM Stage-2", color="#2b6cb0", alpha=0.9)
    axes[1].barh(yg - h / 2, g_plot["xgb_shap_share_pct"], height=h, label="XGBoost Stage-2", color="#dd6b20", alpha=0.9)
    for i, (_, r) in enumerate(g_plot.iterrows()):
        axes[1].text(r["lgb_shap_share_pct"] + 0.4, i + h / 2, f"{r['lgb_shap_share_pct']:.1f}%", va="center", fontsize=8)
        axes[1].text(r["xgb_shap_share_pct"] + 0.4, i - h / 2, f"{r['xgb_shap_share_pct']:.1f}%", va="center", fontsize=8)
    axes[1].set_yticks(yg)
    axes[1].set_yticklabels(g_plot["domain_group"], fontsize=8.5)
    axes[1].set_xlabel("Share of Total Mean(|SHAP|) (%)")
    axes[1].set_title("B. Attribution Concentration: Borrower Fundamentals vs. Proxies")
    axes[1].set_xlim(0.0, float(max(g_plot["lgb_shap_share_pct"].max(), g_plot["xgb_shap_share_pct"].max())) * 1.18)
    axes[1].legend(loc="lower right", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def _generate_local_waterfalls(
    cases: list[dict[str, Any]],
    fig_path: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "axes.titlesize": 10.0, "axes.titleweight": "bold"})

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 9.6))
    for ax, case in zip(axes.ravel(), cases):
        drivers = case["top_drivers"][:8]
        # Reverse so top driver sits on top
        drivers_rev = list(reversed(drivers))
        labels = [f"{d['feature']} = {d['feature_value']:.3g}" for d in drivers_rev]
        vals = [d["shap_value"] for d in drivers_rev]
        colors = ["#c53030" if v > 0 else "#2b6cb0" for v in vals]

        y = np.arange(len(labels))
        ax.barh(y, vals, color=colors, alpha=0.88, edgecolor="black", linewidth=0.4)
        ax.axvline(0.0, color="black", linewidth=0.9)
        max_abs = max(0.05, max(abs(v) for v in vals))
        for i, v in enumerate(vals):
            offset = max_abs * 0.03 * (1 if v >= 0 else -1)
            ha = "left" if v >= 0 else "right"
            ax.text(v + offset, i, f"{v:+.3f}", va="center", ha=ha, fontsize=7.5)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.0)
        ax.set_xlim(-max_abs * 1.28, max_abs * 1.28)
        ax.set_xlabel("SHAP Contribution (Log-Odds)")
        ax.set_title(
            f"{case['case_type']} (Actual y={case['actual_y']} | Pred Prob={case['pred_prob']:.3f} | Base={case['base_prob']:.3f})"
        )

    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    root = get_project_root()
    base_cfg = load_config()
    model_cfg = load_model_config()
    seed = int(model_cfg["benchmark"]["random_seed"])
    n_threads = int(model_cfg["benchmark"]["n_threads"])
    tier = str(model_cfg["benchmark"]["default_feature_tier"])

    results_dir = root / "experiments" / "results"
    fig_dir = root / "experiments" / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    best_params_doc = json.loads((results_dir / "m5_opt_best_params.json").read_text(encoding="utf-8"))
    opt_oof = dict(np.load(results_dir / "m5_opt_oof_predictions.npz"))
    dev_df, y_dev, cv_folds, _, disbursed_amt = load_dev_only_data(base_cfg)

    tr_idx = np.where(cv_folds != 4)[0]
    va_idx = np.where(cv_folds == 4)[0]
    assert len(set(tr_idx) & set(va_idx)) == 0 and len(tr_idx) + len(va_idx) == 182501

    winsor_q = tuple(base_cfg["preprocessing"]["winsor_quantiles"])
    te_smooth = float(base_cfg["preprocessing"]["target_encoding_smoothing"])
    te_folds = int(base_cfg["preprocessing"]["target_encoding_cv_folds"])

    # Measure unseen cold-start rates across CV folds for operational/geography IDs
    cold_start_rates: dict[str, list[float]] = {c: [] for c in ("branch_id", "supplier_id", "Current_pincode_ID")}
    for f in range(5):
        m_tr = cv_folds != f
        m_va = cv_folds == f
        for c in cold_start_rates:
            tr_set = set(dev_df.loc[m_tr, c].astype(str))
            va_vals = dev_df.loc[m_va, c].astype(str)
            cold_start_rates[c].append(float((~va_vals.isin(tr_set)).mean()))

    logger.info("=== Fitting F2_safe pipelines on Dev folds 0..3 (n=%d) for unseen Fold 4 (n=%d) explainability ===", len(tr_idx), len(va_idx))
    tree_pipe = build_preprocessing_pipeline(tier, "tree", False, winsor_q, te_smooth, te_folds, seed)
    X_tr_tree = tree_pipe.fit_transform(dev_df.iloc[tr_idx], y_dev[tr_idx]).astype(np.float32)
    X_va_tree = tree_pipe.transform(dev_df.iloc[va_idx]).astype(np.float32)
    tree_feat_names = [canonical_feature_name(c) for c in tree_pipe.named_steps["preprocessor"].get_feature_names_out()]

    lin_pipe = build_preprocessing_pipeline(tier, "linear", True, winsor_q, te_smooth, te_folds, seed)
    X_tr_lin = lin_pipe.fit_transform(dev_df.iloc[tr_idx], y_dev[tr_idx]).astype(np.float32)
    lin_feat_names = [canonical_feature_name(c) for c in lin_pipe.named_steps["preprocessor"].get_feature_names_out()]

    y_tr = y_dev[tr_idx]
    y_va = y_dev[va_idx]

    lgb_clf = build_model("lightgbm", best_params_doc["models"]["lightgbm"], seed, n_threads).fit(X_tr_tree, y_tr)
    xgb_clf = build_model("xgboost", best_params_doc["models"]["xgboost"], seed, n_threads).fit(X_tr_tree, y_tr)
    lr_clf = build_model("logistic_regression", best_params_doc["models"]["logistic_regression"], seed, n_threads).fit(X_tr_lin, y_tr)
    lin_tr_std = np.std(X_tr_lin, axis=0)
    del X_tr_tree, X_tr_lin
    gc.collect()

    # Stratified n=10,000 unseen validation sample from cv_fold == 4
    sss10 = StratifiedShuffleSplit(n_splits=1, train_size=10000, random_state=seed)
    exp_rel_idx, _ = next(sss10.split(X_va_tree, y_va))
    X_exp = X_va_tree[exp_rel_idx]
    y_exp = y_va[exp_rel_idx]

    # Stratified n=1,000 subsample for pairwise XGBoost TreeSHAP interactions
    sss1k = StratifiedShuffleSplit(n_splits=1, train_size=1000, random_state=seed)
    inter_rel_idx, _ = next(sss1k.split(X_exp, y_exp))
    X_inter = X_exp[inter_rel_idx]

    logger.info("=== Computing Exact Native C++ TreeSHAP (LightGBM & XGBoost) on n=10,000 ===")
    lgb_contrib = lgb_clf.predict(X_exp, pred_contrib=True)
    lgb_raw_margin = lgb_clf.predict(X_exp, raw_score=True)
    lgb_add_err = float(np.max(np.abs(lgb_contrib.sum(axis=1) - lgb_raw_margin)))
    assert lgb_add_err < 1e-5, f"LightGBM TreeSHAP additivity failed: {lgb_add_err}"
    lgb_shap = lgb_contrib[:, :-1]

    dm_exp = xgb.DMatrix(X_exp)
    xgb_contrib = xgb_clf.get_booster().predict(dm_exp, pred_contribs=True)
    xgb_raw_margin = xgb_clf.get_booster().predict(dm_exp, output_margin=True)
    xgb_add_err = float(np.max(np.abs(xgb_contrib.sum(axis=1) - xgb_raw_margin)))
    assert xgb_add_err < 1e-4, f"XGBoost TreeSHAP additivity failed: {xgb_add_err}"
    xgb_shap = xgb_contrib[:, :-1]

    logger.info("=== Computing Permutation Importance (ROC-AUC, 5 repeats) on exact same n=10,000 ===")
    perm = permutation_importance(lgb_clf, X_exp, y_exp, scoring="roc_auc", n_repeats=5, random_state=seed, n_jobs=1)

    # Extract gain/split & LR coefficients
    lgb_gain = lgb_clf.booster_.feature_importance(importance_type="gain")
    lgb_split = lgb_clf.booster_.feature_importance(importance_type="split")
    xgb_gain_dict = xgb_clf.get_booster().get_score(importance_type="gain")
    xgb_gain = np.array([xgb_gain_dict.get(f"f{i}", 0.0) for i in range(len(tree_feat_names))], dtype=np.float64)

    lr_coef_map = dict(zip(lin_feat_names, lr_clf.coef_.ravel().tolist()))
    lr_std_coef_map = dict(zip(lin_feat_names, (lr_clf.coef_.ravel() * lin_tr_std).tolist()))

    lgb_dirs = _compute_spearman_directions(X_exp, lgb_shap)
    xgb_dirs = _compute_spearman_directions(X_exp, xgb_shap)

    lgb_mean_abs = np.mean(np.abs(lgb_shap), axis=0)
    xgb_mean_abs = np.mean(np.abs(xgb_shap), axis=0)
    lgb_gain_norm = lgb_gain / np.sum(lgb_gain) if np.sum(lgb_gain) > 0 else lgb_gain

    feat_rows = []
    for j, fname in enumerate(tree_feat_names):
        lr_b = lr_coef_map.get(fname, np.nan)
        lr_sb = lr_std_coef_map.get(fname, np.nan)
        feat_rows.append(
            {
                "feature": fname,
                "domain_group": assign_domain_group(fname),
                "lgb_mean_abs_shap": round(float(lgb_mean_abs[j]), 6),
                "lgb_shap_std": round(float(np.std(lgb_shap[:, j])), 6),
                "lgb_spearman_dir": round(float(lgb_dirs[j]), 5),
                "xgb_mean_abs_shap": round(float(xgb_mean_abs[j]), 6),
                "xgb_spearman_dir": round(float(xgb_dirs[j]), 5),
                "lgb_perm_importance": round(float(perm.importances_mean[j]), 6),
                "lgb_perm_std": round(float(perm.importances_std[j]), 6),
                "lgb_gain_norm": round(float(lgb_gain_norm[j]), 6),
                "lgb_split_count": int(lgb_split[j]),
                "xgb_gain": round(float(xgb_gain[j]), 4),
                "lr_beta": round(float(lr_b), 6) if np.isfinite(lr_b) else np.nan,
                "lr_abs_beta": round(float(abs(lr_b)), 6) if np.isfinite(lr_b) else np.nan,
                "lr_std_beta": round(float(lr_sb), 6) if np.isfinite(lr_sb) else np.nan,
                "lr_abs_std_beta": round(float(abs(lr_sb)), 6) if np.isfinite(lr_sb) else np.nan,
            }
        )

    feat_df = pd.DataFrame(feat_rows).sort_values("lgb_mean_abs_shap", ascending=False).reset_index(drop=True)
    feat_df["lgb_shap_rank"] = np.arange(1, len(feat_df) + 1)
    feat_df.to_csv(results_dir / "m5_explainability_feature_ranking.csv", index=False)

    # Group SHAP breakdown & Cross-method agreement
    group_agg = (
        feat_df.groupby("domain_group", as_index=False)[["lgb_mean_abs_shap", "xgb_mean_abs_shap", "lgb_perm_importance"]]
        .sum()
    )
    group_agg["lgb_shap_share_pct"] = (group_agg["lgb_mean_abs_shap"] / group_agg["lgb_mean_abs_shap"].sum() * 100.0).round(2)
    group_agg["xgb_shap_share_pct"] = (group_agg["xgb_mean_abs_shap"] / group_agg["xgb_mean_abs_shap"].sum() * 100.0).round(2)

    idx_df = feat_df.set_index("feature")
    common_lr = idx_df.dropna(subset=["lr_abs_beta"])
    agreement_doc = {
        "n_explain_samples": int(len(X_exp)),
        "n_interaction_samples": int(len(X_inter)),
        "explain_split": "oot_dev_unseen_cv_fold_4",
        "lgb_treeshap_max_additivity_error": lgb_add_err,
        "xgb_treeshap_max_additivity_error": xgb_add_err,
        "cross_method_agreement": {
            "lgb_shap_vs_xgb_shap": {
                "spearman_rho": round(float(spearmanr(idx_df["lgb_mean_abs_shap"], idx_df["xgb_mean_abs_shap"]).statistic), 4),
                "top15_jaccard": round(_top_k_jaccard(idx_df["lgb_mean_abs_shap"], idx_df["xgb_mean_abs_shap"], 15), 4),
            },
            "lgb_shap_vs_perm_auc": {
                "spearman_rho": round(float(spearmanr(idx_df["lgb_mean_abs_shap"], idx_df["lgb_perm_importance"]).statistic), 4),
                "top15_jaccard": round(_top_k_jaccard(idx_df["lgb_mean_abs_shap"], idx_df["lgb_perm_importance"], 15), 4),
            },
            "lgb_shap_vs_lgb_gain": {
                "spearman_rho": round(float(spearmanr(idx_df["lgb_mean_abs_shap"], idx_df["lgb_gain_norm"]).statistic), 4),
                "top15_jaccard": round(_top_k_jaccard(idx_df["lgb_mean_abs_shap"], idx_df["lgb_gain_norm"], 15), 4),
            },
            "lgb_shap_vs_lr_abs_beta_common120": {
                "spearman_rho": round(float(spearmanr(common_lr["lgb_mean_abs_shap"], common_lr["lr_abs_beta"]).statistic), 4),
                "top15_jaccard": round(_top_k_jaccard(common_lr["lgb_mean_abs_shap"], common_lr["lr_abs_beta"], 15), 4),
            },
            "lgb_shap_vs_lr_abs_std_beta_common120": {
                "spearman_rho": round(float(spearmanr(common_lr["lgb_mean_abs_shap"], common_lr["lr_abs_std_beta"]).statistic), 4),
                "top15_jaccard": round(_top_k_jaccard(common_lr["lgb_mean_abs_shap"], common_lr["lr_abs_std_beta"], 15), 4),
            },
        },
        "domain_group_shares": group_agg.sort_values("lgb_shap_share_pct", ascending=False).to_dict(orient="records"),
        "cv_entity_cold_start_rate_pct": {
            c: round(float(np.mean(vals) * 100.0), 3) for c, vals in cold_start_rates.items()
        },
    }

    # Exact Pairwise XGBoost TreeSHAP Interactions on n=1,000
    logger.info("=== Computing Exact Pairwise XGBoost TreeSHAP Interactions (n=1,000) ===")
    dm_inter = xgb.DMatrix(X_inter)
    inter = xgb_clf.get_booster().predict(dm_inter, pred_interactions=True)
    inter_err = float(np.max(np.abs(inter.sum(axis=(1, 2)) - xgb_clf.get_booster().predict(dm_inter, output_margin=True))))
    assert inter_err < 1e-4, f"XGBoost SHAP interaction additivity failed: {inter_err}"
    agreement_doc["xgb_interaction_max_additivity_error"] = inter_err

    D = len(tree_feat_names)
    inter_feats = inter[:, :D, :D]
    mean_abs_inter = 2.0 * np.mean(np.abs(inter_feats), axis=0)
    tri_i, tri_j = np.triu_indices(D, k=1)
    pair_strengths = mean_abs_inter[tri_i, tri_j]
    top_pair_order = np.argsort(-pair_strengths)[:20]
    inter_rows = [
        {
            "rank": int(r + 1),
            "feature_1": tree_feat_names[tri_i[idx]],
            "feature_2": tree_feat_names[tri_j[idx]],
            "mean_abs_interaction_shap": round(float(pair_strengths[idx]), 6),
        }
        for r, idx in enumerate(top_pair_order)
    ]
    pd.DataFrame(inter_rows).to_csv(results_dir / "m5_shap_interactions.csv", index=False)
    del inter, inter_feats
    gc.collect()

    # 4 Individual Case Studies from unseen Fold 4
    logger.info("=== Extracting 4 Individual Local Case Studies (TP, TN, FN, FP) from unseen Fold 4 ===")
    va_probs = lgb_clf.predict_proba(X_va_tree)[:, 1]
    case_pos_map = _select_case_study_indices(y_va, va_probs, thresh=0.205)
    case_rel_positions = [case_pos_map[k] for k in ("TP", "TN", "FN", "FP")]
    case_contribs = lgb_clf.predict(X_va_tree[case_rel_positions], pred_contrib=True)

    cases_out: list[dict[str, Any]] = []
    raw_audit_cols = [
        "disbursed_amount",
        "asset_cost",
        "ltv",
        "PERFORM_CNS.SCORE",
        "Employment.Type",
        "PRI.NO.OF.ACCTS",
        "PRI.ACTIVE.ACCTS",
        "PRI.OVERDUE.ACCTS",
        "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS",
        "NO.OF_INQUIRIES",
        "State_ID",
        "branch_id",
        "supplier_id",
        "Current_pincode_ID",
    ]
    for idx_case, (c_type, rel_pos) in enumerate(zip(("TP", "TN", "FN", "FP"), case_rel_positions)):
        dev_pos = int(va_idx[rel_pos])
        phi = case_contribs[idx_case, :-1]
        base_val = float(case_contribs[idx_case, -1])
        margin = float(base_val + np.sum(phi))
        top_j = np.argsort(-np.abs(phi))[:10]
        raw_row = dev_df.iloc[dev_pos]
        cases_out.append(
            {
                "case_type": c_type,
                "dev_row_index": dev_pos,
                "actual_y": int(y_va[rel_pos]),
                "pred_prob": round(float(va_probs[rel_pos]), 6),
                "log_odds_margin": round(margin, 6),
                "base_value_log_odds": round(base_val, 6),
                "base_prob": round(float(expit(base_val)), 6),
                "raw_attributes": {c: (str(raw_row[c]) if isinstance(raw_row[c], str) or pd.isna(raw_row[c]) else float(raw_row[c])) for c in raw_audit_cols},
                "top_drivers": [
                    {
                        "feature": tree_feat_names[j],
                        "feature_value": round(float(X_va_tree[rel_pos, j]), 4),
                        "shap_value": round(float(phi[j]), 6),
                    }
                    for j in top_j
                ],
            }
        )

    (results_dir / "m5_local_case_studies.json").write_text(json.dumps(cases_out, indent=2, ensure_ascii=False), encoding="utf-8")
    (results_dir / "m5_explainability_agreement.json").write_text(json.dumps(agreement_doc, indent=2, ensure_ascii=False), encoding="utf-8")

    # Dev-Only Subgroup Slice Audit across full 182,501 Dev OOF predictions
    logger.info("=== Running Dev-Only Subgroup Slice Audit (Bureau Regime, Employment.Type, LTV Quartiles) ===")
    lgb_oof_full = opt_oof["lightgbm"]
    lr_oof_full = opt_oof["logistic_regression"]

    bureau_slice = np.where(dev_df["PERFORM_CNS.SCORE"] >= 300, "Scored (>=300)", "Thin-file / Unscored (<300)")
    emp_slice = dev_df["Employment.Type"].fillna("MISSING").astype(str).to_numpy()
    ltv_q = pd.qcut(dev_df["ltv"], q=4, labels=["LTV_Q1_Lowest", "LTV_Q2", "LTV_Q3", "LTV_Q4_Highest"]).astype(str).to_numpy()

    slice_defs = [
        ("Bureau Regime", bureau_slice),
        ("Employment Type", emp_slice),
        ("LTV Quartile", ltv_q),
    ]
    m5_thresh_path = results_dir / "m5_threshold_comparison.csv"
    if m5_thresh_path.exists():
        t_df = pd.read_csv(m5_thresh_path)
        lgb_f1_t = float(t_df.loc[(t_df["model"] == "lightgbm_stage2_natural") & (t_df["regime"] == "F1_Optimal"), "nested_thresh_mean"].iloc[0])
        lr_f1_t = float(t_df.loc[(t_df["model"] == "logistic_regression_s1") & (t_df["regime"] == "F1_Optimal"), "nested_thresh_mean"].iloc[0])
    else:
        lgb_f1_t, lr_f1_t = 0.205, 0.192

    sub_rows = []
    for dim_name, arr in slice_defs:
        for val in pd.unique(arr):
            mask = arr == val
            for m_label, p_arr, opt_t in [
                ("lightgbm_stage2", lgb_oof_full, lgb_f1_t),
                ("logistic_regression_s1", lr_oof_full, lr_f1_t),
            ]:
                m_sub = compute_binary_metrics(y_dev[mask], p_arr[mask], threshold=0.5, loan_amounts=disbursed_amt[mask], opt_threshold=opt_t)
                flag_rate_opt = float(np.mean(p_arr[mask] >= opt_t))
                sub_rows.append(
                    {
                        "slice_dimension": dim_name,
                        "slice_value": str(val),
                        "model": m_label,
                        "n": int(mask.sum()),
                        "default_rate": round(float(np.mean(y_dev[mask])), 6),
                        "roc_auc": m_sub["roc_auc"],
                        "pr_auc": m_sub["pr_auc"],
                        "brier_score": m_sub["brier_score"],
                        "ece": m_sub["ece"],
                        "recall_at_opt": m_sub["recall_opt"],
                        "precision_at_opt": m_sub["precision_opt"],
                        "flag_rate_at_opt": round(flag_rate_opt, 6),
                        "expected_cost_5to1_bayes": m_sub["expected_cost_5to1_bayes"],
                    }
                )
    pd.DataFrame(sub_rows).to_csv(results_dir / "m5_dev_subgroup_audit.csv", index=False)

    # Generate 4 Explainability Figures
    _generate_shap_global_and_beeswarm(feat_df, X_exp, lgb_shap, tree_feat_names, fig_dir / "exp_01_shap_global_and_beeswarm.png")
    _generate_dependence_figure(X_exp, lgb_shap, tree_feat_names, fig_dir / "exp_02_shap_dependence_interactions.png")
    _generate_agreement_figure(feat_df, group_agg, fig_dir / "exp_03_cross_model_agreement.png")
    _generate_local_waterfalls(cases_out, fig_dir / "exp_04_local_waterfall_cases.png")

    logger.info("[OK] Saved all explainability tables and 4 figures in %s", fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
