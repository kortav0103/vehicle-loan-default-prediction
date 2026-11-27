"""Reproducible CLI orchestrator and figure generator for Milestone 4 Benchmark."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_m4_figures() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "axes.titlesize": 11, "axes.titleweight": "bold"})

    results_dir = ROOT / "experiments" / "results"
    fig_dir = ROOT / "experiments" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    base_df = pd.read_csv(results_dir / "m4_baseline_comparison.csv")
    tune_df = pd.read_csv(results_dir / "m4_tuning_comparison.csv")
    abl_df = pd.read_csv(results_dir / "m4_feature_ablation.csv")
    imb_df = pd.read_csv(results_dir / "m4_imbalance_calibration.csv")

    # -------------------------------------------------------------------------
    # Figure 1: Model Benchmark Comparison (Baseline vs. Tuned ROC-AUC & PR-AUC)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))

    base_plot = base_df.sort_values("cv_roc_auc_mean", ascending=True)
    y_pos = np.arange(len(base_plot))
    axes[0].barh(
        y_pos,
        base_plot["cv_roc_auc_mean"],
        xerr=base_plot["cv_roc_auc_std"],
        capsize=3,
        color="#2b6cb0",
        alpha=0.88,
        edgecolor="black",
        linewidth=0.5,
        label="Baseline (5-Fold CV Mean +/- SD)",
    )
    tuned_map_auc = dict(zip(tune_df["model"], tune_df["tuned_cv_auc"]))
    tuned_auc_labeled = False
    for i, m_name in enumerate(base_plot["model"]):
        b_val = float(base_plot.iloc[i]["cv_roc_auc_mean"])
        std_val = float(base_plot.iloc[i]["cv_roc_auc_std"])
        if m_name in tuned_map_auc:
            t_val = float(tuned_map_auc[m_name])
            lbl = "Tuned (5-Fold CV)" if not tuned_auc_labeled else None
            tuned_auc_labeled = True
            axes[0].scatter([t_val], [i], color="#e53e3e", s=55, zorder=5, label=lbl)
            x_txt = max(b_val + std_val, t_val) + 0.004
            axes[0].text(x_txt, i, f"{b_val:.4f} → {t_val:.4f}", va="center", fontsize=8)
        else:
            axes[0].text(b_val + std_val + 0.004, i, f"{b_val:.4f}", va="center", fontsize=8)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(base_plot["model"])
    axes[0].set_xlabel("5-Fold Dev CV ROC-AUC")
    axes[0].set_title("Model Benchmark: Baseline vs. Tuned ROC-AUC (Dev CV)")
    axes[0].set_xlim(0.48, max(base_plot["cv_roc_auc_mean"].max(), max(tuned_map_auc.values())) + 0.085)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(), loc="lower right", fontsize=8)

    tuned_map_pr = dict(zip(tune_df["model"], tune_df["pr_auc"]))
    axes[1].barh(y_pos, base_plot["pr_auc"], color="#38a169", alpha=0.85, edgecolor="black", linewidth=0.5, label="Baseline OOF PR-AUC")
    tuned_pr_labeled = False
    for i, m_name in enumerate(base_plot["model"]):
        pr_val = float(base_plot.iloc[i]["pr_auc"])
        if m_name in tuned_map_pr:
            t_pr = float(tuned_map_pr[m_name])
            lbl = "Tuned OOF PR-AUC" if not tuned_pr_labeled else None
            tuned_pr_labeled = True
            axes[1].scatter([t_pr], [i], color="#e53e3e", s=55, zorder=5, label=lbl)
            x_txt = max(pr_val, t_pr) + 0.004
            axes[1].text(x_txt, i, f"{pr_val:.4f} → {t_pr:.4f}", va="center", fontsize=8)
        else:
            axes[1].text(pr_val + 0.004, i, f"{pr_val:.4f}", va="center", fontsize=8)
    axes[1].axvline(0.2063, color="#4a5568", linestyle="--", linewidth=1.0, label="Dev Positive Baseline (0.2063)")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(base_plot["model"])
    axes[1].set_xlabel("OOF PR-AUC (Average Precision)")
    axes[1].set_title("Model Benchmark: PR-AUC Comparison (Dev CV)")
    axes[1].set_xlim(0.18, max(base_plot["pr_auc"].max(), max(tuned_map_pr.values())) + 0.075)
    h2, l2 = axes[1].get_legend_handles_labels()
    by_l2 = dict(zip(l2, h2))
    axes[1].legend(by_l2.values(), by_l2.keys(), loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(fig_dir / "m4_model_comparison_cv.png", bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 2: Feature Tier Ablation (5-Fold CV vs. Dev Expanding Temporal)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2))
    tier_order = ["F0", "F1", "F2_safe", "F2_plus_emp"]

    cv_abl = abl_df[abl_df["split_protocol"] == "oot_dev_5fold_cv"].copy()
    for m_name, grp in cv_abl.groupby("model"):
        grp_s = grp.set_index("feature_tier").reindex(tier_order)
        axes[0].plot(tier_order, grp_s["roc_auc"], marker="o", linewidth=2.0, label=m_name)
    axes[0].set_title("RQ2 Feature Ablation on 5-Fold Stratified Dev CV")
    axes[0].set_xlabel("Feature Engineering Tier")
    axes[0].set_ylabel("CV ROC-AUC")
    axes[0].legend(fontsize=8.5)

    tmp_abl = abl_df[abl_df["split_protocol"] == "dev_temporal_block_0..2_to_3"].copy()
    for m_name, grp in tmp_abl.groupby("model"):
        grp_s = grp.set_index("feature_tier").reindex(tier_order)
        axes[1].plot(tier_order, grp_s["roc_auc"], marker="s", linewidth=2.0, label=f"{m_name} (Dev Block 3 Temporal)")
    axes[1].set_title("RQ2 Feature Ablation on Future Dev Temporal Block 3")
    axes[1].set_xlabel("Feature Engineering Tier")
    axes[1].set_ylabel("Temporal Block 3 ROC-AUC")
    axes[1].legend(fontsize=8.5)

    fig.tight_layout()
    fig.savefig(fig_dir / "m4_feature_ablation.png", bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 3: Class Imbalance & Calibration Trade-offs (ROC-AUC vs ECE / Brier)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))

    # Pick the continuous GBDT model evaluated in Track D (lightgbm or xgboost)
    cont_models = [m for m in imb_df["model"].unique() if m in {"lightgbm", "xgboost"}]
    focus_gbdt = cont_models[0] if cont_models else imb_df["model"].unique()[0]
    sub_gbdt = imb_df[imb_df["model"] == focus_gbdt].copy()

    strat_order = ["natural", "class_weight", "rus_0.5", "smote_0.5", "smote_1.0"]
    sub_uncal = sub_gbdt[sub_gbdt["calibration"] == "uncalibrated"].set_index("imbalance_strategy").reindex(strat_order)

    x_s = np.arange(len(strat_order))
    b_auc = axes[0].bar(x_s, sub_uncal["roc_auc"], width=0.52, color="#2b6cb0", alpha=0.9)
    b_ece = axes[1].bar(x_s, sub_uncal["ece"], width=0.52, color="#c53030", alpha=0.85)
    max_ece = float(sub_uncal["ece"].max())
    for bar, v in zip(b_auc, sub_uncal["roc_auc"]):
        if np.isfinite(v):
            axes[0].text(bar.get_x() + bar.get_width() / 2, float(v) + 0.001, f"{float(v):.4f}", ha="center", va="bottom", fontsize=8)
    for bar, v in zip(b_ece, sub_uncal["ece"]):
        if np.isfinite(v):
            axes[1].text(bar.get_x() + bar.get_width() / 2, float(v) + max_ece * 0.02, f"{float(v):.4f}", ha="center", va="bottom", fontsize=8)

    axes[0].set_xticks(x_s)
    axes[0].set_xticklabels(strat_order, rotation=15)
    axes[0].set_ylabel("OOF ROC-AUC")
    axes[0].set_title(f"{focus_gbdt}: Discrimination (ROC-AUC) across Imbalance Strategies")
    axes[0].set_ylim(max(0.55, float(sub_uncal["roc_auc"].min()) - 0.015), float(sub_uncal["roc_auc"].max()) + 0.012)

    axes[1].set_xticks(x_s)
    axes[1].set_xticklabels(strat_order, rotation=15)
    axes[1].set_ylabel("Expected Calibration Error (ECE, lower is better)")
    axes[1].set_title(f"{focus_gbdt}: Calibration Error (ECE) across Imbalance Strategies")
    axes[1].set_ylim(0.0, max_ece * 1.18)

    fig.tight_layout()
    fig.savefig(fig_dir / "m4_imbalance_and_calibration.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Generated 3 Milestone 4 figures in {fig_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Milestone 4 Model Benchmark Pipeline")
    parser.add_argument("--tracks", default="all", choices=["all", "baseline", "tune", "imbalance", "plots"])
    args = parser.parse_args()

    if args.tracks in {"all", "baseline"}:
        from src.models.train_benchmark import main as run_baseline
        assert run_baseline() == 0
    if args.tracks in {"all", "tune"}:
        from src.models.tune_models import main as run_tune
        assert run_tune() == 0
    if args.tracks in {"all", "imbalance"}:
        from src.models.imbalance_calibration import main as run_imb
        assert run_imb() == 0
    if args.tracks in {"all", "plots"}:
        generate_m4_figures()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
