"""Structured JSON + CSV experiment tracker and environment snapshot logger."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

LOG_COLUMNS = [
    "experiment_id",
    "timestamp_utc",
    "track",
    "model",
    "feature_tier",
    "preprocessing",
    "dataset_split",
    "random_seed",
    "training_duration_sec",
    "status",
    "cv_roc_auc_mean",
    "cv_roc_auc_std",
    "roc_auc",
    "pr_auc",
    "ks_stat",
    "brier_score",
    "log_loss",
    "ece",
    "recall_at_top10pct",
    "recall_at_top20pct",
    "accuracy_05",
    "precision_05",
    "recall_05",
    "f1_05",
    "tn_05",
    "fp_05",
    "fn_05",
    "tp_05",
    "f1_opt_thresh",
    "accuracy_opt",
    "precision_opt",
    "recall_opt",
    "f1_opt",
    "expected_cost_5to1_bayes",
    "expected_cost_amount_weighted_bayes",
    "best_parameters_json",
    "notes",
]


class ExperimentTracker:
    """Persist structured experiment records into JSON files and a consolidated CSV log."""

    def __init__(self, results_dir: Path | str) -> None:
        self.results_dir = Path(results_dir)
        self.runs_dir = self.results_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv = self.results_dir / "experiment_log.csv"

    def log_experiment(
        self,
        experiment_id: str,
        track: str,
        model: str,
        feature_tier: str,
        preprocessing: str,
        dataset_split: str,
        random_seed: int,
        training_duration_sec: float,
        status: str,
        metrics: dict[str, Any],
        best_parameters: dict[str, Any] | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        ts = datetime.now(timezone.utc).isoformat()
        record: dict[str, Any] = {
            "experiment_id": experiment_id,
            "timestamp_utc": ts,
            "track": track,
            "model": model,
            "feature_tier": feature_tier,
            "preprocessing": preprocessing,
            "dataset_split": dataset_split,
            "random_seed": int(random_seed),
            "training_duration_sec": round(float(training_duration_sec), 3),
            "status": status,
            "best_parameters": best_parameters or {},
            "metrics": metrics,
            "notes": notes,
        }

        run_path = self.runs_dir / f"{experiment_id}.json"
        run_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

        csv_row: dict[str, Any] = {
            "experiment_id": experiment_id,
            "timestamp_utc": ts,
            "track": track,
            "model": model,
            "feature_tier": feature_tier,
            "preprocessing": preprocessing,
            "dataset_split": dataset_split,
            "random_seed": int(random_seed),
            "training_duration_sec": round(float(training_duration_sec), 3),
            "status": status,
            "best_parameters_json": json.dumps(best_parameters or {}, ensure_ascii=False),
            "notes": notes,
        }
        for col in LOG_COLUMNS:
            if col not in csv_row:
                csv_row[col] = metrics.get(col, "")

        # Rewrite or upsert by experiment_id so reruns never create duplicate stale rows
        existing_rows: list[dict[str, Any]] = []
        if self.log_csv.exists():
            with self.log_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get("experiment_id") != experiment_id:
                        existing_rows.append(r)

        existing_rows.append({col: csv_row.get(col, "") for col in LOG_COLUMNS})
        with self.log_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writeheader()
            writer.writerows(existing_rows)

        return record


def save_environment_snapshot(results_dir: Path | str, random_seed: int = 42) -> Path:
    """Record Python, OS, CPU/RAM, and modeling library versions for reproducibility."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    packages = [
        "numpy", "pandas", "scipy", "scikit-learn", "xgboost", "lightgbm",
        "imbalanced-learn", "optuna", "matplotlib", "seaborn", "pyyaml",
    ]
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "NOT_INSTALLED"

    snapshot = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_logical_count": os.cpu_count(),
        "total_ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "packages": versions,
    }
    path = out_dir / "m4_environment_versions.json"
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
