"""Independent audit verification script (Milestone 1).

Uses only Python standard-library modules (`csv`, `hashlib`, `json`, `pathlib`)
to independently verify raw file immutability, row/column counts, target class
counts, table column coverage, chart integrity, and absence of trained models.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

EXPECTED_FIGURES = [
    "m1_target_distribution.png",
    "m1_missing_values.png",
    "m1_bureau_score_audit.png",
    "m1_numeric_distributions_outliers.png",
    "m1_high_cardinality_drift.png",
]

FORBIDDEN_MODEL_SUFFIXES = {
    ".pkl", ".pickle", ".joblib", ".pt", ".pth", ".cbm", ".xgb", ".lgb", ".h5", ".onnx", ".safetensors"
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv_stats(path: Path, target_col: str | None = None) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        row_count = 0
        target_counts: dict[str, int] = {}
        target_idx = header.index(target_col) if (target_col and target_col in header) else None
        for row in reader:
            row_count += 1
            if target_idx is not None:
                val = row[target_idx]
                target_counts[val] = target_counts.get(val, 0) + 1
    return {
        "header": header,
        "n_cols": len(header),
        "n_rows": row_count,
        "target_counts": target_counts,
    }


def read_table_column_values(path: Path, col_name: str) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [row[col_name] for row in reader]


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    raw_dir = root / "data" / "raw"
    audit_dir = root / "reports" / "audit"
    tables_dir = root / "reports" / "tables"
    figures_dir = root / "reports" / "figures"

    metadata = json.loads((audit_dir / "dataset_metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((audit_dir / "audit_summary.json").read_text(encoding="utf-8"))

    checks: list[dict[str, Any]] = []

    # 1. Verify raw file hashes & sizes match metadata exactly
    hash_mismatches: list[str] = []
    for entry in metadata["raw_files"]:
        fp = root / entry["relative_path"]
        if not fp.exists():
            hash_mismatches.append(f"Missing: {entry['relative_path']}")
            continue
        actual_size = fp.stat().st_size
        actual_hash = sha256_file(fp)
        if actual_size != entry["size_bytes"] or actual_hash != entry["sha256"]:
            hash_mismatches.append(entry["relative_path"])
    checks.append(
        {
            "check": "raw_files_immutable_sha256_and_size",
            "passed": len(hash_mismatches) == 0,
            "details": {"checked_files": len(metadata["raw_files"]), "mismatches": hash_mismatches},
        }
    )

    # 2. Independent raw CSV row/col/target counts
    train_stats = read_csv_stats(raw_dir / summary["inventory"]["train_file"], target_col=summary["target_verification"]["target_column"])
    test_stats = read_csv_stats(raw_dir / summary["inventory"]["test_file"])

    counts_match = (
        train_stats["n_rows"] == summary["inventory"]["train_rows"]
        and train_stats["n_cols"] == summary["inventory"]["train_cols"]
        and test_stats["n_rows"] == summary["inventory"]["test_rows"]
        and test_stats["n_cols"] == summary["inventory"]["test_cols"]
    )
    checks.append(
        {
            "check": "csv_dimensions_match_audit_summary",
            "passed": counts_match,
            "details": {
                "raw_train": {"rows": train_stats["n_rows"], "cols": train_stats["n_cols"]},
                "summary_train": {"rows": summary["inventory"]["train_rows"], "cols": summary["inventory"]["train_cols"]},
                "raw_test": {"rows": test_stats["n_rows"], "cols": test_stats["n_cols"]},
                "summary_test": {"rows": summary["inventory"]["test_rows"], "cols": summary["inventory"]["test_cols"]},
            },
        }
    )

    # 3. Independent target count check
    raw_0 = train_stats["target_counts"].get("0", 0)
    raw_1 = train_stats["target_counts"].get("1", 0)
    target_match = (
        raw_0 == summary["target_verification"]["class_0_count"]
        and raw_1 == summary["target_verification"]["class_1_count"]
        and (raw_0 + raw_1) == train_stats["n_rows"]
        and set(train_stats["target_counts"].keys()) == {"0", "1"}
    )
    checks.append(
        {
            "check": "target_counts_exact_match",
            "passed": target_match,
            "details": {"raw_target_counts": train_stats["target_counts"]},
        }
    )

    # 4. Exact 100% column coverage across audit CSV tables
    train_header_set = set(train_stats["header"])
    table_coverage: dict[str, Any] = {}
    all_tables_match = True
    for table_name in ("m1_column_profile.csv", "m1_leakage_audit.csv", "m1_feature_classification.csv"):
        cols = read_table_column_values(tables_dir / table_name, "column")
        col_set = set(cols)
        match = (len(cols) == len(train_stats["header"])) and (col_set == train_header_set)
        if not match:
            all_tables_match = False
        table_coverage[table_name] = {
            "row_count": len(cols),
            "missing_from_table": sorted(train_header_set - col_set),
            "extra_in_table": sorted(col_set - train_header_set),
            "exact_match": match,
        }
    checks.append(
        {
            "check": "audit_tables_cover_100pct_columns",
            "passed": all_tables_match,
            "details": table_coverage,
        }
    )

    # 5. Verify PNG charts exist, non-empty, and valid PNG signature
    png_details: list[dict[str, Any]] = []
    all_pngs_ok = True
    for fig_name in EXPECTED_FIGURES:
        fp = figures_dir / fig_name
        exists = fp.exists()
        size = fp.stat().st_size if exists else 0
        magic_ok = fp.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" if exists and size >= 8 else False
        ok = exists and size > 10_000 and magic_ok
        if not ok:
            all_pngs_ok = False
        png_details.append({"file": fig_name, "exists": exists, "size_bytes": size, "valid_png_magic": magic_ok, "ok": ok})
    checks.append(
        {
            "check": "diagnostic_figures_valid_png",
            "passed": all_pngs_ok,
            "details": png_details,
        }
    )

    # 6. Verify no trained model files exist anywhere in workspace (excluding .venv)
    found_models: list[str] = []
    for p in root.rglob("*"):
        if ".venv" in p.parts:
            continue
        if p.is_file() and p.suffix.lower() in FORBIDDEN_MODEL_SUFFIXES:
            found_models.append(p.relative_to(root).as_posix())
    checks.append(
        {
            "check": "no_trained_models_created",
            "passed": len(found_models) == 0,
            "details": {"forbidden_model_files_found": found_models},
        }
    )

    overall_passed = all(c["passed"] for c in checks)
    report = {
        "overall_status": "PASS" if overall_passed else "FAIL",
        "checks": checks,
    }

    out_path = audit_dir / "verification_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{'PASS' if overall_passed else 'FAIL'}] Verification report saved to {out_path}")
    for c in checks:
        print(f"  - {c['check']}: {'PASS' if c['passed'] else 'FAIL'}")

    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
