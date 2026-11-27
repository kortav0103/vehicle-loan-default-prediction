"""Independent verification checks for Milestone 2 (Literature Review deliverables)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

EXPECTED_COLUMNS = [
    "paper_title",
    "authors",
    "year",
    "dataset_domain",
    "problem",
    "models",
    "evaluation_metrics",
    "main_findings",
    "limitations",
    "relevance_to_our_project",
    "verified_source_link",
]

FORBIDDEN_STRINGS = [
    "10.1016/j.eswa.2021.115906",
    "doi.0rg",
]

FORBIDDEN_MODEL_SUFFIXES = {
    ".pkl", ".pickle", ".joblib", ".pt", ".pth", ".cbm", ".xgb", ".lgb", ".h5", ".onnx", ".safetensors"
}


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    csv_path = root / "reports" / "tables" / "m2_literature_comparison.csv"
    lit_path = root / "docs" / "literature_review.md"
    rq_path = root / "docs" / "research_questions.md"

    errors: list[str] = []

    if not csv_path.exists():
        errors.append(f"Missing {csv_path}")
    if not lit_path.exists():
        errors.append(f"Missing {lit_path}")
    if not rq_path.exists():
        errors.append(f"Missing {rq_path}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != EXPECTED_COLUMNS:
            errors.append(f"CSV headers mismatch: {reader.fieldnames} != {EXPECTED_COLUMNS}")
        rows = list(reader)

    if len(rows) != 27:
        errors.append(f"Expected 27 verified papers in CSV, found {len(rows)}")

    titles: list[str] = []
    links: list[str] = []
    for idx, row in enumerate(rows, start=1):
        for col in EXPECTED_COLUMNS:
            val = row.get(col)
            if val is None or not str(val).strip():
                errors.append(f"Row {idx} has empty/None column '{col}'")
        if None in row:
            errors.append(f"Row {idx} has extra unaligned CSV columns: {row[None]}")
        link = str(row.get("verified_source_link", "")).strip()
        if not link.startswith("https://"):
            errors.append(f"Row {idx} invalid link: {link}")
        titles.append(str(row.get("paper_title", "")).strip())
        links.append(link)

    if len(set(titles)) != len(titles):
        errors.append("Duplicate paper titles detected in CSV")
    if len(set(links)) != len(links):
        errors.append("Duplicate source links detected in CSV")

    lit_text = lit_path.read_text(encoding="utf-8")
    rq_text = rq_path.read_text(encoding="utf-8")

    for bad in FORBIDDEN_STRINGS:
        if bad in lit_text or bad in rq_text or bad in csv_path.read_text(encoding="utf-8"):
            errors.append(f"Forbidden/corrupted string found: {bad}")

    for i in range(1, 8):
        if f"Chủ đề {i}:" not in lit_text:
            errors.append(f"Missing 'Chủ đề {i}:' in literature_review.md")

    for link in links:
        if link not in lit_text:
            errors.append(f"Source link from CSV missing in literature_review.md: {link}")

    for rq in ("## RQ1:", "## RQ2:", "## RQ3:", "## RQ4:"):
        if rq not in rq_text:
            errors.append(f"Missing section '{rq}' in research_questions.md")

    models_found = [
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if ".venv" not in p.parts and p.is_file() and p.suffix.lower() in FORBIDDEN_MODEL_SUFFIXES
    ]
    if models_found:
        errors.append(f"Forbidden trained model files found: {models_found}")

    summary = {
        "overall_status": "PASS" if not errors else "FAIL",
        "csv_rows": len(rows),
        "csv_columns": len(EXPECTED_COLUMNS),
        "unique_titles": len(set(titles)),
        "unique_links": len(set(links)),
        "themes_checked": 7,
        "rqs_checked": 4,
        "trained_models_found": len(models_found),
        "errors": errors,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
