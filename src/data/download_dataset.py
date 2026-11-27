"""Legitimate and reproducible download script for the LTFS Vehicle Loan Default dataset.

Downloads the public Kaggle dataset `mamtadhaker/lt-vehicle-loan-default-prediction`
into `data/raw/` without modifying any raw dataset contents, and writes provenance
and SHA-256 integrity metadata to `reports/audit/dataset_metadata.json`.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

DATASET_SLUG = "mamtadhaker/lt-vehicle-loan-default-prediction"
DATASET_PAGE_URL = f"https://www.kaggle.com/datasets/{DATASET_SLUG}"
KAGGLE_DOWNLOAD_API_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_SLUG}"
KAGGLE_VIEW_API_URL = f"https://www.kaggle.com/api/v1/datasets/view/{DATASET_SLUG}"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_public_kaggle_metadata() -> dict[str, Any]:
    """Attempt to query Kaggle's public dataset view API for provenance and license fields."""
    info: dict[str, Any] = {
        "endpoint": KAGGLE_VIEW_API_URL,
        "status_code": None,
        "accessible_without_auth": False,
        "payload": None,
        "error": None,
    }
    try:
        resp = requests.get(KAGGLE_VIEW_API_URL, timeout=30)
        info["status_code"] = resp.status_code
        if resp.status_code == 200:
            info["accessible_without_auth"] = True
            info["payload"] = resp.json()
        else:
            info["error"] = f"HTTP {resp.status_code}: {resp.text[:300]}"
    except Exception as exc:  # pragma: no cover - network dependent
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def download_via_kaggle_api_v1(cache_dir: Path, raw_dir: Path) -> dict[str, Any]:
    """Download public Kaggle dataset ZIP via API v1 and extract cleanly to raw_dir."""
    archive_path = cache_dir / "lt-vehicle-loan-default-prediction.zip"
    with requests.get(KAGGLE_DOWNLOAD_API_URL, stream=True, allow_redirects=True, timeout=180) as resp:
        status_code = resp.status_code
        if status_code != 200:
            raise RuntimeError(f"Kaggle API v1 returned HTTP {status_code}: {resp.text[:300]}")
        with archive_path.open("wb") as out:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out.write(chunk)

    if not zipfile.is_zipfile(archive_path):
        head = archive_path.read_bytes()[:250]
        archive_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is not a valid ZIP archive. Head bytes: {head!r}")

    archive_sha256 = sha256_file(archive_path)
    archive_size = archive_path.stat().st_size

    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(raw_dir)

    return {
        "method": "kaggle_public_api_v1",
        "download_url": KAGGLE_DOWNLOAD_API_URL,
        "archive_cache_path": str(archive_path),
        "archive_size_bytes": archive_size,
        "archive_sha256": archive_sha256,
    }


def download_via_kagglehub(raw_dir: Path) -> dict[str, Any]:
    """Fallback download via official kagglehub library, copying files byte-for-byte to raw_dir."""
    import kagglehub

    cached_path = Path(kagglehub.dataset_download(DATASET_SLUG))
    copied_files: list[str] = []
    for item in sorted(cached_path.rglob("*")):
        if item.is_file():
            rel = item.relative_to(cached_path)
            dest = raw_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)
            copied_files.append(str(rel))

    return {
        "method": "kagglehub_dataset_download",
        "download_url": DATASET_PAGE_URL,
        "kagglehub_cache_path": str(cached_path),
        "copied_relative_paths": copied_files,
    }


def collect_raw_inventory(raw_dir: Path, project_root: Path) -> list[dict[str, Any]]:
    """Collect immutable file inventory and SHA-256 hashes for all files in raw_dir."""
    entries: list[dict[str, Any]] = []
    for file_path in sorted(p for p in raw_dir.rglob("*") if p.is_file()):
        size_bytes = file_path.stat().st_size
        entries.append(
            {
                "file_name": file_path.name,
                "relative_path": file_path.relative_to(project_root).as_posix(),
                "format": file_path.suffix.lower().lstrip("."),
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / (1024 * 1024), 4),
                "sha256": sha256_file(file_path),
            }
        )
    return entries


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / "data" / "raw"
    cache_dir = project_root / "data" / "interim" / "_download_cache"
    audit_dir = project_root / "reports" / "audit"
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    existing_raw_files = [p for p in raw_dir.rglob("*") if p.is_file()]
    download_details: dict[str, Any]

    if existing_raw_files:
        print(f"[INFO] Found {len(existing_raw_files)} existing file(s) in {raw_dir}. Skipping download to preserve raw files.")
        download_details = {
            "method": "existing_raw_files_preserved",
            "download_url": DATASET_PAGE_URL,
        }
    else:
        errors: list[str] = []
        try:
            print(f"[INFO] Attempting public Kaggle API v1 download for {DATASET_SLUG}...")
            download_details = download_via_kaggle_api_v1(cache_dir, raw_dir)
        except Exception as exc1:
            errors.append(f"API v1 error: {exc1}")
            print(f"[WARN] Kaggle API v1 failed: {exc1}")
            print(f"[INFO] Falling back to kagglehub.dataset_download('{DATASET_SLUG}')...")
            try:
                download_details = download_via_kagglehub(raw_dir)
            except Exception as exc2:
                errors.append(f"kagglehub error: {exc2}")
                print("[ERROR] Both legitimate download methods failed.", file=sys.stderr)
                for err in errors:
                    print(f"  - {err}", file=sys.stderr)
                return 1

    raw_inventory = collect_raw_inventory(raw_dir, project_root)
    if not raw_inventory:
        print("[ERROR] data/raw is empty after download attempt.", file=sys.stderr)
        return 1

    kaggle_view_meta = fetch_public_kaggle_metadata()

    metadata = {
        "dataset_slug": DATASET_SLUG,
        "dataset_page_url": DATASET_PAGE_URL,
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "download_details": download_details,
        "kaggle_public_view_metadata": kaggle_view_meta,
        "raw_files": raw_inventory,
    }

    metadata_path = audit_dir / "dataset_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] Saved provenance metadata to {metadata_path}")
    for entry in raw_inventory:
        print(f"  - {entry['relative_path']} ({entry['size_mb']} MB) sha256={entry['sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
