"""Helpers for pushing result files to a Kaggle dataset.

Shells out to the `kaggle` CLI rather than depending on the `kaggle` python
package directly, since the CLI already handles credential discovery
(~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY env vars) and versioning
semantics we'd otherwise have to reimplement.
"""

import json
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _require_kaggle_cli() -> None:
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "kaggle CLI not found on PATH -- install with `pip install kaggle` and "
            "place credentials at ~/.kaggle/kaggle.json (or set KAGGLE_USERNAME/KAGGLE_KEY)"
        )


def ensure_dataset_metadata(staging_dir: Path, dataset_slug: str, title: str) -> Path:
    """Writes the dataset-metadata.json Kaggle's CLI needs to create/version a dataset.

    dataset_slug: "<kaggle-username>/<dataset-name>".
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = staging_dir / "dataset-metadata.json"
    metadata_path.write_text(
        json.dumps({"title": title, "id": dataset_slug, "licenses": [{"name": "CC0-1.0"}]}, indent=2)
    )
    return metadata_path


def upload_dataset(
    staging_dir: Path,
    dataset_slug: str,
    title: str,
    version_notes: str = "checkpoint update",
) -> None:
    """Creates the dataset on first call, versions it on subsequent calls.

    staging_dir must contain the files to upload (plus the dataset-metadata.json
    written by ensure_dataset_metadata). Determines create vs. version by asking
    the Kaggle API whether the dataset slug already exists.
    """
    _require_kaggle_cli()
    ensure_dataset_metadata(staging_dir, dataset_slug, title)

    status = subprocess.run(
        ["kaggle", "datasets", "status", dataset_slug],
        capture_output=True,
        text=True,
    )
    dataset_exists = status.returncode == 0

    if dataset_exists:
        cmd = ["kaggle", "datasets", "version", "-p", str(staging_dir), "-m", version_notes, "-r", "zip"]
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(staging_dir), "-r", "zip"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"kaggle CLI failed ({' '.join(cmd)}):\n{result.stdout}\n{result.stderr}")
    logger.info("kaggle upload ok (%s): %s", "version" if dataset_exists else "create", result.stdout.strip())


def sync_file_to_dataset(
    file_path: Path,
    dataset_slug: str,
    title: str,
    staging_dir: Path | None = None,
    version_notes: str = "checkpoint update",
) -> None:
    """Copies a single result file into a staging dir and uploads it as a Kaggle dataset.

    Convenience wrapper for the common case (one CSV, no other artifacts) --
    for multi-file datasets, copy files into staging_dir yourself and call
    upload_dataset directly.
    """
    staging_dir = staging_dir or file_path.parent / "kaggle_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file_path, staging_dir / file_path.name)
    upload_dataset(staging_dir, dataset_slug, title, version_notes)
