"""
ingestion.py
------------
Loads a raw CSV or Excel file into a pandas DataFrame, records basic
schema information, and preserves an untouched copy of the raw data so
every later stage can be compared back against the original.
"""

import hashlib
import os
import shutil
from dataclasses import dataclass, field
from typing import Dict

import pandas as pd

from src.config import config
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    dataframe: pd.DataFrame          # working copy, cleaning will mutate this
    raw_dataframe: pd.DataFrame      # frozen original, never mutated
    source_path: str
    dataset_name: str
    row_count: int
    column_count: int
    dtypes: Dict[str, str]
    dataset_hash: str                # sha256 of raw file bytes, used for idempotency
    archived_raw_path: str = field(default="")


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def _hash_file(path: str) -> str:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _archive_raw_file(path: str) -> str:
    """Copy the raw input into data/raw/ so it is preserved untouched."""
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    filename = os.path.basename(path)
    dest = os.path.join(config.RAW_DATA_DIR, filename)
    if os.path.abspath(path) != os.path.abspath(dest):
        shutil.copy2(path, dest)
    return dest


def load_dataset(path: str) -> IngestionResult:
    """
    Load a CSV or Excel file and return an IngestionResult containing:
      - a working DataFrame (df) that later stages are allowed to clean
      - a frozen raw_dataframe copy that is never modified
      - basic schema metadata (row/column counts, dtypes)
      - a content hash used to keep MySQL loads idempotent across re-runs
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input dataset not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported types: {SUPPORTED_EXTENSIONS}"
        )

    logger.info("Ingesting dataset: %s", path)

    dataset_hash = _hash_file(path)
    archived_path = _archive_raw_file(path)

    if ext == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    raw_df = df.copy(deep=True)

    dataset_name = os.path.splitext(os.path.basename(path))[0]
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

    logger.info(
        "Loaded dataset '%s' with %d rows and %d columns",
        dataset_name, df.shape[0], df.shape[1],
    )
    logger.debug("Detected dtypes: %s", dtypes)

    return IngestionResult(
        dataframe=df,
        raw_dataframe=raw_df,
        source_path=path,
        dataset_name=dataset_name,
        row_count=df.shape[0],
        column_count=df.shape[1],
        dtypes=dtypes,
        dataset_hash=dataset_hash,
        archived_raw_path=archived_path,
    )
