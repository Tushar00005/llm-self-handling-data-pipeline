"""
profiling.py
------------
Generates a structured statistical profile of a DataFrame:
row/column counts, dtypes, missing values, duplicates, uniqueness,
numeric statistics and categorical frequency distributions.

Profiling is purely observational - it never modifies the data.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


def _numeric_stats(series: pd.Series) -> Dict[str, Any]:
    clean = series.dropna()
    if clean.empty:
        return {}
    return {
        "min": float(clean.min()),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std()) if len(clean) > 1 else 0.0,
        "q1": float(clean.quantile(0.25)),
        "q3": float(clean.quantile(0.75)),
    }


def _categorical_frequencies(series: pd.Series, top_n: int = 10) -> Dict[str, int]:
    counts = series.dropna().astype(str).value_counts().head(top_n)
    return {str(k): int(v) for k, v in counts.items()}


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Build a profile dict describing the dataset's shape and content."""
    logger.info("Profiling dataset with %d rows, %d columns", df.shape[0], df.shape[1])

    profile: Dict[str, Any] = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "duplicate_row_count": int(df.duplicated().sum()),
        "columns": {},
    }

    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        missing = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))

        col_profile: Dict[str, Any] = {
            "dtype": dtype,
            "missing_count": missing,
            "missing_pct": round((missing / len(df)) * 100, 2) if len(df) else 0.0,
            "unique_count": unique,
        }

        if pd.api.types.is_numeric_dtype(series):
            col_profile["numeric_stats"] = _numeric_stats(series)
        else:
            col_profile["top_values"] = _categorical_frequencies(series)

        profile["columns"][col] = col_profile

    logger.info(
        "Profiling complete: %d duplicate rows, %d columns profiled",
        profile["duplicate_row_count"], len(profile["columns"]),
    )
    return profile
