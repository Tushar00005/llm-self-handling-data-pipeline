"""
quality_checks.py
------------------
Deterministic, rule-based detection of common data-quality problems.

IMPORTANT: detection never modifies the DataFrame. It only produces a
list of Issue objects describing what is wrong. Correction happens
later, in cleaning_engine.py, after the LLM (for ambiguous cases) or a
deterministic rule (for clear-cut cases) has proposed a fix and that
fix has passed validation.
"""

import re
from typing import List

import numpy as np
import pandas as pd

from src.config import config
from src.logger import get_logger
from src.models import Issue

logger = get_logger(__name__)

EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
PHONE_DIGITS_REGEX = re.compile(r"\D")
TEXTUAL_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
}

EMAIL_COLUMN_HINTS = ("email", "e-mail", "mail")
PHONE_COLUMN_HINTS = ("phone", "mobile", "contact", "tel")
DATE_COLUMN_HINTS = ("date", "dob", "birth", "created", "updated", "timestamp")

KNOWN_DATE_FORMATS = [
    "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
    "%d %B %Y", "%B %d, %Y", "%d-%b-%Y", "%Y.%m.%d",
]


def _column_hint_match(col_name: str, hints) -> bool:
    lowered = col_name.lower()
    return any(hint in lowered for hint in hints)


def _looks_like_email_column(series: pd.Series, col_name: str) -> bool:
    if _column_hint_match(col_name, EMAIL_COLUMN_HINTS):
        return True
    sample = series.dropna().astype(str).head(20)
    if sample.empty:
        return False
    hits = sum(1 for v in sample if "@" in v)
    return hits / len(sample) > 0.3


def _looks_like_phone_column(series: pd.Series, col_name: str) -> bool:
    if _column_hint_match(col_name, PHONE_COLUMN_HINTS):
        return True
    return False


def _looks_like_date_column(series: pd.Series, col_name: str) -> bool:
    if _column_hint_match(col_name, DATE_COLUMN_HINTS):
        return True
    return False


def _try_parse_date(value: str):
    for fmt in KNOWN_DATE_FORMATS:
        try:
            return pd.to_datetime(value, format=fmt)
        except (ValueError, TypeError):
            continue
    try:
        return pd.to_datetime(value, errors="raise")
    except Exception:
        return None


def _detect_missing_values(df: pd.DataFrame) -> List[Issue]:
    issues = []
    for col in df.columns:
        missing_idx = df.index[df[col].isna()]
        for idx in missing_idx:
            issues.append(Issue(
                row_index=int(idx),
                column=col,
                issue_type="missing_value",
                original_value=None,
                description=f"Missing value in column '{col}'",
                requires_llm=True,  # the right fill strategy is context-dependent
            ))
    return issues


def _detect_duplicate_rows(df: pd.DataFrame) -> List[Issue]:
    issues = []
    dup_mask = df.duplicated(keep="first")
    for idx in df.index[dup_mask]:
        issues.append(Issue(
            row_index=int(idx),
            column="__row__",
            issue_type="duplicate_row",
            original_value=df.loc[idx].to_dict(),
            description="Row is an exact duplicate of an earlier row",
            requires_llm=False,
            suggested_action="drop_duplicate_row",
            suggested_value=None,
            confidence="high",
            source="rule",
        ))
    return issues


def _detect_whitespace_issues(df: pd.DataFrame) -> List[Issue]:
    issues = []
    for col in df.columns:
        if not pd.api.types.is_string_dtype(df[col]):
            continue
        for idx, val in df[col].items():
            if isinstance(val, str) and val != val.strip():
                issues.append(Issue(
                    row_index=int(idx),
                    column=col,
                    issue_type="whitespace",
                    original_value=val,
                    description=f"Leading/trailing whitespace in '{col}'",
                    requires_llm=False,
                    suggested_action="trim_whitespace",
                    suggested_value=val.strip(),
                    confidence="high",
                    source="rule",
                ))
    return issues


def _detect_capitalization_issues(df: pd.DataFrame) -> List[Issue]:
    """Flag categorical columns where the same value appears under
    different casing, e.g. 'Male', 'MALE', 'male'."""
    issues = []
    for col in df.columns:
        if not pd.api.types.is_string_dtype(df[col]):
            continue
        series = df[col].dropna().astype(str)
        if series.empty:
            continue
        # Only treat as "categorical" if cardinality is low relative to size
        if series.nunique() > max(20, len(series) * 0.5):
            continue
        groups = {}
        for val in series.unique():
            groups.setdefault(val.lower().strip(), []).append(val)
        for lower_val, variants in groups.items():
            if len(variants) <= 1:
                continue
            # canonical = most frequent variant
            counts = series.value_counts()
            canonical = max(variants, key=lambda v: counts.get(v, 0))
            for idx, val in df[col].items():
                if isinstance(val, str) and val in variants and val != canonical:
                    issues.append(Issue(
                        row_index=int(idx),
                        column=col,
                        issue_type="inconsistent_capitalization",
                        original_value=val,
                        description=f"Inconsistent casing in '{col}': '{val}' vs canonical '{canonical}'",
                        requires_llm=False,
                        suggested_action="normalize_capitalization",
                        suggested_value=canonical,
                        confidence="high",
                        source="rule",
                    ))
    return issues


def _detect_invalid_numeric_values(df: pd.DataFrame) -> List[Issue]:
    """Detect columns that are mostly numeric but contain non-numeric
    entries (e.g. 'twenty five', 'N/A', 'unknown')."""
    issues = []
    for col in df.columns:
        if not pd.api.types.is_string_dtype(df[col]):
            continue
        series = df[col].dropna().astype(str)
        if series.empty:
            continue
        numeric_like = pd.to_numeric(series, errors="coerce")
        numeric_ratio = numeric_like.notna().mean()
        if numeric_ratio < 0.6:
            continue  # column doesn't look predominantly numeric
        for idx, val in df[col].items():
            if pd.isna(val):
                continue
            val_str = str(val)
            if pd.to_numeric(val_str, errors="coerce") is not None and not pd.isna(
                pd.to_numeric(val_str, errors="coerce")
            ):
                continue
            words = re.findall(r"[a-zA-Z]+", val_str.lower())
            looks_textual_number = bool(words) and all(
                w in TEXTUAL_NUMBER_WORDS for w in words
            )
            issues.append(Issue(
                row_index=int(idx),
                column=col,
                issue_type="invalid_numeric_value",
                original_value=val,
                description=(
                    f"Non-numeric value '{val}' in predominantly numeric column '{col}'"
                    + (" (looks like a spelled-out number)" if looks_textual_number else "")
                ),
                requires_llm=True,  # interpreting the intended value needs reasoning
            ))
    return issues


def _detect_invalid_email_format(df: pd.DataFrame) -> List[Issue]:
    issues = []
    for col in df.columns:
        if not pd.api.types.is_string_dtype(df[col]):
            continue
        if not _looks_like_email_column(df[col], col):
            continue
        for idx, val in df[col].items():
            if pd.isna(val):
                continue
            val_str = str(val).strip()
            if val_str == "" or EMAIL_REGEX.match(val_str):
                continue
            issues.append(Issue(
                row_index=int(idx),
                column=col,
                issue_type="invalid_email",
                original_value=val,
                description=f"Value '{val}' in column '{col}' does not match a valid email format",
                requires_llm=True,  # guessing the intended address is inherently uncertain
            ))
    return issues


def _detect_invalid_phone_format(df: pd.DataFrame) -> List[Issue]:
    issues = []
    for col in df.columns:
        if not pd.api.types.is_string_dtype(df[col]):
            continue
        if not _looks_like_phone_column(df[col], col):
            continue
        for idx, val in df[col].items():
            if pd.isna(val):
                continue
            val_str = str(val).strip()
            digits = PHONE_DIGITS_REGEX.sub("", val_str)
            if 7 <= len(digits) <= 15:
                continue  # plausible phone number length
            issues.append(Issue(
                row_index=int(idx),
                column=col,
                issue_type="invalid_phone",
                original_value=val,
                description=f"Value '{val}' in column '{col}' does not look like a valid phone number",
                requires_llm=True,
            ))
    return issues


def _detect_inconsistent_date_formats(df: pd.DataFrame) -> List[Issue]:
    issues = []
    for col in df.columns:
        if not pd.api.types.is_string_dtype(df[col]):
            continue
        if not _looks_like_date_column(df[col], col):
            continue
        parsed_formats_seen = set()
        rows = []
        for idx, val in df[col].items():
            if pd.isna(val):
                continue
            val_str = str(val).strip()
            parsed = None
            matched_fmt = None
            for fmt in KNOWN_DATE_FORMATS:
                try:
                    parsed = pd.to_datetime(val_str, format=fmt)
                    matched_fmt = fmt
                    break
                except (ValueError, TypeError):
                    continue
            if parsed is None:
                issues.append(Issue(
                    row_index=int(idx),
                    column=col,
                    issue_type="invalid_date",
                    original_value=val,
                    description=f"Value '{val}' in column '{col}' could not be parsed as a date",
                    requires_llm=True,
                ))
            else:
                parsed_formats_seen.add(matched_fmt)
                rows.append((idx, val_str, matched_fmt, parsed))

        if len(parsed_formats_seen) > 1:
            # column mixes multiple date formats - normalize to ISO 8601
            majority_fmt = max(
                parsed_formats_seen,
                key=lambda f: sum(1 for r in rows if r[2] == f),
            )
            for idx, val_str, fmt, parsed in rows:
                if fmt != majority_fmt:
                    issues.append(Issue(
                        row_index=int(idx),
                        column=col,
                        issue_type="inconsistent_date_format",
                        original_value=val_str,
                        description=(
                            f"Date '{val_str}' in column '{col}' uses format '{fmt}' "
                            f"while the majority of the column uses '{majority_fmt}'"
                        ),
                        requires_llm=False,
                        suggested_action="normalize_date_to_iso8601",
                        suggested_value=parsed.strftime("%Y-%m-%d"),
                        confidence="high",
                        source="rule",
                    ))
    return issues


def _detect_incorrect_data_types(df: pd.DataFrame, dtype_hints: dict) -> List[Issue]:
    """Very lightweight structural check: a column that pandas loaded as
    object but whose non-null values are *all* cleanly numeric is
    probably supposed to be numeric (e.g. IDs stored as strings)."""
    issues = []
    for col in df.columns:
        if not pd.api.types.is_string_dtype(df[col]):
            continue
        series = df[col].dropna().astype(str)
        if series.empty:
            continue
        numeric_like = pd.to_numeric(series, errors="coerce")
        if numeric_like.notna().mean() == 1.0 and len(series) > 1:
            issues.append(Issue(
                row_index=-1,
                column=col,
                issue_type="incorrect_data_type",
                original_value=str(df[col].dtype),
                description=(
                    f"Column '{col}' is stored as text but every value is numeric; "
                    f"likely should be a numeric dtype"
                ),
                requires_llm=False,
                suggested_action="cast_column_numeric",
                suggested_value="numeric",
                confidence="medium",
                source="rule",
            ))
    return issues


def _detect_outliers(df: pd.DataFrame) -> List[Issue]:
    """IQR-based outlier detection for numeric columns. Outliers are
    flagged for manual review, never auto-corrected - a value being
    statistically unusual doesn't mean it's wrong."""
    issues = []
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        series = df[col].dropna()
        if len(series) < 5:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - config.OUTLIER_IQR_MULTIPLIER * iqr
        upper = q3 + config.OUTLIER_IQR_MULTIPLIER * iqr
        outlier_idx = series[(series < lower) | (series > upper)].index
        for idx in outlier_idx:
            issues.append(Issue(
                row_index=int(idx),
                column=col,
                issue_type="outlier",
                original_value=df.loc[idx, col],
                description=(
                    f"Value {df.loc[idx, col]} in '{col}' falls outside the "
                    f"expected range [{lower:.2f}, {upper:.2f}]"
                ),
                requires_llm=True,  # LLM assesses plausibility, doesn't just delete data
            ))
    return issues


def detect_issues(df: pd.DataFrame, dtype_hints: dict = None) -> List[Issue]:
    """Run every deterministic detector and return the combined issue list.
    Detection is read-only: df is never modified here."""
    dtype_hints = dtype_hints or {}
    logger.info("Starting data quality detection on %d rows", len(df))

    issues: List[Issue] = []
    detectors = [
        _detect_missing_values,
        _detect_duplicate_rows,
        _detect_whitespace_issues,
        _detect_capitalization_issues,
        _detect_invalid_numeric_values,
        _detect_invalid_email_format,
        _detect_invalid_phone_format,
        _detect_inconsistent_date_formats,
        _detect_outliers,
    ]
    for detector in detectors:
        found = detector(df)
        logger.info("%s -> %d issue(s)", detector.__name__, len(found))
        issues.extend(found)

    issues.extend(_detect_incorrect_data_types(df, dtype_hints))

    logger.info("Detection complete: %d total issue(s) found", len(issues))
    return issues
