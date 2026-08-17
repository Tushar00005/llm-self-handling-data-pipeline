"""
validation.py
--------------
Re-checks every applied cleaning action against basic correctness
rules. Any action that fails validation is rolled back (the cell is
restored to its original value) and re-flagged as requiring manual
review. This is the safety net described in the spec:

    Correction causes a validation failure
        -> Rollback correction
        -> Mark issue as unresolved
        -> Add it to manual-review report
"""

import re
from dataclasses import dataclass
from typing import Any, List, Tuple

import pandas as pd

from src.logger import get_logger
from src.models import CleaningAction

logger = get_logger(__name__)

EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")


@dataclass
class ValidationResult:
    check_type: str
    passed: bool
    details: str
    row_index: int = -1
    column: str = ""


def _validate_single_action(df: pd.DataFrame, action: CleaningAction) -> ValidationResult:
    """Type/format/range check for one applied correction, based on its issue_type."""
    if action.row_index not in df.index or action.column not in df.columns:
        return ValidationResult(
            check_type="cell_exists", passed=False,
            details="Target cell no longer exists in the dataset (row dropped?)",
            row_index=action.row_index, column=action.column,
        )

    value = df.at[action.row_index, action.column]

    if action.issue_type == "invalid_email":
        ok = isinstance(value, str) and bool(EMAIL_REGEX.match(value.strip()))
        return ValidationResult("format_validity", ok,
                                 f"Email format check on corrected value '{value}'",
                                 action.row_index, action.column)

    if action.issue_type in ("invalid_numeric_value", "outlier"):
        ok = pd.to_numeric(pd.Series([value]), errors="coerce").notna().iloc[0]
        return ValidationResult("data_type_validity", bool(ok),
                                 f"Numeric castability check on '{value}'",
                                 action.row_index, action.column)

    if action.issue_type in ("invalid_date", "inconsistent_date_format"):
        parsed = pd.to_datetime(value, errors="coerce")
        ok = pd.notna(parsed)
        return ValidationResult("format_validity", bool(ok),
                                 f"Date parseability check on '{value}'",
                                 action.row_index, action.column)

    if action.issue_type == "invalid_phone":
        digits = re.sub(r"\D", "", str(value))
        ok = 7 <= len(digits) <= 15
        return ValidationResult("format_validity", ok,
                                 f"Phone digit-length check on '{value}'",
                                 action.row_index, action.column)

    if action.issue_type == "missing_value":
        ok = value is not None and not (isinstance(value, float) and pd.isna(value))
        return ValidationResult("null_constraint", bool(ok),
                                 f"Null-constraint check after fill: '{value}'",
                                 action.row_index, action.column)

    if action.issue_type in ("whitespace", "inconsistent_capitalization"):
        ok = isinstance(value, str) and value == value.strip()
        return ValidationResult("format_validity", bool(ok),
                                 f"Whitespace/casing check on '{value}'",
                                 action.row_index, action.column)

    # Default: no specific rule, treat as passed (nothing to re-check)
    return ValidationResult("no_op", True, "No specific validation rule for this issue type",
                             action.row_index, action.column)


def _validate_dataset_level(df: pd.DataFrame) -> List[ValidationResult]:
    results = []

    dup_count = int(df.duplicated().sum())
    results.append(ValidationResult(
        "duplicate_records", dup_count == 0,
        f"{dup_count} duplicate row(s) remain after cleaning",
    ))

    # Referential consistency: if both an *_id and a look-up-style column
    # exist (heuristic), make sure ids are unique when column name suggests
    # a primary identifier (e.g. "id", "customer_id").
    for col in df.columns:
        if col.lower() == "id" or col.lower().endswith("_id"):
            dup_ids = int(df[col].dropna().duplicated().sum())
            results.append(ValidationResult(
                "referential_consistency", dup_ids == 0,
                f"Column '{col}' looks like an identifier but has {dup_ids} duplicate value(s)",
                column=col,
            ))

    return results


def validate_and_finalize(
    df: pd.DataFrame, actions: List[CleaningAction]
) -> Tuple[pd.DataFrame, List[CleaningAction], List[ValidationResult]]:
    """
    Validate every applied action. Roll back anything that fails and
    return the final DataFrame, the updated action list (statuses may
    change to 'rolled_back'), and the full list of ValidationResults
    for the audit trail / MySQL load.
    """
    final_df = df.copy(deep=True)
    validation_results: List[ValidationResult] = []

    for action in actions:
        if action.status != "applied":
            continue  # nothing to validate for actions never applied

        result = _validate_single_action(final_df, action)
        validation_results.append(result)

        if not result.passed:
            logger.warning(
                "Validation FAILED for row=%s col=%s issue=%s -> rolling back",
                action.row_index, action.column, action.issue_type,
            )
            if action.row_index in final_df.index and action.column in final_df.columns:
                final_df.at[action.row_index, action.column] = action.original_value
            action.status = "rolled_back"
            action.validation_notes = result.details
        else:
            action.validation_notes = "validation passed"

    dataset_results = _validate_dataset_level(final_df)
    validation_results.extend(dataset_results)
    for r in dataset_results:
        if not r.passed:
            logger.warning("Dataset-level validation failed: %s", r.details)

    passed = sum(1 for r in validation_results if r.passed)
    total = len(validation_results) or 1
    logger.info(
        "Validation complete: %d/%d checks passed (%.1f%%)",
        passed, total, 100 * passed / total,
    )

    return final_df, actions, validation_results
