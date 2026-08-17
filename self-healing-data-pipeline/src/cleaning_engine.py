"""
cleaning_engine.py
-------------------
The self-healing / controlled correction engine.

For deterministic issues (whitespace, capitalization, exact duplicate
rows, date-format normalization, obvious dtype casts) it applies the
rule-generated fix directly - these are safe by construction.

For ambiguous issues it calls llm_engine.get_recommendation() and only
auto-applies the result when confidence == "high" AND
safe_to_auto_apply is True. Anything else is left untouched and routed
to manual review. Nothing here is allowed to silently overwrite data
without producing a CleaningAction record for the audit log.
"""

from typing import List, Tuple

import pandas as pd

from src.config import config
from src.llm_engine import get_recommendation
from src.logger import get_logger
from src.models import CleaningAction, Issue

logger = get_logger(__name__)


def _column_sample_context(df: pd.DataFrame, column: str, n: int = 5) -> dict:
    if column not in df.columns:
        return {}
    valid_samples = df[column].dropna().astype(str).unique()[:n].tolist()
    return {"sample_valid_values": valid_samples, "dtype": str(df[column].dtype)}


def _apply_deterministic(df: pd.DataFrame, issue: Issue) -> Tuple[pd.DataFrame, CleaningAction]:
    """Apply a rule-generated fix. Returns the (possibly mutated) df and
    the CleaningAction describing what happened."""
    action = issue.suggested_action

    if action == "drop_duplicate_row":
        # Row is dropped later in a batch (see clean_dataset) to keep
        # index alignment stable while iterating; here we just record intent.
        return df, CleaningAction(
            row_index=issue.row_index, column=issue.column,
            original_value=issue.original_value, corrected_value=None,
            issue_type=issue.issue_type, action_taken="drop_duplicate_row",
            confidence="high", source="rule", status="applied",
        )

    if action == "trim_whitespace":
        df.at[issue.row_index, issue.column] = issue.suggested_value
        return df, CleaningAction(
            row_index=issue.row_index, column=issue.column,
            original_value=issue.original_value, corrected_value=issue.suggested_value,
            issue_type=issue.issue_type, action_taken="trim_whitespace",
            confidence="high", source="rule", status="applied",
        )

    if action == "normalize_capitalization":
        df.at[issue.row_index, issue.column] = issue.suggested_value
        return df, CleaningAction(
            row_index=issue.row_index, column=issue.column,
            original_value=issue.original_value, corrected_value=issue.suggested_value,
            issue_type=issue.issue_type, action_taken="normalize_capitalization",
            confidence="high", source="rule", status="applied",
        )

    if action == "normalize_date_to_iso8601":
        df.at[issue.row_index, issue.column] = issue.suggested_value
        return df, CleaningAction(
            row_index=issue.row_index, column=issue.column,
            original_value=issue.original_value, corrected_value=issue.suggested_value,
            issue_type=issue.issue_type, action_taken="normalize_date_to_iso8601",
            confidence="high", source="rule", status="applied",
        )

    if action == "cast_column_numeric":
        # Column-level cast is applied once by the caller, not per-row.
        return df, CleaningAction(
            row_index=issue.row_index, column=issue.column,
            original_value=issue.original_value, corrected_value="numeric",
            issue_type=issue.issue_type, action_taken="cast_column_numeric",
            confidence="medium", source="rule", status="manual_review",
        )

    # Unknown deterministic action - be conservative
    logger.warning("Unrecognized deterministic action '%s', routing to manual review", action)
    return df, CleaningAction(
        row_index=issue.row_index, column=issue.column,
        original_value=issue.original_value, corrected_value=None,
        issue_type=issue.issue_type, action_taken="unknown",
        confidence="low", source="rule", status="manual_review",
    )


def _apply_llm_recommendation(df: pd.DataFrame, issue: Issue) -> Tuple[pd.DataFrame, CleaningAction]:
    context = _column_sample_context(df, issue.column)
    rec = get_recommendation(issue, column_context=context)

    if rec.confidence == "high" and rec.safe_to_auto_apply and rec.proposed_correction is not None:
        try:
            df.at[issue.row_index, issue.column] = rec.proposed_correction
            status = "applied"
        except Exception as e:  # noqa: BLE001
            logger.error("Failed applying LLM correction at row=%s col=%s: %s",
                        issue.row_index, issue.column, e)
            status = "manual_review"
    else:
        status = "manual_review"

    return df, CleaningAction(
        row_index=issue.row_index,
        column=issue.column,
        original_value=issue.original_value,
        corrected_value=rec.proposed_correction if status == "applied" else None,
        issue_type=issue.issue_type,
        action_taken=rec.recommended_action,
        confidence=rec.confidence,
        source="llm",
        status=status,
    )


def clean_dataset(df: pd.DataFrame, issues: List[Issue]) -> Tuple[pd.DataFrame, List[CleaningAction]]:
    """
    Apply safe corrections to a working copy of df and return
    (cleaned_df, cleaning_actions). Nothing here is final - validation.py
    re-checks every applied action afterward and can roll it back.
    """
    working = df.copy(deep=True)
    actions: List[CleaningAction] = []
    rows_to_drop = []
    columns_to_cast = set()

    for issue in issues:
        if not issue.requires_llm:
            working, action = _apply_deterministic(working, issue)
        else:
            working, action = _apply_llm_recommendation(working, issue)

        actions.append(action)

        if action.action_taken == "drop_duplicate_row" and action.status == "applied":
            rows_to_drop.append(issue.row_index)
        if action.action_taken == "cast_column_numeric":
            columns_to_cast.add(issue.column)

    if rows_to_drop:
        working = working.drop(index=[r for r in rows_to_drop if r in working.index])
        logger.info("Dropped %d duplicate row(s)", len(rows_to_drop))

    logger.info(
        "Cleaning pass complete: %d action(s) recorded (%d applied, %d manual review)",
        len(actions),
        sum(1 for a in actions if a.status == "applied"),
        sum(1 for a in actions if a.status == "manual_review"),
    )

    return working, actions
