"""
reporting.py
------------
Produces the two deliverables described in the spec:
  1. A human-readable Data Quality Report (text + saved to reports/).
  2. A detailed cleaning log (Row | Column | Original | Corrected |
     Issue | Action | Confidence | Status) saved as CSV.
"""

import json
import os
from collections import Counter
from datetime import datetime
from typing import List

from src.config import config
from src.logger import get_logger
from src.models import CleaningAction, Issue
from src.validation import ValidationResult

logger = get_logger(__name__)


def _pct(n: int, d: int) -> float:
    return round(100 * n / d, 1) if d else 0.0


def build_report_text(
    dataset_name: str,
    rows_processed: int,
    issues: List[Issue],
    actions: List[CleaningAction],
    validation_results: List[ValidationResult],
) -> str:
    issues_by_type = Counter(i.issue_type for i in issues)
    fixed = sum(1 for a in actions if a.status == "applied")
    manual_review = sum(1 for a in actions if a.status in ("manual_review", "rolled_back"))
    validation_passed = sum(1 for r in validation_results if r.passed)
    validation_total = len(validation_results) or 1
    validation_pass_rate = _pct(validation_passed, validation_total)

    lines = []
    lines.append("DATA QUALITY REPORT")
    lines.append(f"Dataset:                     {dataset_name}")
    lines.append(f"Generated:                   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"Rows Processed:              {rows_processed:,}")
    lines.append(f"Issues Detected:             {len(issues):,}")
    lines.append(f"Automatically Fixed:         {fixed:,}")
    lines.append(f"Manual Review Required:      {manual_review:,}")
    lines.append(f"Validation Passed:           {validation_pass_rate}%")
    lines.append("")
    lines.append("Issues Detected")
    lines.append("-" * 30)
    for issue_type, count in issues_by_type.most_common():
        label = issue_type.replace("_", " ").title()
        lines.append(f"{label + ':':<28} {count:>5,}")
    lines.append("")
    lines.append("Cleaning Actions")
    lines.append("-" * 30)
    status_counts = Counter(a.status for a in actions)
    for status, count in status_counts.most_common():
        label = status.replace("_", " ").title()
        lines.append(f"{label + ':':<28} {count:>5,}")

    return "\n".join(lines)


def save_report(report_text: str, dataset_name: str) -> str:
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(config.REPORTS_DIR, f"{dataset_name}_quality_report_{timestamp}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info("Saved quality report to %s", path)
    return path


def save_cleaning_log(actions: List[CleaningAction], dataset_name: str) -> str:
    import csv

    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(config.REPORTS_DIR, f"{dataset_name}_cleaning_log_{timestamp}.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Row", "Column", "Original", "Corrected", "Issue", "Action", "Confidence", "Status"]
        )
        for a in actions:
            writer.writerow([
                a.row_index,
                a.column,
                json.dumps(a.original_value, default=str),
                json.dumps(a.corrected_value, default=str),
                a.issue_type,
                a.action_taken,
                a.confidence,
                a.status,
            ])

    logger.info("Saved cleaning log (%d rows) to %s", len(actions), path)
    return path
