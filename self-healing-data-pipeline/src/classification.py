"""
classification.py
------------------
Sits between detection and remediation. Groups raw Issue objects into:
  - deterministic issues: solvable with plain Python/Pandas rules,
    no LLM call needed (whitespace, capitalization, exact duplicates,
    date-format normalization, obvious dtype casts).
  - ambiguous issues: require contextual judgement, routed to the LLM
    reasoning layer (missing-value strategy, malformed emails/phones,
    spelled-out numbers, outlier plausibility, unparseable dates).

This keeps LLM usage scoped to genuinely ambiguous decisions, per the
project's engineering requirements.
"""

from collections import defaultdict
from typing import Dict, List

from src.logger import get_logger
from src.models import Issue

logger = get_logger(__name__)


def classify_issues(issues: List[Issue]) -> Dict[str, List[Issue]]:
    deterministic = [i for i in issues if not i.requires_llm]
    needs_llm = [i for i in issues if i.requires_llm]

    by_type = defaultdict(int)
    for issue in issues:
        by_type[issue.issue_type] += 1

    logger.info(
        "Classified %d issue(s): %d deterministic, %d require LLM reasoning",
        len(issues), len(deterministic), len(needs_llm),
    )
    for issue_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
        logger.info("  - %s: %d", issue_type, count)

    return {
        "deterministic": deterministic,
        "needs_llm": needs_llm,
        "counts_by_type": dict(by_type),
    }
