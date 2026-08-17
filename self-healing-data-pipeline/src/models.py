"""
models.py
---------
Shared dataclasses passed between pipeline stages. Keeping these in one
module (instead of importing between quality_checks / llm_engine /
cleaning_engine) avoids circular imports.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Issue:
    """A single detected data-quality problem."""
    row_index: int
    column: str
    issue_type: str            # e.g. "missing_value", "invalid_email"
    original_value: Any
    description: str
    # Some issue types are simple enough to fix deterministically without
    # ever calling the LLM (e.g. trimming whitespace). This flag lets the
    # cleaning engine skip the LLM round-trip for those.
    requires_llm: bool = True
    # Deterministic engines can pre-populate a suggested fix; the LLM
    # engine fills this in for issues where requires_llm=True.
    suggested_action: Optional[str] = None
    suggested_value: Optional[Any] = None
    confidence: Optional[str] = None   # "high" | "medium" | "low"
    source: str = "rule"               # "rule" | "llm"


@dataclass
class Recommendation:
    """Structured response describing what should be done about an Issue."""
    issue_type: str
    column: str
    description: str
    recommended_action: str
    proposed_correction: Any
    confidence: str             # "high" | "medium" | "low"
    safe_to_auto_apply: bool
    source: str = "llm"         # "llm" | "rule"


@dataclass
class CleaningAction:
    """Record of what actually happened to a single cell."""
    row_index: int
    column: str
    original_value: Any
    corrected_value: Any
    issue_type: str
    action_taken: str
    confidence: str
    source: str                 # "rule" | "llm"
    status: str                 # "applied" | "rolled_back" | "manual_review"
    validation_notes: str = field(default="")
