"""
llm_engine.py
-------------
The LLM reasoning layer. For issues that deterministic rules can't
safely resolve on their own (missing-value strategy, malformed emails,
spelled-out numbers, unparseable dates, outlier plausibility), this
module sends ONLY the minimal necessary context to the Claude API and
asks for a structured recommendation.

Hard rule: the LLM never touches the DataFrame. It returns a
Recommendation object; the cleaning_engine decides whether and how to
apply it, and validation.py can still roll it back.
"""

import json
import re
from typing import Optional

from src.config import config
from src.logger import get_logger
from src.models import Issue, Recommendation

logger = get_logger(__name__)

_client = None


def _get_client():
    """Lazily import and construct the Anthropic client. Importing the
    anthropic package only when actually needed means the rest of the
    pipeline still runs (and is testable) even in environments where
    the LLM is disabled or the package isn't installed."""
    global _client
    if not config.LLM_ENABLED:
        return None
    if not config.ANTHROPIC_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set - LLM reasoning disabled, "
                        "ambiguous issues will fall back to manual review")
        return None
    if _client is None:
        try:
            import anthropic
        except ImportError:
            logger.error("anthropic package not installed - run: pip install anthropic")
            return None
        _client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    return _client


SYSTEM_PROMPT = """You are a data quality reasoning assistant embedded in an \
automated data cleaning pipeline. You NEVER modify data directly - you only \
analyze a single reported issue and return a structured JSON recommendation.

Respond with ONLY a JSON object (no markdown fences, no commentary) with \
exactly these fields:
{
  "issue_type": string,
  "column": string,
  "problem_description": string,
  "recommended_action": string,       // short imperative description
  "proposed_correction": string|number|null,  // the actual replacement value, or null if none should be applied automatically
  "confidence": "high" | "medium" | "low",
  "safe_to_auto_apply": boolean       // true only if you are highly confident AND the correction cannot plausibly destroy information
}

Rules you must follow:
- Prefer caution. If there is any reasonable ambiguity about the correct value, \
set confidence to "low" or "medium" and safe_to_auto_apply to false.
- Only set safe_to_auto_apply=true when confidence="high".
- For missing values, only propose an automatic fill when it is a safe, \
conservative choice (e.g. a clearly implied constant); otherwise recommend \
manual review.
- For malformed emails/phones where the intended value is genuinely unclear \
(e.g. missing "@" with no way to know the domain), set confidence low and \
do not guess.
- Never invent facts not implied by the given value and column context.
"""


def _build_user_prompt(issue: Issue, column_context: dict) -> str:
    context = {
        "issue_type": issue.issue_type,
        "column": issue.column,
        "row_index": issue.row_index,
        "original_value": _safe_serialize(issue.original_value),
        "description": issue.description,
        "column_context": column_context,
    }
    return (
        "Analyze this single data quality issue and return the JSON "
        "recommendation described in your instructions.\n\n"
        f"Issue:\n{json.dumps(context, indent=2, default=str)}"
    )


def _safe_serialize(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    text = re.sub(r"^```json\s*|^```\s*|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _fallback_recommendation(issue: Issue, reason: str) -> Recommendation:
    """Used when the LLM is unavailable or returns something unusable.
    Always defaults to the safe option: manual review."""
    return Recommendation(
        issue_type=issue.issue_type,
        column=issue.column,
        description=f"{issue.description} (LLM unavailable: {reason})",
        recommended_action="manual_review",
        proposed_correction=None,
        confidence="low",
        safe_to_auto_apply=False,
        source="fallback",
    )


def get_recommendation(issue: Issue, column_context: dict = None) -> Recommendation:
    """
    Send a single ambiguous issue to the LLM and return a structured
    Recommendation. Only the minimal necessary fields (issue type,
    column name, the offending value, and a small amount of column
    context such as sample valid values) are sent - not the full dataset.
    """
    column_context = column_context or {}
    client = _get_client()
    if client is None:
        return _fallback_recommendation(issue, "no client configured")

    try:
        logger.info(
            "Requesting LLM recommendation for issue_type=%s column=%s row=%s",
            issue.issue_type, issue.column, issue.row_index,
        )
        response = client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=config.LLM_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_prompt(issue, column_context)}],
        )
        text_blocks = [b.text for b in response.content if getattr(b, "type", "") == "text"]
        raw_text = "\n".join(text_blocks)
        parsed = _extract_json(raw_text)

        if not parsed:
            logger.warning("Could not parse LLM response as JSON for issue row=%s col=%s",
                            issue.row_index, issue.column)
            return _fallback_recommendation(issue, "unparseable response")

        confidence = str(parsed.get("confidence", "low")).lower()
        if confidence not in ("high", "medium", "low"):
            confidence = "low"

        safe_flag = bool(parsed.get("safe_to_auto_apply", False))
        # Hard safety gate: never trust safe_to_auto_apply unless confidence is also high
        safe_to_auto_apply = safe_flag and confidence == "high"

        recommendation = Recommendation(
            issue_type=parsed.get("issue_type", issue.issue_type),
            column=parsed.get("column", issue.column),
            description=parsed.get("problem_description", issue.description),
            recommended_action=parsed.get("recommended_action", "manual_review"),
            proposed_correction=parsed.get("proposed_correction"),
            confidence=confidence,
            safe_to_auto_apply=safe_to_auto_apply,
            source="llm",
        )
        logger.info(
            "LLM recommendation: action=%s confidence=%s auto_apply=%s",
            recommendation.recommended_action, recommendation.confidence,
            recommendation.safe_to_auto_apply,
        )
        return recommendation

    except Exception as e:  # noqa: BLE001 - pipeline must never crash on LLM failure
        # Covers anthropic.APIError and any other transport/parsing failure.
        logger.error("Unexpected error getting LLM recommendation: %s", e)
        return _fallback_recommendation(issue, f"unexpected error: {e}")
