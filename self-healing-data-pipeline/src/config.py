"""
config.py
---------
Central configuration for the self-healing data pipeline.
All secrets and environment-specific values are read from environment
variables (loaded from a local .env file via python-dotenv). Nothing
sensitive is hard-coded in source.
"""

import os
from dotenv import load_dotenv

# Load variables from a .env file if present. In production, real
# environment variables (e.g. injected by the shell / CI / secrets
# manager) take precedence over anything in .env.
load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    try:
        return float(val) if val is not None else default
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    try:
        return int(val) if val is not None else default
    except ValueError:
        return default


class Config:
    # ---- MySQL ------------------------------------------------------
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT = _get_int("MYSQL_PORT", 3306)
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "data_quality_pipeline")

    # ---- LLM ----------------------------------------------------------
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-6")
    LLM_MAX_TOKENS = _get_int("LLM_MAX_TOKENS", 1024)
    LLM_ENABLED = _get_bool("LLM_ENABLED", True)
    # If the LLM is disabled or unreachable, the pipeline still runs but
    # every ambiguous issue is routed straight to manual review instead
    # of being guessed at.

    # ---- Safety thresholds --------------------------------------------
    # Only corrections whose confidence is >= this threshold are ever
    # auto-applied. Everything else -> manual review, no exceptions.
    AUTO_APPLY_CONFIDENCE_THRESHOLD = os.getenv(
        "AUTO_APPLY_CONFIDENCE_THRESHOLD", "high"
    ).lower()  # one of: "high"

    OUTLIER_IQR_MULTIPLIER = _get_float("OUTLIER_IQR_MULTIPLIER", 1.5)

    # ---- Paths ----------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
    CLEANED_DATA_DIR = os.path.join(BASE_DIR, "data", "cleaned")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    REPORTS_DIR = os.path.join(BASE_DIR, "reports")

    # ---- Logging --------------------------------------------------------
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


config = Config()
