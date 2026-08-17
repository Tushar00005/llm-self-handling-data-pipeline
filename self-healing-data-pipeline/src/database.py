"""
database.py
------------
MySQL persistence layer. Creates a normalized schema (if it doesn't
already exist) and loads:
  - pipeline run metadata
  - detected quality issues
  - cleaning actions (the audit log)
  - validation results
  - the final cleaned dataset itself (dynamic table, one per dataset name)

Idempotency: pipeline_runs.dataset_hash has a UNIQUE constraint. If the
same raw file is processed again, the previous run's rows are deleted
(cascade) before the new ones are inserted, so re-running the pipeline
never creates duplicate records.
"""

import json
import re
from typing import Any, List

import mysql.connector
import pandas as pd
from mysql.connector import errorcode

from src.config import config
from src.logger import get_logger
from src.models import CleaningAction
from src.validation import ValidationResult

logger = get_logger(__name__)


def _connect(use_database: bool = True):
    kwargs = dict(
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        user=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
    )
    if use_database:
        kwargs["database"] = config.MYSQL_DATABASE
    return mysql.connector.connect(**kwargs)


def ensure_database_exists():
    """Create the target database if it doesn't already exist."""
    conn = _connect(use_database=False)
    try:
        cur = conn.cursor()
        cur.execute(
            f"CREATE DATABASE IF NOT EXISTS `{config.MYSQL_DATABASE}` "
            f"CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        conn.commit()
        logger.info("Ensured database '%s' exists", config.MYSQL_DATABASE)
    finally:
        conn.close()


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS pipeline_runs (
        run_id            BIGINT AUTO_INCREMENT PRIMARY KEY,
        dataset_name      VARCHAR(255) NOT NULL,
        source_file       VARCHAR(1024) NOT NULL,
        dataset_hash      CHAR(64) NOT NULL,
        run_timestamp     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        rows_processed    INT NOT NULL DEFAULT 0,
        issues_detected   INT NOT NULL DEFAULT 0,
        issues_fixed      INT NOT NULL DEFAULT 0,
        manual_review_count INT NOT NULL DEFAULT 0,
        validation_pass_rate DECIMAL(5,2) NOT NULL DEFAULT 0,
        status            VARCHAR(32) NOT NULL DEFAULT 'completed',
        UNIQUE KEY uq_dataset_hash (dataset_hash)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS quality_issues (
        issue_id      BIGINT AUTO_INCREMENT PRIMARY KEY,
        run_id        BIGINT NOT NULL,
        row_index     INT NOT NULL,
        column_name   VARCHAR(255) NOT NULL,
        issue_type    VARCHAR(64) NOT NULL,
        description   TEXT,
        original_value TEXT,
        detected_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
        INDEX idx_qi_run (run_id),
        INDEX idx_qi_type (issue_type)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS cleaning_actions (
        action_id       BIGINT AUTO_INCREMENT PRIMARY KEY,
        run_id          BIGINT NOT NULL,
        row_index       INT NOT NULL,
        column_name     VARCHAR(255) NOT NULL,
        original_value  TEXT,
        corrected_value TEXT,
        issue_type      VARCHAR(64) NOT NULL,
        action_taken    VARCHAR(255),
        confidence      VARCHAR(16),
        source          VARCHAR(16),
        status          VARCHAR(32) NOT NULL,
        validation_notes TEXT,
        created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
        INDEX idx_ca_run (run_id),
        INDEX idx_ca_status (status)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS validation_results (
        validation_id   BIGINT AUTO_INCREMENT PRIMARY KEY,
        run_id          BIGINT NOT NULL,
        check_type      VARCHAR(64) NOT NULL,
        passed          BOOLEAN NOT NULL,
        details         TEXT,
        row_index       INT DEFAULT -1,
        column_name     VARCHAR(255),
        created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
        INDEX idx_vr_run (run_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    """
    CREATE TABLE IF NOT EXISTS processing_logs (
        log_id      BIGINT AUTO_INCREMENT PRIMARY KEY,
        run_id      BIGINT NOT NULL,
        level       VARCHAR(16) NOT NULL,
        message     TEXT NOT NULL,
        created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
        INDEX idx_pl_run (run_id)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
]


def initialize_schema():
    ensure_database_exists()
    conn = _connect()
    try:
        cur = conn.cursor()
        for statement in SCHEMA_STATEMENTS:
            cur.execute(statement)
        conn.commit()
        logger.info("Core schema ensured (pipeline_runs, quality_issues, "
                     "cleaning_actions, validation_results, processing_logs)")
    finally:
        conn.close()


def _sanitize_identifier(name: str) -> str:
    """Turn an arbitrary column/dataset name into a safe MySQL identifier."""
    safe = re.sub(r"[^0-9a-zA-Z_]", "_", str(name)).strip("_")
    if not safe:
        safe = "col"
    if safe[0].isdigit():
        safe = f"c_{safe}"
    return safe.lower()[:64]


def _mysql_type_for(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "BIGINT"
    if pd.api.types.is_float_dtype(series):
        return "DOUBLE"
    if pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "DATETIME"
    return "TEXT"


def _find_existing_run(cur, dataset_hash: str):
    cur.execute("SELECT run_id FROM pipeline_runs WHERE dataset_hash = %s", (dataset_hash,))
    row = cur.fetchone()
    return row[0] if row else None


def load_pipeline_results(
    dataset_name: str,
    source_file: str,
    dataset_hash: str,
    cleaned_df: pd.DataFrame,
    issues: List[Any],
    actions: List[CleaningAction],
    validation_results: List[ValidationResult],
) -> int:
    """
    Persist a full pipeline run to MySQL. If a run with the same
    dataset_hash already exists, it is deleted first (cascade removes
    its child rows) so re-running the pipeline is idempotent rather
    than appending duplicates.

    Returns the new run_id.
    """
    initialize_schema()
    conn = _connect()
    try:
        cur = conn.cursor()

        existing_run_id = _find_existing_run(cur, dataset_hash)
        if existing_run_id:
            logger.info(
                "Dataset hash already processed as run_id=%s; deleting old run "
                "before reloading to keep MySQL idempotent",
                existing_run_id,
            )
            cur.execute("DELETE FROM pipeline_runs WHERE run_id = %s", (existing_run_id,))
            conn.commit()

        issues_fixed = sum(1 for a in actions if a.status == "applied")
        manual_review = sum(1 for a in actions if a.status in ("manual_review", "rolled_back"))
        passed = sum(1 for r in validation_results if r.passed)
        total = len(validation_results) or 1
        pass_rate = round(100 * passed / total, 2)

        cur.execute(
            """
            INSERT INTO pipeline_runs
                (dataset_name, source_file, dataset_hash, rows_processed,
                 issues_detected, issues_fixed, manual_review_count,
                 validation_pass_rate, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                dataset_name, source_file, dataset_hash, len(cleaned_df),
                len(issues), issues_fixed, manual_review, pass_rate, "completed",
            ),
        )
        run_id = cur.lastrowid
        conn.commit()
        logger.info("Created pipeline_runs row run_id=%s", run_id)

        if issues:
            cur.executemany(
                """
                INSERT INTO quality_issues
                    (run_id, row_index, column_name, issue_type, description, original_value)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                [
                    (run_id, i.row_index, i.column, i.issue_type, i.description,
                     json.dumps(i.original_value, default=str))
                    for i in issues
                ],
            )
            conn.commit()
            logger.info("Inserted %d quality_issues row(s)", len(issues))

        if actions:
            cur.executemany(
                """
                INSERT INTO cleaning_actions
                    (run_id, row_index, column_name, original_value, corrected_value,
                     issue_type, action_taken, confidence, source, status, validation_notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    (
                        run_id, a.row_index, a.column,
                        json.dumps(a.original_value, default=str),
                        json.dumps(a.corrected_value, default=str),
                        a.issue_type, a.action_taken, a.confidence, a.source,
                        a.status, a.validation_notes,
                    )
                    for a in actions
                ],
            )
            conn.commit()
            logger.info("Inserted %d cleaning_actions row(s)", len(actions))

        if validation_results:
            cur.executemany(
                """
                INSERT INTO validation_results
                    (run_id, check_type, passed, details, row_index, column_name)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                [
                    (run_id, r.check_type, r.passed, r.details, r.row_index, r.column)
                    for r in validation_results
                ],
            )
            conn.commit()
            logger.info("Inserted %d validation_results row(s)", len(validation_results))

        _load_cleaned_dataframe(cur, conn, run_id, dataset_name, cleaned_df)

        return run_id
    finally:
        conn.close()


def _load_cleaned_dataframe(cur, conn, run_id: int, dataset_name: str, df: pd.DataFrame):
    """Create (if needed) a dynamic table matching the cleaned dataset's
    schema and load the cleaned rows into it, tagged with run_id."""
    table_name = f"cleaned_{_sanitize_identifier(dataset_name)}"

    column_defs = []
    safe_columns = []
    for col in df.columns:
        safe_col = _sanitize_identifier(col)
        safe_columns.append((col, safe_col))
        column_defs.append(f"`{safe_col}` {_mysql_type_for(df[col])}")

    create_stmt = f"""
        CREATE TABLE IF NOT EXISTS `{table_name}` (
            record_id BIGINT AUTO_INCREMENT PRIMARY KEY,
            run_id BIGINT NOT NULL,
            source_row_index INT NOT NULL,
            {', '.join(column_defs)},
            loaded_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES pipeline_runs(run_id) ON DELETE CASCADE,
            UNIQUE KEY uq_run_row (run_id, source_row_index),
            INDEX idx_run (run_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    cur.execute(create_stmt)
    conn.commit()

    if df.empty:
        logger.info("Cleaned dataset is empty, nothing to load into %s", table_name)
        return

    col_list = ", ".join(f"`{s}`" for _, s in safe_columns)
    placeholders = ", ".join(["%s"] * (len(safe_columns) + 2))
    insert_stmt = (
        f"INSERT INTO `{table_name}` (run_id, source_row_index, {col_list}) "
        f"VALUES ({placeholders})"
    )

    rows = []
    for idx, row in df.iterrows():
        values = [run_id, int(idx)]
        for orig_col, _ in safe_columns:
            val = row[orig_col]
            if pd.isna(val):
                values.append(None)
            elif isinstance(val, (pd.Timestamp,)):
                values.append(val.to_pydatetime())
            else:
                values.append(val)
        rows.append(tuple(values))

    cur.executemany(insert_stmt, rows)
    conn.commit()
    logger.info("Loaded %d row(s) into %s", len(rows), table_name)
