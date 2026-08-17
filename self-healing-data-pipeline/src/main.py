"""
main.py
-------
Orchestrates the full self-healing data cleaning pipeline:

    Raw CSV/Excel
        -> Ingestion
        -> Profiling
        -> Quality Detection
        -> Issue Classification
        -> LLM Reasoning (ambiguous issues only)
        -> Cleaning Rule Generation / Safe Automated Correction
        -> Post-Cleaning Validation (with rollback)
        -> Quality Report
        -> MySQL

Usage:
    python -m src.main --input data/raw/customers.csv
    python -m src.main --input data/raw/customers.csv --skip-db
"""

import argparse
import os
import sys
import time

from src.classification import classify_issues
from src.cleaning_engine import clean_dataset
from src.config import config
from src.ingestion import load_dataset
from src.logger import get_logger
from src.profiling import profile_dataset
from src.quality_checks import detect_issues
from src.reporting import build_report_text, save_cleaning_log, save_report
from src.validation import validate_and_finalize

logger = get_logger(__name__)


def run_pipeline(input_path: str, skip_db: bool = False) -> dict:
    start = time.time()
    logger.info("=" * 70)
    logger.info("PIPELINE START: %s", input_path)
    logger.info("=" * 70)

    try:
        # 1. Ingestion --------------------------------------------------
        ingestion_result = load_dataset(input_path)
        df = ingestion_result.dataframe

        # 2. Profiling ----------------------------------------------------
        profile = profile_dataset(df)

        # 3. Quality Detection ---------------------------------------------
        issues = detect_issues(df, dtype_hints=ingestion_result.dtypes)

        # 4. Issue Classification -------------------------------------------
        classification = classify_issues(issues)

        # 5 & 6. LLM Reasoning + Safe Automated Correction ------------------
        #    (cleaning_engine calls the LLM internally for ambiguous issues)
        cleaned_df, actions = clean_dataset(df, issues)

        # 7. Post-Cleaning Validation (with rollback) -----------------------
        final_df, actions, validation_results = validate_and_finalize(cleaned_df, actions)

        # 8. Quality Report --------------------------------------------------
        report_text = build_report_text(
            ingestion_result.dataset_name, ingestion_result.row_count,
            issues, actions, validation_results,
        )
        print("\n" + report_text + "\n")
        report_path = save_report(report_text, ingestion_result.dataset_name)
        log_path = save_cleaning_log(actions, ingestion_result.dataset_name)

        # Save cleaned dataset to disk regardless of DB availability
        os.makedirs(config.CLEANED_DATA_DIR, exist_ok=True)
        cleaned_csv_path = os.path.join(
            config.CLEANED_DATA_DIR, f"{ingestion_result.dataset_name}_cleaned.csv"
        )
        final_df.to_csv(cleaned_csv_path, index=False)
        logger.info("Saved cleaned dataset to %s", cleaned_csv_path)

        # 9. MySQL -------------------------------------------------------
        run_id = None
        if not skip_db:
            try:
                from src.database import load_pipeline_results
                run_id = load_pipeline_results(
                    dataset_name=ingestion_result.dataset_name,
                    source_file=ingestion_result.archived_raw_path,
                    dataset_hash=ingestion_result.dataset_hash,
                    cleaned_df=final_df,
                    issues=issues,
                    actions=actions,
                    validation_results=validation_results,
                )
            except Exception as e:  # noqa: BLE001
                logger.error("MySQL load failed, pipeline outputs are still saved locally: %s", e)
        else:
            logger.info("--skip-db set, not loading results into MySQL")

        elapsed = time.time() - start
        logger.info("PIPELINE COMPLETE in %.2fs (run_id=%s)", elapsed, run_id)

        return {
            "dataset_name": ingestion_result.dataset_name,
            "row_count": ingestion_result.row_count,
            "issues_detected": len(issues),
            "issues_fixed": sum(1 for a in actions if a.status == "applied"),
            "manual_review_count": sum(
                1 for a in actions if a.status in ("manual_review", "rolled_back")
            ),
            "cleaned_csv_path": cleaned_csv_path,
            "report_path": report_path,
            "cleaning_log_path": log_path,
            "mysql_run_id": run_id,
            "profile": profile,
        }

    except Exception:
        logger.exception("Pipeline run failed")
        raise


def main():
    parser = argparse.ArgumentParser(description="Self-healing data cleaning pipeline")
    parser.add_argument("--input", required=True, help="Path to raw CSV or Excel file")
    parser.add_argument("--skip-db", action="store_true", help="Skip loading results into MySQL")
    args = parser.parse_args()

    try:
        run_pipeline(args.input, skip_db=args.skip_db)
    except Exception as e:  # noqa: BLE001
        print(f"Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
