# Self-Healing Data Cleaning Pipeline

A Python data-engineering pipeline that automatically detects data-quality
problems, uses an LLM (Claude) to reason about ambiguous corrections,
applies only *safe* automated fixes, validates every correction (rolling
back anything that fails), and loads the cleaned dataset plus a full
audit trail into MySQL.

The LLM never touches your data directly. It only returns a structured
recommendation (action, proposed value, confidence, and whether it's
safe to auto-apply). A separate correction engine decides what to
actually apply, and a validation layer re-checks the result and rolls
back anything that doesn't hold up.

## Architecture

```
Raw CSV / Excel
     |
     v
Ingestion            (load file, freeze a raw copy, detect schema)
     |
     v
Profiling            (row/col counts, dtypes, missing %, stats, top values)
     |
     v
Quality Detection    (rule-based: missing, dupes, formats, outliers, ...)
     |
     v
Issue Classification (deterministic vs needs-LLM-reasoning)
     |
     v
LLM Reasoning        (Claude proposes a fix ONLY for ambiguous issues)
     |
     v
Safe Automated Correction  (apply only high-confidence, safe fixes)
     |
     v
Post-Cleaning Validation   (re-check every fix; rollback on failure)
     |
     v
Quality Report + Cleaning Log
     |
     v
MySQL  (normalized schema, idempotent re-runs)
```

## Why the LLM is used sparingly

Deterministic Python/Pandas rules handle anything that's mechanically
solvable: trimming whitespace, normalizing casing against the majority
variant, standardizing a mixed-format date column to ISO 8601, and
dropping exact-duplicate rows. These never need a model call.

The LLM is only consulted for genuinely ambiguous decisions: what to do
about a missing value, how to interpret a malformed email/phone, what
"twenty five" should become numerically, whether a statistical outlier
looks like a real data-entry problem, or how to fix a date that doesn't
parse at all. Even then, its recommendation is only auto-applied when
it reports **high confidence and explicitly marks the fix as safe** -
otherwise the row is routed to manual review. Nothing is ever guessed
into your dataset silently.

## Project Structure

```
self-healing-data-pipeline/
├── data/
│   ├── raw/            # ingested files are archived here untouched
│   └── cleaned/         # final cleaned CSV output per run
├── src/
│   ├── config.py         # env-var driven settings
│   ├── logger.py         # shared logging setup (console + file)
│   ├── models.py         # Issue / Recommendation / CleaningAction dataclasses
│   ├── ingestion.py       # stage 1
│   ├── profiling.py       # stage 2
│   ├── quality_checks.py  # stage 3 (rule-based detection)
│   ├── classification.py  # stage 4 (deterministic vs needs-LLM)
│   ├── llm_engine.py       # stage 5 (Claude reasoning, structured JSON)
│   ├── cleaning_engine.py  # stages 6-7 (controlled correction engine)
│   ├── validation.py        # stage 8 (post-cleaning validation + rollback)
│   ├── reporting.py          # stage 9 (quality report + cleaning log)
│   ├── database.py            # stage 10 (MySQL schema + idempotent loads)
│   └── main.py                 # orchestrator / CLI entrypoint
├── logs/                # timestamped run logs
├── reports/             # quality_report_*.txt and cleaning_log_*.csv
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # then edit .env with your real MySQL credentials and ANTHROPIC_API_KEY
   ```
   The pipeline never reads secrets from source code - only from
   environment variables (see `src/config.py`).

3. **Have a MySQL server reachable** with the credentials in `.env`.
   The pipeline creates the database and all tables automatically on
   first run (`CREATE DATABASE IF NOT EXISTS`, `CREATE TABLE IF NOT
   EXISTS`) - no manual schema setup required.

## Usage

```bash
# Full run: clean a CSV/Excel file and load results into MySQL
python -m src.main --input data/raw/your_file.csv

# Dry run without touching MySQL (useful for testing/CI)
python -m src.main --input data/raw/your_file.csv --skip-db
```

A sample dataset with deliberately seeded issues (whitespace, mixed
capitalization, mixed date formats, an invalid email, an invalid
phone number, a spelled-out numeric value, a missing value, and an
exact duplicate row) is included at `data/raw/sample_customers.csv`
so you can try the pipeline immediately:

```bash
python -m src.main --input data/raw/sample_customers.csv --skip-db
```

Each run produces:
- `data/cleaned/<dataset>_cleaned.csv` - the final cleaned dataset
- `reports/<dataset>_quality_report_<timestamp>.txt` - summary report
- `reports/<dataset>_cleaning_log_<timestamp>.csv` - full row-by-row audit log
- `logs/pipeline_<timestamp>.log` - detailed execution log
- (if not `--skip-db`) rows in MySQL across `pipeline_runs`,
  `quality_issues`, `cleaning_actions`, `validation_results`, and a
  dynamic `cleaned_<dataset_name>` table holding the final data.

## MySQL Schema

| Table | Purpose |
|---|---|
| `pipeline_runs` | One row per pipeline execution; `dataset_hash` has a UNIQUE constraint so re-running the same file replaces its prior results instead of duplicating them. |
| `quality_issues` | Every issue detected, before any correction. |
| `cleaning_actions` | Every attempted correction: original value, corrected value, confidence, source (rule/llm), and final status (`applied` / `manual_review` / `rolled_back`). |
| `validation_results` | Every post-correction validation check and whether it passed. |
| `processing_logs` | Reserved for structured log persistence (the pipeline currently logs to file via `logs/`; this table is available if you want to mirror logs into MySQL too). |
| `cleaned_<dataset_name>` | Dynamically created to match the cleaned dataset's own columns, tagged with `run_id` and `source_row_index`. |

Re-running the pipeline on an unchanged input file is idempotent: the
file's SHA-256 hash is checked against `pipeline_runs.dataset_hash`,
and if a match is found the old run (and all its child rows, via
`ON DELETE CASCADE`) is deleted before the new results are inserted.

## Safety Model

- The LLM **never** writes to the DataFrame. It only returns a
  structured `Recommendation`.
- A correction is only auto-applied when confidence is `"high"` **and**
  the model explicitly marked it `safe_to_auto_apply` - both are
  required.
- Every applied correction is re-validated afterward (type, format,
  null, and dataset-level constraints). Anything that fails validation
  is **rolled back** to its original value and re-flagged as needing
  manual review.
- Statistical outliers are always routed to manual review, never
  auto-modified - being unusual isn't the same as being wrong.
- Nothing is corrected, dropped, or overwritten without a
  `CleaningAction` record in the audit log.

## Extending

- Add new detection rules in `src/quality_checks.py`; set
  `requires_llm=False` and provide a `suggested_action` /
  `suggested_value` for anything mechanically solvable, or leave
  `requires_llm=True` for anything context-dependent.
- Add a corresponding branch in `cleaning_engine._apply_deterministic`
  if you introduce a new deterministic `suggested_action`.
- Add a corresponding format check in
  `validation._validate_single_action` if you introduce a new
  `issue_type` so corrections of that type get validated.
