"""
dags/etl_pipeline.py
---------------------
Prefect flow orchestrating the full Extract → Transform → Load pipeline
for NYC TLC Yellow Taxi trip data.

Prompt used with Claude Desktop:
  "Write a Prefect 2 flow that orchestrates the full ETL pipeline,
   with individual tasks for extract, transform, and load, retry logic,
   artifact logging, and a configurable run for multiple months."
"""

import logging
from pathlib import Path
from datetime import datetime
from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact

# Local modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from extractors.nyc_tlc_extractor import extract
from transformers.trip_transformer import create_spark_session, read_raw, transform
from loaders.duckdb_loader import DuckDBLoader

# ── Config ──────────────────────────────────────────────────────────────────
RAW_DIR = Path("data/raw")
STAGING_DIR = Path("data/processed/staging")
DB_PATH = Path("data/warehouse.duckdb")


# ── Tasks ────────────────────────────────────────────────────────────────────

@task(retries=2, retry_delay_seconds=30, name="Extract TLC Data")
def extract_task(year: int, months: list[int], taxi_type: str = "yellow") -> list[str]:
    """Download raw Parquet files from TLC open data."""
    logger = get_run_logger()
    logger.info(f"Extracting {taxi_type} taxi data for {year}, months={months}")

    paths = extract(
        year=year,
        months=months,
        taxi_type=taxi_type,
        output_dir=RAW_DIR,
        skip_existing=True,
    )
    return [str(p) for p in paths]


@task(name="Transform Trip Data")
def transform_task(raw_paths: list[str]) -> int:
    """Apply PySpark transformations to raw data and write to staging."""
    logger = get_run_logger()

    if not raw_paths:
        logger.warning("No raw files to transform. Skipping.")
        return 0

    spark = create_spark_session("NYC-TLC-Transform")

    try:
        df_raw = spark.read.parquet(*raw_paths)
        logger.info(f"Loaded {df_raw.count():,} raw rows from {len(raw_paths)} file(s)")

        df_transformed = transform(df_raw)

        # Write to staging partitioned by year/month
        writer = (
            df_transformed.write
            .mode("overwrite")
            .partitionBy("pickup_year", "pickup_month")
            .parquet(str(STAGING_DIR))
        )

        count = df_transformed.count()
        logger.info(f"Staged {count:,} transformed rows to {STAGING_DIR}")
        return count
    finally:
        spark.stop()


@task(name="Load to DuckDB")
def load_task() -> dict:
    """Load staged Parquet data into DuckDB warehouse."""
    logger = get_run_logger()
    loader = DuckDBLoader(DB_PATH)

    try:
        # Load directly from staged Parquet (skip Spark)
        import duckdb
        conn = loader.conn
        parquet_glob = f"{STAGING_DIR}/**/*.parquet"

        conn.execute("DROP TABLE IF EXISTS nyc_tlc.trips")
        conn.execute(f"""
            CREATE TABLE nyc_tlc.trips AS
            SELECT * FROM read_parquet('{parquet_glob}', hive_partitioning=true)
        """)

        row_count = conn.execute("SELECT COUNT(*) FROM nyc_tlc.trips").fetchone()[0]
        logger.info(f"Loaded {row_count:,} rows into nyc_tlc.trips")

        loader.create_analytics_views()
        quality = loader.run_data_quality_checks()
        return quality
    finally:
        loader.close()


@task(name="Publish Run Summary")
def publish_summary_task(
    year: int,
    months: list[int],
    raw_files: list[str],
    transformed_rows: int,
    quality: dict,
):
    """Create a Prefect markdown artifact summarising the run."""
    date_range = quality.get("date_range", ("N/A", "N/A"))
    summary = f"""
# ETL Pipeline Run Summary

**Run time:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Dataset:** NYC TLC Yellow Taxi — {year}, months {months}

## Pipeline Results

| Stage      | Result |
|------------|--------|
| Files extracted | {len(raw_files)} |
| Rows transformed | {transformed_rows:,} |
| Rows loaded | {quality.get('total_rows', 0):,} |
| Null fares | {quality.get('null_fare_count', 0)} |
| Avg trip distance | {quality.get('avg_trip_distance', 0)} miles |

## Date Range in Warehouse
- **From:** {date_range[0]}
- **To:**   {date_range[1]}

## Analytics Views Created
- `nyc_tlc.v_daily_summary`
- `nyc_tlc.v_hourly_patterns`
- `nyc_tlc.v_payment_breakdown`
- `nyc_tlc.v_top_pickup_zones`
"""
    create_markdown_artifact(key="etl-run-summary", markdown=summary)


# ── Flow ─────────────────────────────────────────────────────────────────────

@flow(name="NYC TLC ETL Pipeline", log_prints=True)
def nyc_tlc_etl_pipeline(
    year: int = 2023,
    months: list[int] = [1],
    taxi_type: str = "yellow",
):
    """
    Full ETL pipeline for NYC TLC trip data.

    Args:
        year: Year to process
        months: List of months to process
        taxi_type: Taxi type ('yellow', 'green', 'fhv')
    """
    # Extract
    raw_files = extract_task(year=year, months=months, taxi_type=taxi_type)

    # Transform
    transformed_rows = transform_task(raw_files)

    # Load
    quality = load_task()

    # Summarise
    publish_summary_task(
        year=year,
        months=months,
        raw_files=raw_files,
        transformed_rows=transformed_rows,
        quality=quality,
    )

    return quality


if __name__ == "__main__":
    # Run the pipeline for Jan 2023
    result = nyc_tlc_etl_pipeline(year=2023, months=[1])
    print("\n=== Pipeline complete ===")
    print(result)