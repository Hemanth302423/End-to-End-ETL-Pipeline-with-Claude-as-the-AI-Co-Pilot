"""
loaders/duckdb_loader.py
--------------------------
Loads transformed Spark DataFrames into DuckDB as the local data warehouse.
Supports upsert, partitioned writes, and schema evolution.

Prompt used with Claude Desktop:
  "Write a DuckDB loader that accepts a PySpark DataFrame, writes it
   to a local DuckDB warehouse, supports upsert by partition key,
   and creates summary analytics views automatically."
"""

import logging
from pathlib import Path
from pyspark.sql import DataFrame
import duckdb

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/warehouse.duckdb")


class DuckDBLoader:
    """
    Loads PySpark DataFrames into DuckDB tables.
    Uses Parquet as the interchange format (Spark → Parquet → DuckDB).
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        logger.info(f"Connected to DuckDB at {self.db_path}")
        self._init_schema()

    def _init_schema(self):
        """Create warehouse schema and configure DuckDB settings."""
        self.conn.execute("CREATE SCHEMA IF NOT EXISTS nyc_tlc")
        self.conn.execute("SET memory_limit='2GB'")
        self.conn.execute("SET threads=4")
        logger.info("DuckDB schema initialised")

    def load(
        self,
        df: DataFrame,
        table_name: str,
        staging_path: str | Path = "data/processed/staging",
        partition_cols: list[str] | None = None,
        mode: str = "overwrite",
    ) -> int:
        """
        Write Spark DataFrame to DuckDB table.

        Args:
            df: Transformed Spark DataFrame
            table_name: Target table name in nyc_tlc schema
            staging_path: Temp Parquet staging area
            partition_cols: Columns to partition Parquet by (e.g. ['pickup_year','pickup_month'])
            mode: 'overwrite' or 'append'

        Returns:
            Number of rows loaded
        """
        staging_path = Path(staging_path)
        staging_path.mkdir(parents=True, exist_ok=True)

        # Step 1: Write Spark DF → Parquet staging
        logger.info(f"Writing to staging: {staging_path}")
        writer = df.write.mode(mode).format("parquet")
        if partition_cols:
            writer = writer.partitionBy(*partition_cols)
        writer.save(str(staging_path))

        # Step 2: Load from Parquet → DuckDB
        full_table = f"nyc_tlc.{table_name}"
        parquet_glob = f"{staging_path}/**/*.parquet"

        logger.info(f"Loading into DuckDB table: {full_table}")
        if mode == "overwrite":
            self.conn.execute(f"DROP TABLE IF EXISTS {full_table}")
            self.conn.execute(
                f"CREATE TABLE {full_table} AS SELECT * FROM read_parquet('{parquet_glob}', hive_partitioning=true)"
            )
        else:  # append
            # Create table if it doesn't exist
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {full_table} AS
                SELECT * FROM read_parquet('{parquet_glob}', hive_partitioning=true) WHERE 1=0
            """)
            self.conn.execute(
                f"INSERT INTO {full_table} SELECT * FROM read_parquet('{parquet_glob}', hive_partitioning=true)"
            )

        row_count = self.conn.execute(f"SELECT COUNT(*) FROM {full_table}").fetchone()[0]
        logger.info(f"Loaded {row_count:,} rows into {full_table}")
        return row_count

    def create_analytics_views(self):
        """Create pre-built analytics views for downstream querying."""
        views = {
            "v_daily_summary": """
                CREATE OR REPLACE VIEW nyc_tlc.v_daily_summary AS
                SELECT
                    pickup_year,
                    pickup_month,
                    pickup_day,
                    COUNT(*)                        AS total_trips,
                    ROUND(AVG(trip_distance), 2)    AS avg_distance_miles,
                    ROUND(AVG(trip_duration_minutes), 2) AS avg_duration_mins,
                    ROUND(AVG(fare_amount), 2)      AS avg_fare,
                    ROUND(AVG(tip_pct), 2)          AS avg_tip_pct,
                    ROUND(SUM(total_amount), 2)     AS total_revenue,
                    ROUND(AVG(avg_speed_mph), 2)    AS avg_speed_mph
                FROM nyc_tlc.trips
                GROUP BY 1, 2, 3
                ORDER BY 1, 2, 3
            """,
            "v_hourly_patterns": """
                CREATE OR REPLACE VIEW nyc_tlc.v_hourly_patterns AS
                SELECT
                    pickup_hour,
                    is_weekend,
                    COUNT(*)                        AS trip_count,
                    ROUND(AVG(fare_amount), 2)      AS avg_fare,
                    ROUND(AVG(trip_duration_minutes), 2) AS avg_duration_mins,
                    ROUND(AVG(tip_pct), 2)          AS avg_tip_pct
                FROM nyc_tlc.trips
                GROUP BY 1, 2
                ORDER BY 1, 2
            """,
            "v_payment_breakdown": """
                CREATE OR REPLACE VIEW nyc_tlc.v_payment_breakdown AS
                SELECT
                    payment_type_desc,
                    COUNT(*)                        AS trip_count,
                    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_trips,
                    ROUND(AVG(tip_pct), 2)          AS avg_tip_pct,
                    ROUND(SUM(total_amount), 2)     AS total_revenue
                FROM nyc_tlc.trips
                GROUP BY 1
                ORDER BY 2 DESC
            """,
            "v_top_pickup_zones": """
                CREATE OR REPLACE VIEW nyc_tlc.v_top_pickup_zones AS
                SELECT
                    pickup_location_id,
                    COUNT(*)                        AS trip_count,
                    ROUND(AVG(fare_amount), 2)      AS avg_fare,
                    ROUND(AVG(tip_pct), 2)          AS avg_tip_pct
                FROM nyc_tlc.trips
                GROUP BY 1
                ORDER BY 2 DESC
                LIMIT 50
            """,
        }

        for view_name, sql in views.items():
            self.conn.execute(sql)
            logger.info(f"Created view: nyc_tlc.{view_name}")

    def run_data_quality_checks(self, table_name: str = "trips") -> dict:
        """Run post-load data quality assertions."""
        full_table = f"nyc_tlc.{table_name}"
        checks = {}

        checks["total_rows"] = self.conn.execute(
            f"SELECT COUNT(*) FROM {full_table}"
        ).fetchone()[0]

        checks["null_fare_count"] = self.conn.execute(
            f"SELECT COUNT(*) FROM {full_table} WHERE fare_amount IS NULL"
        ).fetchone()[0]

        checks["negative_fare_count"] = self.conn.execute(
            f"SELECT COUNT(*) FROM {full_table} WHERE fare_amount < 0"
        ).fetchone()[0]

        checks["avg_trip_distance"] = self.conn.execute(
            f"SELECT ROUND(AVG(trip_distance), 3) FROM {full_table}"
        ).fetchone()[0]

        checks["date_range"] = self.conn.execute(
            f"SELECT MIN(pickup_datetime), MAX(pickup_datetime) FROM {full_table}"
        ).fetchone()

        # Assert expectations
        assert checks["total_rows"] > 0,            "❌ No rows loaded!"
        assert checks["null_fare_count"] == 0,      "❌ Null fares found!"
        assert checks["negative_fare_count"] == 0,  "❌ Negative fares found!"
        assert checks["avg_trip_distance"] > 0,     "❌ Avg trip distance is 0!"

        logger.info("✅ All data quality checks passed")
        logger.info(f"   Total rows: {checks['total_rows']:,}")
        logger.info(f"   Date range: {checks['date_range']}")
        logger.info(f"   Avg distance: {checks['avg_trip_distance']} miles")
        return checks

    def close(self):
        self.conn.close()
        logger.info("DuckDB connection closed")