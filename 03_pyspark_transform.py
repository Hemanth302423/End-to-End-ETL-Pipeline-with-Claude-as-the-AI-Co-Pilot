"""
transformers/trip_transformer.py
----------------------------------
PySpark-based transformation layer for NYC TLC Yellow Taxi trip data.
Applies cleaning, enrichment, and business logic transformations.

Prompt used with Claude Desktop:
  "Write a PySpark transformer for NYC taxi data that cleans nulls,
   removes outliers, adds time-based features, calculates derived
   metrics like tip percentage and speed, and partitions output
   by pickup year and month."
"""

import logging
from pathlib import Path
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

logger = logging.getLogger(__name__)

# ── Business rule constants ───────────────────────────────────────────────────
MIN_TRIP_DISTANCE_MILES = 0.1
MAX_TRIP_DISTANCE_MILES = 200.0
MIN_FARE_AMOUNT = 2.50          # NYC TLC minimum fare
MAX_FARE_AMOUNT = 1000.0
MIN_TRIP_DURATION_SECONDS = 30
MAX_TRIP_DURATION_SECONDS = 7 * 3600  # 7 hours
MAX_PASSENGER_COUNT = 6
MIN_SPEED_MPH = 0.5
MAX_SPEED_MPH = 100.0
NYC_BBOX = {"lat_min": 40.4, "lat_max": 41.0, "lon_min": -74.3, "lon_max": -73.6}


def create_spark_session(app_name: str = "NYC-TLC-ETL") -> SparkSession:
    """Create and configure a SparkSession."""
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )


def read_raw(spark: SparkSession, input_path: str | Path) -> DataFrame:
    """Read raw Parquet file(s) from the given path."""
    path = str(input_path)
    df = spark.read.parquet(path)
    logger.info(f"Read {df.count():,} rows from {path}")
    return df


def standardise_columns(df: DataFrame) -> DataFrame:
    """
    Rename and cast columns to a consistent schema.
    Handles both old (2014) and new (2015+) TLC column naming.
    """
    column_map = {
        "tpep_pickup_datetime":  "pickup_datetime",
        "tpep_dropoff_datetime": "dropoff_datetime",
        "PULocationID":          "pickup_location_id",
        "DOLocationID":          "dropoff_location_id",
        "RatecodeID":            "rate_code_id",
        "VendorID":              "vendor_id",
    }
    for old_col, new_col in column_map.items():
        if old_col in df.columns:
            df = df.withColumnRenamed(old_col, new_col)

    # Cast to correct types
    df = (
        df
        .withColumn("passenger_count",  F.col("passenger_count").cast("int"))
        .withColumn("trip_distance",     F.col("trip_distance").cast(DoubleType()))
        .withColumn("fare_amount",       F.col("fare_amount").cast(DoubleType()))
        .withColumn("tip_amount",        F.col("tip_amount").cast(DoubleType()))
        .withColumn("total_amount",      F.col("total_amount").cast(DoubleType()))
        .withColumn("tolls_amount",      F.col("tolls_amount").cast(DoubleType()))
        .withColumn("extra",             F.col("extra").cast(DoubleType()))
        .withColumn("mta_tax",           F.col("mta_tax").cast(DoubleType()))
        .withColumn("improvement_surcharge", F.col("improvement_surcharge").cast(DoubleType()))
    )
    return df


def drop_nulls(df: DataFrame) -> DataFrame:
    """Drop rows with nulls in critical columns."""
    critical_cols = ["pickup_datetime", "dropoff_datetime", "fare_amount", "trip_distance"]
    before = df.count()
    df = df.dropna(subset=critical_cols)
    after = df.count()
    logger.info(f"drop_nulls: removed {before - after:,} rows ({(before - after) / before * 100:.2f}%)")
    return df


def add_time_features(df: DataFrame) -> DataFrame:
    """Derive time-based features from pickup datetime."""
    return (
        df
        .withColumn("pickup_year",        F.year("pickup_datetime"))
        .withColumn("pickup_month",       F.month("pickup_datetime"))
        .withColumn("pickup_day",         F.dayofmonth("pickup_datetime"))
        .withColumn("pickup_hour",        F.hour("pickup_datetime"))
        .withColumn("pickup_day_of_week", F.dayofweek("pickup_datetime"))  # 1=Sun, 7=Sat
        .withColumn("pickup_week",        F.weekofyear("pickup_datetime"))
        .withColumn("is_weekend",
            F.col("pickup_day_of_week").isin([1, 7]).cast("boolean")
        )
        .withColumn("time_of_day",
            F.when(F.col("pickup_hour").between(5, 11),  F.lit("morning"))
             .when(F.col("pickup_hour").between(12, 16), F.lit("afternoon"))
             .when(F.col("pickup_hour").between(17, 20), F.lit("evening"))
             .otherwise(F.lit("night"))
        )
    )


def add_trip_metrics(df: DataFrame) -> DataFrame:
    """Calculate derived trip metrics."""
    return (
        df
        # Trip duration in seconds
        .withColumn("trip_duration_seconds",
            F.unix_timestamp("dropoff_datetime") - F.unix_timestamp("pickup_datetime")
        )
        .withColumn("trip_duration_minutes",
            F.round(F.col("trip_duration_seconds") / 60.0, 2)
        )
        # Speed in mph
        .withColumn("avg_speed_mph",
            F.when(
                F.col("trip_duration_seconds") > 0,
                F.round(
                    F.col("trip_distance") / (F.col("trip_duration_seconds") / 3600.0), 2
                )
            ).otherwise(F.lit(None).cast(DoubleType()))
        )
        # Tip percentage
        .withColumn("tip_pct",
            F.when(
                F.col("fare_amount") > 0,
                F.round(F.col("tip_amount") / F.col("fare_amount") * 100.0, 2)
            ).otherwise(F.lit(0.0))
        )
        # Fare per mile
        .withColumn("fare_per_mile",
            F.when(
                F.col("trip_distance") > 0,
                F.round(F.col("fare_amount") / F.col("trip_distance"), 2)
            ).otherwise(F.lit(None).cast(DoubleType()))
        )
    )


def apply_quality_filters(df: DataFrame) -> DataFrame:
    """
    Remove rows that violate business rules / are clearly erroneous.
    Each filter is logged individually for observability.
    """
    filters = {
        "valid_distance":  (F.col("trip_distance").between(MIN_TRIP_DISTANCE_MILES, MAX_TRIP_DISTANCE_MILES)),
        "valid_fare":      (F.col("fare_amount").between(MIN_FARE_AMOUNT, MAX_FARE_AMOUNT)),
        "valid_duration":  (F.col("trip_duration_seconds").between(MIN_TRIP_DURATION_SECONDS, MAX_TRIP_DURATION_SECONDS)),
        "valid_passengers":(F.col("passenger_count").between(1, MAX_PASSENGER_COUNT)),
        "valid_speed":     (F.col("avg_speed_mph").isNull() | F.col("avg_speed_mph").between(MIN_SPEED_MPH, MAX_SPEED_MPH)),
        "non_negative_tip":(F.col("tip_amount") >= 0),
        "chronological":   (F.col("dropoff_datetime") > F.col("pickup_datetime")),
    }

    before = df.count()
    for name, condition in filters.items():
        count_before = df.count()
        df = df.filter(condition)
        removed = count_before - df.count()
        if removed > 0:
            logger.info(f"  Filter '{name}': removed {removed:,} rows")

    logger.info(f"apply_quality_filters: {before - df.count():,} total rows removed")
    return df


def enrich_payment_type(df: DataFrame) -> DataFrame:
    """Map numeric payment type codes to human-readable labels."""
    payment_map = {1: "credit_card", 2: "cash", 3: "no_charge", 4: "dispute", 5: "unknown", 6: "voided"}
    expr = F.lit("unknown")
    for code, label in payment_map.items():
        expr = F.when(F.col("payment_type") == code, F.lit(label)).otherwise(expr)

    return df.withColumn("payment_type_desc", expr)


def transform(df: DataFrame) -> DataFrame:
    """
    Run the full transformation pipeline.
    Order matters: standardise → nulls → time → metrics → filters → enrich
    """
    logger.info("Starting transformation pipeline...")
    df = standardise_columns(df)
    df = drop_nulls(df)
    df = add_time_features(df)
    df = add_trip_metrics(df)
    df = apply_quality_filters(df)
    df = enrich_payment_type(df)
    logger.info(f"Transformation complete. Final row count: {df.count():,}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    spark = create_spark_session()

    raw_path = "data/raw/yellow_tripdata_2023-01.parquet"
    df_raw = read_raw(spark, raw_path)
    df_transformed = transform(df_raw)
    df_transformed.printSchema()
    df_transformed.show(5, truncate=False)

    spark.stop()