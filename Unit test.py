"""
tests/test_transformer.py
--------------------------
Unit tests for the PySpark trip transformer.
Uses a small in-memory DataFrame to avoid needing real TLC data.

Prompt used with Claude Desktop:
  "Write pytest unit tests for the PySpark transformer covering
   column standardisation, time features, trip metrics, quality
   filters, and payment enrichment. Use a minimal synthetic dataset."
"""

import pytest
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, IntegerType, DoubleType, TimestampType, LongType
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers.trip_transformer import (
    standardise_columns,
    add_time_features,
    add_trip_metrics,
    apply_quality_filters,
    enrich_payment_type,
    drop_nulls,
)


@pytest.fixture(scope="session")
def spark():
    """Shared SparkSession for all tests."""
    return (
        SparkSession.builder
        .master("local[1]")
        .appName("test-transformer")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )


@pytest.fixture
def sample_df(spark):
    """A minimal synthetic DataFrame mimicking TLC schema."""
    pickup = datetime(2023, 1, 15, 8, 30, 0)
    dropoff = pickup + timedelta(minutes=25)

    data = [
        (1, pickup, dropoff, 2, 3.5, 1, 12.50, 0.5, 0.5, 2.80, 0.3, 16.60, 1),
        (2, pickup, dropoff, 1, 1.2, 1,  7.00, 0.0, 0.5, 0.00, 0.3,  8.30, 2),
        # bad row: zero distance
        (1, pickup, dropoff, 1, 0.0, 1,  5.00, 0.0, 0.5, 0.00, 0.3,  6.30, 1),
        # bad row: null fare
        (1, pickup, dropoff, 1, 2.0, 1,  None, 0.0, 0.5, 0.00, 0.3,  None, 1),
    ]

    schema = StructType([
        StructField("VendorID",              IntegerType(), True),
        StructField("tpep_pickup_datetime",  TimestampType(), True),
        StructField("tpep_dropoff_datetime", TimestampType(), True),
        StructField("passenger_count",       IntegerType(), True),
        StructField("trip_distance",         DoubleType(), True),
        StructField("RatecodeID",            IntegerType(), True),
        StructField("fare_amount",           DoubleType(), True),
        StructField("extra",                 DoubleType(), True),
        StructField("mta_tax",               DoubleType(), True),
        StructField("tip_amount",            DoubleType(), True),
        StructField("tolls_amount",          DoubleType(), True),
        StructField("total_amount",          DoubleType(), True),
        StructField("payment_type",          IntegerType(), True),
    ])

    return spark.createDataFrame(data, schema=schema)


class TestColumnStandardisation:
    def test_renames_pickup_datetime(self, sample_df):
        df = standardise_columns(sample_df)
        assert "pickup_datetime" in df.columns
        assert "tpep_pickup_datetime" not in df.columns

    def test_renames_dropoff_datetime(self, sample_df):
        df = standardise_columns(sample_df)
        assert "dropoff_datetime" in df.columns

    def test_renames_vendor_id(self, sample_df):
        df = standardise_columns(sample_df)
        assert "vendor_id" in df.columns

    def test_renames_location_ids(self, sample_df):
        df = standardise_columns(sample_df)
        # Original columns should be gone
        assert "PULocationID" not in df.columns
        assert "DOLocationID" not in df.columns


class TestDropNulls:
    def test_removes_null_fare_rows(self, sample_df):
        df = standardise_columns(sample_df)
        before = df.count()
        df = drop_nulls(df)
        assert df.count() == before - 1  # one null fare row

    def test_no_nulls_in_critical_cols(self, sample_df):
        df = standardise_columns(sample_df)
        df = drop_nulls(df)
        for col in ["pickup_datetime", "dropoff_datetime", "fare_amount", "trip_distance"]:
            null_count = df.filter(df[col].isNull()).count()
            assert null_count == 0, f"Found nulls in {col}"


class TestTimeFeatures:
    def test_pickup_year(self, sample_df):
        df = standardise_columns(sample_df)
        df = drop_nulls(df)
        df = add_time_features(df)
        years = {row["pickup_year"] for row in df.select("pickup_year").collect()}
        assert years == {2023}

    def test_pickup_hour(self, sample_df):
        df = standardise_columns(sample_df)
        df = drop_nulls(df)
        df = add_time_features(df)
        hours = {row["pickup_hour"] for row in df.select("pickup_hour").collect()}
        assert 8 in hours

    def test_time_of_day_is_morning(self, sample_df):
        df = standardise_columns(sample_df)
        df = drop_nulls(df)
        df = add_time_features(df)
        tod = {row["time_of_day"] for row in df.select("time_of_day").collect()}
        assert "morning" in tod

    def test_is_weekend_false_on_weekday(self, sample_df):
        # Jan 15 2023 is a Sunday → is_weekend = True
        df = standardise_columns(sample_df)
        df = drop_nulls(df)
        df = add_time_features(df)
        row = df.select("is_weekend").first()
        assert row["is_weekend"] is True


class TestTripMetrics:
    def test_trip_duration_positive(self, sample_df):
        df = standardise_columns(sample_df)
        df = drop_nulls(df)
        df = add_time_features(df)
        df = add_trip_metrics(df)
        durations = [r["trip_duration_seconds"] for r in df.select("trip_duration_seconds").collect()]
        assert all(d > 0 for d in durations)

    def test_tip_pct_non_negative(self, sample_df):
        df = standardise_columns(sample_df)
        df = drop_nulls(df)
        df = add_time_features(df)
        df = add_trip_metrics(df)
        tips = [r["tip_pct"] for r in df.select("tip_pct").collect()]
        assert all(t >= 0 for t in tips)

    def test_avg_speed_positive(self, sample_df):
        df = standardise_columns(sample_df)
        df = drop_nulls(df)
        df = add_time_features(df)
        df = add_trip_metrics(df)
        # Only rows with non-null speed
        rows = df.filter(df["avg_speed_mph"].isNotNull()).select("avg_speed_mph").collect()
        assert all(r["avg_speed_mph"] > 0 for r in rows)


class TestQualityFilters:
    def test_removes_zero_distance(self, sample_df):
        df = standardise_columns(sample_df)
        df = drop_nulls(df)
        df = add_time_features(df)
        df = add_trip_metrics(df)
        before = df.count()
        df = apply_quality_filters(df)
        assert df.count() < before  # zero-distance row should be removed

    def test_all_remaining_distances_in_range(self, sample_df):
        df = standardise_columns(sample_df)
        df = drop_nulls(df)
        df = add_time_features(df)
        df = add_trip_metrics(df)
        df = apply_quality_filters(df)
        bad = df.filter((df["trip_distance"] < 0.1) | (df["trip_distance"] > 200)).count()
        assert bad == 0


class TestPaymentEnrichment:
    def test_credit_card_label(self, sample_df):
        df = standardise_columns(sample_df)
        df = drop_nulls(df)
        df = add_time_features(df)
        df = add_trip_metrics(df)
        df = apply_quality_filters(df)
        df = enrich_payment_type(df)
        labels = {r["payment_type_desc"] for r in df.select("payment_type_desc").collect()}
        assert "credit_card" in labels

    def test_cash_label(self, sample_df):
        df = standardise_columns(sample_df)
        df = drop_nulls(df)
        df = add_time_features(df)
        df = add_trip_metrics(df)
        df = apply_quality_filters(df)
        df = enrich_payment_type(df)
        labels = {r["payment_type_desc"] for r in df.select("payment_type_desc").collect()}
        assert "cash" in labels