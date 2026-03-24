"""
data_loader.py — Utilities to load and preprocess financial transaction data.
Supports both Pandas (local) and PySpark (Databricks) workflows.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(filepath: str) -> pd.DataFrame:
    """Load transaction CSV, parse dates, and validate schema."""
    expected_columns = [
        "transaction_id", "user_id", "date", "category",
        "amount", "payment_method", "merchant", "description",
    ]
    df = pd.read_csv(filepath, parse_dates=["date"])
    missing = set(expected_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns useful for downstream analysis."""
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")
    df["year"] = df["date"].dt.year
    df["is_income"] = df["amount"] > 0
    df["abs_amount"] = df["amount"].abs()
    return df


def get_user_data(df: pd.DataFrame, user_id: str) -> pd.DataFrame:
    """Return transactions for a single user."""
    user_df = df[df["user_id"] == user_id].copy()
    if user_df.empty:
        raise ValueError(f"No data found for user '{user_id}'")
    return user_df


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Compute high-level financial summary statistics."""
    total_income = df.loc[df["amount"] > 0, "amount"].sum()
    total_expenses = df.loc[df["amount"] < 0, "amount"].sum()  # negative
    net_savings = total_income + total_expenses  # expenses are negative
    savings_rate = (net_savings / total_income * 100) if total_income > 0 else 0.0
    return {
        "total_income": round(total_income, 2),
        "total_expenses": round(abs(total_expenses), 2),
        "net_savings": round(net_savings, 2),
        "savings_rate": round(savings_rate, 2),
        "transaction_count": len(df),
    }


def load_data_spark(spark, filepath: str):
    """Load transaction CSV into a PySpark DataFrame (for Databricks)."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StructType, StructField, StringType, FloatType, DateType,
    )

    schema = StructType([
        StructField("transaction_id", StringType(), True),
        StructField("user_id", StringType(), True),
        StructField("date", StringType(), True),
        StructField("category", StringType(), True),
        StructField("amount", FloatType(), True),
        StructField("payment_method", StringType(), True),
        StructField("merchant", StringType(), True),
        StructField("description", StringType(), True),
    ])

    sdf = spark.read.csv(filepath, header=True, schema=schema)
    sdf = sdf.withColumn("date", F.to_date(F.col("date")))
    return sdf
