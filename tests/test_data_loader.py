"""Tests for src/data_loader.py"""

import os
import pytest
import pandas as pd
import numpy as np

from src.data_loader import load_data, preprocess, get_user_data, get_summary_stats


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "virtual_financial_advisor_data.csv")


@pytest.fixture
def raw_df():
    return load_data(DATA_PATH)


@pytest.fixture
def prep_df(raw_df):
    return preprocess(raw_df)


class TestLoadData:
    def test_loads_csv(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)

    def test_row_count(self, raw_df):
        assert len(raw_df) == 5200

    def test_columns(self, raw_df):
        expected = {
            "transaction_id", "user_id", "date", "category",
            "amount", "payment_method", "merchant", "description",
        }
        assert expected.issubset(set(raw_df.columns))

    def test_date_parsed(self, raw_df):
        assert pd.api.types.is_datetime64_any_dtype(raw_df["date"])

    def test_date_range(self, raw_df):
        assert raw_df["date"].min().year >= 2023
        assert raw_df["date"].max().year <= 2024

    def test_missing_column_raises(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("col_a,col_b\n1,2\n")
        with pytest.raises(ValueError, match="Missing columns"):
            load_data(str(bad_csv))


class TestPreprocess:
    def test_derived_columns(self, prep_df):
        assert "month" in prep_df.columns
        assert "year" in prep_df.columns
        assert "is_income" in prep_df.columns
        assert "abs_amount" in prep_df.columns

    def test_is_income_flag(self, prep_df):
        income_rows = prep_df[prep_df["is_income"]]
        assert (income_rows["amount"] > 0).all()

    def test_abs_amount(self, prep_df):
        assert (prep_df["abs_amount"] >= 0).all()


class TestGetUserData:
    def test_returns_single_user(self, prep_df):
        user_df = get_user_data(prep_df, "user_1")
        assert (user_df["user_id"] == "user_1").all()
        assert len(user_df) > 0

    def test_invalid_user_raises(self, prep_df):
        with pytest.raises(ValueError, match="No data found"):
            get_user_data(prep_df, "nonexistent_user")


class TestGetSummaryStats:
    def test_keys(self, prep_df):
        stats = get_summary_stats(prep_df)
        expected_keys = {
            "total_income", "total_expenses", "net_savings",
            "savings_rate", "transaction_count",
        }
        assert expected_keys == set(stats.keys())

    def test_transaction_count(self, prep_df):
        stats = get_summary_stats(prep_df)
        assert stats["transaction_count"] == 5200

    def test_income_positive(self, prep_df):
        stats = get_summary_stats(prep_df)
        assert stats["total_income"] > 0

    def test_expenses_positive(self, prep_df):
        stats = get_summary_stats(prep_df)
        assert stats["total_expenses"] > 0  # stored as absolute value

    def test_savings_rate_range(self, prep_df):
        user_df = get_user_data(prep_df, "user_1")
        stats = get_summary_stats(user_df)
        assert -100 <= stats["savings_rate"] <= 100
