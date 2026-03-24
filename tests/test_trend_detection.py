"""Tests for src/trend_detection.py"""

import os
import pytest
import pandas as pd
import numpy as np

from src.data_loader import load_data, preprocess, get_user_data
from src.trend_detection import (
    monthly_trends,
    spending_trend,
    detect_risky_patterns,
    financial_health_score,
    category_trend,
)


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "virtual_financial_advisor_data.csv")


@pytest.fixture
def user_df():
    df = preprocess(load_data(DATA_PATH))
    return get_user_data(df, "user_1")


class TestMonthlyTrends:
    def test_returns_dataframe(self, user_df):
        result = monthly_trends(user_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, user_df):
        result = monthly_trends(user_df)
        for col in ["month", "income", "expenses", "net_savings", "savings_rate"]:
            assert col in result.columns

    def test_month_count(self, user_df):
        result = monthly_trends(user_df)
        # 2 years of data → up to 24 months
        assert 1 <= len(result) <= 24

    def test_income_non_negative(self, user_df):
        result = monthly_trends(user_df)
        assert (result["income"] >= 0).all()

    def test_expenses_non_negative(self, user_df):
        result = monthly_trends(user_df)
        assert (result["expenses"] >= 0).all()


class TestSpendingTrend:
    def test_has_rolling_avg(self, user_df):
        result = spending_trend(user_df)
        assert "rolling_avg" in result.columns
        assert "trend_direction" in result.columns

    def test_directions_valid(self, user_df):
        result = spending_trend(user_df)
        valid = {"increasing", "decreasing", "stable"}
        assert set(result["trend_direction"].unique()).issubset(valid)


class TestDetectRiskyPatterns:
    def test_returns_list(self, user_df):
        result = detect_risky_patterns(user_df)
        assert isinstance(result, list)

    def test_alert_structure(self, user_df):
        result = detect_risky_patterns(user_df)
        for alert in result:
            assert "risk" in alert
            assert "severity" in alert
            assert "detail" in alert
            assert alert["severity"] in ("high", "medium", "low")

    def test_detects_overspending_with_synthetic(self):
        """Create data guaranteed to overspend."""
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        rows = []
        for d in dates:
            rows.append({"date": d, "amount": -500, "category": "Dining", "user_id": "u"})
        # Only tiny income
        rows.append({"date": dates[0], "amount": 100, "category": "Salary", "user_id": "u"})
        df = pd.DataFrame(rows)
        df["transaction_id"] = range(len(df))
        df["payment_method"] = "Cash"
        df["merchant"] = "Test"
        df["description"] = "Test"

        risks = detect_risky_patterns(df)
        risk_names = [r["risk"] for r in risks]
        assert "Overspending" in risk_names


class TestFinancialHealthScore:
    def test_returns_dict(self, user_df):
        result = financial_health_score(user_df)
        assert isinstance(result, dict)
        assert "score" in result
        assert "breakdown" in result

    def test_score_range(self, user_df):
        result = financial_health_score(user_df)
        assert 0 <= result["score"] <= 100

    def test_breakdown_keys(self, user_df):
        result = financial_health_score(user_df)
        bd = result["breakdown"]
        assert "savings_pts" in bd
        assert "volatility_pts" in bd
        assert "income_stability_pts" in bd
        assert "discretionary_pts" in bd


class TestCategoryTrend:
    def test_returns_dataframe(self, user_df):
        result = category_trend(user_df, "Groceries")
        assert isinstance(result, pd.DataFrame)
        assert "month" in result.columns
        assert "total" in result.columns

    def test_empty_category(self, user_df):
        result = category_trend(user_df, "NonexistentCategory")
        assert len(result) == 0
