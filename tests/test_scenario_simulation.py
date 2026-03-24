"""Tests for src/scenario_simulation.py"""

import os
import pytest
import pandas as pd
import numpy as np

from src.data_loader import load_data, preprocess, get_user_data
from src.scenario_simulation import (
    simulate_savings_increase,
    simulate_expense_reduction,
    simulate_income_change,
    compare_scenarios,
    project_balance,
)


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "virtual_financial_advisor_data.csv")


@pytest.fixture
def user_df():
    df = preprocess(load_data(DATA_PATH))
    return get_user_data(df, "user_1")


class TestSimulateSavingsIncrease:
    def test_returns_dict(self, user_df):
        result = simulate_savings_increase(user_df, 10)
        assert isinstance(result, dict)

    def test_projected_greater_than_baseline(self, user_df):
        result = simulate_savings_increase(user_df, 10, months=12)
        assert result["projected_cumulative"] > result["baseline_cumulative"]

    def test_additional_savings_positive(self, user_df):
        result = simulate_savings_increase(user_df, 10)
        assert result["additional_savings"] > 0

    def test_zero_increase(self, user_df):
        result = simulate_savings_increase(user_df, 0)
        assert result["additional_savings"] == 0

    def test_has_scenario_name(self, user_df):
        result = simulate_savings_increase(user_df, 15)
        assert "15" in result["scenario"]


class TestSimulateExpenseReduction:
    def test_returns_dict(self, user_df):
        result = simulate_expense_reduction(user_df, "Dining", 20)
        assert isinstance(result, dict)

    def test_projected_greater_than_baseline(self, user_df):
        result = simulate_expense_reduction(user_df, "Dining", 20, months=12)
        assert result["projected_cumulative"] >= result["baseline_cumulative"]

    def test_has_category_expense(self, user_df):
        result = simulate_expense_reduction(user_df, "Dining", 20)
        assert "avg_monthly_category_expense" in result


class TestSimulateIncomeChange:
    def test_increase(self, user_df):
        result = simulate_income_change(user_df, 15)
        assert result["projected_cumulative"] > result["baseline_cumulative"]

    def test_decrease(self, user_df):
        result = simulate_income_change(user_df, -10)
        assert result["projected_cumulative"] < result["baseline_cumulative"]


class TestCompareScenarios:
    def test_returns_dataframe(self, user_df):
        s1 = simulate_savings_increase(user_df, 10)
        s2 = simulate_expense_reduction(user_df, "Dining", 20)
        s3 = simulate_income_change(user_df, 15)
        result = compare_scenarios([s1, s2, s3])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_has_expected_columns(self, user_df):
        s1 = simulate_savings_increase(user_df, 10)
        result = compare_scenarios([s1])
        for col in ["Scenario", "Months", "Baseline Cumulative", "Projected Cumulative", "Additional Savings"]:
            assert col in result.columns


class TestProjectBalance:
    def test_returns_dataframe(self, user_df):
        result = project_balance(user_df, 5000, 12)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 12

    def test_month_column(self, user_df):
        result = project_balance(user_df, 5000, 6)
        assert list(result["month"]) == [1, 2, 3, 4, 5, 6]

    def test_initial_balance_affects_projection(self, user_df):
        low = project_balance(user_df, 1000, 12)
        high = project_balance(user_df, 10000, 12)
        assert high.iloc[-1]["projected_balance"] > low.iloc[-1]["projected_balance"]
