"""Tests for src/expense_classifier.py"""

import os
import pytest
import pandas as pd
import numpy as np

from src.expense_classifier import (
    classify_transaction,
    classify_all,
    get_category_breakdown,
    detect_unusual_expenses,
    CATEGORY_MAP,
    INCOME_CATEGORIES,
    ESSENTIAL_CATEGORIES,
    DISCRETIONARY_CATEGORIES,
    SAVINGS_CATEGORIES,
)
from src.data_loader import load_data, preprocess


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "virtual_financial_advisor_data.csv")


@pytest.fixture
def df():
    return preprocess(load_data(DATA_PATH))


class TestClassifyTransaction:
    def test_income_categories(self):
        for cat in INCOME_CATEGORIES:
            assert classify_transaction(cat, 3000) == "Income"

    def test_essential_categories(self):
        for cat in ESSENTIAL_CATEGORIES:
            assert classify_transaction(cat, -200) == "Essential Expense"

    def test_discretionary_categories(self):
        for cat in DISCRETIONARY_CATEGORIES:
            assert classify_transaction(cat, -50) == "Discretionary Expense"

    def test_savings_category(self):
        assert classify_transaction("Savings Transfer", -500) == "Savings"

    def test_unknown_category(self):
        assert classify_transaction("UnknownCat", -10) == "Other"

    def test_zero_amount(self):
        assert classify_transaction("Salary", 0.0) == "Income"


class TestClassifyAll:
    def test_adds_classification_column(self, df):
        result = classify_all(df)
        assert "classification" in result.columns

    def test_no_unknowns_in_dataset(self, df):
        result = classify_all(df)
        assert "Other" not in result["classification"].values

    def test_all_rows_classified(self, df):
        result = classify_all(df)
        assert result["classification"].notna().all()


class TestGetCategoryBreakdown:
    def test_returns_dataframe(self, df):
        result = get_category_breakdown(df)
        assert isinstance(result, pd.DataFrame)
        assert "classification" in result.columns
        assert "category" in result.columns
        assert "total_amount" in result.columns

    def test_all_categories_represented(self, df):
        result = get_category_breakdown(df)
        cats_in_result = set(result["category"].unique())
        cats_in_data = set(df["category"].unique())
        assert cats_in_data == cats_in_result


class TestDetectUnusualExpenses:
    def test_returns_dataframe(self, df):
        result = detect_unusual_expenses(df)
        assert isinstance(result, pd.DataFrame)

    def test_flagged_are_expenses(self, df):
        result = detect_unusual_expenses(df)
        if not result.empty:
            assert (result["amount"] < 0).all()

    def test_higher_threshold_fewer_flags(self, df):
        loose = detect_unusual_expenses(df, threshold_multiplier=3.0)
        strict = detect_unusual_expenses(df, threshold_multiplier=1.5)
        assert len(loose) <= len(strict)
