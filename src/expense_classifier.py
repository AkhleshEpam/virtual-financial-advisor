"""
expense_classifier.py — Rule-based expense classification functions.
No ML required — uses predefined category-to-classification mapping.
"""

import pandas as pd
import numpy as np


# ── Category mapping ──────────────────────────────────────────────────────────

INCOME_CATEGORIES = {"Salary", "Bonus", "Interest"}
ESSENTIAL_CATEGORIES = {
    "Rent", "Utilities", "Groceries", "Healthcare", "Transport", "Education",
}
DISCRETIONARY_CATEGORIES = {"Entertainment", "Dining"}
SAVINGS_CATEGORIES = {"Savings Transfer"}

CATEGORY_MAP: dict[str, str] = {}
for cat in INCOME_CATEGORIES:
    CATEGORY_MAP[cat] = "Income"
for cat in ESSENTIAL_CATEGORIES:
    CATEGORY_MAP[cat] = "Essential Expense"
for cat in DISCRETIONARY_CATEGORIES:
    CATEGORY_MAP[cat] = "Discretionary Expense"
for cat in SAVINGS_CATEGORIES:
    CATEGORY_MAP[cat] = "Savings"


# ── Public functions ──────────────────────────────────────────────────────────

def classify_transaction(category: str, amount: float) -> str:
    """Return the high-level classification for a single transaction."""
    return CATEGORY_MAP.get(category, "Other")


def classify_all(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'classification' column to the dataframe."""
    df = df.copy()
    df["classification"] = df.apply(
        lambda row: classify_transaction(row["category"], row["amount"]), axis=1
    )
    return df


def get_category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate spending by classification and category."""
    if "classification" not in df.columns:
        df = classify_all(df)
    breakdown = (
        df.groupby(["classification", "category"])
        .agg(
            total_amount=("amount", "sum"),
            abs_total=("amount", lambda x: x.abs().sum()),
            count=("amount", "count"),
        )
        .reset_index()
        .sort_values("abs_total", ascending=False)
    )
    return breakdown


def detect_unusual_expenses(
    df: pd.DataFrame, threshold_multiplier: float = 2.0
) -> pd.DataFrame:
    """
    Flag individual expense transactions whose |amount| exceeds
    threshold_multiplier × the mean |amount| for that category.
    """
    expenses = df[df["amount"] < 0].copy()
    expenses["abs_amount"] = expenses["amount"].abs()

    category_means = expenses.groupby("category")["abs_amount"].transform("mean")
    flagged = expenses[expenses["abs_amount"] > threshold_multiplier * category_means]
    return flagged
