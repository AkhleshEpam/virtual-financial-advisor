"""
scenario_simulation.py — "What-if" projection logic for savings, expenses, and income.
Pure Python/Pandas — no ML models.
"""

import pandas as pd
import numpy as np


def _avg_monthly(df: pd.DataFrame) -> dict:
    """Compute average monthly income and expenses from transaction data."""
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")
    monthly = df.groupby("month")["amount"].sum()
    income_months = df[df["amount"] > 0].groupby("month")["amount"].sum()
    expense_months = df[df["amount"] < 0].groupby("month")["amount"].sum().abs()
    return {
        "avg_income": income_months.mean() if len(income_months) else 0,
        "avg_expenses": expense_months.mean() if len(expense_months) else 0,
        "avg_net": monthly.mean() if len(monthly) else 0,
        "months_of_data": len(monthly),
    }


def simulate_savings_increase(
    df: pd.DataFrame, increase_pct: float, months: int = 12
) -> dict:
    """
    Project cumulative savings if the user saves `increase_pct`% more each month.
    Returns baseline vs projected savings over the projection period.
    """
    avgs = _avg_monthly(df)
    baseline_monthly_savings = avgs["avg_income"] - avgs["avg_expenses"]
    additional = avgs["avg_income"] * (increase_pct / 100)
    projected_monthly_savings = baseline_monthly_savings + additional

    baseline_cumulative = baseline_monthly_savings * months
    projected_cumulative = projected_monthly_savings * months

    return {
        "scenario": f"Save {increase_pct}% more of income each month",
        "months": months,
        "baseline_monthly_savings": round(baseline_monthly_savings, 2),
        "projected_monthly_savings": round(projected_monthly_savings, 2),
        "baseline_cumulative": round(baseline_cumulative, 2),
        "projected_cumulative": round(projected_cumulative, 2),
        "additional_savings": round(projected_cumulative - baseline_cumulative, 2),
    }


def simulate_expense_reduction(
    df: pd.DataFrame, category: str, reduction_pct: float, months: int = 12
) -> dict:
    """
    Project savings if expenses in `category` are reduced by `reduction_pct`%.
    """
    avgs = _avg_monthly(df)
    cat_expenses = df[
        (df["amount"] < 0) & (df["category"] == category)
    ]["amount"].abs()

    df_copy = df.copy()
    df_copy["month"] = df_copy["date"].dt.to_period("M")
    n_months = df_copy["month"].nunique()
    avg_cat_monthly = cat_expenses.sum() / max(n_months, 1)
    monthly_saving = avg_cat_monthly * (reduction_pct / 100)

    baseline_monthly_savings = avgs["avg_income"] - avgs["avg_expenses"]
    projected_monthly_savings = baseline_monthly_savings + monthly_saving

    return {
        "scenario": f"Reduce {category} expenses by {reduction_pct}%",
        "months": months,
        "avg_monthly_category_expense": round(avg_cat_monthly, 2),
        "monthly_saving": round(monthly_saving, 2),
        "baseline_monthly_savings": round(baseline_monthly_savings, 2),
        "projected_monthly_savings": round(projected_monthly_savings, 2),
        "baseline_cumulative": round(baseline_monthly_savings * months, 2),
        "projected_cumulative": round(projected_monthly_savings * months, 2),
        "additional_savings": round(monthly_saving * months, 2),
    }


def simulate_income_change(
    df: pd.DataFrame, change_pct: float, months: int = 12
) -> dict:
    """Project impact of a percentage change in income."""
    avgs = _avg_monthly(df)
    income_change = avgs["avg_income"] * (change_pct / 100)
    baseline_monthly_savings = avgs["avg_income"] - avgs["avg_expenses"]
    projected_monthly_savings = baseline_monthly_savings + income_change

    return {
        "scenario": f"Income {'increase' if change_pct > 0 else 'decrease'} of {abs(change_pct)}%",
        "months": months,
        "avg_income": round(avgs["avg_income"], 2),
        "projected_income": round(avgs["avg_income"] + income_change, 2),
        "baseline_monthly_savings": round(baseline_monthly_savings, 2),
        "projected_monthly_savings": round(projected_monthly_savings, 2),
        "baseline_cumulative": round(baseline_monthly_savings * months, 2),
        "projected_cumulative": round(projected_monthly_savings * months, 2),
        "additional_savings": round(income_change * months, 2),
    }


def compare_scenarios(scenarios: list[dict]) -> pd.DataFrame:
    """Side-by-side comparison of multiple scenario results."""
    rows = []
    for s in scenarios:
        rows.append({
            "Scenario": s["scenario"],
            "Months": s["months"],
            "Baseline Cumulative": s["baseline_cumulative"],
            "Projected Cumulative": s["projected_cumulative"],
            "Additional Savings": s["additional_savings"],
        })
    return pd.DataFrame(rows)


def project_balance(
    df: pd.DataFrame, initial_balance: float, months: int = 12
) -> pd.DataFrame:
    """
    Month-by-month balance projection based on average income/expense patterns.
    """
    avgs = _avg_monthly(df)
    avg_net = avgs["avg_income"] - avgs["avg_expenses"]

    projection = []
    balance = initial_balance
    for m in range(1, months + 1):
        balance += avg_net
        projection.append({
            "month": m,
            "projected_balance": round(balance, 2),
            "monthly_net": round(avg_net, 2),
        })
    return pd.DataFrame(projection)
