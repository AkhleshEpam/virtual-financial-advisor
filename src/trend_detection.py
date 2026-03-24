"""
trend_detection.py — Financial trend analysis and anomaly/risk detection.
Uses Pandas and NumPy for aggregation and pattern identification.
"""

import pandas as pd
import numpy as np


def monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions by month → income, expenses, net savings, savings rate."""
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")

    income = df[df["amount"] > 0].groupby("month")["amount"].sum().rename("income")
    expenses = (
        df[df["amount"] < 0]
        .groupby("month")["amount"]
        .sum()
        .abs()
        .rename("expenses")
    )

    trends = pd.DataFrame({"income": income, "expenses": expenses}).fillna(0)
    trends["net_savings"] = trends["income"] - trends["expenses"]
    trends["savings_rate"] = np.where(
        trends["income"] > 0,
        (trends["net_savings"] / trends["income"]) * 100,
        0.0,
    )
    trends = trends.sort_index().reset_index()
    trends["month"] = trends["month"].astype(str)
    return trends


def spending_trend(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Compute rolling average of monthly expenses and label the trend direction."""
    trends = monthly_trends(df)
    trends["rolling_avg"] = trends["expenses"].rolling(window, min_periods=1).mean()

    def _direction(row_idx):
        if row_idx == 0:
            return "stable"
        diff = trends.loc[row_idx, "rolling_avg"] - trends.loc[row_idx - 1, "rolling_avg"]
        if diff > 50:
            return "increasing"
        elif diff < -50:
            return "decreasing"
        return "stable"

    trends["trend_direction"] = [_direction(i) for i in trends.index]
    return trends


def detect_risky_patterns(df: pd.DataFrame) -> list[dict]:
    """Identify risky financial patterns and return a list of alert dicts."""
    from src.expense_classifier import classify_all

    alerts: list[dict] = []
    trends = monthly_trends(df)

    # 1. Overspending months — expenses exceed income
    overspend = trends[trends["expenses"] > trends["income"]]
    if not overspend.empty:
        months = overspend["month"].tolist()
        alerts.append({
            "risk": "Overspending",
            "severity": "high",
            "detail": f"Expenses exceeded income in {len(months)} month(s): {months[:5]}",
        })

    # 2. Idle savings — savings rate consistently < 5%
    low_savings = trends[trends["savings_rate"] < 5]
    if len(low_savings) >= len(trends) * 0.5:
        alerts.append({
            "risk": "Idle Savings",
            "severity": "medium",
            "detail": (
                f"Savings rate below 5% in {len(low_savings)} of {len(trends)} months. "
                "Consider increasing savings contributions."
            ),
        })

    # 3. Expense spikes — month-over-month increase > 30%
    for i in range(1, len(trends)):
        prev = trends.loc[i - 1, "expenses"]
        curr = trends.loc[i, "expenses"]
        if prev > 0 and ((curr - prev) / prev) > 0.30:
            alerts.append({
                "risk": "Expense Spike",
                "severity": "medium",
                "detail": f"Expenses spiked {((curr - prev) / prev * 100):.0f}% in {trends.loc[i, 'month']}",
            })

    # 4. High discretionary ratio — discretionary > 40% of total expenses
    classified = classify_all(df)
    expenses_only = classified[classified["amount"] < 0]
    if not expenses_only.empty:
        total_exp = expenses_only["amount"].abs().sum()
        disc = expenses_only[expenses_only["classification"] == "Discretionary Expense"]["amount"].abs().sum()
        disc_ratio = disc / total_exp * 100 if total_exp > 0 else 0
        if disc_ratio > 40:
            alerts.append({
                "risk": "High Discretionary Spending",
                "severity": "medium",
                "detail": f"Discretionary spending is {disc_ratio:.1f}% of total expenses (threshold: 40%).",
            })

    return alerts


def financial_health_score(df: pd.DataFrame) -> dict:
    """
    Composite financial health score (0–100) based on:
      - Savings rate (40 pts)
      - Expense volatility (20 pts)
      - Income stability (20 pts)
      - Discretionary ratio (20 pts)
    """
    from src.expense_classifier import classify_all

    trends = monthly_trends(df)
    score = 0.0
    breakdown = {}

    # Savings rate component (40 pts) — higher is better
    avg_savings_rate = trends["savings_rate"].mean()
    savings_pts = min(avg_savings_rate / 30 * 40, 40)  # 30% savings → full marks
    breakdown["savings_rate_avg"] = round(avg_savings_rate, 2)
    breakdown["savings_pts"] = round(savings_pts, 2)
    score += savings_pts

    # Expense volatility (20 pts) — lower CV is better
    if trends["expenses"].mean() > 0:
        cv = trends["expenses"].std() / trends["expenses"].mean()
        vol_pts = max(0, 20 - cv * 20)
    else:
        vol_pts = 20.0
    breakdown["expense_cv"] = round(cv if trends["expenses"].mean() > 0 else 0, 2)
    breakdown["volatility_pts"] = round(vol_pts, 2)
    score += vol_pts

    # Income stability (20 pts) — lower CV is better
    income_series = trends[trends["income"] > 0]["income"]
    if len(income_series) > 1 and income_series.mean() > 0:
        income_cv = income_series.std() / income_series.mean()
        income_pts = max(0, 20 - income_cv * 20)
    else:
        income_pts = 10.0
    breakdown["income_stability_pts"] = round(income_pts, 2)
    score += income_pts

    # Discretionary ratio (20 pts) — lower is better
    classified = classify_all(df)
    expenses_only = classified[classified["amount"] < 0]
    if not expenses_only.empty:
        total_exp = expenses_only["amount"].abs().sum()
        disc = expenses_only[
            expenses_only["classification"] == "Discretionary Expense"
        ]["amount"].abs().sum()
        disc_ratio = disc / total_exp if total_exp > 0 else 0
    else:
        disc_ratio = 0
    disc_pts = max(0, 20 - disc_ratio * 50)
    breakdown["discretionary_ratio"] = round(disc_ratio * 100, 2)
    breakdown["discretionary_pts"] = round(disc_pts, 2)
    score += disc_pts

    return {
        "score": round(min(max(score, 0), 100), 1),
        "breakdown": breakdown,
    }


def category_trend(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Monthly trend for a specific transaction category."""
    df = df.copy()
    df["month"] = df["date"].dt.to_period("M")
    cat_df = df[df["category"] == category]
    trend = (
        cat_df.groupby("month")
        .agg(total=("amount", "sum"), count=("amount", "count"))
        .sort_index()
        .reset_index()
    )
    trend["month"] = trend["month"].astype(str)
    return trend
