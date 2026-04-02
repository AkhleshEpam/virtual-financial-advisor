"""
agent_core.py — LangChain ReAct agent with custom financial-analysis tools.
Orchestrates data loading, classification, trend detection, simulation, and advice.
"""

import json
import os
from typing import Optional

from langchain.agents import create_agent
from langchain_core.tools import tool

from src.agent.memory import SessionMemory

# ── Module-level state (set via init_agent) ───────────────────────────────────
_df = None          # Full preprocessed DataFrame
_user_df = None     # Filtered user DataFrame
_session = None     # SessionMemory instance
_data_path = None   # Path to CSV


def _ensure_data():
    if _user_df is None:
        raise RuntimeError("No user data loaded. Call 'load_user_data' tool first.")


# ── Tool definitions ──────────────────────────────────────────────────────────

@tool
def load_user_data(user_id: str) -> str:
    """Load and summarize financial data for a specific user. Input: user_id (e.g. 'Aarav_Sharma')."""
    global _df, _user_df, _session

    from src.data_loader import load_data, preprocess, get_user_data, get_summary_stats

    path = _data_path or os.getenv(
        "DATA_PATH", "data/virtual_financial_advisor_data_v2.csv"
    )
    _df = preprocess(load_data(path))
    _user_df = get_user_data(_df, user_id)
    stats = get_summary_stats(_user_df)

    if _session:
        _session.set_user_profile(stats)

    return json.dumps(stats, indent=2)


@tool
def analyze_spending(dummy: str = "") -> str:
    """Analyze the loaded user's spending trends and financial health score. No input needed."""
    _ensure_data()
    from src.trend_detection import monthly_trends, financial_health_score

    trends = monthly_trends(_user_df)
    health = financial_health_score(_user_df)

    summary = {
        "months_analyzed": len(trends),
        "avg_monthly_income": round(trends["income"].mean(), 2),
        "avg_monthly_expenses": round(trends["expenses"].mean(), 2),
        "avg_savings_rate": round(trends["savings_rate"].mean(), 2),
        "health_score": health["score"],
        "score_breakdown": health["breakdown"],
    }
    return json.dumps(summary, indent=2)


@tool
def classify_expenses(dummy: str = "") -> str:
    """Classify all expenses into Essential, Discretionary, and Savings. No input needed."""
    _ensure_data()
    from src.expense_classifier import classify_all, get_category_breakdown

    classified = classify_all(_user_df)
    breakdown = get_category_breakdown(classified)

    result = breakdown.to_dict(orient="records")
    return json.dumps(result, indent=2, default=str)


@tool
def detect_risks(dummy: str = "") -> str:
    """Detect risky financial patterns (overspending, idle savings, spikes). No input needed."""
    _ensure_data()
    from src.trend_detection import detect_risky_patterns

    risks = detect_risky_patterns(_user_df)
    if not risks:
        return "No significant financial risks detected."
    return json.dumps(risks, indent=2)


@tool
def simulate_scenario(params: str) -> str:
    """
    Run a what-if financial scenario. Input: JSON string with keys:
    - type: 'savings_increase' | 'expense_reduction' | 'income_change'
    - value: percentage (float)
    - category: (required for expense_reduction) expense category name
    - months: (optional, default 12) projection months
    Example: {"type": "savings_increase", "value": 10}
    """
    _ensure_data()
    from src.scenario_simulation import (
        simulate_savings_increase,
        simulate_expense_reduction,
        simulate_income_change,
    )

    try:
        p = json.loads(params)
    except json.JSONDecodeError:
        return "Error: Input must be a valid JSON string."

    sim_type = p.get("type", "savings_increase")
    value = float(p.get("value", 10))
    months = int(p.get("months", 12))

    if sim_type == "savings_increase":
        result = simulate_savings_increase(_user_df, value, months)
    elif sim_type == "expense_reduction":
        category = p.get("category", "Dining")
        result = simulate_expense_reduction(_user_df, category, value, months)
    elif sim_type == "income_change":
        result = simulate_income_change(_user_df, value, months)
    else:
        return f"Unknown scenario type: {sim_type}"

    return json.dumps(result, indent=2)


@tool
def get_advice(context: str = "") -> str:
    """
    Generate personalized financial advice using GenAI based on the user's analysis.
    Optional input: additional context or question from the user.
    """
    _ensure_data()
    from src.data_loader import get_summary_stats
    from src.trend_detection import detect_risky_patterns, financial_health_score
    from src.genai_interface import generate_personalized_advice

    stats = get_summary_stats(_user_df)
    risks = detect_risky_patterns(_user_df)
    health = financial_health_score(_user_df)

    user_data = {
        **stats,
        "health_score": health["score"],
    }

    advice = generate_personalized_advice(
        user_data=user_data,
        risks=risks,
        scenarios={"user_context": context} if context else {},
    )
    return advice


# ── Agent Setup ───────────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """\
You are a Virtual Financial Advisor Agent. You help bank customers understand
their financial health by analyzing transactions, detecting risks, running
simulations, and providing personalized advice.

When a user asks a question:
1. If no user data is loaded yet, use `load_user_data` first.
2. Analyze their spending with `analyze_spending` and `classify_expenses`.
3. Check for risks with `detect_risks`.
4. If the user asks about "what-if" scenarios, use `simulate_scenario`.
5. For personalized advice, use `get_advice`.

Always provide clear, helpful, and data-driven responses.
"""


TOOLS = [load_user_data, analyze_spending, classify_expenses, detect_risks, simulate_scenario, get_advice]


def init_agent(
    data_path: str = "data/virtual_financial_advisor_data_v2.csv",
    llm=None,
    session_memory: Optional[SessionMemory] = None,
):
    """
    Initialize and return the Financial Advisor agent graph.

    Parameters
    ----------
    data_path : str
        Path to the transactions CSV.
    llm : BaseChatModel, optional
        LangChain chat model. If None, uses genai_interface.get_llm().
    session_memory : SessionMemory, optional
        Session memory instance. Creates a new one if None.
    """
    global _data_path, _session

    _data_path = data_path

    if llm is None:
        from src.genai_interface import get_llm
        llm = get_llm()

    _session = session_memory or SessionMemory()

    agent = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=AGENT_SYSTEM_PROMPT,
    )

    return agent
