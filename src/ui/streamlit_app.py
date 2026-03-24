"""
streamlit_app.py — Streamlit UI for the Virtual Financial Advisor.
Four tabs: Dashboard, Analysis, Scenarios, Advisor Chat.
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

from src.data_loader import load_data, preprocess, get_user_data, get_summary_stats
from src.expense_classifier import classify_all, get_category_breakdown, detect_unusual_expenses
from src.trend_detection import monthly_trends, spending_trend, detect_risky_patterns, financial_health_score
from src.scenario_simulation import (
    simulate_savings_increase,
    simulate_expense_reduction,
    simulate_income_change,
    compare_scenarios,
    project_balance,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Virtual Financial Advisor", page_icon="💰", layout="wide")
st.title("💰 Virtual Financial Advisor")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    data_path = st.text_input("Data path", value="data/virtual_financial_advisor_data.csv")
    user_ids = [f"user_{i}" for i in range(1, 21)]
    selected_user = st.selectbox("Select User", user_ids)
    st.markdown("---")
    st.subheader("LLM Settings")
    llm_model = st.text_input(
        "LLM Model",
        value=os.getenv("LLM_MODEL", "ollama/llama3.1"),
        help="e.g. databricks/databricks-meta-llama-3-1-70b-instruct, ollama/llama3.1",
    )
    os.environ["LLM_MODEL"] = llm_model

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare(path, user_id):
    df = preprocess(load_data(path))
    user_df = get_user_data(df, user_id)
    return df, user_df

try:
    full_df, user_df = load_and_prepare(data_path, selected_user)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

stats = get_summary_stats(user_df)
trends = monthly_trends(user_df)
health = financial_health_score(user_df)
classified = classify_all(user_df)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_dash, tab_analysis, tab_scenarios, tab_chat = st.tabs(
    ["📊 Dashboard", "🔍 Analysis", "🔮 Scenarios", "💬 Advisor Chat"]
)

# ════════════════════════  TAB 1 — Dashboard  ═════════════════════════════════
with tab_dash:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Income", f"${stats['total_income']:,.2f}")
    col2.metric("Total Expenses", f"${stats['total_expenses']:,.2f}")
    col3.metric("Net Savings", f"${stats['net_savings']:,.2f}")
    col4.metric("Savings Rate", f"{stats['savings_rate']:.1f}%")

    st.markdown("---")

    # Health score gauge
    col_score, col_chart = st.columns([1, 2])
    with col_score:
        st.subheader("Financial Health Score")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health["score"],
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 40], "color": "#ff4b4b"},
                    {"range": [40, 70], "color": "#ffa500"},
                    {"range": [70, 100], "color": "#00cc96"},
                ],
            },
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=20, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_chart:
        st.subheader("Monthly Income vs Expenses")
        fig_trend = px.line(
            trends, x="month", y=["income", "expenses"],
            labels={"value": "Amount ($)", "month": "Month"},
        )
        fig_trend.update_layout(height=300)
        st.plotly_chart(fig_trend, use_container_width=True)

    # Category breakdown
    st.subheader("Expense Category Breakdown")
    expenses_only = classified[classified["amount"] < 0].copy()
    expenses_only["abs_amount"] = expenses_only["amount"].abs()
    cat_totals = expenses_only.groupby("category")["abs_amount"].sum().reset_index()
    fig_pie = px.pie(cat_totals, names="category", values="abs_amount", hole=0.4)
    fig_pie.update_layout(height=350)
    st.plotly_chart(fig_pie, use_container_width=True)


# ════════════════════════  TAB 2 — Analysis  ══════════════════════════════════
with tab_analysis:
    # Risk alerts
    st.subheader("🚨 Risk Alerts")
    risks = detect_risky_patterns(user_df)
    if risks:
        for r in risks:
            severity_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(r["severity"], "⚪")
            st.markdown(f"{severity_color} **{r['risk']}** — {r['detail']}")
    else:
        st.success("No significant financial risks detected!")

    st.markdown("---")

    # Spending trend
    st.subheader("Spending Trend (3-month rolling avg)")
    sp_trend = spending_trend(user_df)
    fig_sp = px.line(sp_trend, x="month", y=["expenses", "rolling_avg"],
                     labels={"value": "Amount ($)", "month": "Month"})
    fig_sp.update_layout(height=350)
    st.plotly_chart(fig_sp, use_container_width=True)

    # Classification breakdown
    st.subheader("Classification Breakdown")
    class_totals = (
        expenses_only.groupby("classification")["abs_amount"]
        .sum()
        .reset_index()
        .sort_values("abs_amount", ascending=False)
    )
    fig_class = px.bar(class_totals, x="classification", y="abs_amount",
                       labels={"abs_amount": "Total ($)", "classification": "Classification"})
    st.plotly_chart(fig_class, use_container_width=True)

    # Unusual expenses
    st.subheader("Unusual Expenses")
    unusual = detect_unusual_expenses(user_df)
    if not unusual.empty:
        st.dataframe(unusual[["date", "category", "amount", "merchant", "description"]].head(20))
    else:
        st.info("No unusual expenses detected.")

    # Health score breakdown
    st.subheader("Health Score Breakdown")
    bd = health["breakdown"]
    score_df = pd.DataFrame([
        {"Component": "Savings Rate", "Points": bd["savings_pts"], "Max": 40},
        {"Component": "Expense Stability", "Points": bd["volatility_pts"], "Max": 20},
        {"Component": "Income Stability", "Points": bd["income_stability_pts"], "Max": 20},
        {"Component": "Discretionary Control", "Points": bd["discretionary_pts"], "Max": 20},
    ])
    fig_score = px.bar(score_df, x="Component", y=["Points", "Max"], barmode="group")
    st.plotly_chart(fig_score, use_container_width=True)


# ════════════════════════  TAB 3 — Scenarios  ═════════════════════════════════
with tab_scenarios:
    st.subheader("What-If Scenario Simulator")

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        savings_pct = st.slider("Extra savings (% of income)", 0, 50, 10)
    with col_s2:
        expense_cat = st.selectbox("Reduce category", ["Dining", "Entertainment", "Transport", "Groceries"])
        expense_reduction = st.slider("Reduction %", 0, 100, 20)
    with col_s3:
        income_change = st.slider("Income change %", -50, 100, 15)

    proj_months = st.slider("Projection months", 3, 60, 12)

    if st.button("Run Simulations", type="primary"):
        s1 = simulate_savings_increase(user_df, savings_pct, proj_months)
        s2 = simulate_expense_reduction(user_df, expense_cat, expense_reduction, proj_months)
        s3 = simulate_income_change(user_df, income_change, proj_months)

        # Comparison table
        comparison = compare_scenarios([s1, s2, s3])
        st.subheader("Scenario Comparison")
        st.dataframe(comparison, use_container_width=True)

        # Projected vs baseline chart
        fig_sc = go.Figure()
        for i, sc in enumerate([s1, s2, s3]):
            fig_sc.add_trace(go.Bar(
                name=sc["scenario"],
                x=["Baseline", "Projected"],
                y=[sc["baseline_cumulative"], sc["projected_cumulative"]],
            ))
        fig_sc.update_layout(barmode="group", title="Cumulative Savings: Baseline vs Projected")
        st.plotly_chart(fig_sc, use_container_width=True)

        # Balance projection
        st.subheader("12-Month Balance Projection")
        balance_proj = project_balance(user_df, initial_balance=5000, months=proj_months)
        fig_bal = px.line(balance_proj, x="month", y="projected_balance",
                          labels={"projected_balance": "Balance ($)", "month": "Month"})
        st.plotly_chart(fig_bal, use_container_width=True)


# ════════════════════════  TAB 4 — Advisor Chat  ═════════════════════════════
with tab_chat:
    st.subheader("💬 Chat with Your Financial Advisor")
    st.caption(f"Using model: {llm_model} | User: {selected_user}")

    # Session state for chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = None

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask your financial advisor...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    # Lazy-init agent
                    if st.session_state.agent_executor is None:
                        from src.agent.agent_core import init_agent
                        st.session_state.agent_executor = init_agent(
                            data_path=data_path,
                        )

                    response = st.session_state.agent_executor.invoke(
                        {"input": user_input}
                    )
                    answer = response.get("output", str(response))
                except Exception as e:
                    answer = (
                        f"⚠️ Agent error: {e}\n\n"
                        "Make sure the LLM endpoint is accessible. "
                        "For local dev, run `ollama serve` with the configured model."
                    )

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
