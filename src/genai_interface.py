"""
genai_interface.py — LangChain wrapper for LLM-powered financial advice.
Model-agnostic: supports Databricks Foundation Model APIs, Ollama, and
OpenAI-compatible endpoints via environment variables.
"""

import os
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from src.env_utils import default_llm_model


# ── LLM Factory ───────────────────────────────────────────────────────────────

def get_llm(model_name: str | None = None, **kwargs) -> BaseChatModel:
    """
    Return a LangChain chat model instance.

    Resolution order:
      1. Explicit `model_name` parameter
      2. LLM_MODEL environment variable
      3. Default to 'databricks' provider

    Supported providers (prefix-based):
      - "databricks/<model>"  → ChatDatabricks (Foundation Model APIs)
      - "ollama/<model>"      → ChatOllama     (local dev)
      - "openai/<model>"      → ChatOpenAI     (OpenAI-compatible)
    """
    model_name = model_name or default_llm_model()

    if model_name.startswith("databricks/"):
        from langchain_community.chat_models import ChatDatabricks
        endpoint = model_name.split("/", 1)[1]
        return ChatDatabricks(endpoint=endpoint, **kwargs)

    if model_name.startswith("ollama/"):
        from langchain_community.chat_models import ChatOllama
        model = model_name.split("/", 1)[1]
        return ChatOllama(model=model, **kwargs)

    if model_name.startswith("openai/"):
        from langchain_community.chat_models import ChatOpenAI
        model = model_name.split("/", 1)[1]
        return ChatOpenAI(model=model, **kwargs)

    # Fallback — treat as Databricks endpoint name
    from langchain_community.chat_models import ChatDatabricks
    return ChatDatabricks(endpoint=model_name, **kwargs)


# ── Prompt Templates ──────────────────────────────────────────────────────────

_SUMMARY_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "You are a professional financial advisor AI. Provide clear, concise, and actionable financial analysis."),
    ("human", (
        "Here is a user's financial summary:\n\n"
        "Total Income: ${total_income}\n"
        "Total Expenses: ${total_expenses}\n"
        "Net Savings: ${net_savings}\n"
        "Savings Rate: {savings_rate}%\n"
        "Transaction Count: {transaction_count}\n\n"
        "Monthly Trends:\n{trends}\n\n"
        "Risk Alerts:\n{risks}\n\n"
        "Provide a concise financial health summary in 3-5 bullet points."
    )),
])

_ADVICE_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "You are a professional financial advisor AI. Give personalized, actionable advice based on the user's data."),
    ("human", (
        "User Financial Profile:\n"
        "- Total Income: ${total_income}\n"
        "- Total Expenses: ${total_expenses}\n"
        "- Savings Rate: {savings_rate}%\n"
        "- Health Score: {health_score}/100\n\n"
        "Detected Risks:\n{risks}\n\n"
        "Scenario Simulations:\n{scenarios}\n\n"
        "Based on this analysis, provide 3-5 specific, personalized recommendations "
        "the user can act on immediately to improve their financial health."
    )),
])

_SCENARIO_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "You are a financial advisor AI. Explain financial scenarios in plain, easy-to-understand language."),
    ("human", (
        "Scenario: {scenario}\n"
        "Projection Period: {months} months\n"
        "Baseline Monthly Savings: ${baseline_monthly_savings}\n"
        "Projected Monthly Savings: ${projected_monthly_savings}\n"
        "Additional Savings Over Period: ${additional_savings}\n\n"
        "Explain this scenario result in 2-3 sentences a non-financial person can understand."
    )),
])


# ── Generation Functions ──────────────────────────────────────────────────────

def generate_financial_summary(user_data: dict, llm: BaseChatModel | None = None) -> str:
    """Generate a natural-language financial health summary."""
    llm = llm or get_llm()
    chain = _SUMMARY_TEMPLATE | llm
    response = chain.invoke(user_data)
    return response.content


def generate_personalized_advice(
    user_data: dict,
    risks: list,
    scenarios: dict,
    llm: BaseChatModel | None = None,
) -> str:
    """Generate personalized financial advice based on analysis results."""
    llm = llm or get_llm()
    payload = {
        **user_data,
        "risks": "\n".join(
            f"- [{r['severity'].upper()}] {r['risk']}: {r['detail']}" for r in risks
        ) if risks else "No significant risks detected.",
        "scenarios": str(scenarios),
    }
    chain = _ADVICE_TEMPLATE | llm
    response = chain.invoke(payload)
    return response.content


def explain_scenario(scenario_result: dict, llm: BaseChatModel | None = None) -> str:
    """Convert scenario simulation numbers into a human-readable narrative."""
    llm = llm or get_llm()
    chain = _SCENARIO_TEMPLATE | llm
    response = chain.invoke(scenario_result)
    return response.content
