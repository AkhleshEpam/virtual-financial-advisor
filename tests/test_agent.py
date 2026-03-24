"""Tests for src/agent/ — agent_core.py and memory.py"""

import os
import json
import pytest

from src.agent.memory import SessionMemory


class TestSessionMemory:
    def test_save_and_get_history(self):
        mem = SessionMemory()
        mem.save_context("Hello", "Hi there!")
        history = mem.get_history()
        assert len(history) == 2  # human + ai messages

    def test_get_history_str(self):
        mem = SessionMemory()
        mem.save_context("What is my balance?", "Your balance is $5,000.")
        text = mem.get_history_str()
        assert "User:" in text
        assert "Advisor:" in text

    def test_clear(self):
        mem = SessionMemory()
        mem.save_context("test", "response")
        mem.set_user_profile({"income": 5000})
        mem.clear()
        assert len(mem.get_history()) == 0
        assert mem.get_user_profile() is None

    def test_user_profile(self):
        mem = SessionMemory()
        assert mem.get_user_profile() is None
        mem.set_user_profile({"savings_rate": 20})
        assert mem.get_user_profile()["savings_rate"] == 20

    def test_langchain_memory_property(self):
        mem = SessionMemory()
        chat_hist = mem.chat_history
        assert chat_hist is not None


class TestAgentTools:
    """Test individual agent tools with real data (no LLM needed)."""

    @pytest.fixture(autouse=True)
    def setup_agent_state(self):
        """Pre-load user data into module-level state so tools can work."""
        import src.agent.agent_core as ac
        from src.data_loader import load_data, preprocess, get_user_data

        data_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "virtual_financial_advisor_data.csv"
        )
        ac._data_path = data_path
        ac._df = preprocess(load_data(data_path))
        ac._user_df = get_user_data(ac._df, "user_1")
        ac._session = SessionMemory()

    def test_load_user_data_tool(self):
        from src.agent.agent_core import load_user_data
        result = load_user_data.invoke("user_1")
        data = json.loads(result)
        assert "total_income" in data
        assert "total_expenses" in data

    def test_analyze_spending_tool(self):
        from src.agent.agent_core import analyze_spending
        result = analyze_spending.invoke("")
        data = json.loads(result)
        assert "health_score" in data
        assert 0 <= data["health_score"] <= 100

    def test_classify_expenses_tool(self):
        from src.agent.agent_core import classify_expenses
        result = classify_expenses.invoke("")
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) > 0

    def test_detect_risks_tool(self):
        from src.agent.agent_core import detect_risks
        result = detect_risks.invoke("")
        # Either a JSON list or "No significant risks" message
        assert isinstance(result, str)

    def test_simulate_scenario_tool(self):
        from src.agent.agent_core import simulate_scenario
        params = json.dumps({"type": "savings_increase", "value": 10, "months": 12})
        result = simulate_scenario.invoke(params)
        data = json.loads(result)
        assert data["additional_savings"] > 0

    def test_simulate_scenario_bad_json(self):
        from src.agent.agent_core import simulate_scenario
        result = simulate_scenario.invoke("not valid json")
        assert "Error" in result


class TestAgentInit:
    def test_tools_registered(self):
        from src.agent.agent_core import TOOLS
        tool_names = [t.name for t in TOOLS]
        assert "load_user_data" in tool_names
        assert "analyze_spending" in tool_names
        assert "classify_expenses" in tool_names
        assert "detect_risks" in tool_names
        assert "simulate_scenario" in tool_names
        assert "get_advice" in tool_names
        assert len(TOOLS) == 6
