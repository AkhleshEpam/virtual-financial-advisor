"""
cli.py — Command-line REPL interface for the Virtual Financial Advisor agent.
Quick-start for testing without Streamlit.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.env_utils import is_databricks, default_llm_model, default_data_path
from src.agent.agent_core import init_agent


def main():
    data_path = default_data_path()
    print("=" * 60)
    print("  Virtual Financial Advisor — CLI")
    print("=" * 60)
    env_label = "Databricks" if is_databricks() else "Local"
    print(f"Environment: {env_label}")
    print(f"Data: {data_path}")
    print(f"LLM:  {default_llm_model()}")
    print("Type 'quit' or 'exit' to stop.\n")

    try:
        agent = init_agent(data_path=data_path)
    except Exception as e:
        print(f"Error initializing agent: {e}")
        sys.exit(1)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        try:
            response = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
            messages = response.get("messages", [])
            # Extract the last AI message content
            answer = ""
            for msg in reversed(messages):
                if hasattr(msg, "content") and getattr(msg, "type", None) == "ai" and msg.content:
                    answer = msg.content
                    break
            if not answer:
                answer = str(response)
        except Exception as e:
            answer = f"Error: {e}"

        print(f"\nAdvisor: {answer}\n")


if __name__ == "__main__":
    main()
