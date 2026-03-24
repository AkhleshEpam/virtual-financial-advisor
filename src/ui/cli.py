"""
cli.py — Command-line REPL interface for the Virtual Financial Advisor agent.
Quick-start for testing without Streamlit.
"""

import sys
import os

#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

sys.path.insert(0, '/Workspace/Users/akhlesh_kumar@epam.com/virtual-financial-advisor')

import subprocess
import sys

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'langchain-core<1.0', 'langchain-community', 'pandas>=2.0'])

dbutils.library.restartPython()

from src.agent.agent_core import init_agent


def main():
    data_path = os.getenv("DATA_PATH", "data/virtual_financial_advisor_data.csv")
    print("=" * 60)
    print("  Virtual Financial Advisor — CLI")
    print("=" * 60)
    print(f"Data: {data_path}")
    print(f"LLM:  {os.getenv('LLM_MODEL', 'databricks/databricks-meta-llama-3-1-70b-instruct')}")
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
            response = agent.invoke({"input": user_input})
            answer = response.get("output", str(response))
        except Exception as e:
            answer = f"Error: {e}"

        print(f"\nAdvisor: {answer}\n")


if __name__ == "__main__":
    main()
