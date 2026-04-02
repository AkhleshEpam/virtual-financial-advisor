"""
env_utils.py — Environment detection for Databricks vs local execution.
Auto-detects the runtime and returns sensible defaults for each environment.
"""

import os


def is_databricks() -> bool:
    """Return True if running inside a Databricks environment."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def default_llm_model() -> str:
    """Return the default LLM model string based on the current environment."""
    if os.getenv("LLM_MODEL"):
        return os.environ["LLM_MODEL"]
    if is_databricks():
        return "databricks/databricks-meta-llama-3-1-70b-instruct"
    return "ollama/llama3.1"


def default_data_path() -> str:
    """Return the default data file path based on the current environment."""
    if os.getenv("DATA_PATH"):
        return os.environ["DATA_PATH"]
    if is_databricks():
        return "/dbfs/FileStore/virtual_financial_advisor_data_v2.csv"
    return "data/virtual_financial_advisor_data_v2.csv"
