# Virtual Financial Advisor

An AI-powered virtual financial advisor that analyzes spending, detects risky patterns, simulates scenarios, and provides personalized advice using a LangChain ReAct agent with open-source LLMs.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Streamlit UI / CLI                     │
│  Dashboard │ Analysis │ Scenarios │ Advisor Chat         │
├──────────────────────────────────────────────────────────┤
│              LangChain ReAct Agent                       │
│  Tools: load_data│analyze│classify│risks│simulate│advise │
├──────────┬───────────┬────────────┬──────────────────────┤
│ data_    │ expense_  │ trend_     │ scenario_            │
│ loader   │ classifier│ detection  │ simulation           │
├──────────┴───────────┴────────────┴──────────────────────┤
│             GenAI Interface (LangChain)                  │
│  Databricks Foundation Model APIs / Ollama / OpenAI      │
├──────────────────────────────────────────────────────────┤
│               Transaction Data (CSV / PySpark)           │
└──────────────────────────────────────────────────────────┘
```

## Project Structure

```
virtual-financial-advisor/
├── data/
│   └── virtual_financial_advisor_data.csv
├── notebooks/
│   ├── 01_data_loading_and_eda.ipynb
│   ├── 02_expense_classification.ipynb
│   ├── 03_financial_trend_detection.ipynb
│   ├── 04_scenario_simulation.ipynb
│   ├── 05_genai_advice_generation.ipynb
│   └── 06_agent_loop_prototype.ipynb
├── src/
│   ├── data_loader.py
│   ├── expense_classifier.py
│   ├── trend_detection.py
│   ├── scenario_simulation.py
│   ├── genai_interface.py
│   ├── agent/
│   │   ├── agent_core.py
│   │   └── memory.py
│   └── ui/
│       ├── streamlit_app.py
│       └── cli.py
├── tests/
│   ├── test_data_loader.py
│   ├── test_expense_classifier.py
│   ├── test_trend_detection.py
│   ├── test_scenario_simulation.py
│   └── test_agent.py
├── requirements.txt
├── README.md
└── Dockerfile
```

## Features

| Feature | Implementation |
|---|---|
| Financial trend detection | Pandas, NumPy |
| Expense classification | Rule-based (no ML) |
| Scenario simulations | Simple projection logic |
| Natural language summaries | LangChain + open-source LLMs |
| Multi-step decision agent | LangChain ReAct Agent |
| UI + deployment | Streamlit + Docker |

## Setup

### Prerequisites

- Python 3.11+
- (Optional) [Ollama](https://ollama.ai) for local LLM inference
- (Optional) Azure Databricks cluster for PySpark and Foundation Model APIs

### Local Development

```bash
# Clone and install
cd virtual-financial-advisor
pip install -r requirements.txt

# Generate sample data (if not present)
python -c "exec(open('generate_data.py').read())"

# Run tests
pytest tests/ -v

# Launch Streamlit UI
streamlit run src/ui/streamlit_app.py

# Or use the CLI
python -m src.ui.cli
```

### LLM Configuration

Set the `LLM_MODEL` environment variable:

```bash
# Databricks Foundation Model APIs (on Databricks cluster)
export LLM_MODEL="databricks/databricks-meta-llama-3-1-70b-instruct"

# Local Ollama
export LLM_MODEL="ollama/llama3.1"

# OpenAI-compatible endpoint
export LLM_MODEL="openai/gpt-4"
export OPENAI_API_KEY="sk-..."
```

### Azure Databricks

1. Create a cluster with **Databricks Runtime 15.x LTS (ML)**
2. Upload `data/virtual_financial_advisor_data.csv` to DBFS
3. Import notebooks from `notebooks/` into your Databricks workspace
4. Install additional packages on the cluster:
   ```
   langchain langchain-community plotly
   ```
5. Run notebooks 01–06 sequentially
6. For Foundation Model APIs, enable the serving endpoint and set `LLM_MODEL` accordingly

### Docker

```bash
docker build -t virtual-financial-advisor .
docker run -p 8501:8501 -e LLM_MODEL="ollama/llama3.1" virtual-financial-advisor
```

Access the app at `http://localhost:8501`.

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `LLM_MODEL` | LLM provider/model string | `databricks/databricks-meta-llama-3-1-70b-instruct` |
| `DATA_PATH` | Path to transaction CSV | `data/virtual_financial_advisor_data.csv` |
| `OPENAI_API_KEY` | OpenAI API key (only for openai/ provider) | — |

## Dataset

Synthetic dataset with 5,200 transactions for 20 users over 2 years (2023–2024):

- **Income categories**: Salary, Bonus, Interest
- **Expense categories**: Groceries, Rent, Utilities, Entertainment, Dining, Transport, Healthcare, Education, Savings Transfer
- **Payment methods**: Credit Card, Debit Card, Cash, Transfer

## Tech Stack

- **Data**: Pandas, NumPy, PySpark (Databricks)
- **Classification**: Rule-based category mapping
- **Simulation**: Python projection functions
- **GenAI**: LangChain + Databricks Foundation Model APIs / Ollama
- **Agent**: LangChain ReAct Agent with 6 custom tools
- **UI**: Streamlit
- **Deployment**: Docker
- **Platform**: Azure Databricks
