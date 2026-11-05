# LangSmith Setup & Connection Guide (Siri Multi-Agent System)

This guide explains how **LangSmith** is connected and configured in our project (`siri`), covering installation, environment setup, automatic LangGraph tracing, evaluation datasets, and per-node token tracking.

---

## Table of Contents
1. [Prerequisites & Environment Variables](#1-prerequisites--environment-variables)
2. [Package Installation](#2-package-installation)
3. [Global Tracing Configuration (`config/config.py`)](#3-global-tracing-configuration-configconfigpy)
4. [Agent Graph Tracing with `@traceable` (`main.py`)](#4-agent-graph-tracing-with-traceable-mainpy)
5. [Evaluation Datasets & LangSmith Client (`evaluate/`)](#5-evaluation-datasets--langsmith-client-evaluate)
6. [Token Usage & Metadata Tracking (`services/token_tracker.py`)](#6-token-usage--metadata-tracking-servicestoken_trackerpy)
7. [How to Run & Verify Traces in the LangSmith Dashboard](#7-how-to-run--verify-traces-in-the-langsmith-dashboard)

---

## 1. Prerequisites & Environment Variables

To connect your local project to your LangSmith account:

1. Create a free account at [https://smith.langchain.com](https://smith.langchain.com).
2. Generate an API Key in **Settings > API Keys**.
3. Add the following variables to your `siri/.env` file:

```env
# LangSmith Configuration
LANGSMITH_API_KEY=lsv2_pt_your_api_key_here
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=siri
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

---

## 2. Package Installation

LangSmith is installed via `pip`:

```bash
pip install langsmith
```

Ensure `langsmith` is included in your project dependencies alongside `langchain` and `langgraph`.

---

## 3. Global Tracing Configuration (`config/config.py`)

In `siri/config/config.py`, tracing is globally enabled as soon as the configuration module is imported:

```python
# config/config.py
import os
from dotenv import load_dotenv
from langsmith import traceable

# 1. Load environment variables (.env)
load_dotenv()

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# 2. Automatically route all LangChain & LangGraph telemetry to LangSmith
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "siri"

class Config:
    """Configuration class for agents"""
```

---

## 4. Agent Graph Tracing with `@traceable` (`main.py`)

In `siri/main.py`, the `@traceable` decorator wraps the graph execution chain so that every user question, agent hop, and final output is logged as a distinct run in LangSmith:

```python
# main.py (excerpt)
from langsmith import evaluate, Client
from langsmith import traceable

client = Client()
dataset_name = "Agents Loop"

@traceable(run_type="chain", name="Siri Agent Execution")
def target_function(inputs: dict) -> dict:
    """Runs the multi-agent graph and records execution traces in LangSmith."""
    user_question = inputs["question"]
    
    # Initialize turn state
    initial_state = builder.create_turn_state(
        [{"role": "user", "content": user_question}],
    )
    
    # Stream through the LangGraph agents
    hop_count = 0
    final_state = initial_state
    for state in builder.graph.stream(initial_state, stream_mode="values"):
        final_state = state
        hop_count += 1
        
    hop_count = max(0, hop_count - 1)
    
    # Extract final assistant response
    final_output = ""
    if "messages" in final_state and len(final_state["messages"]) > 0:
        final_output = final_state["messages"][-1].content

    return {
        "query_input": user_question,
        "final_output": final_output,
        "hops": hop_count,
    }
```

---

## 5. Evaluation Datasets & LangSmith Client (`evaluate/`)

We manage test datasets in LangSmith using the `Client` SDK across two files:

### 1. `evaluate/datasets_data.py`
Defines test prompts categorized by domain (Weather, Research, Action, General Chat):

```python
# evaluate/datasets_data.py
example_inputs = [
    # Weather queries
    "Whats the weather in new york?",
    "Will it rain in London tomorrow?",
    
    # Research queries
    "Which location does Antarctica exist?",
    "Can you summarize the theory of relativity?",
    
    # Action queries
    "Create a new Python file called hello.py and write a print statement in it.",
    "Can you list all the files in my current directory?",
    
    # General conversation
    "Hello! How are you doing today?",
]
```

### 2. `evaluate/datasets_evaluate.py`
Creates the remote dataset on LangSmith and populates test examples:

```python
# evaluate/datasets_evaluate.py
from .datasets_data import example_inputs
from langsmith import Client

client = Client()
dataset_name = "Agents Loop" 

def upload_datasets():
    """Uploads benchmark examples to the LangSmith 'Agents Loop' dataset."""
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"Dataset '{dataset_name}' already exists. Skipping creation.")
        return
    except Exception:
        pass # Dataset does not exist, create it

    # 1. Create a new dataset in LangSmith
    dataset = client.create_dataset(dataset_name=dataset_name)
    print(f"Your new dataset ID is: {dataset.id}")

    # 2. Upload test cases as examples
    for question in example_inputs:
        client.create_example(
            inputs={"question": question},
            dataset_id=dataset.id,
        )
    print("Finished uploading examples!")
```

---

## 6. Token Usage & Metadata Tracking (`services/token_tracker.py`)

To track exact prompt tokens, completion tokens, and per-node attribution (Supervisor, Weather, Researcher, Action) from LLM runs:

```python
# services/token_tracker.py
from typing import Any, Dict
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

class LangSmithTokenTracker(BaseCallbackHandler):
    """Callback handler accumulating token metrics for LangSmith and API responses."""

    def __init__(self):
        super().__init__()
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.node_breakdown: Dict[str, int] = {}
        self.current_node: str = "supervisor"

    def set_current_node(self, node_name: str) -> None:
        if node_name:
            self.current_node = str(node_name).strip().lower()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Extract exact token usage from LLM response metadata."""
        p_tokens, c_tokens, t_tokens = 0, 0, 0

        # Standard token usage metadata from Groq / OpenAI / LangSmith
        if response.llm_output and isinstance(response.llm_output, dict):
            usage = response.llm_output.get("token_usage") or response.llm_output.get("usage", {})
            if isinstance(usage, dict):
                p_tokens = usage.get("prompt_tokens", 0)
                c_tokens = usage.get("completion_tokens", 0)
                t_tokens = usage.get("total_tokens", p_tokens + c_tokens)

        self.prompt_tokens += p_tokens
        self.completion_tokens += c_tokens
        self.total_tokens += t_tokens

        node_key = self.current_node or "general"
        self.node_breakdown[node_key] = self.node_breakdown.get(node_key, 0) + t_tokens

    def get_usage(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "node_breakdown": dict(self.node_breakdown),
            "source": "langsmith",
        }
```

---

## 7. How to Run & Verify Traces in the LangSmith Dashboard

### 1. Test Dataset Upload
In Python or `main.py`, uncomment and run:
```python
from evaluate.datasets_evaluate import upload_datasets
upload_datasets()
```

### 2. Run the Main Agent Pipeline
```bash
python main.py
```

### 3. Check Traces in LangSmith UI
1. Navigate to [https://smith.langchain.com](https://smith.langchain.com).
2. Click on the **`siri`** project.
3. You will see:
   - **Trace Tree**: `guardrail` $\rightarrow$ `supervisor` $\rightarrow$ sub-agents $\rightarrow$ `output_sanitizer`.
   - **Latency**: Duration spent in each agent node.
   - **Token Consumption**: Exact prompt and completion token counts.
   - **Inputs & Outputs**: Raw messages exchanged between nodes.
