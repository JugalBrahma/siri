"""LangSmith & LangChain Token Usage Tracking Service.

Captures real-time, exact prompt tokens, completion tokens, and per-node breakdown
from LLM invocations during LangGraph execution turns.
"""

from typing import Any, Dict, List, Optional
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult


class LangSmithTokenTracker(BaseCallbackHandler):
    """LangChain callback handler that accumulates exact token metrics across graph nodes."""

    def __init__(self):
        super().__init__()
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.node_breakdown: Dict[str, int] = {}
        self.current_node: str = "supervisor"

    def set_current_node(self, node_name: str) -> None:
        """Update active node context for token attribution."""
        if node_name:
            self.current_node = str(node_name).strip().lower()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Extract exact token usage from LLM response metadata."""
        p_tokens = 0
        c_tokens = 0
        t_tokens = 0

        # 1. Standard LLMResult.llm_output (Groq, OpenAI, NVIDIA, LangSmith)
        if response.llm_output and isinstance(response.llm_output, dict):
            usage = response.llm_output.get("token_usage") or response.llm_output.get("usage", {})
            if isinstance(usage, dict):
                p_tokens = usage.get("prompt_tokens") or usage.get("input_tokens", 0)
                c_tokens = usage.get("completion_tokens") or usage.get("output_tokens", 0)
                t_tokens = usage.get("total_tokens") or (p_tokens + c_tokens)

        # 2. Modern LangChain generation message usage_metadata / response_metadata
        if not t_tokens and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    if msg:
                        usage_meta = getattr(msg, "usage_metadata", None)
                        if usage_meta and isinstance(usage_meta, dict):
                            p = usage_meta.get("input_tokens", 0)
                            c = usage_meta.get("output_tokens", 0)
                            t = usage_meta.get("total_tokens", p + c)
                            p_tokens += p
                            c_tokens += c
                            t_tokens += t
                        elif hasattr(msg, "response_metadata") and isinstance(msg.response_metadata, dict):
                            u = msg.response_metadata.get("token_usage", {})
                            if isinstance(u, dict):
                                p = u.get("prompt_tokens") or u.get("input_tokens", 0)
                                c = u.get("completion_tokens") or u.get("output_tokens", 0)
                                t = u.get("total_tokens", p + c)
                                p_tokens += p
                                c_tokens += c
                                t_tokens += t

        self.prompt_tokens += p_tokens
        self.completion_tokens += c_tokens
        self.total_tokens += (t_tokens if t_tokens > 0 else (p_tokens + c_tokens))

        if t_tokens > 0 or (p_tokens + c_tokens) > 0:
            node_key = self.current_node or "general"
            node_cost = t_tokens if t_tokens > 0 else (p_tokens + c_tokens)
            self.node_breakdown[node_key] = self.node_breakdown.get(node_key, 0) + node_cost

    def get_usage(self) -> Dict[str, Any]:
        """Return exact token telemetry for API responses and UI dashboards."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "input": self.prompt_tokens,
            "output": self.completion_tokens,
            "total": self.total_tokens,
            "node_breakdown": dict(self.node_breakdown),
            "source": "langsmith",
        }
