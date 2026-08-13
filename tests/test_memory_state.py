"""Fast tests for state ownership and one-fetch semantic-memory behavior.

These tests use a fake store. They never make an API request or require
Pinecone credentials.
"""

import sys
import types
import unittest


def _install_langgraph_stub() -> None:
    """Allow state-contract tests to run even in a minimal Python environment."""
    if "langgraph.graph" in sys.modules:
        return

    langgraph = types.ModuleType("langgraph")
    graph = types.ModuleType("langgraph.graph")

    class FakeStateGraph:
        def __init__(self, *_args, **_kwargs):
            pass

    graph.MessagesState = dict
    graph.StateGraph = FakeStateGraph
    graph.END = "END"
    langgraph.graph = graph
    sys.modules["langgraph"] = langgraph
    sys.modules["langgraph.graph"] = graph


_install_langgraph_stub()

from graph_builder.graph import GraphBuilder
from state.message_state import (
    build_worker_state,
    create_turn_state,
    get_supervisor_state,
    new_supervisor_state,
    record_agent_output,
)


class FakeMemoryStore:
    """A controlled test double with the same graph-facing API as the real store."""

    def __init__(self, profile: str):
        self.profile = profile
        self.fetch_count = 0

    def format_for_injection(self) -> str:
        self.fetch_count += 1
        return self.profile

    def get_last_retrieval_metrics(self) -> dict:
        return {"fact_count": 1, "confidence_min": 0.9, "confidence_max": 0.9}


class FakeMemoryObserver:
    def __init__(self):
        self.successes = []
        self.failures = []

    def fetch_succeeded(self, **event) -> None:
        self.successes.append(event)

    def fetch_failed(self, **event) -> None:
        self.failures.append(event)


class SemanticMemoryStateTests(unittest.TestCase):
    def test_memory_is_fetched_once_per_turn(self):
        store = FakeMemoryStore("- The user prefers concise answers [high confidence]")
        observer = FakeMemoryObserver()
        builder = GraphBuilder(
            supervisor=None,
            sub_infosupervisor=None,
            sub_actionsupervisor=None,
            researcher=None,
            weather=None,
            action=None,
            guardrail=None,
            output_sanitizer=None,
            memory_store=store,
            memory_observer=observer,
        )

        turn = builder.create_turn_state([{"role": "user", "content": "Hello"}])

        self.assertEqual(store.fetch_count, 1)
        self.assertEqual(
            turn["main_supervisor"]["semantic_memory"],
            "- The user prefers concise answers [high confidence]",
        )
        self.assertEqual(len(observer.successes), 1)

    def test_supervisor_scopes_are_isolated(self):
        turn = create_turn_state([{"role": "user", "content": "Search the weather"}], "- Metric units")
        main = get_supervisor_state(turn, "main_supervisor")
        info = new_supervisor_state(main["semantic_memory"])
        info = record_agent_output(info, "weather", "30°C and rain")

        self.assertEqual(main["agent_outputs"], {})
        self.assertEqual(info["agent_outputs"], {"weather": "30°C and rain"})
        self.assertEqual(main["semantic_memory"], info["semantic_memory"])

    def test_agent_output_overwrites_previous_result(self):
        scope = new_supervisor_state()
        scope = record_agent_output(scope, "researcher", "temporary failed answer")
        scope = record_agent_output(scope, "researcher", "latest successful answer")

        self.assertEqual(scope["agent_outputs"], {"researcher": "latest successful answer"})

    def test_worker_scratchpad_never_enters_graph_state(self):
        turn = create_turn_state(
            [{"role": "user", "content": "Find Mumbai weather"}],
            "- The user prefers Celsius [high confidence]",
        )
        turn["info_supervisor"] = new_supervisor_state(
            turn["main_supervisor"]["semantic_memory"],
        )
        worker = build_worker_state(turn, "info_supervisor")
        worker["retry_count"] = 2
        worker["last_error"] = "tool timeout"

        self.assertNotIn("retry_count", turn)
        self.assertNotIn("last_error", turn)
        self.assertNotIn("latest_tool_result", turn)
        self.assertEqual(worker["semantic_memory"], turn["info_supervisor"]["semantic_memory"])


if __name__ == "__main__":
    unittest.main()
