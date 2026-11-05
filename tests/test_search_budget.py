import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agents.search_budget import BLOCKED_MARKER, AdaptiveSearchLimit, search_failed

GOOD = "• **Python 3.15.2**\n  Released October 2026\n  Source: https://python.org"


def _ask(*call_ids):
    """AI message requesting one tavily_search per id."""
    return AIMessage(content="", tool_calls=[
        {"name": "tavily_search", "args": {"query": f"q{i}"}, "id": i} for i in call_ids
    ])


def _result(call_id, content):
    return ToolMessage(content=content, tool_call_id=call_id, name="tavily_search")


def _blocked_ids(update):
    return [m.tool_call_id for m in (update or {}).get("messages", [])]


class TestSearchFailed(unittest.TestCase):
    def test_detects_errors_and_empty_results(self):
        for text in ["Error running search: timeout", "Error: the MCP server returned no search tools.",
                     "", "   ", "{'results': []}", "No results found"]:
            self.assertTrue(search_failed(text), text)

    def test_real_results_are_not_failures(self):
        self.assertFalse(search_failed(GOOD))


class TestAdaptiveSearchLimit(unittest.TestCase):
    def setUp(self):
        self.mw = AdaptiveSearchLimit(normal_limit=2, retry_limit=3)
        self.start = [HumanMessage(content="latest python?")]

    def _run(self, messages):
        return _blocked_ids(self.mw.after_model({"messages": messages}, None))

    def test_first_two_searches_allowed(self):
        self.assertEqual(self._run(self.start + [_ask("a")]), [])
        history = self.start + [_ask("a"), _result("a", GOOD), _ask("b")]
        self.assertEqual(self._run(history), [])

    def test_third_search_blocked_when_first_succeeded(self):
        history = self.start + [_ask("a"), _result("a", GOOD), _ask("b"), _result("b", GOOD), _ask("c")]
        self.assertEqual(self._run(history), ["c"])

    def test_third_search_allowed_when_first_failed(self):
        history = self.start + [_ask("a"), _result("a", "Error running search: timeout"),
                                _ask("b"), _result("b", GOOD), _ask("c")]
        self.assertEqual(self._run(history), [])

    def test_fourth_search_blocked_even_after_failure(self):
        history = self.start + [_ask("a"), _result("a", "Error running search: timeout"),
                                _ask("b"), _result("b", GOOD), _ask("c"), _result("c", GOOD), _ask("d")]
        self.assertEqual(self._run(history), ["d"])

    def test_parallel_batch_trimmed_to_limit(self):
        # Three searches requested at once before any result: only 2 may run.
        self.assertEqual(self._run(self.start + [_ask("a", "b", "c")]), ["c"])

    def test_blocked_messages_do_not_count_as_searches(self):
        history = self.start + [_ask("a"), _result("a", GOOD), _ask("b", "c"),
                                _result("b", GOOD),
                                ToolMessage(content=f"{BLOCKED_MARKER} (2 searches)", tool_call_id="c",
                                            name="tavily_search", status="error")]
        limit, used = self.mw.limit_and_used(history)
        self.assertEqual((limit, used), (2, 2))

    def test_other_tools_are_ignored(self):
        msg = AIMessage(content="", tool_calls=[{"name": "ask_human", "args": {}, "id": "h"}])
        self.assertEqual(self._run(self.start + [msg]), [])


if __name__ == "__main__":
    unittest.main()
