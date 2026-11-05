import unittest
from unittest.mock import MagicMock

from graph_builder.graph import GraphBuilder
from services.text import message_text
from services.token_tracker import LangSmithTokenTracker, build_run_config


class TestMessageText(unittest.TestCase):
    def test_string_is_returned_unchanged(self):
        self.assertEqual(message_text("hello"), "hello")

    def test_list_parts_are_joined(self):
        parts = ["a", {"type": "text", "text": "b"}, {"type": "tool_use"}, 3]
        self.assertEqual(message_text(parts), "a b  3")
        self.assertEqual(message_text(parts, sep=""), "ab3")

    def test_other_types_use_str(self):
        self.assertEqual(message_text(None), "None")


class TestRunConfig(unittest.TestCase):
    def test_build_run_config(self):
        tracker = LangSmithTokenTracker()
        config = build_run_config(42, tracker)
        self.assertEqual(config["configurable"], {"thread_id": "42"})
        self.assertEqual(config["callbacks"], [tracker])


class TestEnsureGraph(unittest.TestCase):
    def _builder(self):
        return GraphBuilder(
            supervisor=None, sub_infosupervisor=None, sub_actionsupervisor=None,
            researcher=None, weather=None, action=None, guardrail=None, output_sanitizer=None,
        )

    def test_builds_once_and_reuses(self):
        builder = self._builder()
        builder.build = MagicMock(side_effect=lambda: setattr(builder, "graph", "G") or "G")
        self.assertEqual(builder._ensure_graph(), "G")
        self.assertEqual(builder._ensure_graph(), "G")
        builder.build.assert_called_once()

    def test_raises_when_build_returns_none(self):
        builder = self._builder()
        builder.build = MagicMock(return_value=None)
        with self.assertRaises(RuntimeError):
            builder._ensure_graph()


if __name__ == "__main__":
    unittest.main()
