"""What past conversation the agents see: last 2 questions, answers trimmed."""

import unittest

from langchain_core.messages import AIMessage, HumanMessage

from state.message_state import (
    build_worker_state,
    create_turn_state,
    format_recent_history,
    with_recent_history,
)


class FormatRecentHistoryTests(unittest.TestCase):
    def test_keeps_only_the_last_two_turns(self):
        history = [
            {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "q3"}, {"role": "assistant", "content": "a3"},
        ]
        self.assertEqual(format_recent_history(history), "User: q2\nSiri: a2\nUser: q3\nSiri: a3")

    def test_trims_answers_but_not_questions(self):
        question = "Q" * 500
        history = [{"role": "user", "content": question}, {"role": "assistant", "content": "A" * 1000}]
        lines = format_recent_history(history, answer_chars=300).split("\n")
        self.assertEqual(lines[0], f"User: {question}")
        self.assertEqual(lines[1], "Siri: " + "A" * 300 + "…")

    def test_accepts_message_objects(self):
        history = [HumanMessage(content="Weather in Tokyo?"), AIMessage(content="18°C")]
        self.assertEqual(format_recent_history(history), "User: Weather in Tokyo?\nSiri: 18°C")

    def test_empty_history(self):
        self.assertEqual(format_recent_history([]), "")
        self.assertEqual(with_recent_history("hello", ""), "hello")


class WorkerTaskTests(unittest.TestCase):
    def test_worker_task_carries_recent_conversation(self):
        state = create_turn_state(
            [{"role": "user", "content": "what about tomorrow?"}],
            "",
            "User: Weather in Tokyo?\nSiri: 18°C",
        )
        task = build_worker_state(state, "info_supervisor")["task"]
        self.assertIn("User: Weather in Tokyo?\nSiri: 18°C", task)
        self.assertTrue(task.endswith("Current request: what about tomorrow?"))

    def test_worker_task_unchanged_without_history(self):
        state = create_turn_state([{"role": "user", "content": "Weather in Tokyo?"}])
        self.assertEqual(build_worker_state(state, "info_supervisor")["task"], "Weather in Tokyo?")


if __name__ == "__main__":
    unittest.main()
