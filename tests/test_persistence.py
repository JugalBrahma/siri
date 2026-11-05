"""Crash/restart persistence: SQLite checkpoints plus chat history.

A "restart" is simulated by building a second GraphBuilder on the same
SQLite file, so nothing carries over in memory between the two.
"""

import tempfile
import unittest
from collections import Counter
from pathlib import Path

from langchain_core.messages import AIMessage
from langgraph.types import Command, interrupt

from graph_builder.graph import GraphBuilder, open_sqlite_checkpointer, pending_interrupt
from memory.chat_history import ChatHistory


def _thread(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def _nodes(calls: Counter, *, crash_weather_once=False, ask_city=False, supervisor_asks=False):
    """Stand-in agents that follow the real graph's edges and count their runs."""

    def guardrail(state):
        calls["guardrail"] += 1
        return Command(goto="supervisor")

    def supervisor(state):
        calls["supervisor"] += 1
        if supervisor_asks and calls["supervisor"] == 1:
            return Command(
                goto="ask_user",
                update={"clarification": {"question": "Which city?", "return_to": "supervisor"}},
            )
        return Command(goto="researcher" if ask_city else "weather")

    def weather(state):
        calls["weather"] += 1
        if crash_weather_once and calls["weather"] == 1:
            raise RuntimeError("laptop shut down")
        return {"messages": [AIMessage(content="18°C and sunny", name="weather")]}

    def researcher(state):
        calls["researcher"] += 1
        city = interrupt({"type": "ask_human", "question": "Which city?"})
        return {"messages": [AIMessage(content=f"Weather for {city}", name="researcher")]}

    def sub_infosupervisor(state):
        calls["sub_infosupervisor"] += 1
        found = state["messages"][-1].content
        return Command(
            goto="output_sanitizer",
            update={"messages": [AIMessage(content=f"Answer: {found}", name="supervisor")]},
        )

    def output_sanitizer(state):
        calls["output_sanitizer"] += 1
        return {}

    def unused(state):
        raise AssertionError("not part of these test runs")

    return dict(
        guardrail=guardrail,
        supervisor=supervisor,
        weather=weather,
        researcher=researcher,
        sub_infosupervisor=sub_infosupervisor,
        output_sanitizer=output_sanitizer,
        sub_actionsupervisor=unused,
        action=unused,
    )


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "siri.db"
        self.calls = Counter()
        self._builders = []

    def tearDown(self):
        for builder in self._builders:
            builder.checkpointer.conn.close()
            builder.chat_history.close()
        self._tmp.cleanup()

    def _builder(self, **node_options):
        """A fresh builder on the shared SQLite file — i.e. an app (re)start."""
        builder = GraphBuilder(
            **_nodes(self.calls, **node_options),
            checkpointer=open_sqlite_checkpointer(self.db_path),
            chat_history=ChatHistory(self.db_path),
        )
        builder.build()
        self._builders.append(builder)
        return builder

    def _run(self, builder, thread_id, question):
        state = builder.prepare_turn(thread_id, [{"role": "user", "content": question}])
        for _ in builder.stream(state, stream_mode="updates", config=_thread(thread_id)):
            pass

    def test_crash_resumes_from_the_agent_that_was_running(self):
        first = self._builder(crash_weather_once=True)
        with self.assertRaises(RuntimeError):
            self._run(first, "siri_user", "Weather in Tokyo?")
        self.assertEqual(first.get_state(_thread("siri_user")).next, ("weather",))

        restarted = self._builder(crash_weather_once=True)
        results = restarted.resume_unfinished()

        self.assertEqual([r["status"] for r in results], ["success"])
        self.assertEqual(results[0]["response"], "Answer: 18°C and sunny")
        # Finished agents ran once; only the interrupted one ran again.
        self.assertEqual(self.calls["guardrail"], 1)
        self.assertEqual(self.calls["supervisor"], 1)
        self.assertEqual(self.calls["weather"], 2)
        # The clean result is in history and the checkpoint is gone.
        self.assertEqual(
            restarted.chat_history.recent("siri_user"),
            [
                {"role": "user", "content": "Weather in Tokyo?"},
                {"role": "assistant", "content": "Answer: 18°C and sunny"},
            ],
        )
        self.assertEqual(list(restarted.checkpointer.list(None)), [])

    def test_ask_human_pause_survives_restart(self):
        first = self._builder(ask_city=True)
        self._run(first, "siri_user", "What's the weather?")
        self.assertEqual(
            pending_interrupt(first.get_state(_thread("siri_user"))),
            {"type": "ask_human", "question": "Which city?"},
        )

        restarted = self._builder(ask_city=True)
        # Startup resume leaves a paused question alone.
        self.assertEqual(restarted.resume_unfinished(), [])

        for _ in restarted.stream(Command(resume="Tokyo"), stream_mode="updates", config=_thread("siri_user")):
            pass
        snapshot = restarted.get_state(_thread("siri_user"))
        self.assertIsNone(pending_interrupt(snapshot))
        turn = restarted.complete_turn("siri_user", snapshot.values["messages"])

        self.assertEqual(turn[-1].content, "Answer: Weather for Tokyo")
        self.assertEqual(self.calls["guardrail"], 1)
        self.assertEqual(self.calls["supervisor"], 1)

    def test_supervisor_question_survives_restart(self):
        first = self._builder(supervisor_asks=True)
        self._run(first, "siri_user", "What's the weather?")
        self.assertEqual(
            pending_interrupt(first.get_state(_thread("siri_user"))),
            {"type": "ask_human", "question": "Which city?"},
        )

        restarted = self._builder(supervisor_asks=True)
        for _ in restarted.stream(Command(resume="Tokyo"), stream_mode="updates", config=_thread("siri_user")):
            pass
        snapshot = restarted.get_state(_thread("siri_user"))
        turn = restarted.complete_turn("siri_user", snapshot.values["messages"])

        # The answer was added to the request and the supervisor routed again.
        self.assertEqual(
            turn[0].content,
            "What's the weather?\n(When asked \"Which city?\", the user answered: Tokyo)",
        )
        self.assertEqual(turn[-1].content, "Answer: 18°C and sunny")
        self.assertEqual(self.calls["supervisor"], 2)
        self.assertEqual(self.calls["guardrail"], 1)
        self.assertEqual(restarted.chat_history.recent("siri_user")[0]["content"], turn[0].content)

    def test_next_turn_gets_history_and_learns_only_from_itself(self):
        builder = self._builder()
        builder.chat_history.add_turn("siri_user", "Weather in Tokyo?", "18°C")

        state = builder.prepare_turn("siri_user", [{"role": "user", "content": "And tomorrow?"}])
        self.assertEqual(
            [m["content"] for m in state["messages"]],
            ["Weather in Tokyo?", "18°C", "And tomorrow?"],
        )

        for _ in builder.stream(state, stream_mode="updates", config=_thread("siri_user")):
            pass
        turn = builder.complete_turn("siri_user", builder.get_state(_thread("siri_user")).values["messages"])

        self.assertEqual(turn[0].content, "And tomorrow?")
        self.assertEqual(len(builder.chat_history.recent("siri_user")), 4)

    def test_new_message_replaces_an_abandoned_paused_turn(self):
        builder = self._builder(ask_city=True)
        self._run(builder, "siri_user", "What's the weather?")
        self.assertIsNotNone(pending_interrupt(builder.get_state(_thread("siri_user"))))

        builder.prepare_turn("siri_user", [{"role": "user", "content": "Never mind"}])

        self.assertEqual(list(builder.checkpointer.list(_thread("siri_user"))), [])


class ChatHistoryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.history = ChatHistory(Path(self._tmp.name) / "siri.db")

    def tearDown(self):
        self.history.close()
        self._tmp.cleanup()

    def test_recent_returns_latest_messages_oldest_first(self):
        for n in range(3):
            self.history.add_turn("t1", f"q{n}", f"a{n}")
        self.assertEqual(
            [m["content"] for m in self.history.recent("t1", limit=4)],
            ["q1", "a1", "q2", "a2"],
        )

    def test_threads_are_separate_and_clearable(self):
        self.history.add_turn("t1", "q", "a")
        self.history.add_turn("t2", "other", "reply")
        self.assertEqual(self.history.clear("t1"), 2)
        self.assertEqual(self.history.recent("t1"), [])
        self.assertEqual(len(self.history.recent("t2")), 2)


if __name__ == "__main__":
    unittest.main()
