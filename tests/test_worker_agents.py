"""Behaviour tests for the research and action worker agents, with the agent faked."""

import importlib
import unittest
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

import agents.actionagent as action_mod
import agents.research_agent as research_mod
from state.message_state import create_turn_state, new_supervisor_state, record_agent_output

MEMORY = "- The user prefers Celsius [high confidence]"


def _patch_create_agent(stack, fake):
    """Patch create_agent wherever the worker code imports it from."""
    for name in ("agents.research_agent", "agents.actionagent", "agents.worker"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(module, "create_agent"):
            stack.enter_context(patch.object(module, "create_agent", fake))


def _fake_agent(answer):
    executor = MagicMock()
    executor.invoke.return_value = {"messages": [AIMessage(content=answer)]}
    return MagicMock(return_value=executor), executor


def _state(text, scope):
    state = create_turn_state([{"role": "user", "content": text}], MEMORY)
    state[scope] = new_supervisor_state(MEMORY)
    return state


class TestResearchAgent(unittest.TestCase):
    def test_runs_task_and_records_output(self):
        create_agent, executor = _fake_agent("Cats purr.")
        with ExitStack() as stack:
            _patch_create_agent(stack, create_agent)
            cmd = research_mod.research_agent(_state("tell me about cats", "info_supervisor"))

        _, tools = create_agent.call_args.args
        self.assertEqual([t.name for t in tools], ["tavily_search", "ask_human"])
        prompt = create_agent.call_args.kwargs["system_prompt"]
        self.assertTrue(prompt.startswith("You are a web researcher"))
        self.assertIn(MEMORY, prompt)
        executor.invoke.assert_called_once_with(
            {"messages": [{"role": "user", "content": "tell me about cats"}]}
        )

        self.assertEqual(cmd.goto, "sub_infosupervisor")
        self.assertEqual(cmd.update["info_supervisor"]["agent_outputs"], {"researcher": "Cats purr."})
        message = cmd.update["messages"][0]
        self.assertEqual((message.content, message.name), ("Cats purr.", "researcher"))


class TestActionAgent(unittest.TestCase):
    def test_runs_task_and_records_output(self):
        create_agent, executor = _fake_agent("Opened safari.")
        with ExitStack() as stack:
            _patch_create_agent(stack, create_agent)
            stack.enter_context(patch.object(action_mod, "get_word_tools", return_value=[]))
            cmd = action_mod.action_agent(_state("open safari", "action_supervisor"))

        _, tools = create_agent.call_args.args
        self.assertEqual([t.name for t in tools], ["open_app", "modify_file", "read_file", "ask_human"])
        prompt = create_agent.call_args.kwargs["system_prompt"]
        self.assertTrue(prompt.startswith("You are an automation agent"))
        self.assertIn(MEMORY, prompt)
        executor.invoke.assert_called_once_with(
            {"messages": [{"role": "user", "content": "open safari"}]}
        )

        self.assertEqual(cmd.goto, "sub_actionsupervisor")
        self.assertEqual(cmd.update["action_supervisor"]["agent_outputs"], {"action": "Opened safari."})
        message = cmd.update["messages"][0]
        self.assertEqual((message.content, message.name), ("Opened safari.", "action"))

    def test_research_results_are_added_to_task(self):
        create_agent, executor = _fake_agent("Wrote doc.")
        state = _state("write a document about cats", "action_supervisor")
        state["main_supervisor"] = record_agent_output(state["main_supervisor"], "sub_infosupervisor", "Cats purr.")
        with ExitStack() as stack:
            _patch_create_agent(stack, create_agent)
            stack.enter_context(patch.object(action_mod, "get_word_tools", return_value=[]))
            action_mod.action_agent(state)

        task = executor.invoke.call_args.args[0]["messages"][0]["content"]
        self.assertEqual(
            task,
            "write a document about cats\n\nGathered Information / Research Content from previous step:\nCats purr.",
        )


if __name__ == "__main__":
    unittest.main()
