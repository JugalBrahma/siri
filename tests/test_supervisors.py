"""Behaviour tests for the three supervisors, with the LLM faked.

They pin down routing decisions and the exact messages sent to the LLM, so
refactors of the supervisor code can be checked without real API calls.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import agents.supervisor_agent as sup
from agents.prompt import sub_action_supervisor_prompt, sub_info_supervisor_prompt, system_prompt
from state.message_state import (
    ActionSupervisorDecision,
    InfoSupervisorDecision,
    MainSupervisorDecision,
    create_turn_state,
    new_supervisor_state,
    record_agent_output,
)

MEMORY = "- The user prefers Celsius [high confidence]"


def _fake_llm(decision):
    """An llm whose with_structured_output(...).invoke(...) returns `decision`."""
    structured = MagicMock()
    structured.invoke.return_value = decision
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm, structured


def _turn(text):
    return create_turn_state([{"role": "user", "content": text}], MEMORY)


class TestFastPaths(unittest.TestCase):
    def test_main_fast_path(self):
        self.assertEqual(sup._fast_path_main("what is the weather"), "sub_infosupervisor")
        self.assertEqual(sup._fast_path_main("create a file"), "sub_actionsupervisor")
        self.assertIsNone(sup._fast_path_main("research cats and write a document"))
        self.assertIsNone(sup._fast_path_main("hmm"))
        self.assertIsNone(sup._fast_path_main(""))

    def test_info_fast_path(self):
        self.assertEqual(sup._fast_path_info("forecast for Tokyo"), "weather")
        self.assertEqual(sup._fast_path_info("search the news"), "researcher")
        self.assertIsNone(sup._fast_path_info("search the weather forecast"))
        self.assertIsNone(sup._fast_path_info("Mumbai today"))
        self.assertIsNone(sup._fast_path_info(""))


class TestMainSupervisor(unittest.TestCase):
    def test_fast_path_skips_llm(self):
        llm, _ = _fake_llm(None)
        with patch.object(sup, "llm", llm):
            cmd = sup.superVisorAgent(_turn("what is the weather in Paris"))
        llm.with_structured_output.assert_not_called()
        self.assertEqual(cmd.goto, "sub_infosupervisor")
        self.assertEqual(cmd.update["main_supervisor"]["routing_plan"], ["sub_infosupervisor"])
        self.assertIn("info_supervisor", cmd.update)

    def test_llm_decision_dict(self):
        """Mixed info+action text falls through to the LLM; a dict answer is accepted."""
        llm, structured = _fake_llm({"next": "sub_actionsupervisor", "reasoning": "needs a file"})
        with patch.object(sup, "llm", llm):
            cmd = sup.superVisorAgent(_turn("research cats and write a document"))
        llm.with_structured_output.assert_called_once_with(MainSupervisorDecision)
        messages = structured.invoke.call_args.args[0]
        self.assertEqual(messages[0]["role"], "system")
        self.assertTrue(messages[0]["content"].startswith(system_prompt))
        self.assertIn(MEMORY, messages[0]["content"])
        self.assertEqual(messages[1], {"role": "user", "content": "User Request: research cats and write a document"})
        self.assertEqual(cmd.goto, "sub_actionsupervisor")
        self.assertIn("action_supervisor", cmd.update)

    def test_llm_decision_object(self):
        """A pydantic-style object answer is accepted too."""
        llm, _ = _fake_llm(SimpleNamespace(next="sub_infosupervisor", reasoning="info"))
        with patch.object(sup, "llm", llm):
            cmd = sup.superVisorAgent(_turn("hmm"))
        self.assertEqual(cmd.goto, "sub_infosupervisor")

    def test_finish_with_no_outputs_returns_reasoning(self):
        llm, _ = _fake_llm({"next": "FINISH", "reasoning": "Nothing to do."})
        with patch.object(sup, "llm", llm):
            cmd = sup.superVisorAgent(_turn("hmm"))
        self.assertEqual(cmd.goto, "output_sanitizer")
        self.assertEqual(cmd.update["messages"][0].content, "Nothing to do.")

    def test_completed_outputs_are_listed_in_prompt(self):
        state = _turn("research cats and write a document")
        state["main_supervisor"] = record_agent_output(state["main_supervisor"], "sub_infosupervisor", "Cats purr.")
        state["main_supervisor"]["routing_plan"] = ["sub_infosupervisor"]
        llm, structured = _fake_llm({"next": "sub_infosupervisor", "reasoning": "again"})
        with patch.object(sup, "llm", llm):
            cmd = sup.superVisorAgent(state)
        system = structured.invoke.call_args.args[0][0]["content"]
        self.assertIn("**sub_infosupervisor** [STATUS: DONE]", system)
        self.assertIn("Cats purr.", system)
        # Guard: info is done and the text asks for an action, so go to action.
        self.assertEqual(cmd.goto, "sub_actionsupervisor")

    def test_action_done_fast_finishes(self):
        state = _turn("write a file")
        state["main_supervisor"] = record_agent_output(state["main_supervisor"], "sub_actionsupervisor", "File written.")
        llm, _ = _fake_llm(None)
        with patch.object(sup, "llm", llm):
            cmd = sup.superVisorAgent(state)
        llm.with_structured_output.assert_not_called()
        self.assertEqual(cmd.goto, "output_sanitizer")
        self.assertEqual(cmd.update["messages"][0].content, "File written.")


class TestInfoSupervisor(unittest.TestCase):
    def _state(self, text):
        state = _turn(text)
        state["info_supervisor"] = new_supervisor_state(MEMORY)
        return state

    def test_fast_path_weather(self):
        llm, _ = _fake_llm(None)
        with patch.object(sup, "llm", llm):
            cmd = sup.sub_infoSuperVisorAgent(self._state("forecast for Tokyo"))
        llm.with_structured_output.assert_not_called()
        self.assertEqual(cmd.goto, "weather")

    def test_llm_decision(self):
        llm, structured = _fake_llm({"next": "researcher", "reasoning": "look it up"})
        with patch.object(sup, "llm", llm):
            cmd = sup.sub_infoSuperVisorAgent(self._state("Mumbai today"))
        llm.with_structured_output.assert_called_once_with(InfoSupervisorDecision)
        messages = structured.invoke.call_args.args[0]
        self.assertTrue(messages[0]["content"].startswith(sub_info_supervisor_prompt))
        self.assertIn(MEMORY, messages[0]["content"])
        self.assertEqual(messages[1]["content"], "User Request: Mumbai today")
        self.assertEqual(cmd.goto, "researcher")
        self.assertEqual(cmd.update["info_supervisor"]["routing_plan"], ["researcher"])

    def test_done_returns_results_to_main(self):
        state = self._state("Mumbai today")
        state["info_supervisor"] = record_agent_output(state["info_supervisor"], "weather", "30°C")
        state["info_supervisor"]["routing_plan"] = ["weather"]
        llm, _ = _fake_llm({"next": "supervisor", "reasoning": "done"})
        with patch.object(sup, "llm", llm):
            cmd = sup.sub_infoSuperVisorAgent(state)
        self.assertEqual(cmd.goto, "supervisor")
        self.assertEqual(cmd.update["main_supervisor"]["agent_outputs"]["sub_infosupervisor"], "30°C")


class TestActionSupervisor(unittest.TestCase):
    def _state(self, text):
        state = _turn(text)
        state["action_supervisor"] = new_supervisor_state(MEMORY)
        return state

    def test_routes_to_action(self):
        llm, structured = _fake_llm({"next": "action", "reasoning": "do it"})
        with patch.object(sup, "llm", llm):
            cmd = sup.sub_actionSuperVisorAgent(self._state("open safari"))
        llm.with_structured_output.assert_called_once_with(ActionSupervisorDecision)
        messages = structured.invoke.call_args.args[0]
        self.assertTrue(messages[0]["content"].startswith(sub_action_supervisor_prompt))
        self.assertEqual(messages[1]["content"], "User Request: open safari")
        self.assertEqual(cmd.goto, "action")

    def test_returns_to_main_after_action(self):
        state = self._state("open safari")
        state["action_supervisor"] = record_agent_output(state["action_supervisor"], "action", "Opened safari.")
        llm, _ = _fake_llm({"next": "action", "reasoning": "again"})
        with patch.object(sup, "llm", llm):
            cmd = sup.sub_actionSuperVisorAgent(state)
        self.assertEqual(cmd.goto, "supervisor")
        self.assertEqual(cmd.update["main_supervisor"]["agent_outputs"]["sub_actionsupervisor"], "Opened safari.")

    def test_missing_next_returns_reasoning(self):
        llm, _ = _fake_llm(SimpleNamespace(reasoning="unclear"))
        with patch.object(sup, "llm", llm):
            cmd = sup.sub_actionSuperVisorAgent(self._state("open safari"))
        self.assertEqual(cmd.goto, "supervisor")
        self.assertEqual(cmd.update["main_supervisor"]["agent_outputs"]["sub_actionsupervisor"], "unclear")


class TestSupervisorAsksUser(unittest.TestCase):
    """Each supervisor can pause for the user via the ask_user node."""

    def _decide(self, decision):
        return _fake_llm(decision)[0]

    def test_main_asks_user(self):
        llm = self._decide({"next": "ask_user", "reasoning": "unclear", "question": "What should I do?"})
        with patch.object(sup, "llm", llm):
            cmd = sup.superVisorAgent(_turn("hmm"))
        self.assertEqual(cmd.goto, "ask_user")
        self.assertEqual(cmd.update["clarification"], {"question": "What should I do?", "return_to": "supervisor"})
        self.assertEqual(cmd.update["main_supervisor"]["routing_plan"], ["ask_user"])

    def test_info_asks_user_and_returns_to_itself(self):
        state = _turn("Mumbai today")
        state["info_supervisor"] = new_supervisor_state(MEMORY)
        llm = self._decide({"next": "ask_user", "reasoning": "no topic", "question": "Weather or news?"})
        with patch.object(sup, "llm", llm):
            cmd = sup.sub_infoSuperVisorAgent(state)
        self.assertEqual(cmd.goto, "ask_user")
        self.assertEqual(cmd.update["clarification"]["return_to"], "sub_infosupervisor")
        self.assertEqual(cmd.update["info_supervisor"]["routing_plan"], ["ask_user"])

    def test_action_question_falls_back_to_reasoning(self):
        state = _turn("make something")
        state["action_supervisor"] = new_supervisor_state(MEMORY)
        llm = self._decide({"next": "ask_user", "reasoning": "Which file should I create?"})
        with patch.object(sup, "llm", llm):
            cmd = sup.sub_actionSuperVisorAgent(state)
        self.assertEqual(cmd.goto, "ask_user")
        self.assertEqual(
            cmd.update["clarification"],
            {"question": "Which file should I create?", "return_to": "sub_actionsupervisor"},
        )

    def test_does_not_ask_more_than_twice(self):
        state = _turn("hmm")
        state["main_supervisor"]["routing_plan"] = ["ask_user", "ask_user"]
        llm = self._decide({"next": "ask_user", "reasoning": "still unclear", "question": "?"})
        with patch.object(sup, "llm", llm):
            cmd = sup.superVisorAgent(state)
        self.assertEqual(cmd.goto, "output_sanitizer")


if __name__ == "__main__":
    unittest.main()
