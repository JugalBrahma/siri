"""screen_reader / action_taker hand-offs and the screen tools, with the LLM and screen faked."""

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agents.screen_agents as screen_mod
import tools.computer as computer
import tools.terminal as terminal
from graph_builder.graph import GRAPH_STEP_LIMIT, _run_config
from tools import macos_windows
from tools.macos_windows import Window
from tools.screen_target import screen_target
from state.message_state import ScreenDecision, create_turn_state, new_screen_task, new_supervisor_state

REQUEST = "send hi to my friend on WhatsApp"
SCREENSHOT_CALL = ToolMessage(content="Chat open, 'hi' typed.", name="screenshot", tool_call_id="1")


def _screen(**fields):
    return {**new_screen_task(), **fields}


def _state(screen=None, answer=None, question="Send 'hi'?"):
    messages = [{"role": "user", "content": REQUEST}]
    if answer is not None:  # what the ask_user node adds after a question
        messages.append(HumanMessage(content=f'{REQUEST}\n(When asked "{question}", the user answered: {answer})'))
    state = create_turn_state(messages)
    state["action_supervisor"] = new_supervisor_state()
    state["screen"] = screen
    return state


def _fake_reader(*decisions, screenshot_taken=True):
    executor = MagicMock()
    messages = [SCREENSHOT_CALL] if screenshot_taken else []
    executor.invoke.side_effect = [{"messages": messages, "structured_response": d} for d in decisions]
    return MagicMock(return_value=executor), executor


def _run_reader(state, *decisions, **kwargs):
    create_agent, executor = _fake_reader(*decisions, **kwargs)
    with patch.object(screen_mod, "create_agent", create_agent):
        return screen_mod.screen_reader_agent(state), create_agent, executor


class TestScreenReader(unittest.TestCase):
    def test_hands_instruction_to_action_taker(self):
        cmd, create_agent, _ = _run_reader(_state(), ScreenDecision(next="action_taker", instruction="press cmd+f"))
        _, tools = create_agent.call_args.args
        self.assertEqual([t.name for t in tools], ["open_app", "list_apps", "screenshot"])
        self.assertIs(create_agent.call_args.kwargs["response_format"].schema, ScreenDecision)
        self.assertEqual(cmd.goto, "action_taker")
        self.assertEqual(cmd.update["screen"], _screen(instruction="press cmd+f", rounds=1))

    def test_reports_progress_back_to_itself(self):
        screen = _screen(instruction="press cmd+f", report="Done: pressed cmd+f.", rounds=1)
        _, _, executor = _run_reader(_state(screen), ScreenDecision(next="done", summary="Sent."))
        task = executor.invoke.call_args.args[0]["messages"][0]["content"]
        self.assertIn("Your last instruction to action_taker: press cmd+f", task)
        self.assertIn("action_taker reported (not yet checked on screen): Done: pressed cmd+f.", task)

    def test_every_question_goes_through_ask_user_node_and_is_remembered(self):
        decision = ScreenDecision(next="ask_user", question="Send 'hi'?")
        cmd, _, _ = _run_reader(_state(), decision)
        self.assertEqual(cmd.goto, "ask_user")
        self.assertEqual(cmd.update["clarification"], {"question": "Send 'hi'?", "return_to": "screen_reader"})
        self.assertEqual(cmd.update["screen"]["confirm_question"], "Send 'hi'?")
        self.assertEqual(cmd.update["screen"]["rounds"], 0)  # questions do not use up rounds

    def test_requested_risky_step_goes_ahead_without_asking(self):
        decision = ScreenDecision(next="action_taker", instruction="click the Like button",
                                  risky=True, requested=True)
        cmd, _, _ = _run_reader(_state(), decision)
        self.assertEqual(cmd.goto, "action_taker")

    def test_unrequested_risky_step_is_turned_into_a_question(self):
        decision = ScreenDecision(next="action_taker", instruction="press enter", risky=True)
        cmd, _, _ = _run_reader(_state(), decision)
        self.assertEqual(cmd.goto, "ask_user")
        self.assertEqual(cmd.update["screen"]["confirm_question"], "About to: press enter. Go ahead?")

    def test_a_yes_holds_for_the_rest_of_the_task(self):
        state = _state(_screen(confirm_question="Send 'hi'?"), answer="yes don't ask again")
        decision = ScreenDecision(next="action_taker", instruction="press enter", risky=True)
        cmd, _, executor = _run_reader(state, decision)
        self.assertEqual(cmd.goto, "action_taker")
        self.assertTrue(cmd.update["screen"]["approved"])  # still approved after this step
        self.assertIn("Approved", executor.invoke.call_args.args[0]["messages"][0]["content"])

        # A retry later in the same task does not ask again.
        retry = _state(cmd.update["screen"])
        cmd, _, executor = _run_reader(retry, ScreenDecision(next="action_taker", instruction="press enter", risky=True))
        self.assertEqual(cmd.goto, "action_taker")
        self.assertIn("already approved", executor.invoke.call_args.args[0]["messages"][0]["content"])

    def test_an_edit_is_not_approval(self):
        state = _state(_screen(confirm_question="Send 'hi'?", approved=True), answer="write hi how are you")
        decision = ScreenDecision(next="action_taker", instruction="type it, press enter", risky=True)
        cmd, _, executor = _run_reader(state, decision)
        self.assertEqual(cmd.goto, "ask_user")
        self.assertIn("Not a yes", executor.invoke.call_args.args[0]["messages"][0]["content"])

    def test_what_counts_as_a_yes(self):
        for answer in ["yes", "Yes!", "ok do it", "yes go ahead", "yes don't ask again", "Yes, don’t ask again.", "sure"]:
            self.assertTrue(screen_mod._is_clear_yes(answer), answer)
        for answer in ["no", "wait", "yes but write hello", "don't send", "yes don't send", "try again later", ""]:
            self.assertFalse(screen_mod._is_clear_yes(answer), answer)

    def test_round_limit_still_allows_a_final_check(self):
        screen = _screen(instruction="press enter", report="Done.", rounds=screen_mod.MAX_SCREEN_ROUNDS)
        cmd, _, _ = _run_reader(_state(screen), ScreenDecision(next="done", summary="Sent 'hi'."))
        self.assertEqual(cmd.update["action_supervisor"]["agent_outputs"], {"screen_reader": "Sent 'hi'."})

    def test_round_limit_stops_another_action(self):
        screen = _screen(instruction="x", report="Done.", rounds=screen_mod.MAX_SCREEN_ROUNDS)
        cmd, _, _ = _run_reader(_state(screen), ScreenDecision(next="action_taker", instruction="press tab"))
        self.assertEqual(cmd.goto, "sub_actionsupervisor")
        self.assertIn("Stopped after", cmd.update["action_supervisor"]["agent_outputs"]["screen_reader"])

    def test_must_check_screen_after_an_action(self):
        screen = _screen(instruction="type hi", report="Done: typed hi.", rounds=1)
        decision = ScreenDecision(next="done", summary="Typed hi.")
        cmd, _, executor = _run_reader(_state(screen), decision, decision, screenshot_taken=False)
        self.assertEqual(executor.invoke.call_count, 2)
        retry_task = executor.invoke.call_args.args[0]["messages"][0]["content"]
        self.assertIn("You did not take a screenshot", retry_task)

    def test_keeps_the_app_that_open_app_opened(self):
        def opens_chrome(*_args, **_kwargs):
            from tools.screen_target import set_target_app
            set_target_app("Google Chrome")  # what open_app does on success
            return {"messages": [SCREENSHOT_CALL],
                    "structured_response": ScreenDecision(next="action_taker", instruction="press cmd+l")}
        executor = MagicMock()
        executor.invoke.side_effect = opens_chrome
        with patch.object(screen_mod, "create_agent", MagicMock(return_value=executor)):
            cmd = screen_mod.screen_reader_agent(_state())
        self.assertEqual(cmd.update["screen"]["app"], "Google Chrome")

    def test_tools_run_inside_the_task_app(self):
        seen = []
        executor = MagicMock()
        executor.invoke.side_effect = lambda _: seen.append(computer.target_app()) or {
            "messages": [SCREENSHOT_CALL], "structured_response": ScreenDecision(next="done", summary="ok")}
        with patch.object(screen_mod, "create_agent", MagicMock(return_value=executor)):
            screen_mod.screen_reader_agent(_state(_screen(app="WhatsApp")))
        self.assertEqual(seen, ["WhatsApp"])

    def test_stops_when_reader_hits_its_step_limit(self):
        executor = MagicMock()
        executor.invoke.return_value = {"messages": []}  # ended by the model-call limit
        with patch.object(screen_mod, "create_agent", MagicMock(return_value=executor)):
            cmd = screen_mod.screen_reader_agent(_state())
        self.assertEqual(cmd.goto, "sub_actionsupervisor")
        self.assertIn("hit its limit", cmd.update["action_supervisor"]["agent_outputs"]["screen_reader"])

    def test_a_failed_decision_stops_cleanly_instead_of_crashing(self):
        executor = MagicMock()
        executor.invoke.side_effect = ValueError("Extra data: line 1 column 321")
        with patch.object(screen_mod, "create_agent", MagicMock(return_value=executor)):
            cmd = screen_mod.screen_reader_agent(_state())
        self.assertEqual(cmd.goto, "sub_actionsupervisor")
        self.assertIn("could not decide", cmd.update["action_supervisor"]["agent_outputs"]["screen_reader"])

    def test_agents_have_step_limits(self):
        create_agent, _ = _fake_reader(ScreenDecision(next="done", summary="ok"))
        with patch.object(screen_mod, "create_agent", create_agent):
            screen_mod.screen_reader_agent(_state())
        limits = create_agent.call_args.kwargs["middleware"]
        self.assertEqual(sorted(type(m).__name__ for m in limits),
                         ["ModelCallLimitMiddleware", "ToolCallLimitMiddleware", "ToolCallLimitMiddleware"])

    def test_done_records_output_and_clears_screen(self):
        screen = _screen(instruction="press enter", report="Done.", rounds=3)
        cmd, _, _ = _run_reader(_state(screen), ScreenDecision(next="done", summary="Sent 'hi'."))
        self.assertEqual(cmd.goto, "sub_actionsupervisor")
        self.assertIsNone(cmd.update["screen"])


class TestActionTaker(unittest.TestCase):
    def test_runs_instruction_and_hands_back(self):
        executor = MagicMock()
        executor.invoke.return_value = {"messages": [AIMessage(content="Done: pressed cmd+f.")]}
        create_agent = MagicMock(return_value=executor)
        screen = _screen(instruction="press cmd+f", rounds=1)
        with patch.object(screen_mod, "create_agent", create_agent):
            cmd = screen_mod.action_taker_agent(_state(screen))

        _, tools = create_agent.call_args.args
        self.assertEqual([t.name for t in tools], ["type_text", "press_keys", "click_on"])
        executor.invoke.assert_called_once_with({"messages": [{"role": "user", "content": "press cmd+f"}]})
        self.assertEqual(cmd.goto, "screen_reader")
        self.assertEqual(cmd.update["screen"], {**screen, "report": "Done: pressed cmd+f."})

    def test_reports_when_it_hits_its_step_limit(self):
        executor = MagicMock()
        executor.invoke.return_value = {"messages": [AIMessage(content="")]}
        with patch.object(screen_mod, "create_agent", MagicMock(return_value=executor)):
            cmd = screen_mod.action_taker_agent(_state(_screen(instruction="press tab", rounds=1)))
        self.assertIn("Stopped: hit the limit", cmd.update["screen"]["report"])


CHROME_WINDOW = Window(id=42, x=1280, y=100, width=800, height=600)  # on a second display


class TestComputerTools(unittest.TestCase):
    def setUp(self):
        self.gui = MagicMock()
        self.gui.size.return_value = (1280, 832)
        self.gui.KEYBOARD_KEYS = ["command", "shift", "f", "enter", "escape", "v"]
        self.windows = MagicMock()
        self.windows.focus_app.return_value = True
        self.windows.front_window.return_value = CHROME_WINDOW
        self.windows.on_a_display.return_value = True
        for patcher in (patch.object(computer, "_gui", return_value=self.gui),
                        patch.object(computer, "macos_windows", self.windows),
                        patch.object(computer.time, "sleep")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_press_keys_maps_aliases(self):
        self.assertEqual(computer.press_keys.invoke({"keys": "cmd+F"}), "Pressed cmd+F.")
        self.gui.hotkey.assert_called_once_with("command", "f")
        computer.press_keys.invoke({"keys": "esc"})
        self.gui.hotkey.assert_called_with("escape")

    def test_press_keys_rejects_unknown_keys(self):
        self.assertIn("Unknown key", computer.press_keys.invoke({"keys": "cmd+banana"}))
        self.gui.hotkey.assert_not_called()

    def test_actions_bring_the_task_app_to_the_front_first(self):
        with screen_target("Google Chrome"):
            computer.press_keys.invoke({"keys": "enter"})
        self.windows.focus_app.assert_called_with("Google Chrome")
        self.gui.hotkey.assert_called_once_with("enter")

    def test_nothing_is_typed_when_the_app_cannot_be_brought_forward(self):
        self.windows.focus_app.return_value = False
        self.windows.is_running.return_value = False
        with screen_target("WhatsApp"), patch.object(computer.subprocess, "run") as run:
            pressed = computer.press_keys.invoke({"keys": "enter"})
            typed = computer.type_text.invoke({"text": "hi"})
        self.assertEqual(pressed, "WhatsApp is not open, so nothing was done.")
        self.assertEqual(typed, "WhatsApp is not open, so nothing was done.")
        self.gui.hotkey.assert_not_called()
        run.assert_not_called()

    def test_type_text_pastes_and_restores_clipboard(self):
        with patch.object(computer.subprocess, "run") as run:
            run.return_value.stdout = "old clipboard"
            computer.type_text.invoke({"text": "héllo 👋"})
        pasted = [c.kwargs.get("input") for c in run.call_args_list if c.args[0] == ["pbcopy"]]
        self.assertEqual(pasted, ["héllo 👋", "old clipboard"])
        self.gui.hotkey.assert_called_once_with("command", "v")

    def _capture(self, **fields):
        values = {"image_b64": "QUJD", "width": 800, "height": 600, "origin_x": 1280,
                  "origin_y": 100, "scale": 1.0, "what": "the Google Chrome window"}
        return computer.Capture(**{**values, **fields})

    def test_click_on_clicks_inside_the_window_on_any_display(self):
        with screen_target("Google Chrome"), \
             patch.object(computer, "_ask_vision", return_value=("At 190, 120.", self._capture())) as ask:
            result = computer.click_on.invoke({"target": "the search result named Alex"})
        self.assertIn("the search result named Alex", ask.call_args.args[0])
        self.gui.click.assert_called_once_with(1470, 220, clicks=1)
        self.assertIn("Clicked at 1470,220", result)

    def test_click_on_converts_retina_pixels_to_screen_points(self):
        retina = self._capture(width=1600, height=1200, scale=0.5)  # 800x600-point window
        with patch.object(computer, "_ask_vision", return_value=("At 380, 240.", retina)):
            computer.click_on.invoke({"target": "the message box"})
        self.gui.click.assert_called_once_with(1280 + 190, 100 + 120, clicks=1)

    def test_click_on_refuses_points_off_every_display(self):
        self.windows.on_a_display.return_value = False
        with patch.object(computer, "_ask_vision", return_value=("At 10, 10.", self._capture())):
            self.assertIn("not on any screen", computer.click_on.invoke({"target": "the box"}))
        self.gui.click.assert_not_called()

    def test_an_on_off_button_is_clicked_only_once_per_task(self):
        with screen_target("Safari") as target, \
             patch.object(computer, "_ask_vision", return_value=("At 100, 100.", self._capture())):
            first = computer.click_on.invoke({"target": "the thumbs-up Like button"})
            second = computer.click_on.invoke({"target": "the Like button below the video"})
            other = computer.click_on.invoke({"target": "the Subscribe button"})
        self.assertIn("Clicked", first)
        self.assertIn("already clicked once", second)
        self.assertIn("Clicked", other)  # a different on/off button is fine
        self.assertEqual(self.gui.click.call_count, 2)
        self.assertEqual(target.toggled, {"like", "subscribe"})

    def test_toggle_names(self):
        for text, kind in [("thumbs-up Like button", "like"), ("the thumbs up icon", "like"),
                           ("Dislike", "dislike"), ("the Follow button", "follow"),
                           ("the 'remember me' checkbox", "checkbox"), ("the search box", None)]:
            self.assertEqual(computer._toggle_kind(text), kind, text)

    def test_click_on_does_not_click_when_not_found_or_outside(self):
        for answer in ["NOT FOUND", "At 900, 50."]:  # 900 is wider than the 800-wide window
            with patch.object(computer, "_ask_vision", return_value=(answer, self._capture())):
                self.assertIn("Could not find", computer.click_on.invoke({"target": "the Send button"}))
        self.gui.click.assert_not_called()

    def test_screenshot_sends_the_window_to_the_vision_model(self):
        vision = MagicMock()
        vision.invoke.return_value = AIMessage(content="YouTube results are showing.")
        with screen_target("Google Chrome"), \
             patch.object(computer, "_capture", return_value=self._capture()), \
             patch("models.models.vision_llm", vision):
            answer = computer.screenshot.invoke({"question": "Are results showing?"})
        self.assertEqual(answer, "YouTube results are showing.")
        content = vision.invoke.call_args.args[0][0].content
        self.assertTrue(content[0]["text"].startswith(
            "This is a screenshot of the Google Chrome window (800x600 pixels). Are results showing?"))
        self.assertEqual(content[1]["image_url"]["url"], "data:image/jpeg;base64,QUJD")
        self.windows.focus_app.assert_called_with("Google Chrome")


LSAPPINFO_LIST = """ 10) "Finder" ASN:0x0-0x10010:
    bundleID="com.apple.finder"
    bundle path="/System/Library/CoreServices/Finder.app"
    pid = 462 token=[sess=100002 pid=462] type="Foreground"
 11) "Safari" ASN:0x0-0x2a02a:
    bundle path="/System/Volumes/Preboot/Cryptexes/App/System/Applications/Safari.app"
    pid = 28425 type="Foreground"
 12) "Code" ASN:0x0-0x166166:
    bundle path="/Applications/Visual Studio Code.app"
    pid = 16557 type="Foreground"
"""


class TestRunningApps(unittest.TestCase):
    """App checks read `lsappinfo`, which is current from any thread."""

    def _fake_lsappinfo(self, front_pid):
        def run(*args):
            if args == ("list",):
                return LSAPPINFO_LIST
            if args == ("front",):
                return "ASN:0x0-0x1:"
            return f'"x" ASN:0x0-0x1:\n    pid = {front_pid} type="Foreground"'
        return patch.object(macos_windows, "_lsappinfo", side_effect=run)

    def test_finds_apps_by_bundle_or_display_name(self):
        with self._fake_lsappinfo(front_pid=16557):
            self.assertEqual(macos_windows._running_app("Safari").pid, 28425)
            self.assertEqual(macos_windows._running_app("Visual Studio Code").pid, 16557)
            self.assertFalse(macos_windows.is_running("Chess"))

    def test_knows_which_app_is_in_front(self):
        with self._fake_lsappinfo(front_pid=28425):
            self.assertTrue(macos_windows.is_frontmost("Safari"))
            self.assertFalse(macos_windows.is_frontmost("Finder"))


class TestOpenAppSetsTarget(unittest.TestCase):
    def test_open_app_points_the_screen_task_at_the_app(self):
        with patch.object(terminal.sys, "platform", "darwin"), \
             patch.object(terminal, "installed_apps", return_value={"google chrome": "Google Chrome"}), \
             patch.object(terminal.macos_windows, "is_running", return_value=False), \
             patch.object(terminal, "interrupt", return_value="yes"), \
             patch.object(terminal.subprocess, "run") as run, \
             screen_target() as target:
            run.return_value.returncode = 0
            terminal.open_app.invoke({"app": "chrome"})
        self.assertEqual(target.app, "Google Chrome")

    def test_an_app_that_is_already_open_is_brought_forward_without_asking(self):
        with patch.object(terminal.sys, "platform", "darwin"), \
             patch.object(terminal, "installed_apps", return_value={"safari": "Safari"}), \
             patch.object(terminal.macos_windows, "is_running", return_value=True), \
             patch.object(terminal.macos_windows, "focus_app", return_value=True), \
             patch.object(terminal, "interrupt") as ask, \
             screen_target() as target:
            result = terminal.open_app.invoke({"app": "safari"})
        ask.assert_not_called()
        self.assertIn("already open", result)
        self.assertEqual(target.app, "Safari")


class TestGraphStepLimit(unittest.TestCase):
    def test_step_limit_is_added_but_never_overrides_the_caller(self):
        self.assertEqual(_run_config({"configurable": {"thread_id": "t"}})["recursion_limit"], GRAPH_STEP_LIMIT)
        self.assertEqual(_run_config({"recursion_limit": 5})["recursion_limit"], 5)


if __name__ == "__main__":
    unittest.main()
