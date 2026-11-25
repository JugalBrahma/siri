"""screen_reader and action_taker: two agents that operate apps on screen together.

screen_reader looks (opens apps, takes screenshots) and decides the next step.
action_taker acts (types, presses keys, clicks) and hands back to screen_reader
to check the result. They pass work through ``state["screen"]``.

Asking the user (a missing detail, or "Send this?") goes through the graph's
ask_user node, never an in-agent interrupt. A resumed node re-runs from its
start, so action_taker pausing itself would repeat its typing on resume.

Both agents run their tools inside ``screen_target(app)``: every tool brings
that app to the front first and screenshots only its window, so it works on
any display and keys never land in the window the user answered from.
"""

from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware, ToolCallLimitMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from models.models import llm, reasoning_llm
from prompts import load_prompt
from state.message_state import (
    ScreenDecision,
    ScreenTask,
    State,
    build_worker_state,
    get_supervisor_state,
    latest_user_task,
    new_screen_task,
    record_agent_output,
    semantic_memory_prompt,
)
from tools.computer import click_on, press_keys, screenshot, type_text
from tools.screen_target import screen_target
from tools.terminal import list_apps, open_app
from .worker import gathered_info_suffix

# screen_reader -> action_taker hand-offs allowed per task before it stops.
MAX_SCREEN_ROUNDS = 6

# Limits inside one run of each agent, so a confused model cannot loop on
# paid, slow vision calls. A tool past its limit is refused and the model
# must answer with what it has.
READER_MODEL_CALLS = 8
READER_SCREENSHOTS = 3
READER_OPEN_APPS = 2
TAKER_MODEL_CALLS = 10
TAKER_CLICK_ONS = 4

SCREEN_READER_PROMPT = load_prompt("screen_reader", max_rounds=MAX_SCREEN_ROUNDS)
ACTION_TAKER_PROMPT = load_prompt("action_taker")

# How a "yes" may be worded. An answer starts with a yes-word and may only add
# filler ("yes go ahead", "ok do it") or "don't ask again". Anything else, such
# as an edit ("yes but write hello"), "no" or "wait", is not a yes.
YES_WORDS = {"y", "yes", "yeah", "yep", "yup", "ok", "okay", "sure", "go", "send",
             "confirm", "confirmed", "approve", "approved", "allow", "do"}
YES_FILLER = YES_WORDS | {"it", "ahead", "please", "now", "just", "that", "fine"}
DONT_ASK_AGAIN = ("don't ask again", "dont ask again", "do not ask again",
                  "no need to ask", "stop asking")


def _screen_of(state: State) -> ScreenTask:
    """The current screen task, with any missing fields filled in."""
    screen = new_screen_task()
    screen.update(state.get("screen") or {})
    return screen


def _answer_to(question: str, state: State) -> str:
    """The user's answer to `question`, as the ask_user node wrote it into the request."""
    text = latest_user_task(list(state.get("messages", [])))
    marker = f'(When asked "{question}", the user answered: '
    start = text.rfind(marker)
    if start < 0:
        return ""
    return text[start + len(marker):].rstrip().removesuffix(")")


def _is_clear_yes(answer: str) -> bool:
    text = answer.lower().replace("’", "'")
    for phrase in DONT_ASK_AGAIN:
        text = text.replace(phrase, " ")
    words = "".join(ch if ch.isalnum() or ch in " '" else " " for ch in text).split()
    return bool(words) and words[0] in YES_WORDS and all(w in YES_FILLER for w in words[1:])


def _progress(screen: ScreenTask, answer: str) -> str:
    lines = [f"Working in: {screen['app']} (brought to the front automatically)."] if screen["app"] else []
    if screen["toggled"]:
        lines.append(f"Already clicked once (on/off, never click again): {', '.join(screen['toggled'])}.")
    if screen["instruction"]:
        lines += [
            f"Rounds used: {screen['rounds']} of {MAX_SCREEN_ROUNDS}.",
            f"Your last instruction to action_taker: {screen['instruction']}",
            f"action_taker reported (not yet checked on screen): {screen['report'] or 'nothing'}",
        ]
    if answer:
        verdict = ("Approved. Go ahead, and do not ask again in this task (retries included)."
                   if screen["approved"] else
                   "Not a yes. Use the answer (a detail or a change); do nothing risky "
                   "the user has not asked for or approved.")
        lines.append(f"The user answered your question: {answer!r} -> {verdict}")
    elif screen["approved"]:
        lines.append("The user already approved this task. Do not ask again.")
    return "\n\nProgress so far:\n" + "\n".join(lines) if lines else ""


def _took_screenshot(result: dict) -> bool:
    return any(isinstance(m, ToolMessage) and m.name == "screenshot" for m in result.get("messages", []))


def _ask(screen: ScreenTask, question: str) -> Command:
    """Pause in the ask_user node; it brings the answer back to screen_reader.

    Every question is remembered, so a clear yes to it approves the task,
    whether or not the model meant it as a confirmation.
    """
    return Command(
        goto="ask_user",
        update={
            "clarification": {"question": question, "return_to": "screen_reader"},
            "screen": {**screen, "confirm_question": question},
            "next": "ask_user",
        },
    )


def _finish(state: State, summary: str) -> Command:
    """Report the result to sub_actionsupervisor and clear the screen task."""
    scope_state = record_agent_output(
        get_supervisor_state(state, "action_supervisor"), "screen_reader", summary
    )
    print(f"Result: {summary}")
    return Command(
        goto="sub_actionsupervisor",
        update={
            "messages": [AIMessage(content=summary, name="screen_reader")],
            "action_supervisor": scope_state,
            "screen": None,
        },
    )


def screen_reader_agent(state: State) -> Command[Literal["action_taker", "ask_user", "sub_actionsupervisor"]]:
    print("--- SCREEN READER ---")
    screen: ScreenTask = _screen_of(state)

    # Coming back from a question: a clear yes approves the rest of the task;
    # anything else (an edit, "no") withdraws approval until asked again.
    answer = ""
    if screen["confirm_question"]:
        answer = _answer_to(screen["confirm_question"], state)
        screen = {**screen, "confirm_question": "", "approved": _is_clear_yes(answer)}

    worker_state = build_worker_state(state, "action_supervisor")
    # Typed explicitly: the limit middlewares have different state types, so an
    # inline list is inferred as a union that create_agent's overloads reject.
    middleware: list[AgentMiddleware[Any, Any, Any]] = [
        ModelCallLimitMiddleware(run_limit=READER_MODEL_CALLS),
        ToolCallLimitMiddleware(tool_name="screenshot", run_limit=READER_SCREENSHOTS),
        ToolCallLimitMiddleware(tool_name="open_app", run_limit=READER_OPEN_APPS),
    ]
    agent = create_agent(
        reasoning_llm,  # thinks before deciding: asks less, recovers better
        [open_app, list_apps, screenshot],
        system_prompt=SCREEN_READER_PROMPT + semantic_memory_prompt(worker_state["semantic_memory"]),
        # The decision is submitted as a tool call: a malformed one is sent back
        # to the model to fix, instead of failing the turn.
        response_format=ToolStrategy(ScreenDecision),
        middleware=middleware,
    )
    task = worker_state["task"] + gathered_info_suffix(state) + _progress(screen, answer)
    try:
        with screen_target(screen["app"], screen["toggled"]) as target:
            result = agent.invoke({"messages": [{"role": "user", "content": task}]})
            if screen["report"] and not _took_screenshot(result):
                # action_taker cannot see the screen, so its report must be checked.
                task += "\n\nYou did not take a screenshot. Call screenshot to check the last action before deciding."
                result = agent.invoke({"messages": [{"role": "user", "content": task}]})
    except GraphInterrupt:
        raise  # open_app asking for permission
    except Exception as exc:
        print(f"  ↳ screen_reader failed: {exc}")
        return _finish(state, "Stopped: the screen reader could not decide the next step "
                              f"({type(exc).__name__}). Nothing else was done.")
    screen = {**screen, "app": target.app}  # open_app may have set it

    decision: ScreenDecision | None = result.get("structured_response")
    if decision is None:
        return _finish(state, f"Stopped: the screen reader hit its limit of {READER_MODEL_CALLS} "
                              "steps in one go without deciding.")
    print(f"  ↳ {decision.next}: {decision.instruction or decision.question or decision.summary}")

    if decision.next == "action_taker" and decision.instruction.strip():
        # Safety net: a risky step the user neither asked for nor approved.
        if decision.risky and not decision.requested and not screen["approved"]:
            question = decision.question.strip() or f"About to: {decision.instruction}. Go ahead?"
            return _ask(screen, question)
        if screen["rounds"] >= MAX_SCREEN_ROUNDS:
            return _finish(state, f"Stopped after {MAX_SCREEN_ROUNDS} rounds without finishing. "
                                  f"Last step: {screen['report'] or screen['instruction']}")
        return Command(
            goto="action_taker",
            update={"screen": {**screen, "instruction": decision.instruction, "report": "",
                               "rounds": screen["rounds"] + 1}},
        )
    if decision.next == "ask_user" and decision.question.strip():
        return _ask(screen, decision.question)
    return _finish(state, decision.summary.strip() or "Finished the screen task.")


def action_taker_agent(state: State) -> Command[Literal["screen_reader"]]:
    print("--- ACTION TAKER ---")
    screen: ScreenTask = _screen_of(state)
    if not screen["instruction"]:
        return Command(goto="screen_reader", update={"screen": {**screen, "report": "No instruction given."}})

    middleware: list[AgentMiddleware[Any, Any, Any]] = [
        ModelCallLimitMiddleware(run_limit=TAKER_MODEL_CALLS),
        ToolCallLimitMiddleware(tool_name="click_on", run_limit=TAKER_CLICK_ONS),
    ]
    agent = create_agent(
        llm,
        [type_text, press_keys, click_on],
        system_prompt=ACTION_TAKER_PROMPT,
        middleware=middleware,
    )
    with screen_target(screen["app"], screen["toggled"]) as target:
        result = agent.invoke({"messages": [{"role": "user", "content": screen["instruction"]}]})
    report = str(result["messages"][-1].content).strip() or (
        f"Stopped: hit the limit of {TAKER_MODEL_CALLS} steps before finishing.")
    screen = {**screen, "toggled": sorted(target.toggled)}
    print(f"Result: {report}")
    return Command(goto="screen_reader", update={"screen": {**screen, "report": report}})
