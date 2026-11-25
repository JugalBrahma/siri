from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.types import Command

from state.message_state import (
    ActionSupervisorDecision,
    InfoSupervisorDecision,
    MainSupervisorDecision,
    State,
    SupervisorState,
    get_supervisor_state,
    latest_user_task,
    new_supervisor_state,
    record_agent_output,
    semantic_memory_prompt,
    set_routing_plan,
    with_recent_history,
)
from models.models import llm, reasoning_llm
from prompts import load_prompt

from .fast_path import ACTION_KEYWORDS, INFO_FAST_PATH, MAIN_FAST_PATH, pick_by_keywords

MAIN_SUPERVISOR_PROMPT = load_prompt("main_supervisor")
INFO_SUPERVISOR_PROMPT = load_prompt("info_supervisor")
ACTION_SUPERVISOR_PROMPT = load_prompt("action_supervisor")

def format_agent_outputs_prompt(agent_outputs: dict[str, str]) -> str:
    """Format completed agent outputs with [STATUS: DONE] for the supervisor prompt."""
    if not agent_outputs:
        return ""
    lines = ["\n\n### Completed Tasks & Agent Outputs:"]
    for agent_name, output in agent_outputs.items():
        lines.append(f"- **{agent_name}** [STATUS: DONE]:\n{output.strip()}")
    lines.append("\n**CRITICAL RULE**: NEVER route to any agent or sub-supervisor marked [STATUS: DONE] above.")
    return "\n".join(lines)


def _llm_decide(base_prompt: str, scope: SupervisorState, user_text: str, schema, label: str, recent_history: str = "",
                model=None,
                method: Literal["function_calling", "json_mode", "json_schema"] | None = None) -> tuple:
    """Ask the LLM for a routing decision and return (next, reasoning).

    The system prompt is the supervisor's own prompt plus the shared semantic
    memory and the outputs already completed in this scope.
    """
    messages = [
        {
            "role": "system",
            "content": (
                base_prompt
                + semantic_memory_prompt(scope["semantic_memory"])
                + format_agent_outputs_prompt(scope["agent_outputs"])
            ),
        },
        {
            "role": "user",
            "content": f"User Request: {user_text}" if not recent_history.strip()
            else with_recent_history(user_text, recent_history),
        },
    ]
    structured = (model or llm).with_structured_output(schema, method=method) if method \
        else (model or llm).with_structured_output(schema)
    response = structured.invoke(messages)

    def field(name: str, default=""):
        return response.get(name, default) if isinstance(response, dict) else getattr(response, name, default)

    goto, reasoning, question = field("next", None), field("reasoning"), field("question")

    print(f"{label} Decision: {goto} (Reasoning: {reasoning})")
    return goto, reasoning, question or ""


def _decide(state: State, scope: SupervisorState, user_text: str, fast_path: tuple, base_prompt: str,
            schema, label: str, **model_options) -> tuple:
    """(next, reasoning, question): a keyword fast-path on the first decision, else the LLM."""
    if fast_path and not scope["routing_plan"] and not scope["agent_outputs"]:
        fast = pick_by_keywords(user_text, *fast_path)
        if fast:
            reasoning = f"[fast-path] keyword match → {fast}"
            print(f"{label} Decision: {fast} (Reasoning: {reasoning})")
            return fast, reasoning, ""
    return _llm_decide(base_prompt, scope, user_text, schema, label,
                       state.get("recent_history") or "", **model_options)


def _route(scope_key: str, scope: SupervisorState, goto: str, **extra) -> Command:
    """Go to `goto`, recording it in this scope's routing plan."""
    return Command(
        goto=goto,
        update={scope_key: set_routing_plan(scope, scope["routing_plan"] + [goto]), "next": goto, **extra},
    )


def _ask_user(scope_key: str, scope: SupervisorState, question: str, reasoning: str, return_to: str) -> Command:
    """Pause for the user's answer via the ask_user node, then come back to `return_to`."""
    clarification = {"question": question.strip() or reasoning.strip(), "return_to": return_to}
    return _route(scope_key, scope, "ask_user", clarification=clarification)


def _report_to_main(state: State, name: str, scope: SupervisorState, reasoning: str) -> Command:
    """Record a sub-supervisor's combined result with the main supervisor and go back to it."""
    result = "\n\n".join(scope["agent_outputs"].values()) if scope["agent_outputs"] else reasoning
    main_state = record_agent_output(get_supervisor_state(state, "main_supervisor"), name, result)
    return Command(goto="supervisor", update={"main_supervisor": main_state, "next": "supervisor"})


def _finish_main(main_state: SupervisorState, content: str) -> Command:
    """Hand the final answer to output_sanitizer."""
    return Command(
        goto="output_sanitizer",
        update={
            "messages": [AIMessage(content=content, name="supervisor")],
            "main_supervisor": set_routing_plan(main_state, main_state["routing_plan"] + ["output_sanitizer"]),
            "next": "output_sanitizer",
        },
    )


def superVisorAgent(state: State) -> Command[Literal['sub_infosupervisor', 'sub_actionsupervisor', 'ask_user', 'output_sanitizer']]:
    print("--- SUPERVISOR AGENT ---")
    main_state = get_supervisor_state(state, "main_supervisor")
    outputs = main_state["agent_outputs"]
    routing_history = main_state["routing_plan"]
    user_text = latest_user_task(state.get("messages", []))

    # The action side always finishes the turn.
    if "sub_actionsupervisor" in outputs:
        print("  ↳ Fast-finish: Action execution complete. Routing to output_sanitizer.")
        return _finish_main(main_state, outputs["sub_actionsupervisor"])

    goto, reasoning, question = _decide(
        state, main_state, user_text, MAIN_FAST_PATH,
        MAIN_SUPERVISOR_PROMPT, MainSupervisorDecision, "Main Supervisor",
    )

    # Never re-route to a sub-supervisor that already finished.
    if goto in outputs:
        print(f"  ↳ Guard: {goto} already completed [STATUS: DONE]. Checking remaining tasks.")
        if goto == "sub_infosupervisor" and ACTION_KEYWORDS.search(user_text):
            goto = "sub_actionsupervisor"
            reasoning = "Information gathered; proceeding to action execution."
        else:
            goto = "FINISH"
            reasoning = "All requested tasks completed."

    if goto == "FINISH" or routing_history.count(goto) >= 2 or not goto:
        return _finish_main(main_state, outputs.get("sub_infosupervisor") or "\n\n".join(outputs.values()) or reasoning)

    if goto == "ask_user":
        return _ask_user("main_supervisor", main_state, question, reasoning, "supervisor")

    fresh = {"sub_infosupervisor": "info_supervisor", "sub_actionsupervisor": "action_supervisor"}
    extra = {fresh[goto]: new_supervisor_state(main_state["semantic_memory"])} if goto in fresh else {}
    return _route("main_supervisor", main_state, goto, **extra)


def sub_infoSuperVisorAgent(state: State) -> Command[Literal['researcher', 'weather', 'ask_user', 'supervisor']]:
    print("--- SUB INFO SUPERVISOR AGENT ---")
    info_state = get_supervisor_state(state, "info_supervisor")
    routing_history = info_state["routing_plan"]
    user_text = latest_user_task(state.get("messages", []))

    goto, reasoning, question = _decide(
        state, info_state, user_text, INFO_FAST_PATH,
        INFO_SUPERVISOR_PROMPT, InfoSupervisorDecision, "Sub-Info",
    )

    # Never re-route to a worker that already finished.
    if goto in info_state["agent_outputs"]:
        print(f"  ↳ Guard: {goto} already completed [STATUS: DONE]. Returning to supervisor.")
        goto = "supervisor"

    if goto == "supervisor" or routing_history.count(goto) >= 2 or not goto:
        return _report_to_main(state, "sub_infosupervisor", info_state, reasoning)

    if goto == "ask_user":
        return _ask_user("info_supervisor", info_state, question, reasoning, "sub_infosupervisor")

    return _route("info_supervisor", info_state, goto)


def sub_actionSuperVisorAgent(state: State) -> Command[Literal['document_writer', 'screen_reader', 'ask_user', 'supervisor']]:
    print("--- SUB ACTION SUPERVISOR AGENT ---")
    action_state = get_supervisor_state(state, "action_supervisor")
    routing_history = action_state["routing_plan"]
    user_text = latest_user_task(state.get("messages", []))

    goto, reasoning, question = _decide(
        state, action_state, user_text, (),
        ACTION_SUPERVISOR_PROMPT, ActionSupervisorDecision, "Sub-Action",
        model=reasoning_llm,  # thinks first: fewer needless questions, better routing
        method="function_calling",  # a tool call parses reliably; JSON mode can add stray text
    )

    # One action worker per turn: once it has reported, go back to the main supervisor.
    if goto == "supervisor" or action_state["agent_outputs"] or routing_history.count(goto) >= 2 or not goto:
        return _report_to_main(state, "sub_actionsupervisor", action_state, reasoning)

    if goto == "ask_user":
        return _ask_user("action_supervisor", action_state, question, reasoning, "sub_actionsupervisor")

    return _route("action_supervisor", action_state, goto)
