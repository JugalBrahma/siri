import re
from langgraph.types import Command
from typing import Literal
from langchain_core.messages import AIMessage
from state.message_state import (
    ActionSupervisorDecision,
    InfoSupervisorDecision,
    MainSupervisorDecision,
    State,
    get_supervisor_state,
    latest_user_task,
    new_supervisor_state,
    record_agent_output,
    semantic_memory_prompt,
    set_routing_plan,
    with_recent_history,
)
from prompts import load_prompt
from models.models import llm, reasoning_llm

MAIN_SUPERVISOR_PROMPT = load_prompt("main_supervisor")
INFO_SUPERVISOR_PROMPT = load_prompt("info_supervisor")
ACTION_SUPERVISOR_PROMPT = load_prompt("action_supervisor")

# ---------------------------------------------------------------------------
# Fast-path keyword detection
# ---------------------------------------------------------------------------
# Patterns that strongly indicate an information-only request (no action).
_INFO_KEYWORDS = re.compile(
    r"\b(weather|forecast|temperature|humidity|rain|snow|wind|climate|"
    r"search|find|look\s?up|research|who\s+is|what\s+is|when\s+is|where\s+is|"
    r"tell\s+me|explain|summarize|summary)\b",
    re.IGNORECASE,
)
# Patterns that indicate the user also needs an action/execution.
_ACTION_KEYWORDS = re.compile(
    r"\b(create|write|make|generate|build|edit|modify|delete|remove|run|execute|"
    r"save|open|file|document|docx|code|script|terminal|command|install|"
    r"send|type|click|press|whatsapp)\b",
    re.IGNORECASE,
)
# Patterns that indicate weather specifically (for sub-info fast-path).
_WEATHER_KEYWORDS = re.compile(
    r"\b(weather|forecast|temperature|humidity|rain|snow|wind|climate|"
    r"celsius|fahrenheit|degrees|storm|sunny|cloudy)\b",
    re.IGNORECASE,
)
# Patterns that indicate research (non-weather information tasks).
_RESEARCH_KEYWORDS = re.compile(
    r"\b(search|find|look\s?up|research|article|news|event|tutorial|"
    r"who\s+is|what\s+is|when\s+is|where\s+is|tell\s+me\s+about|explain|summarize)\b",
    re.IGNORECASE,
)


def _pick_by_keywords(user_text: str, first: tuple, second: tuple) -> str | None:
    """Return the destination whose keyword pattern alone matches, else None.

    `first` and `second` are (pattern, destination) pairs. If both or neither
    match, the request is ambiguous and the caller falls back to LLM routing.
    """
    if not user_text:
        return None
    (first_pattern, first_dest), (second_pattern, second_dest) = first, second
    has_first = bool(first_pattern.search(user_text))
    has_second = bool(second_pattern.search(user_text))
    if has_first and not has_second:
        return first_dest
    if has_second and not has_first:
        return second_dest
    return None


_MAIN_FAST_PATH = ((_INFO_KEYWORDS, "sub_infosupervisor"), (_ACTION_KEYWORDS, "sub_actionsupervisor"))
_INFO_FAST_PATH = ((_WEATHER_KEYWORDS, "weather"), (_RESEARCH_KEYWORDS, "researcher"))


def format_agent_outputs_prompt(agent_outputs: dict[str, str]) -> str:
    """Format completed agent outputs with [STATUS: DONE] for the supervisor prompt."""
    if not agent_outputs:
        return ""
    lines = ["\n\n### Completed Tasks & Agent Outputs:"]
    for agent_name, output in agent_outputs.items():
        lines.append(f"- **{agent_name}** [STATUS: DONE]:\n{output.strip()}")
    lines.append("\n**CRITICAL RULE**: NEVER route to any agent or sub-supervisor marked [STATUS: DONE] above.")
    return "\n".join(lines)


def _llm_decide(base_prompt: str, scope: dict, user_text: str, schema, label: str, recent_history: str = "",
                model=None, method: str | None = None) -> tuple:
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
    options = {"method": method} if method else {}
    response = (model or llm).with_structured_output(schema, **options).invoke(messages)

    if isinstance(response, dict):
        goto = response.get("next")
        reasoning = response.get("reasoning", "")
        question = response.get("question", "")
    else:
        goto = getattr(response, "next", None)
        reasoning = getattr(response, "reasoning", "")
        question = getattr(response, "question", "")

    print(f"{label} Decision: {goto} (Reasoning: {reasoning})")
    return goto, reasoning, question or ""


def _ask_user(scope_key: str, scope_state: dict, routing_history: list, question: str, reasoning: str, return_to: str) -> Command:
    """Pause for the user's answer via the ask_user node, then come back to `return_to`."""
    question = question.strip() or reasoning.strip()
    return Command(
        goto="ask_user",
        update={
            scope_key: set_routing_plan(scope_state, routing_history + ["ask_user"]),
            "clarification": {"question": question, "return_to": return_to},
            "next": "ask_user",
        },
    )


def _decide(state: State, scope: dict, user_text: str, fast_path: tuple, base_prompt: str,
            schema, label: str, **model_options) -> tuple:
    """(next, reasoning, question): a keyword fast-path on the first decision, else the LLM."""
    if fast_path and not scope["routing_plan"] and not scope["agent_outputs"]:
        fast = _pick_by_keywords(user_text, *fast_path)
        if fast:
            reasoning = f"[fast-path] keyword match → {fast}"
            print(f"{label} Decision: {fast} (Reasoning: {reasoning})")
            return fast, reasoning, ""
    return _llm_decide(base_prompt, scope, user_text, schema, label,
                       state.get("recent_history") or "", **model_options)


def _report_to_main(state: State, name: str, scope: dict, reasoning: str) -> Command:
    """Record a sub-supervisor's combined result with the main supervisor and go back to it."""
    result = "\n\n".join(scope["agent_outputs"].values()) if scope["agent_outputs"] else reasoning
    main_state = record_agent_output(get_supervisor_state(state, "main_supervisor"), name, result)
    return Command(goto="supervisor", update={"main_supervisor": main_state, "next": "supervisor"})


def _finish_main(main_state: dict, content: str) -> Command:
    """Hand the final answer to output_sanitizer."""
    return Command(
        goto="output_sanitizer",
        update={
            "messages": [AIMessage(content=content, name="supervisor")],
            "main_supervisor": set_routing_plan(main_state, main_state["routing_plan"] + ["output_sanitizer"]),
            "next": "output_sanitizer",
        },
    )


# ---------------------------------------------------------------------------
# Main Supervisor
# ---------------------------------------------------------------------------

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
        state, main_state, user_text, _MAIN_FAST_PATH,
        MAIN_SUPERVISOR_PROMPT, MainSupervisorDecision, "Main Supervisor",
    )

    # --- Re-routing Guard: Never re-route to an already completed sub-supervisor ---
    if goto in outputs:
        print(f"  ↳ Guard: {goto} already completed [STATUS: DONE]. Checking remaining tasks.")
        if goto == "sub_infosupervisor" and _ACTION_KEYWORDS.search(user_text):
            goto = "sub_actionsupervisor"
            reasoning = "Information gathered; proceeding to action execution."
        else:
            goto = "FINISH"
            reasoning = "All requested tasks completed."

    # --- FINISH or loop guard ---
    if goto == "FINISH" or routing_history.count(goto) >= 2 or not goto:
        return _finish_main(main_state, outputs.get("sub_infosupervisor") or "\n\n".join(outputs.values()) or reasoning)

    if goto == "ask_user":
        return _ask_user("main_supervisor", main_state, routing_history, question, reasoning, "supervisor")

    update = {"main_supervisor": set_routing_plan(main_state, routing_history + [goto]), "next": goto}
    if goto == "sub_infosupervisor":
        update["info_supervisor"] = new_supervisor_state(main_state["semantic_memory"])
    elif goto == "sub_actionsupervisor":
        update["action_supervisor"] = new_supervisor_state(main_state["semantic_memory"])
    return Command(goto=goto, update=update)


# ---------------------------------------------------------------------------
# Sub-Info Supervisor
# ---------------------------------------------------------------------------

def sub_infoSuperVisorAgent(state: State) -> Command[Literal['researcher', 'weather', 'ask_user', 'supervisor']]:
    print("--- SUB INFO SUPERVISOR AGENT ---")
    info_state = get_supervisor_state(state, "info_supervisor")
    routing_history = info_state["routing_plan"]
    user_text = latest_user_task(state.get("messages", []))

    goto, reasoning, question = _decide(
        state, info_state, user_text, _INFO_FAST_PATH,
        INFO_SUPERVISOR_PROMPT, InfoSupervisorDecision, "Sub-Info",
    )

    # --- Guard: Never re-route to an already completed worker ---
    if goto in info_state["agent_outputs"]:
        print(f"  ↳ Guard: {goto} already completed [STATUS: DONE]. Returning to supervisor.")
        goto = "supervisor"

    # --- Route back to Main Supervisor when info tasks are gathered ---
    if goto == "supervisor" or routing_history.count(goto) >= 2 or not goto:
        return _report_to_main(state, "sub_infosupervisor", info_state, reasoning)

    if goto == "ask_user":
        return _ask_user("info_supervisor", info_state, routing_history, question, reasoning, "sub_infosupervisor")

    return Command(
        goto=goto,
        update={"info_supervisor": set_routing_plan(info_state, routing_history + [goto]), "next": goto},
    )


# ---------------------------------------------------------------------------
# Sub-Action Supervisor
# ---------------------------------------------------------------------------

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
        return _ask_user("action_supervisor", action_state, routing_history, question, reasoning, "sub_actionsupervisor")

    return Command(
        goto=goto,
        update={"action_supervisor": set_routing_plan(action_state, routing_history + [goto]), "next": goto},
    )
