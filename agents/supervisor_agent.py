import re
from langgraph.graph import END
from langgraph.types import Command
from typing import Literal
from langchain_core.messages import AIMessage
from state.message_state import (
    ActionSupervisorDecision,
    InfoSupervisorDecision,
    MainSupervisorDecision,
    State,
    get_supervisor_state,
    new_supervisor_state,
    record_agent_output,
    semantic_memory_prompt,
    set_routing_plan,
)
from .prompt import system_prompt, sub_info_supervisor_prompt, sub_action_supervisor_prompt
from models.models import llm

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
    r"save|open|file|document|docx|code|script|terminal|command|install)\b",
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


def _extract_user_text(state: State) -> str:
    """Get the latest user message text for keyword analysis."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, dict):
            if msg.get("role") in {"user", "human"}:
                return msg.get("content", "")
        elif getattr(msg, "type", None) in {"user", "human"}:
            return getattr(msg, "content", "")
    return ""


def _fast_path_main(user_text: str) -> str | None:
    """Return a fast-path destination or None to fall through to LLM routing."""
    if not user_text:
        return None
    has_info = bool(_INFO_KEYWORDS.search(user_text))
    has_action = bool(_ACTION_KEYWORDS.search(user_text))
    if has_info and not has_action:
        return "sub_infosupervisor"
    if has_action and not has_info:
        return "sub_actionsupervisor"
    # Ambiguous or mixed → let the LLM decide
    return None


def _fast_path_info(user_text: str) -> str | None:
    """Return a fast-path worker destination or None to fall through to LLM routing."""
    if not user_text:
        return None
    has_weather = bool(_WEATHER_KEYWORDS.search(user_text))
    has_research = bool(_RESEARCH_KEYWORDS.search(user_text))
    if has_weather and not has_research:
        return "weather"
    if has_research and not has_weather:
        return "researcher"
    # Both or neither → let the LLM decide
    return None


def format_agent_outputs_prompt(agent_outputs: dict[str, str]) -> str:
    """Format completed agent outputs with [STATUS: DONE] for the supervisor prompt."""
    if not agent_outputs:
        return ""
    lines = ["\n\n### Completed Tasks & Agent Outputs:"]
    for agent_name, output in agent_outputs.items():
        lines.append(f"- **{agent_name}** [STATUS: DONE]:\n{output.strip()}")
    lines.append("\n**CRITICAL RULE**: NEVER route to any agent or sub-supervisor marked [STATUS: DONE] above.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main Supervisor
# ---------------------------------------------------------------------------

def superVisorAgent(state: State) -> Command[Literal['sub_infosupervisor', 'sub_actionsupervisor', 'output_sanitizer']]:
    print("--- SUPERVISOR AGENT ---")
    main_state = get_supervisor_state(state, "main_supervisor")
    routing_history = main_state.get("routing_plan", [])
    user_text = _extract_user_text(state)

    # --- Fast-finish: If sub_actionsupervisor has finished, complete immediately ---
    if "sub_actionsupervisor" in main_state["agent_outputs"]:
        final_content = main_state["agent_outputs"]["sub_actionsupervisor"]
        print("  ↳ Fast-finish: Action execution complete. Routing to output_sanitizer.")
        return Command(
            goto="output_sanitizer",
            update={
                "messages": [AIMessage(content=final_content, name="supervisor")],
                "main_supervisor": set_routing_plan(main_state, routing_history + ["output_sanitizer"]),
                "next": "output_sanitizer",
            },
        )

    # --- Fast-path: skip the LLM call on the very first routing decision if unambiguous ---
    goto = None
    reasoning = ""
    if not routing_history and not main_state["agent_outputs"]:
        fast = _fast_path_main(user_text)
        if fast:
            goto = fast
            reasoning = f"[fast-path] keyword match → {fast}"
            print(f"Main Supervisor Decision: {goto} (Reasoning: {reasoning})")

    # --- Fall through to LLM routing if fast-path didn't match ---
    if goto is None:
        messages = [
            {
                "role": "system",
                "content": (
                    system_prompt
                    + semantic_memory_prompt(main_state["semantic_memory"])
                    + format_agent_outputs_prompt(main_state["agent_outputs"])
                ),
            },
            {
                "role": "user",
                "content": f"User Request: {user_text}",
            },
        ]
        llm_with_structured_output = llm.with_structured_output(MainSupervisorDecision)
        response = llm_with_structured_output.invoke(messages)

        if isinstance(response, dict):
            goto = response.get("next")
            reasoning = response.get("reasoning", "")
        else:
            goto = getattr(response, "next", None)
            reasoning = getattr(response, "reasoning", "")

        print(f"Main Supervisor Decision: {goto} (Reasoning: {reasoning})")

    # --- Re-routing Guard: Never re-route to an already completed sub-supervisor ---
    if goto in main_state["agent_outputs"]:
        print(f"  ↳ Guard: {goto} already completed [STATUS: DONE]. Checking remaining tasks.")
        has_action = bool(_ACTION_KEYWORDS.search(user_text))
        if goto == "sub_infosupervisor" and has_action and "sub_actionsupervisor" not in main_state["agent_outputs"]:
            goto = "sub_actionsupervisor"
            reasoning = "Information gathered; proceeding to action execution."
        else:
            goto = "FINISH"
            reasoning = "All requested tasks completed."

    # --- FINISH or loop guard ---
    if goto == "FINISH" or routing_history.count(goto) >= 2 or not goto:
        if "sub_actionsupervisor" in main_state["agent_outputs"]:
            final_content = main_state["agent_outputs"]["sub_actionsupervisor"]
        elif "sub_infosupervisor" in main_state["agent_outputs"]:
            final_content = main_state["agent_outputs"]["sub_infosupervisor"]
        elif main_state["agent_outputs"]:
            final_content = "\n\n".join(main_state["agent_outputs"].values())
        else:
            final_content = reasoning

        return Command(
            goto="output_sanitizer",
            update={
                "messages": [AIMessage(content=final_content, name="supervisor")],
                "main_supervisor": set_routing_plan(main_state, routing_history + ["output_sanitizer"]),
                "next": "output_sanitizer",
            },
        )

    updated_main_state = set_routing_plan(main_state, routing_history + [goto])
    update = {"main_supervisor": updated_main_state, "next": goto}
    if goto == "sub_infosupervisor":
        update["info_supervisor"] = new_supervisor_state(
            main_state["semantic_memory"],
        )
    elif goto == "sub_actionsupervisor":
        update["action_supervisor"] = new_supervisor_state(
            main_state["semantic_memory"],
        )

    return Command(goto=goto, update=update)


# ---------------------------------------------------------------------------
# Sub-Info Supervisor
# ---------------------------------------------------------------------------

def sub_infoSuperVisorAgent(state: State) -> Command[Literal['researcher', 'weather', 'supervisor']]:
    print("--- SUB INFO SUPERVISOR AGENT ---")
    info_state = get_supervisor_state(state, "info_supervisor")
    routing_history = info_state.get("routing_plan", [])
    user_text = _extract_user_text(state)

    # --- Fast-path: skip the LLM call on the first routing decision ---
    goto = None
    reasoning = ""
    if not routing_history and not info_state["agent_outputs"]:
        fast = _fast_path_info(user_text)
        if fast:
            goto = fast
            reasoning = f"[fast-path] keyword match → {fast}"
            print(f"Sub-Info Decision: {goto} (Reasoning: {reasoning})")

    # --- Fall through to LLM routing if fast-path didn't match ---
    if goto is None:
        messages = [
            {
                "role": "system",
                "content": (
                    sub_info_supervisor_prompt
                    + semantic_memory_prompt(info_state["semantic_memory"])
                    + format_agent_outputs_prompt(info_state["agent_outputs"])
                ),
            },
            {
                "role": "user",
                "content": f"User Request: {user_text}",
            },
        ]
        llm_with_structured_output = llm.with_structured_output(InfoSupervisorDecision)
        response = llm_with_structured_output.invoke(messages)

        if isinstance(response, dict):
            goto = response.get("next")
            reasoning = response.get("reasoning", "")
        else:
            goto = getattr(response, "next", None)
            reasoning = getattr(response, "reasoning", "")

        print(f"Sub-Info Decision: {goto} (Reasoning: {reasoning})")

    # --- Guard: Never re-route to an already completed worker ---
    if goto in info_state["agent_outputs"]:
        print(f"  ↳ Guard: {goto} already completed [STATUS: DONE]. Returning to supervisor.")
        goto = "supervisor"

    # --- Route back to Main Supervisor when info tasks are gathered ---
    if goto == "supervisor" or routing_history.count(goto) >= 2 or not goto:
        main_state = get_supervisor_state(state, "main_supervisor")
        combined_result = "\n\n".join(info_state["agent_outputs"].values()) if info_state["agent_outputs"] else reasoning
        main_state = record_agent_output(
            main_state,
            "sub_infosupervisor",
            combined_result,
        )

        # Always return cleanly to main supervisor
        return Command(
            goto="supervisor",
            update={"main_supervisor": main_state, "next": "supervisor"},
        )

    info_state = set_routing_plan(info_state, routing_history + [goto])
    return Command(
        goto=goto,
        update={"info_supervisor": info_state, "next": goto},
    )


# ---------------------------------------------------------------------------
# Sub-Action Supervisor
# ---------------------------------------------------------------------------

def sub_actionSuperVisorAgent(state: State) -> Command[Literal['action', 'supervisor']]:
    print("--- SUB ACTION SUPERVISOR AGENT ---")
    action_state = get_supervisor_state(state, "action_supervisor")
    routing_history = action_state.get("routing_plan", [])
    user_text = _extract_user_text(state)

    messages = [
        {
            "role": "system",
            "content": (
                sub_action_supervisor_prompt
                + semantic_memory_prompt(action_state["semantic_memory"])
                + format_agent_outputs_prompt(action_state["agent_outputs"])
            ),
        },
        {
            "role": "user",
            "content": f"User Request: {user_text}",
        },
    ]
    llm_with_structured_output = llm.with_structured_output(ActionSupervisorDecision)
    response = llm_with_structured_output.invoke(messages)
    
    if isinstance(response, dict):
        goto = response.get("next")
        reasoning = response.get("reasoning", "")
    else:
        goto = getattr(response, "next", None)
        reasoning = getattr(response, "reasoning", "")

    print(f"Sub-Action Decision: {goto} (Reasoning: {reasoning})")

    # --- Guard: Never re-route to action worker if already completed ---
    if goto in action_state["agent_outputs"]:
        print(f"  ↳ Guard: action already completed [STATUS: DONE]. Returning to supervisor.")
        goto = "supervisor"

    if goto == "supervisor" or action_state["agent_outputs"] or routing_history.count(goto) >= 2 or not goto:
        main_state = get_supervisor_state(state, "main_supervisor")
        combined_result = "\n\n".join(action_state["agent_outputs"].values()) if action_state["agent_outputs"] else reasoning
        main_state = record_agent_output(
            main_state,
            "sub_actionsupervisor",
            combined_result,
        )
        return Command(
            goto="supervisor",
            update={"main_supervisor": main_state, "next": "supervisor"},
        )

    action_state = set_routing_plan(action_state, routing_history + [goto])
    return Command(
        goto=goto,
        update={"action_supervisor": action_state, "next": goto},
    )
