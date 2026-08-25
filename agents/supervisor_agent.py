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


# ---------------------------------------------------------------------------
# Main Supervisor
# ---------------------------------------------------------------------------

def superVisorAgent(state: State) -> Command[Literal['sub_infosupervisor', 'sub_actionsupervisor', 'output_sanitizer']]:
    print("--- SUPERVISOR AGENT ---")
    main_state = get_supervisor_state(state, "main_supervisor")
    routing_history = main_state.get("routing_plan", [])

    # --- Fast-path: skip the LLM call on the first routing decision ---
    goto = None
    reasoning = ""
    if not routing_history:
        user_text = _extract_user_text(state)
        fast = _fast_path_main(user_text)
        if fast:
            goto = fast
            reasoning = f"[fast-path] keyword match → {fast}"
            print(f"Main Supervisor Decision: {goto} (Reasoning: {reasoning})")

    # --- Fall through to LLM routing if fast-path didn't match ---
    if goto is None:
        latest_msg = [state["messages"][-1]] if state.get("messages") else []
        messages = [{
            "role": "system",
            "content": system_prompt + semantic_memory_prompt(main_state["semantic_memory"]),
        }] + latest_msg
        llm_with_structured_output = llm.with_structured_output(MainSupervisorDecision)
        response = llm_with_structured_output.invoke(messages)

        if isinstance(response, dict):
            goto = response.get("next")
            reasoning = response.get("reasoning", "")
        else:
            goto = getattr(response, "next", None)
            reasoning = getattr(response, "reasoning", "")

        print(f"Main Supervisor Decision: {goto} (Reasoning: {reasoning})")

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
# Sub-Info Supervisor (with fast-path + return-path shortcut)
# ---------------------------------------------------------------------------

def sub_infoSuperVisorAgent(state: State) -> Command[Literal['researcher', 'weather', 'supervisor', 'output_sanitizer']]:
    print("--- SUB INFO SUPERVISOR AGENT ---")
    info_state = get_supervisor_state(state, "info_supervisor")
    routing_history = info_state.get("routing_plan", [])

    # --- Fast-path: skip the LLM call on the first routing decision ---
    goto = None
    reasoning = ""
    if not routing_history:
        user_text = _extract_user_text(state)
        fast = _fast_path_info(user_text)
        if fast:
            goto = fast
            reasoning = f"[fast-path] keyword match → {fast}"
            print(f"Sub-Info Decision: {goto} (Reasoning: {reasoning})")

    # --- Fall through to LLM routing if fast-path didn't match ---
    if goto is None:
        latest_msg = [state["messages"][-1]] if state.get("messages") else []
        messages = [{
            "role": "system",
            "content": sub_info_supervisor_prompt + semantic_memory_prompt(info_state["semantic_memory"]),
        }] + latest_msg
        llm_with_structured_output = llm.with_structured_output(InfoSupervisorDecision)
        response = llm_with_structured_output.invoke(messages)

        if isinstance(response, dict):
            goto = response.get("next")
            reasoning = response.get("reasoning", "")
        else:
            goto = getattr(response, "next", None)
            reasoning = getattr(response, "reasoning", "")

        print(f"Sub-Info Decision: {goto} (Reasoning: {reasoning})")

    # --- Route back when all info tasks are gathered or loop detected ---
    if goto == "supervisor" or routing_history.count(goto) >= 2 or not goto:
        main_state = get_supervisor_state(state, "main_supervisor")
        combined_result = "\n\n".join(info_state["agent_outputs"].values()) if info_state["agent_outputs"] else reasoning
        main_state = record_agent_output(
            main_state,
            "sub_infosupervisor",
            combined_result,
        )

        # --- Return-path shortcut ---
        # If the main supervisor only dispatched to us (single sub-supervisor),
        # skip the round-trip back through supervisor and go straight to
        # output_sanitizer. This saves one LLM call.
        main_routing = main_state.get("routing_plan", [])
        dispatched_subs = [r for r in main_routing if r in ("sub_infosupervisor", "sub_actionsupervisor")]
        if len(dispatched_subs) <= 1 and "sub_actionsupervisor" not in main_routing:
            print("  ↳ return-path shortcut: sub_infosupervisor → output_sanitizer (skipping supervisor)")
            return Command(
                goto="output_sanitizer",
                update={
                    "messages": [AIMessage(content=combined_result, name="supervisor")],
                    "main_supervisor": set_routing_plan(main_state, main_routing + ["output_sanitizer"]),
                    "next": "output_sanitizer",
                },
            )

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
# Sub-Action Supervisor (unchanged)
# ---------------------------------------------------------------------------

def sub_actionSuperVisorAgent(state: State) -> Command[Literal['action', 'supervisor']]:
    print("--- SUB ACTION SUPERVISOR AGENT ---")
    action_state = get_supervisor_state(state, "action_supervisor")

    latest_msg = [state["messages"][-1]] if state.get("messages") else []
    messages = [{
        "role": "system",
        "content": sub_action_supervisor_prompt + semantic_memory_prompt(action_state["semantic_memory"]),
    }] + latest_msg
    llm_with_structured_output = llm.with_structured_output(ActionSupervisorDecision)
    response = llm_with_structured_output.invoke(messages)
    
    if isinstance(response, dict):
        goto = response.get("next")
        reasoning = response.get("reasoning", "")
    else:
        goto = getattr(response, "next", None)
        reasoning = getattr(response, "reasoning", "")

    print(f"Sub-Action Decision: {goto} (Reasoning: {reasoning})")

    routing_history = action_state.get("routing_plan", [])

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
