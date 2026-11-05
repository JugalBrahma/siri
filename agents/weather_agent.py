from langgraph.types import Command
from typing import Literal
from state.message_state import (
    State,
    build_worker_state,
    get_supervisor_state,
    record_agent_output,
    semantic_memory_prompt,
)
from models.models import llm
from services.text import message_text
from tools.weather import tools as openweather_tools
from tools.human_tool import ask_human
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

weather_tools = list(openweather_tools) + [ask_human]

# Build a name → callable lookup for direct tool execution
_TOOL_MAP = {t.name: t for t in weather_tools}

WEATHER_SYSTEM_PROMPT = (
    "You are a weather bot. You MUST ONLY use the provided tools to find weather and interact with the user. "
    "HUMAN-IN-THE-LOOP RULE: If the user did not specify the target location (city/coordinates) or the location "
    "is ambiguous, use the 'ask_human' tool to ask the user for their city or location before querying weather tools.\n"
    "Do not attempt to use any other tools (like brave_search) if a tool fails; "
    "instead, report the error to the user. "
    "If the user asks for BOTH current weather AND forecast, call BOTH get_current_weather and get_weather_forecast."
)


def weather_agent(state: State) -> Command[Literal["sub_infosupervisor"]]:
    print("--- WEATHER AGENT ---")
    worker_state = build_worker_state(state, "info_supervisor")
    system_prompt = WEATHER_SYSTEM_PROMPT + semantic_memory_prompt(worker_state["semantic_memory"])
    user_task = worker_state["task"]

    # Use only the last user message — not the full conversation history
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_task},
    ]

    # Single LLM call with bound tools
    bound_llm = llm.bind_tools(weather_tools)
    response = bound_llm.invoke(messages)

    # If the LLM returned tool calls, execute them directly and make one
    # final LLM call to compose the answer from the tool results.
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_messages: list[BaseMessage] = [response]  # include the AI message with tool_calls
        for tc in response.tool_calls:
            tool_fn = _TOOL_MAP.get(tc["name"])
            if tool_fn is None:
                tool_result = f"Error: unknown tool '{tc['name']}'"
            else:
                try:
                    tool_result = tool_fn.invoke(tc["args"])
                except Exception as exc:
                    tool_result = f"Tool error: {exc}"
            tool_messages.append(
                ToolMessage(content=str(tool_result), tool_call_id=tc["id"])
            )

        # Compose answer with minimal context: just the question + tool results
        compose_messages = [
            {"role": "system", "content": "Summarize the weather data below into a clear, friendly response for the user."},
            {"role": "user", "content": user_task},
        ] + tool_messages
        final_response = llm.invoke(compose_messages)
        final_output = final_response.content
    else:
        # No tool calls — LLM answered directly (e.g., asked a clarification)
        final_output = response.content

    output_text = message_text(final_output, sep="")

    info_state = record_agent_output(
        get_supervisor_state(state, "info_supervisor"),
        "weather",
        output_text,
    )
    return Command(
        update={
            "messages": [AIMessage(content=output_text, name="weather")],
            "info_supervisor": info_state,
        },
        goto="sub_infosupervisor",
    )

