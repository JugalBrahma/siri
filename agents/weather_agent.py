from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.types import Command

from models.models import llm
from prompts import load_prompt
from services.text import message_text
from state.message_state import (
    State,
    build_worker_state,
    get_supervisor_state,
    record_agent_output,
    semantic_memory_prompt,
)
from tools.human_tool import ask_human
from tools.weather import tools as openweather_tools

WEATHER_SYSTEM_PROMPT = load_prompt("weather")

weather_tools = list(openweather_tools) + [ask_human]
_TOOL_MAP = {t.name: t for t in weather_tools}  # tool name -> tool, for direct execution


def weather_agent(state: State) -> Command[Literal["sub_infosupervisor"]]:
    print("--- WEATHER AGENT ---")
    worker_state = build_worker_state(state, "info_supervisor")
    system_prompt = WEATHER_SYSTEM_PROMPT + semantic_memory_prompt(worker_state["semantic_memory"])
    user_task = worker_state["task"]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_task},
    ]

    bound_llm = llm.bind_tools(weather_tools)
    response = bound_llm.invoke(messages)

    # If the LLM returned tool calls, execute them directly and make one
    # final LLM call to compose the answer from the tool results.
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_messages: list[BaseMessage] = [response]
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

