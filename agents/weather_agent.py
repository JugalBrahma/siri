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
from tools.weather import tools as openweather_tools
from tools.human_tool import ask_human
from langchain.agents import create_agent
from langchain_core.messages import AIMessage

weather_tools = list(openweather_tools) + [ask_human]


def weather_agent(state:State)-> Command[Literal["sub_infosupervisor"]]:
    print("--- WEATHER AGENT ---")
    worker_state = build_worker_state(state, "info_supervisor")
    # Bind tools to the LLM explicitly
    llm_with_tools = llm.bind_tools(weather_tools)
    system_prompt = (
        "You are a weather bot. You MUST ONLY use the provided tools to find weather and interact with the user. "
        "HUMAN-IN-THE-LOOP RULE: If the user did not specify the target location (city/coordinates) or the location is ambiguous, use the 'ask_human' tool to ask the user for their city or location before querying weather tools.\n"
        "Do not attempt to use any other tools (like brave_search) if a tool fails; "
        "instead, report the error to the user."
        + semantic_memory_prompt(worker_state["semantic_memory"])
    )
    weather_agent_exec = create_agent(llm_with_tools, weather_tools, system_prompt=system_prompt)
    result = weather_agent_exec.invoke({"messages": state["messages"]})
    final_output = result["messages"][-1].content
    info_state = record_agent_output(
        get_supervisor_state(state, "info_supervisor"),
        "weather",
        final_output,
    )
    #print(f"Result: {result['messages'][-1].content}")
    return Command(
        update={
            "messages":[
                AIMessage(content=final_output, name="weather")
            ],
            "info_supervisor": info_state,
        },
        goto="sub_infosupervisor",
    )
