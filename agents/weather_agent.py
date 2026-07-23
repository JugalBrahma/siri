from langgraph.types import Command
from typing import Literal
from state.message_state import State
from models.models import llm
from tools.weather import tools
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage


def weather_agent(state:State)-> Command[Literal["sub_infosupervisor"]]:
    print("--- WEATHER AGENT ---")
    # Bind tools to the LLM explicitly
    llm_with_tools = llm.bind_tools(tools)
    system_prompt = "You are a weather bot. You MUST ONLY use the provided tools to find weather. Do not attempt to use any other tools (like brave_search) if a tool fails; instead, report the error to the user."
    weather_agent_exec = create_agent(llm_with_tools, tools, system_prompt=system_prompt)
    result = weather_agent_exec.invoke(state)
    #print(f"Result: {result['messages'][-1].content}")
    return Command(
        update={
            "messages":[
                HumanMessage(content=result["messages"][-1].content, name="weather")
            ]
        },
        goto="sub_infosupervisor",
    )