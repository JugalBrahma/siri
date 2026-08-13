from langgraph.graph import MessagesState,StateGraph,START, END
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
from tools.searchtools import search_tool
from langchain.agents import create_agent
from langchain_core.messages import AIMessage


def research_agent(state:State)-> Command[Literal["sub_infosupervisor"]]:
    print("--- RESEARCH AGENT ---")
    worker_state = build_worker_state(state, "info_supervisor")
    research_agent_exec = create_agent(
        llm,
        [search_tool],
        system_prompt="You are researcher use tools to find" + semantic_memory_prompt(
            worker_state["semantic_memory"],
        ),
    )
    result = research_agent_exec.invoke({"messages": state["messages"]})
    final_output = result["messages"][-1].content
    info_state = record_agent_output(
        get_supervisor_state(state, "info_supervisor"),
        "researcher",
        final_output,
    )
    print(f"Result: {result['messages'][-1].content}")
    return Command(
        update={
            "messages":[
                AIMessage(content=final_output, name="researcher")
            ],
            "info_supervisor": info_state,
        },
        goto="sub_infosupervisor",
    )

