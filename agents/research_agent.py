from langgraph.graph import MessagesState,StateGraph,START, END
from langgraph.types import Command
from typing import Literal
from state.message_state import State
from models.models import llm
from tools.searchtools import search_tool
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage


def research_agent(state:State)-> Command[Literal["sub_infosupervisor"]]:
    print("--- RESEARCH AGENT ---")
    research_agent_exec = create_agent(llm, [search_tool], system_prompt="You are researcher use tools to find")
    result = research_agent_exec.invoke(state)
    print(f"Result: {result['messages'][-1].content}")
    return Command(
        update={
            "messages":[
                HumanMessage(content=result["messages"][-1].content, name="researcher")
            ]
        },
        goto="sub_infosupervisor",
    )

