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
from tools.human_tool import ask_human
from langchain.agents import create_agent
from langchain_core.messages import AIMessage

tools = [search_tool, ask_human]


def research_agent(state:State)-> Command[Literal["sub_infosupervisor"]]:
    print("--- RESEARCH AGENT ---")
    worker_state = build_worker_state(state, "info_supervisor")
    system_prompt = (
        "You are a web researcher on a multi-agent automation team. Your sole responsibility is to search the web and synthesize facts, articles, event info, and summaries.\n"
        "RULES:\n"
        "1. Focus ONLY on web research and information gathering using the search tool.\n"
        "2. Teammates handle weather and document creation. NEVER apologize or state that you lack weather or document tools—simply return the research findings clearly and concisely.\n"
        "3. If the search query is ambiguous or lacks essential criteria, use 'ask_human' to clarify."
        + semantic_memory_prompt(worker_state["semantic_memory"])
    )
    research_agent_exec = create_agent(
        llm,
        tools,
        system_prompt=system_prompt,
    )
    # Use only the last user message — not the full conversation history
    result = research_agent_exec.invoke(
        {"messages": [{"role": "user", "content": worker_state["task"]}]}
    )
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

