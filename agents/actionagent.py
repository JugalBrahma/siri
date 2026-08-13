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
from tools.terminal import run_terminal_command, modify_file,read_file,write_to_open_word,write_to_word
from langchain.agents import create_agent
from langchain_core.messages import AIMessage

tools = [run_terminal_command,modify_file,read_file,write_to_open_word,write_to_word]

def action_agent(state:State)-> Command[Literal["sub_actionsupervisor"]]:
    print("--- ACTION AGENT ---")
    worker_state = build_worker_state(state, "action_supervisor")
    system_prompt = (
        "You are an automation agent. You have tools to execute terminal commands, modify files, and interact with MS Word. "
        "CRITICAL RULE: If the user asks you to 'open MS word' and write something, you MUST FIRST use 'run_terminal_command' with 'start winword' to visually open the application. "
        "THEN, you MUST use 'write_to_open_word' to type the text directly into the live Word window. "
        "DO NOT use 'write_to_word' (which creates a background .docx file) unless explicitly asked to generate a file."
        + semantic_memory_prompt(worker_state["semantic_memory"])
    )
    action_agent_exec = create_agent(llm, tools, system_prompt=system_prompt)
    result = action_agent_exec.invoke({"messages": state["messages"]})
    final_output = result["messages"][-1].content
    action_state = record_agent_output(
        get_supervisor_state(state, "action_supervisor"),
        "action",
        final_output,
    )
    print(f"Result: {result['messages'][-1].content}")
    return Command(
        update={
            "messages":[
                AIMessage(content=final_output, name="action")
            ],
            "action_supervisor": action_state,
        },
        goto="sub_actionsupervisor",
    )
