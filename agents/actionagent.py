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
from tools.terminal import run_terminal_command, modify_file, read_file
from tools.word_tools import get_word_tools, word_tools
from tools.human_tool import ask_human
from langchain.agents import create_agent
from langchain_core.messages import AIMessage


def get_action_tools():
    """Retrieve full list of action tools including dynamic Word MCP tools."""
    mcp_word_tools = get_word_tools()
    base_tools = [run_terminal_command, modify_file, read_file, ask_human]
    return base_tools + list(mcp_word_tools)


tools = [run_terminal_command, modify_file, read_file, ask_human] + list(word_tools)

def action_agent(state:State)-> Command[Literal["sub_actionsupervisor"]]:
    print("--- ACTION AGENT ---")
    worker_state = build_worker_state(state, "action_supervisor")
    system_prompt = (
        "You are an automation agent. You have tools to execute terminal commands, modify files, manipulate MS Word documents via Office Word MCP tools, and ask the human user for clarification.\n"
        "MS WORD RULES:\n"
        "1. Use the dedicated Word MCP tools (create_document, add_heading, add_paragraph, add_table) to create, inspect, and modify Word (.docx) documents.\n"
        "2. AUTOMATIC LAUNCH: After completing the creation or editing of any Word (.docx) document, ALWAYS call the 'open_word_document' tool with the file path so Microsoft Word automatically launches and displays the document to the user.\n"
        "HUMAN-IN-THE-LOOP RULE: If critical information (e.g. file path, specific command parameters, or confirmation for destructive actions) is missing or ambiguous, use the 'ask_human' tool to ask the user before executing."
        + semantic_memory_prompt(worker_state["semantic_memory"])
    )
    active_tools = get_action_tools()
    action_agent_exec = create_agent(llm, active_tools, system_prompt=system_prompt)

    result = action_agent_exec.invoke(
        {"messages": [{"role": "user", "content": worker_state["task"]}]}
    )
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
