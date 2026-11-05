from langgraph.types import Command
from typing import Literal
from state.message_state import State, get_supervisor_state
from tools.terminal import open_app, modify_file, read_file
from tools.word_tools import get_word_tools, word_tools
from tools.human_tool import ask_human
from .worker import run_worker


def get_action_tools():
    """Retrieve full list of action tools including dynamic Word MCP tools."""
    mcp_word_tools = get_word_tools()
    base_tools = [open_app, modify_file, read_file, ask_human]
    return base_tools + list(mcp_word_tools)


tools = [open_app, modify_file, read_file, ask_human] + list(word_tools)

ACTION_SYSTEM_PROMPT = (
    "You are an automation agent. You have tools to open approved apps (open_app), read/modify files, manipulate MS Word documents, and ask the human user for clarification.\n"
    "MS WORD RULES:\n"
    "1. Prefer using 'create_complete_word_document' to write articles, reports, or research summaries into Word in a SINGLE step (it formats headings, bullets, paragraphs, and automatically launches Word).\n"
    "2. If fine-grained editing of an existing document is needed, use create_document, add_heading, add_paragraph, add_table.\n"
    "3. AUTOMATIC LAUNCH: If not using create_complete_word_document, ALWAYS call 'open_word_document' with the file path after editing.\n"
    "HUMAN-IN-THE-LOOP RULE: If critical information (e.g. file path, specific command parameters, or confirmation for destructive actions) is missing or ambiguous, use the 'ask_human' tool to ask the user before executing."
)


def action_agent(state:State)-> Command[Literal["sub_actionsupervisor"]]:
    print("--- ACTION AGENT ---")
    main_state = get_supervisor_state(state, "main_supervisor")
    gathered_info = main_state.get("agent_outputs", {}).get("sub_infosupervisor", "")
    task_suffix = ""
    if gathered_info:
        task_suffix = f"\n\nGathered Information / Research Content from previous step:\n{gathered_info}"

    return run_worker(
        state,
        name="action",
        scope="action_supervisor",
        goto="sub_actionsupervisor",
        tools=get_action_tools(),
        base_prompt=ACTION_SYSTEM_PROMPT,
        task_suffix=task_suffix,
    )
