from langgraph.types import Command
from typing import Literal
from state.message_state import State
from tools.searchtools import search_tool
from tools.human_tool import ask_human
from .worker import run_worker

tools = [search_tool, ask_human]

RESEARCH_SYSTEM_PROMPT = (
    "You are a web researcher on a multi-agent automation team. Your sole responsibility is to search the web and synthesize facts, articles, event info, and summaries.\n"
    "RULES:\n"
    "1. Focus ONLY on web research and information gathering using the search tool.\n"
    "2. Teammates handle weather and document creation. NEVER apologize or state that you lack weather or document tools—simply return the research findings clearly and concisely.\n"
    "3. If the search query is ambiguous or lacks essential criteria, use 'ask_human' to clarify."
)


def research_agent(state:State)-> Command[Literal["sub_infosupervisor"]]:
    print("--- RESEARCH AGENT ---")
    return run_worker(
        state,
        name="researcher",
        scope="info_supervisor",
        goto="sub_infosupervisor",
        tools=tools,
        base_prompt=RESEARCH_SYSTEM_PROMPT,
    )
