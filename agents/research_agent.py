from langgraph.types import Command
from typing import Literal
from prompts import load_prompt
from state.message_state import State
from tools.searchtools import search_tool
from tools.human_tool import ask_human
from .search_budget import AdaptiveSearchLimit
from .worker import run_worker

tools = [search_tool, ask_human]

# Hard cap on web searches per research task: 2 normally, 3 if the first
# search failed (error / no results). The prompt asks for the same; this
# enforces it if the model ignores the prompt. Searches over the limit are
# refused and the agent answers with what it already has.
MAX_SEARCHES = 2
MAX_SEARCHES_AFTER_FAILURE = 3
search_limit = AdaptiveSearchLimit(
    normal_limit=MAX_SEARCHES,
    retry_limit=MAX_SEARCHES_AFTER_FAILURE,
)

RESEARCH_SYSTEM_PROMPT = load_prompt(
    "researcher",
    max_searches=MAX_SEARCHES,
    max_searches_after_failure=MAX_SEARCHES_AFTER_FAILURE,
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
        middleware=[search_limit],
    )
