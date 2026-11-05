from langgraph.types import Command
from typing import Literal
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

RESEARCH_SYSTEM_PROMPT = (
    "You are a web researcher on a multi-agent automation team. Your sole responsibility is to search the web and synthesize facts, articles, event info, and summaries.\n"
    "RULES:\n"
    "1. Focus ONLY on web research and information gathering using the search tool.\n"
    "2. Teammates handle weather and document creation. NEVER apologize or state that you lack weather or document tools—simply return the research findings clearly and concisely.\n"
    "3. If the search query is ambiguous or lacks essential criteria, use 'ask_human' to clarify.\n"
    f"4. SEARCH BUDGET: You may search at most {MAX_SEARCHES} times per task ({MAX_SEARCHES_AFTER_FAILURE} if your first search "
    "returns an error or no results). Start with ONE well-phrased search; "
    "most questions are answered by it. Search again only if the results clearly do not answer the question. "
    "Never repeat or rephrase a search you already ran, and do not search again just to double-check an answer you already have. "
    "As soon as you have the answer, stop searching and write it up."
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
