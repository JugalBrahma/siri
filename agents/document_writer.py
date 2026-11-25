from typing import Literal

from langchain.agents.middleware import ModelCallLimitMiddleware
from langgraph.types import Command

from prompts import load_prompt
from state.message_state import State
from tools.human_tool import ask_human
from tools.terminal import modify_file, read_file
from tools.word_tools import get_word_tools

from .worker import gathered_info_suffix, run_worker

DOCUMENT_WRITER_PROMPT = load_prompt("document_writer")
# Most model calls in one run, so it cannot loop on file or Word tools.
WRITER_MODEL_CALLS = 10


def get_document_tools():
    """Text-file tools, Word tools and ask_human. No screen, no apps."""
    return [read_file, modify_file, ask_human] + list(get_word_tools())


def document_writer_agent(state: State) -> Command[Literal["sub_actionsupervisor"]]:
    print("--- DOCUMENT WRITER ---")
    return run_worker(
        state,
        name="document_writer",
        scope="action_supervisor",
        goto="sub_actionsupervisor",
        tools=get_document_tools(),
        base_prompt=DOCUMENT_WRITER_PROMPT,
        task_suffix=gathered_info_suffix(state),
        middleware=[ModelCallLimitMiddleware(run_limit=WRITER_MODEL_CALLS)],
    )
