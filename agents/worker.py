from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langgraph.types import Command
from state.message_state import (
    State,
    build_worker_state,
    get_supervisor_state,
    record_agent_output,
    semantic_memory_prompt,
)
from models.models import llm


def run_worker(
    state: State,
    *,
    name: str,
    scope: str,
    goto: str,
    tools: list,
    base_prompt: str,
    task_suffix: str = "",
) -> Command:
    """Run a tool-using worker agent on the current task and report back.

    `scope` is the supervisor state the worker belongs to ("info_supervisor"
    or "action_supervisor"); its output is recorded there under `name`, and
    control returns to `goto`.
    """
    worker_state = build_worker_state(state, scope)
    system_prompt = base_prompt + semantic_memory_prompt(worker_state["semantic_memory"])
    agent_exec = create_agent(llm, tools, system_prompt=system_prompt)

    # Use only the latest user task, not the full conversation history
    result = agent_exec.invoke(
        {"messages": [{"role": "user", "content": worker_state["task"] + task_suffix}]}
    )
    final_output = result["messages"][-1].content
    scope_state = record_agent_output(get_supervisor_state(state, scope), name, final_output)
    print(f"Result: {final_output}")
    return Command(
        update={
            "messages": [AIMessage(content=final_output, name=name)],
            scope: scope_state,
        },
        goto=goto,
    )
