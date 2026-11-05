"""Human-in-the-loop for supervisors.

A supervisor that needs more detail routes here with a question in
``state["clarification"]``. This node pauses the run, and when the user
answers, adds the answer to their request and returns to that supervisor.

The pause lives in its own node, with no LLM call before ``interrupt()``,
because a resumed node re-runs from its start: a supervisor pausing itself
would repeat its routing call and might not reach the question again.
"""

from langchain_core.messages import HumanMessage
from langgraph.types import Command, interrupt

from state.message_state import State, latest_user_task
from tools.human_tool import human_answer_text

DEFAULT_QUESTION = "Could you give me a few more details?"


def ask_user_node(state: State) -> Command:
    request = state.get("clarification") or {}
    question = request.get("question") or DEFAULT_QUESTION
    print(f"[ask_user] Supervisor asks: '{question}'")

    answer = human_answer_text(interrupt({"type": "ask_human", "question": question}))

    original = latest_user_task(state.get("messages", []))
    clarified = f'{original}\n(When asked "{question}", the user answered: {answer})'
    return Command(
        goto=request.get("return_to", "supervisor"),
        update={"messages": [HumanMessage(content=clarified)], "clarification": None},
    )
