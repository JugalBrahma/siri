# agents/guardrail_agent.py
from langgraph.graph import END
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from typing import Literal
from state.message_state import State
# pyrefly: ignore [missing-import]
from nemoguardrails import RailsConfig, LLMRails
from models.models import llm

REFUSAL_FALLBACK = (
    "I'm sorry, but I can't process that request. "
    "It appears to violate my safety guidelines."
)

rails_config = RailsConfig.from_path("./config/guardrails")
rails = LLMRails(rails_config, llm=llm)


def guardrail_agent(state: State) -> Command[Literal["supervisor", END]]:
    raw_input = state["messages"][-1].content
    res = rails.generate(messages=[{"role": "user", "content": raw_input}])
    bot_output = res.get("content", "").strip() if isinstance(res, dict) else str(res).strip()

    is_blocked = (
        isinstance(res, dict)
        and res.get("role") == "assistant"
        and (bot_output == "" or bot_output != raw_input.strip())
    )

    if is_blocked:
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=bot_output or REFUSAL_FALLBACK,
                        name="guardrail",
                    )
                ]
            },
            goto=END,
        )

    return Command(goto="supervisor")
