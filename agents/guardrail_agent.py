# agents/guardrail_agent.py
import asyncio
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

from langgraph.graph import END
from langgraph.types import Command
from langchain_core.messages import AIMessage
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
    raw_input = state["messages"][-1].content if state.get("messages") else ""
    if not raw_input:
        return Command(goto="supervisor")

    try:
        res = rails.generate(messages=[{"role": "user", "content": raw_input}])
        explain = rails.explain()
        bot_output = res.get("content", "").strip() if isinstance(res, dict) else str(res).strip()

        # Check if NeMo Guardrails triggered an input rail block
        triggered = getattr(explain, "triggered_input_rails", None)
        is_blocked = bool(triggered) if triggered is not None else False

        # Fallback check on colang history if available
        if not is_blocked and hasattr(explain, "colang_history") and explain.colang_history:
            history = str(explain.colang_history).lower()
            if "refuse" in history or "jailbreak" in history:
                is_blocked = True

        if is_blocked:
            return Command(
                update={
                    "messages": [
                        AIMessage(
                            content=bot_output or REFUSAL_FALLBACK,
                            name="guardrail",
                        )
                    ]
                },
                goto=END,
            )
    except Exception:
        # Fallback to supervisor if guardrail fails unexpectedly
        pass

    return Command(goto="supervisor")
