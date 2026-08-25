# agents/guardrail_agent.py
import asyncio
import re

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
from models.models import guardrail_llm

REFUSAL_FALLBACK = (
    "I'm sorry, but I can't process that request. "
    "It appears to violate my safety guidelines."
)

# Patterns that are overwhelmingly safe — skip the full NeMo pipeline.
# Kept conservative: only well-known benign intents.
_SAFE_PATTERN = re.compile(
    r"(?:what(?:'s| is)|how(?:'s| is)|tell me|show me|get|check|find|give me)?\s*"
    r"(?:the\s+)?(?:weather|forecast|temperature|time|date|hello|hi|hey|"
    r"thanks|thank you|good morning|good evening)",
    re.IGNORECASE,
)

rails_config = RailsConfig.from_path("./config/guardrails")
rails = LLMRails(rails_config, llm=guardrail_llm)  # type: ignore[arg-type]  # NeMo wraps LangChain LLMs internally


def guardrail_agent(state: State) -> Command[Literal["supervisor", "__end__"]]:
    last_msg = state["messages"][-1] if state.get("messages") else None
    if not last_msg:
        return Command(goto="supervisor")

    raw_content = last_msg.content
    if isinstance(raw_content, str):
        raw_input = raw_content
    elif isinstance(raw_content, list):
        raw_input = " ".join(
            item if isinstance(item, str) else str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in raw_content
        )
    else:
        raw_input = str(raw_content)

    raw_input = raw_input.strip()
    if not raw_input:
        return Command(goto="supervisor")

    # Fast-path: skip NeMo for obviously safe, short inputs
    if _SAFE_PATTERN.search(raw_input) and len(raw_input) < 200:
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
