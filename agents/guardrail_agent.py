# agents/guardrail_agent.py
import re

from langgraph.types import Command
from langchain_core.messages import AIMessage
from typing import Literal
from state.message_state import State
# pyrefly: ignore [missing-import]
from nemoguardrails import RailsConfig, LLMRails
from models.models import guardrail_llm
from services.text import message_text

REFUSAL_FALLBACK = (
    "I can't do that. I am only designed to assist with specific tasks and automation."
)

# Strict rule: Pure greetings, gratitude, or pleasantries with no task attached are blocked
_CHITCHAT_GRATITUDE_PATTERN = re.compile(
    r"^(?:hello|hi|hey|thanks|thank you|good morning|good evening|good afternoon|how are you|howdy|appreciate it)\b[\s!.?]*$",
    re.IGNORECASE,
)

# Fast-path safe pattern ONLY for unambiguous task queries (e.g. weather)
_SAFE_PATTERN = re.compile(
    r"^(?:what(?:'s| is)|how(?:'s| is)|tell me|show me|get|check|find)?\s*(?:the\s+)?(?:weather|forecast|temperature|time|date)\b",
    re.IGNORECASE,
)

rails_config = RailsConfig.from_path("./config/guardrails")
rails = LLMRails(rails_config, llm=guardrail_llm)  # type: ignore[arg-type]  # NeMo wraps LangChain LLMs internally


def guardrail_agent(state: State) -> Command[Literal["supervisor", "__end__"]]:
    last_msg = state["messages"][-1] if state.get("messages") else None
    if not last_msg:
        return Command(goto="supervisor")

    raw_input = message_text(last_msg.content)

    raw_input = raw_input.strip()
    if not raw_input:
        return Command(goto="supervisor")

    # Strict refusal for pure greetings, gratitude, or casual pleasantries
    if _CHITCHAT_GRATITUDE_PATTERN.match(raw_input):
        return Command(
            update={
                "messages": [
                    AIMessage(
                        content=REFUSAL_FALLBACK,
                        name="guardrail",
                    )
                ]
            },
            goto="__end__",
        )

    # Fast-path: skip NeMo for obviously safe, short task queries
    if _SAFE_PATTERN.search(raw_input) and len(raw_input) < 200:
        return Command(goto="supervisor")

    try:
        # Run only the input rails (the self-check YES/NO call). A full
        # rails.generate() also runs intent detection and writes a complete
        # bot answer that we discard, costing ~20s and 3 extra LLM calls.
        result = rails.generate(
            messages=[{"role": "user", "content": raw_input}],
            options={"rails": ["input"], "log": {"activated_rails": True}},
        )

        # An input rail that blocks the message sets stop=True.
        activated = getattr(getattr(result, "log", None), "activated_rails", None) or []
        is_blocked = any(
            rail.type == "input" and rail.stop for rail in activated
        )

        if is_blocked:
            return Command(
                update={
                    "messages": [
                        AIMessage(
                            content=REFUSAL_FALLBACK,
                            name="guardrail",
                        )
                    ]
                },
                goto="__end__",
            )
    except Exception:
        # Fallback to supervisor if guardrail fails unexpectedly
        pass

    return Command(goto="supervisor")
