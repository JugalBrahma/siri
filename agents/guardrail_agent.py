import re
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.types import Command
# pyrefly: ignore [missing-import]
from nemoguardrails import LLMRails, RailsConfig

from models.models import guardrail_llm
from services.text import message_text
from state.message_state import State

REFUSAL_FALLBACK = (
    "I can't do that. I am only designed to assist with specific tasks and automation."
)

# Pure greetings or thanks with no task attached are refused.
_CHITCHAT_GRATITUDE_PATTERN = re.compile(
    r"^(?:hello|hi|hey|thanks|thank you|good morning|good evening|good afternoon|how are you|howdy|appreciate it)\b[\s!.?]*$",
    re.IGNORECASE,
)

# Unambiguous task queries (e.g. weather) skip the NeMo check.
_SAFE_PATTERN = re.compile(
    r"^(?:what(?:'s| is)|how(?:'s| is)|tell me|show me|get|check|find)?\s*(?:the\s+)?(?:weather|forecast|temperature|time|date)\b",
    re.IGNORECASE,
)

rails_config = RailsConfig.from_path("./config/guardrails")
rails = LLMRails(rails_config, llm=guardrail_llm)  # type: ignore[arg-type]  # NeMo wraps LangChain LLMs internally


def _refuse() -> Command:
    return Command(update={"messages": [AIMessage(content=REFUSAL_FALLBACK, name="guardrail")]}, goto="__end__")


def guardrail_agent(state: State) -> Command[Literal["supervisor", "__end__"]]:
    messages = state.get("messages")
    raw_input = message_text(messages[-1].content).strip() if messages else ""
    if not raw_input:
        return Command(goto="supervisor")

    if _CHITCHAT_GRATITUDE_PATTERN.match(raw_input):
        return _refuse()

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
        activated = getattr(getattr(result, "log", None), "activated_rails", None) or []
        if any(rail.type == "input" and rail.stop for rail in activated):
            return _refuse()
    except Exception:
        pass  # if the guardrail itself fails, let the supervisor handle the request

    return Command(goto="supervisor")
