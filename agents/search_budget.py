"""Caps web searches per research task: 2, or 3 if the first search failed.

The limit is worked out from the message history each turn, so the middleware
keeps no state. Searches over the limit are refused with an error message.
"""

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage

SEARCH_TOOL = "tavily_search"
BLOCKED_MARKER = "Search limit reached"


def search_failed(content: Any) -> bool:
    """True if a search result is an error or contains no results."""
    text = str(content or "").strip()
    lowered = text.lower()
    return (
        not text
        or lowered.startswith("error")
        or "no results" in lowered
        or "'results': []" in lowered
        or '"results": []' in lowered
    )


class AdaptiveSearchLimit(AgentMiddleware):
    """Allow 2 searches normally, 3 if the first search failed."""

    def __init__(self, normal_limit: int = 2, retry_limit: int = 3, tool_name: str = SEARCH_TOOL):
        super().__init__()
        self.normal_limit = normal_limit
        self.retry_limit = retry_limit
        self.tool_name = tool_name

    def limit_and_used(self, messages: list) -> tuple[int, int]:
        """(current limit, searches already run) based on earlier messages."""
        results = [
            m for m in messages
            if isinstance(m, ToolMessage)
            and m.name == self.tool_name
            and not str(m.content).startswith(BLOCKED_MARKER)
        ]
        first_failed = bool(results) and search_failed(results[0].content)
        limit = self.retry_limit if first_failed else self.normal_limit
        return limit, len(results)

    def after_model(self, state, runtime) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        if not messages or not isinstance(messages[-1], AIMessage):
            return None
        requested = [tc for tc in messages[-1].tool_calls if tc.get("name") == self.tool_name]
        if not requested:
            return None

        limit, used = self.limit_and_used(messages[:-1])
        blocked = requested[max(0, limit - used):]
        if not blocked:
            return None
        return {
            "messages": [
                ToolMessage(
                    content=(
                        f"{BLOCKED_MARKER} ({limit} searches for this task). Do not search "
                        "again; answer using the results you already have."
                    ),
                    tool_call_id=tc["id"],
                    name=self.tool_name,
                    status="error",
                )
                for tc in blocked
            ]
        }

    async def aafter_model(self, state, runtime) -> dict[str, Any] | None:
        return self.after_model(state, runtime)
