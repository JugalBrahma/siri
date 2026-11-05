"""Small text helpers shared by agents."""

from typing import Any


def message_text(content: Any, sep: str = " ") -> str:
    """Flatten message content (a string or a list of parts) into one string.

    Parts that are plain strings are kept, dict parts contribute their "text"
    field (non-text blocks such as tool calls become empty), and anything
    else is converted with str().
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return sep.join(
            part if isinstance(part, str)
            else str(part.get("text", "")) if isinstance(part, dict)
            else str(part)
            for part in content
        )
    return str(content)
