# agents/output_sanitizer.py
import re
from langgraph.graph import END
from langgraph.types import Command
from typing import Literal
from state.message_state import State

# Regex patterns for common sensitive keys, tokens, and credentials
SECRET_PATTERNS = [
    r"sk-[a-zA-Z0-9]{32,}",                # OpenAI API Key
    r"AKIA[0-9A-Z]{16}",                    # AWS Access Key ID
    r"ghp_[a-zA-Z0-9]{36}",                 # GitHub Personal Access Token
    r"AIza[0-9A-Za-z-_]{35}",               # Google API Key
    r"bearer\s+[a-zA-Z0-9\-\._~\+\/]+=*",   # Bearer Token
    r"xox[bpar]-[a-zA-Z0-9]{10,}",          # Slack Token
]

def redact_secrets(text: str) -> str:
    """Scan text for sensitive keys/secrets and replace them with [REDACTED_SECRET]."""
    sanitized = text
    for pattern in SECRET_PATTERNS:
        sanitized = re.sub(pattern, "[REDACTED_SECRET]", sanitized, flags=re.IGNORECASE)
    return sanitized

def output_sanitizer_agent(state: State) -> Command[Literal[END]]:
    """Node that sanitizes the final LLM response before reaching END."""
    messages = state.get("messages", [])
    if not messages:
        return Command(goto=END)

    last_message = messages[-1]
    if hasattr(last_message, "content") and isinstance(last_message.content, str):
        sanitized_content = redact_secrets(last_message.content)
        if sanitized_content != last_message.content:
            last_message.content = sanitized_content

    return Command(goto=END)
