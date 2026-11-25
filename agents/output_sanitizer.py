import re
from typing import Literal

from langgraph.graph import END
from langgraph.types import Command

from state.message_state import State

SECRET_PATTERNS = [
    r"sk-[a-zA-Z0-9]{32,}",                # OpenAI API Key
    r"AKIA[0-9A-Z]{16}",                    # AWS Access Key ID
    r"ghp_[a-zA-Z0-9]{36}",                 # GitHub Personal Access Token
    r"AIza[0-9A-Za-z-_]{35}",               # Google API Key
    r"bearer\s+[a-zA-Z0-9\-\._~\+\/]+=*",   # Bearer Token
    r"xox[bpar]-[a-zA-Z0-9]{10,}",          # Slack Token
    r"tvly-[a-zA-Z0-9_\-]{10,}",            # Tavily API Key
    r"tavilyApiKey=[^&\s\"']+",             # Tavily key inside an MCP URL
]

def redact_secrets(text: str) -> str:
    """Scan text for sensitive keys/secrets and replace them with [REDACTED_SECRET]."""
    for pattern in SECRET_PATTERNS:
        text = re.sub(pattern, "[REDACTED_SECRET]", text, flags=re.IGNORECASE)
    return text


def output_sanitizer_agent(state: State) -> Command[Literal[END]]:
    """Node that sanitizes the final LLM response before reaching END."""
    messages = state.get("messages", [])
    if messages and isinstance(messages[-1].content, str):
        messages[-1].content = redact_secrets(messages[-1].content)
    return Command(goto=END)
