from typing import Any, Mapping
from langchain_core.tools import tool
from langgraph.types import interrupt


@tool
def ask_human(question: str) -> str:
    """Use this tool to ask the human user a question when you need clarification,
    missing details (like location, folder path, or file name), or explicit user confirmation.
    
    Args:
        question: The clear, concise question or confirmation prompt to display to the user.
        
    Returns:
        The user's direct response or answer as a string.
    """
    print(f"[ask_human Tool]: Requesting user input: '{question}'")

    # interrupt() pauses LangGraph execution and surfaces the payload to the caller/UI
    response = interrupt({
        "type": "ask_human",
        "question": question,
    })

    if isinstance(response, Mapping):
        return str(response.get("answer") or response.get("response") or response.get("input") or response)
    
    return str(response)
