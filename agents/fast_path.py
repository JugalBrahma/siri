"""Keyword fast paths: route a request without an LLM call when it clearly goes one way."""

import re

INFO_KEYWORDS = re.compile(
    r"\b(weather|forecast|temperature|humidity|rain|snow|wind|climate|"
    r"search|find|look\s?up|research|who\s+is|what\s+is|when\s+is|where\s+is|"
    r"tell\s+me|explain|summarize|summary)\b",
    re.IGNORECASE,
)
ACTION_KEYWORDS = re.compile(
    r"\b(create|write|make|generate|build|edit|modify|delete|remove|run|execute|"
    r"save|open|file|document|docx|code|script|terminal|command|install|"
    r"send|type|click|press|whatsapp)\b",
    re.IGNORECASE,
)
WEATHER_KEYWORDS = re.compile(
    r"\b(weather|forecast|temperature|humidity|rain|snow|wind|climate|"
    r"celsius|fahrenheit|degrees|storm|sunny|cloudy)\b",
    re.IGNORECASE,
)
RESEARCH_KEYWORDS = re.compile(
    r"\b(search|find|look\s?up|research|article|news|event|tutorial|"
    r"who\s+is|what\s+is|when\s+is|where\s+is|tell\s+me\s+about|explain|summarize)\b",
    re.IGNORECASE,
)


MAIN_FAST_PATH = ((INFO_KEYWORDS, "sub_infosupervisor"), (ACTION_KEYWORDS, "sub_actionsupervisor"))
INFO_FAST_PATH = ((WEATHER_KEYWORDS, "weather"), (RESEARCH_KEYWORDS, "researcher"))


def pick_by_keywords(user_text: str, first: tuple, second: tuple) -> str | None:
    """Return the destination whose keyword pattern alone matches, else None.

    `first` and `second` are (pattern, destination) pairs. If both or neither
    match, the request is ambiguous and the caller falls back to LLM routing.
    """
    if not user_text:
        return None
    (first_pattern, first_dest), (second_pattern, second_dest) = first, second
    has_first = bool(first_pattern.search(user_text))
    has_second = bool(second_pattern.search(user_text))
    if has_first and not has_second:
        return first_dest
    if has_second and not has_first:
        return second_dest
    return None
