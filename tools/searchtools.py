import asyncio
import concurrent.futures
import json
import logging
import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from agents.output_sanitizer import redact_secrets

load_dotenv()

log = logging.getLogger("siri.tools.searchtools")

MAX_CHARS = 2500       # cap for raw/unparsed output
SNIPPET_CHARS = 400    # cap per search result
TOP_RESULTS = 3

_client = None


def get_client() -> MultiServerMCPClient:
    """Create the Tavily MCP client once and reuse it."""
    global _client
    if _client is None:
        api_key = os.getenv("TAVILY_API_KEY") or os.getenv("Tavily_Api_key", "")
        _client = MultiServerMCPClient({
            "tavily": {
                "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={api_key}",
                "transport": "streamable_http",
            }
        })
    return _client


def run_sync(coro):
    """Run a coroutine from sync code, even if an event loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # A loop is already running in this thread, so use a separate thread.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def pick_search_tool(tools):
    """Prefer 'tavily_search', then any other search tool, then whatever is first."""
    for t in tools:
        if t.name == "tavily_search":
            return t
    for t in tools:
        name = t.name.lower()
        if "search" in name and "research" not in name:
            return t
    return tools[0] if tools else None


def format_results(raw) -> str:
    """Trim search output down to a few short results so we don't burn tokens."""
    # MCP returns [{'type': 'text', 'text': '...'}]
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "text" in raw[0]:
        raw = raw[0]["text"]

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return raw[:MAX_CHARS]

    results = raw.get("results", []) if isinstance(raw, dict) else raw
    if not isinstance(results, list):
        return str(raw)[:MAX_CHARS]

    lines = []
    for item in results[:TOP_RESULTS]:
        if not isinstance(item, dict):
            lines.append(str(item)[:SNIPPET_CHARS])
            continue
        title = item.get("title", "No title")
        snippet = (item.get("content") or item.get("snippet") or "")[:SNIPPET_CHARS]
        url = item.get("url", "")
        lines.append(f"• **{title}**\n  {snippet}\n  Source: {url}")

    return "\n\n".join(lines) or str(raw)[:MAX_CHARS]


@tool("tavily_search")
def search_tool(query: str, max_results: int = 3) -> str:
    """Search the web with Tavily for up-to-date facts and articles."""
    async def search():
        tools = await get_client().get_tools()
        tool_to_use = pick_search_tool(tools)
        if tool_to_use is None:
            return "Error: the MCP server returned no search tools."
        raw = await tool_to_use.ainvoke({"query": query, "max_results": max_results})
        return format_results(raw)

    try:
        return run_sync(search())
    except Exception as exc:
        # The MCP URL carries the API key, so keep it out of logs and the LLM.
        message = redact_secrets(str(exc))
        log.error("Search failed: %s", message)
        return f"Error running search: {message}"
