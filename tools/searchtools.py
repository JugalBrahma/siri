import asyncio
import logging
import os
import sys
from typing import Dict, List, Optional
from langchain_core.tools import BaseTool, tool
from langchain_mcp_adapters.client import Connection, MultiServerMCPClient

from dotenv import load_dotenv

load_dotenv()

LOGGER = logging.getLogger("siri.tools.searchtools")


def get_mcp_server_connections() -> dict[str, Connection]:
    """Build connection configuration for Tavily's remote hosted MCP server."""
    api_key = os.getenv("TAVILY_API_KEY", "")
    return {
        "tavily_server": {
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={api_key}",
            "transport": "streamable_http",
        }
    }


MCP_SERVER_CONNECTIONS: dict[str, Connection] = get_mcp_server_connections()

_mcp_client: Optional[MultiServerMCPClient] = None


def get_mcp_client() -> MultiServerMCPClient:
    """Get or create the MultiServerMCPClient instance."""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MultiServerMCPClient(get_mcp_server_connections())
    return _mcp_client


async def fetch_mcp_search_tools_async() -> List[BaseTool]:
    """Fetch search tools from the Tavily MCP Server asynchronously."""
    client = get_mcp_client()
    return await client.get_tools()


def fetch_mcp_search_tools_sync() -> List[BaseTool]:
    """Fetch search tools from the Tavily MCP Server synchronously."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(fetch_mcp_search_tools_async())
        return asyncio.run(fetch_mcp_search_tools_async())
    except Exception as exc:
        LOGGER.warning("Could not fetch tools via event loop, fallback to asyncio.run: %s", exc)
        return asyncio.run(fetch_mcp_search_tools_async())


def _format_search_results(raw_result: Any) -> str:
    """Format and compact search results to prevent massive token inflation."""
    import json
    if isinstance(raw_result, str):
        try:
            parsed = json.loads(raw_result)
            raw_result = parsed
        except Exception:
            return raw_result[:2500] if len(raw_result) > 2500 else raw_result

    if isinstance(raw_result, dict):
        results = raw_result.get("results", [])
    elif isinstance(raw_result, list):
        results = raw_result
    else:
        return str(raw_result)[:2500]

    formatted = []
    for item in results[:3]:  # Top 3 high-relevance results
        if isinstance(item, dict):
            title = item.get("title", "No Title")
            content = item.get("content", item.get("snippet", ""))[:400]
            url = item.get("url", "")
            formatted.append(f"• **{title}**\n  {content}\n  Source: {url}")
        else:
            formatted.append(str(item)[:400])
    return "\n\n".join(formatted) if formatted else str(raw_result)[:2500]


@tool("tavily_search")
def search_tool(query: str, max_results: int = 3) -> str:
    """Search the web using Tavily Search API for up-to-date facts, information, and research articles."""
    try:
        client = get_mcp_client()

        async def _run():
            tools = await client.get_tools()
            for t in tools:
                if "search" in t.name.lower():
                    raw = await t.ainvoke({"query": query, "max_results": max_results})
                    return _format_search_results(raw)
            if tools:
                raw = await tools[0].ainvoke({"query": query, "max_results": max_results})
                return _format_search_results(raw)
            return "Error: No search tools returned from MCP server."

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(_run())
            return asyncio.run(_run())
        except Exception:
            return asyncio.run(_run())
    except Exception as exc:
        LOGGER.error("MCP tool execution failed: %s", exc)
        return f"Error executing search tool via MCP: {str(exc)}"


# Exported tools list
search_tools: List[BaseTool] = [search_tool]