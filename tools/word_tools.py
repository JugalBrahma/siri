import asyncio
import logging
from typing import Dict, List, Optional
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import Connection, MultiServerMCPClient
from dotenv import load_dotenv

load_dotenv()

LOGGER = logging.getLogger("siri.tools.word_tools")


def get_word_mcp_server_connections() -> dict[str, Connection]:
    """Build connection configuration for the local Office Word MCP server via uvx."""
    return {
        "word_server": {
            "command": "uvx",
            "args": ["--from", "office-word-mcp-server", "word_mcp_server"],
            "transport": "stdio",
        }
    }


WORD_MCP_SERVER_CONNECTIONS: dict[str, Connection] = get_word_mcp_server_connections()

_word_mcp_client: Optional[MultiServerMCPClient] = None
_cached_word_tools: Optional[List[BaseTool]] = None


def get_word_mcp_client() -> MultiServerMCPClient:
    """Get or create the MultiServerMCPClient instance for Word MCP Server."""
    global _word_mcp_client
    if _word_mcp_client is None:
        _word_mcp_client = MultiServerMCPClient(get_word_mcp_server_connections())
    return _word_mcp_client


def _wrap_sync_tool(tool: BaseTool) -> BaseTool:
    """Ensure an async StructuredTool supports synchronous execution."""
    if getattr(tool, "func", None) is not None:
        return tool

    coro = getattr(tool, "coroutine", None)
    if coro is None:
        return tool

    def sync_runner(*args, **kwargs):
        async def _execute():
            if kwargs:
                return await coro(**kwargs)
            elif args and isinstance(args[0], dict):
                return await coro(**args[0])
            elif args:
                return await coro(*args)
            return await coro()

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio

                nest_asyncio.apply()
                return loop.run_until_complete(_execute())
            else:
                return asyncio.run(_execute())
        except Exception:
            return asyncio.run(_execute())

    try:
        tool.func = sync_runner  # type: ignore[attr-defined]
    except Exception as exc:
        LOGGER.debug("Could not assign sync func to tool %s: %s", getattr(tool, "name", "unknown"), exc)

    return tool


async def fetch_mcp_word_tools_async() -> List[BaseTool]:
    """Fetch Word tools from the local Office Word MCP Server asynchronously."""
    global _cached_word_tools
    if _cached_word_tools is not None:
        return _cached_word_tools

    client = get_word_mcp_client()
    raw_tools = await client.get_tools()
    tools = [_wrap_sync_tool(t) for t in raw_tools]
    _cached_word_tools = tools
    LOGGER.info("Fetched %d Word tools from Office Word MCP Server.", len(tools))
    return tools


def fetch_mcp_word_tools_sync() -> List[BaseTool]:
    """Fetch Word tools from the local Office Word MCP Server synchronously."""
    global _cached_word_tools
    if _cached_word_tools is not None:
        return _cached_word_tools

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio

            nest_asyncio.apply()
            tools = loop.run_until_complete(fetch_mcp_word_tools_async())
        else:
            tools = asyncio.run(fetch_mcp_word_tools_async())
    except Exception as exc:
        LOGGER.warning("Could not fetch Word tools via event loop, fallback to asyncio.run: %s", exc)
        try:
            tools = asyncio.run(fetch_mcp_word_tools_async())
        except Exception as inner_exc:
            LOGGER.error("Failed to fetch Word MCP tools: %s", inner_exc)
            return []

    tools = [_wrap_sync_tool(t) for t in tools]
    _cached_word_tools = tools
    return tools


from langchain_core.tools import BaseTool, tool


# ==============================================================================
# LOCAL HELPER TOOL: AUTOMATICALLY OPEN WORD DOCUMENT
# ==============================================================================
@tool("open_word_document")
def open_word_document(file_path: str) -> str:
    """
    Opens an MS Word (.docx) document on the host desktop in Microsoft Word for the user to view.
    Always call this after creating or editing a Word document so the user can immediately see the result.

    Args:
        file_path (str): The file path to the Word (.docx) document to open.
    """
    import os
    import subprocess

    try:
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            alt_path = os.path.join(os.getcwd(), file_path)
            if os.path.exists(alt_path):
                abs_path = alt_path
            else:
                return f"Error: Document file '{file_path}' does not exist on disk."

        if hasattr(os, "startfile"):
            os.startfile(abs_path)
        else:
            subprocess.Popen(["start", "", abs_path], shell=True)

        return f"Successfully launched Microsoft Word and opened '{os.path.basename(abs_path)}'."
    except Exception as exc:
        return f"Failed to open Word document: {str(exc)}"


# ==============================================================================
# WORD MCP TOOL FILTERING CONFIGURATION
# ------------------------------------------------------------------------------
# Set ALLOWED_WORD_TOOLS to filter down to essential tools to stay within strict
# TPM rate limits (e.g. Groq free tier 8k limit).
#
# When using a high-capacity LLM (e.g., Llama-3.3-70B, GPT-4o, Claude, NVIDIA NIM),
# simply set ALLOWED_WORD_TOOLS = None (or uncomment the line below) to load ALL tools.
# ==============================================================================
ALLOWED_WORD_TOOLS: Optional[set[str]] = {
    "create_document",
    "add_heading",
    "add_paragraph",
    "add_table",
    "open_word_document",
}
# UNCOMMENT BELOW TO LOAD ALL 9+ WORD TOOLS (WHEN USING HIGH-TIER LLM):
# ALLOWED_WORD_TOOLS = None


def get_word_tools(filter_essential: bool = True) -> List[BaseTool]:
    """Return the list of Word MCP tools plus desktop open_word_document tool."""
    global _cached_word_tools
    if _cached_word_tools is None:
        tools = fetch_mcp_word_tools_sync()
    else:
        tools = list(_cached_word_tools)

    all_tools = [open_word_document] + [t for t in tools if t.name != "open_word_document"]

    if filter_essential and ALLOWED_WORD_TOOLS is not None:
        filtered = [t for t in all_tools if t.name in ALLOWED_WORD_TOOLS]
        return filtered if filtered else all_tools
    return all_tools


# Initial fetch / fallback
try:
    word_tools: List[BaseTool] = get_word_tools()
except Exception as e:
    LOGGER.warning("Initial sync fetch of Word MCP tools deferred: %s", e)
    word_tools = [open_word_document]
