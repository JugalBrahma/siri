import os
from dotenv import load_dotenv
from fastmcp import FastMCP
from tavily import TavilyClient

load_dotenv()

# Initialize FastMCP Server
mcp = FastMCP("TavilySearchMCPServer")

tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None


@mcp.tool(
    name="tavily_search",
    description="Search the web using Tavily Search API for up-to-date facts, information, and research articles.",
)
def tavily_search(query: str, max_results: int = 5) -> str:
    """Search the web using Tavily Search API.

    Args:
        query: The search query string.
        max_results: Maximum number of search results to return (default: 5).

    Returns:
        Formatted search results containing title, url, and content summary.
    """
    global tavily_client
    if tavily_client is None:
        key = os.getenv("TAVILY_API_KEY")
        if key:
            tavily_client = TavilyClient(api_key=key)
        else:
            return "Error: TAVILY_API_KEY is not configured in the environment."

    try:
        response = tavily_client.search(query=query, max_results=max_results)
        results = response.get("results", [])
        if not results:
            return f"No search results found for query: '{query}'"

        formatted_results = []
        for i, item in enumerate(results, start=1):
            title = item.get("title", "No Title")
            url = item.get("url", "")
            content = item.get("content", "")
            formatted_results.append(
                f"[{i}] {title}\nURL: {url}\nSummary: {content}"
            )

        return "\n\n".join(formatted_results)
    except Exception as exc:
        return f"Error executing Tavily search: {str(exc)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
