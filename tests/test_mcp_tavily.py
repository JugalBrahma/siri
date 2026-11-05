import unittest
from unittest.mock import MagicMock, patch
from tools.searchtools import search_tool, get_client, pick_search_tool, format_results
from mcp_server.server import tavily_search as tavily_search_tool

# FastMCP wraps the function in a FunctionTool; .fn is the plain callable.
tavily_search = tavily_search_tool.fn


class TestTavilyMCP(unittest.TestCase):
    def test_mcp_server_tool_registered(self):
        """Verify that tavily_search is registered on the FastMCP server."""
        # FastMCP tools can be checked via mcp.get_tools() or registered functions
        self.assertEqual(tavily_search_tool.name, "tavily_search")

    def test_mcp_server_tavily_search_no_key(self):
        """Verify behavior when TAVILY_API_KEY is not set."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("mcp_server.server.tavily_client", None):
                result = tavily_search("test query")
                self.assertIn("Error", result)

    def test_mcp_server_tavily_search_success(self):
        """Verify search tool formatting with mocked TavilyClient."""
        mock_client = MagicMock()
        mock_client.search.return_value = {
            "results": [
                {
                    "title": "Quantum Computing News",
                    "url": "https://example.com/quantum",
                    "content": "Quantum computing advances rapidly.",
                }
            ]
        }
        with patch("mcp_server.server.tavily_client", mock_client):
            result = tavily_search("quantum computing")
            self.assertIn("Quantum Computing News", result)
            self.assertIn("https://example.com/quantum", result)
            self.assertIn("Quantum computing advances rapidly", result)

    def test_langchain_mcp_client_configuration(self):
        """Verify MultiServerMCPClient connection configuration for remote streamable HTTP."""
        client = get_client()
        self.assertIsNotNone(client)
        self.assertIn("tavily", client.connections)
        conn = client.connections["tavily"]
        self.assertEqual(conn.get("transport"), "streamable_http")
        url = str(conn.get("url", ""))
        self.assertIn("mcp.tavily.com", url)

    def test_pick_search_tool_prefers_tavily(self):
        """Verify the exact tavily_search tool wins over other search-like tools."""
        other = MagicMock()
        other.name = "web_search"
        tavily = MagicMock()
        tavily.name = "tavily_search"
        self.assertIs(pick_search_tool([other, tavily]), tavily)
        self.assertIsNone(pick_search_tool([]))

    def test_format_results_trims_output(self):
        """Verify raw MCP output is reduced to a few short results."""
        raw = [{"type": "text", "text": '{"results": [{"title": "A", "content": "body", "url": "https://a.test"}]}'}]
        text = format_results(raw)
        self.assertIn("A", text)
        self.assertIn("https://a.test", text)

    def test_search_error_hides_api_key(self):
        """A failed search must not leak the Tavily key from the MCP URL."""
        key = "tvly-dev-abc123zzKEYxyz789"
        error = RuntimeError(f"connect failed: https://mcp.tavily.com/mcp/?tavilyApiKey={key}")
        with patch("tools.searchtools.get_client", side_effect=error), \
                self.assertLogs("siri.tools.searchtools", level="ERROR") as logs:
            result = search_tool.invoke({"query": "x"})
        self.assertTrue(result.startswith("Error running search:"))
        self.assertNotIn(key, result)
        self.assertNotIn("zzKEY", " ".join(logs.output))
        self.assertIn("[REDACTED_SECRET]", result)

    def test_search_tool_definition(self):
        """Verify search_tool definition and metadata."""
        self.assertEqual(search_tool.name, "tavily_search")
        self.assertIn("search", search_tool.description.lower())


if __name__ == "__main__":
    unittest.main()
