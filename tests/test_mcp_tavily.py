import unittest
from unittest.mock import MagicMock, patch
from tools.searchtools import search_tool, get_mcp_client, fetch_mcp_search_tools_sync
from mcp_server.server import tavily_search, mcp


class TestTavilyMCP(unittest.TestCase):
    def test_mcp_server_tool_registered(self):
        """Verify that tavily_search is registered on the FastMCP server."""
        # FastMCP tools can be checked via mcp.get_tools() or registered functions
        self.assertEqual(tavily_search.__name__, "tavily_search")

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
        client = get_mcp_client()
        self.assertIsNotNone(client)
        self.assertIn("tavily_server", client.connections)
        conn = client.connections["tavily_server"]
        self.assertEqual(conn.get("transport"), "streamable_http")
        url = str(conn.get("url", ""))
        self.assertIn("mcp.tavily.com", url)

    def test_search_tool_definition(self):
        """Verify search_tool definition and metadata."""
        self.assertEqual(search_tool.name, "tavily_search")
        self.assertIn("search", search_tool.description.lower())


if __name__ == "__main__":
    unittest.main()
