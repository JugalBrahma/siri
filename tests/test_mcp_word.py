import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from tools.word_tools import (
    get_word_mcp_client,
    get_word_mcp_server_connections,
    fetch_mcp_word_tools_async,
    fetch_mcp_word_tools_sync,
    get_word_tools,
)
from tools.terminal import terminal_tools, run_terminal_command, read_file, modify_file
from agents.actionagent import get_action_tools, action_agent
from langchain_core.tools import tool


class TestWordMCP(unittest.TestCase):
    def test_word_mcp_server_connection_config(self):
        """Verify MultiServerMCPClient connection configuration for uvx stdio."""
        connections = get_word_mcp_server_connections()
        self.assertIn("word_server", connections)
        conn = dict(connections["word_server"])
        self.assertEqual(conn.get("transport"), "stdio")
        self.assertEqual(conn.get("command"), "uvx")
        args = conn.get("args", [])
        self.assertIsInstance(args, list)
        self.assertIn("office-word-mcp-server", args)
        self.assertIn("word_mcp_server", args)

    def test_word_mcp_client_singleton(self):
        """Verify get_word_mcp_client returns client instance with word_server config."""
        client = get_word_mcp_client()
        self.assertIsNotNone(client)
        self.assertIn("word_server", client.connections)

    def test_terminal_tools_clean(self):
        """Verify old write_to_word and write_to_open_word are removed from terminal tools."""
        tool_names = [t.name for t in terminal_tools]
        self.assertIn("run_terminal_command", tool_names)
        self.assertIn("read_file", tool_names)
        self.assertIn("modify_file", tool_names)
        self.assertNotIn("write_to_word", tool_names)
        self.assertNotIn("write_to_open_word", tool_names)

    @patch("tools.word_tools.get_word_mcp_client")
    def test_fetch_mcp_word_tools_async(self, mock_get_client):
        """Verify async fetching of Word MCP tools."""
        mock_tool = MagicMock()
        mock_tool.name = "create_document"
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[mock_tool])
        mock_get_client.return_value = mock_client

        with patch("tools.word_tools._cached_word_tools", None):
            import asyncio
            tools = asyncio.run(fetch_mcp_word_tools_async())
            self.assertEqual(len(tools), 1)
            self.assertEqual(tools[0].name, "create_document")

    @patch("tools.word_tools.get_word_mcp_client")
    def test_fetch_mcp_word_tools_sync(self, mock_get_client):
        """Verify sync fetching of Word MCP tools."""
        mock_tool = MagicMock()
        mock_tool.name = "add_heading"
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=[mock_tool])
        mock_get_client.return_value = mock_client

        with patch("tools.word_tools._cached_word_tools", None):
            tools = fetch_mcp_word_tools_sync()
            self.assertEqual(len(tools), 1)
            self.assertEqual(tools[0].name, "add_heading")

    def test_action_agent_tools_assembly(self):
        """Verify action agent combines base terminal tools with Word tools."""
        @tool
        def dummy_word_tool(doc: str) -> str:
            """Dummy word tool"""
            return doc

        with patch("agents.actionagent.get_word_tools", return_value=[dummy_word_tool]):
            action_tools = get_action_tools()
            names = [t.name for t in action_tools]
            self.assertIn("run_terminal_command", names)
            self.assertIn("read_file", names)
            self.assertIn("modify_file", names)
            self.assertIn("ask_human", names)
            self.assertIn("dummy_word_tool", names)
            self.assertNotIn("write_to_word", names)
    def test_wrap_sync_tool_execution(self):
        """Verify async StructuredTool supports synchronous .invoke() with content_and_artifact."""
        from tools.word_tools import _wrap_sync_tool
        from langchain_core.tools import StructuredTool
        import asyncio

        async def dummy_mcp_op(file_path: str, title: str) -> tuple[str, dict]:
            await asyncio.sleep(0.01)
            return f"Created {title} at {file_path}", {"file": file_path}

        raw_async_tool = StructuredTool.from_function(
            func=None,
            coroutine=dummy_mcp_op,
            name="create_document",
            description="Create doc",
            response_format="content_and_artifact",
        )

        wrapped = _wrap_sync_tool(raw_async_tool)
        # Verify sync invocation does not raise NotImplementedError or format errors
        result = wrapped.invoke({"file_path": "test.docx", "title": "Doc Title"})
        self.assertEqual(result, "Created Doc Title at test.docx")

    def test_open_word_document_tool(self):
        """Verify open_word_document tool execution handling."""
        from tools.word_tools import open_word_document
        with patch("os.path.exists", return_value=True), patch("os.startfile", create=True) as mock_startfile:
            res = open_word_document.invoke({"file_path": "test.docx"})
            self.assertIn("Successfully launched Microsoft Word", res)
            mock_startfile.assert_called_once()


if __name__ == "__main__":
    unittest.main()
