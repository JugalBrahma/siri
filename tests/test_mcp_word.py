import os
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.tools import tool

import tools.terminal as terminal
import tools.word_tools as word_tools_module
from agents.actionagent import get_action_tools
from tools.terminal import open_app, read_file, modify_file, terminal_tools
from tools.word_tools import (
    add_paragraph,
    create_complete_word_document,
    create_document,
    get_word_tools,
    open_word_document,
)


class WorkspaceTestCase(unittest.TestCase):
    """Runs each test against a throwaway workspace folder."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = os.path.realpath(self._tmp.name)
        patcher = patch.object(terminal, "WORKSPACE", self.workspace)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)


class TestTerminalTools(WorkspaceTestCase):
    def test_terminal_tools_have_no_shell(self):
        """The raw shell tool is gone; only the safe tools remain."""
        names = [t.name for t in terminal_tools]
        self.assertIn("open_app", names)
        self.assertIn("read_file", names)
        self.assertIn("modify_file", names)
        self.assertNotIn("run_terminal_command", names)

    def test_open_app_rejects_unknown_app(self):
        """Apps outside the allowed list fail validation before anything runs."""
        with patch("subprocess.run") as run:
            with self.assertRaises(Exception):
                open_app.invoke({"app": "rm -rf ~"})
            run.assert_not_called()

    def test_open_app_uses_no_shell(self):
        """An allowed app is launched with an argument list, never a shell string."""
        with patch("tools.terminal.sys.platform", "darwin"), patch("subprocess.run") as run:
            run.return_value.returncode = 0
            result = open_app.invoke({"app": "safari"})
        self.assertIn("Opened", result)
        args, kwargs = run.call_args
        self.assertEqual(args[0], ["open", "-a", "Safari"])
        self.assertNotIn("shell", kwargs)

    def test_file_tools_stay_in_workspace(self):
        """Reads and writes work inside the workspace and are refused outside."""
        self.assertIn("Successfully", modify_file.invoke({"file_path": "a/b.txt", "content": "hi"}))
        self.assertEqual(read_file.invoke({"file_path": "a/b.txt"}), "hi")
        for bad in ["../escape.txt", "/etc/hosts", "~/.zshrc"]:
            self.assertIn("outside the allowed workspace", read_file.invoke({"file_path": bad}))
        self.assertIn(
            "outside the allowed workspace",
            modify_file.invoke({"file_path": "../escape.txt", "content": "x"}),
        )
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(self.workspace), "escape.txt")))


class TestWordTools(WorkspaceTestCase):
    def test_get_word_tools(self):
        names = [t.name for t in get_word_tools()]
        self.assertIn("create_complete_word_document", names)
        self.assertIn("open_word_document", names)

    def test_create_and_edit_inside_workspace(self):
        """Documents are created in the workspace and .docx is added if missing."""
        with patch.object(word_tools_module, "_launch") as launch:
            result = create_complete_word_document.invoke(
                {"file_path": "reports/summary", "title": "T", "content": "# H\n- item\nbody"}
            )
            self.assertIn("Created and opened", result)
            launch.assert_called_once()
        path = os.path.join(self.workspace, "reports", "summary.docx")
        self.assertTrue(os.path.exists(path))
        self.assertIn("Added paragraph", add_paragraph.invoke({"file_path": "reports/summary.docx", "text": "more"}))

    def test_paths_outside_workspace_are_refused(self):
        for bad in ["../x.docx", "/tmp/x.docx", "~/Desktop/x.docx"]:
            self.assertIn("outside the allowed workspace", create_document.invoke({"file_path": bad}))

    def test_open_word_document(self):
        """open_word_document launches existing workspace documents only."""
        create_document.invoke({"file_path": "doc.docx"})
        with patch.object(word_tools_module, "_launch") as launch:
            self.assertIn("Opened", open_word_document.invoke({"file_path": "doc.docx"}))
            launch.assert_called_once()
            self.assertIn("does not exist", open_word_document.invoke({"file_path": "missing.docx"}))
            self.assertIn("outside the allowed workspace", open_word_document.invoke({"file_path": "/etc/hosts"}))
            self.assertEqual(launch.call_count, 1)


class TestActionAgentTools(unittest.TestCase):
    def test_action_agent_tools_assembly(self):
        """The action agent combines the safe base tools with the Word tools."""
        @tool
        def dummy_word_tool(doc: str) -> str:
            """Dummy word tool"""
            return doc

        with patch("agents.actionagent.get_word_tools", return_value=[dummy_word_tool]):
            names = [t.name for t in get_action_tools()]
        for expected in ["open_app", "read_file", "modify_file", "ask_human", "dummy_word_tool"]:
            self.assertIn(expected, names)
        self.assertNotIn("run_terminal_command", names)


if __name__ == "__main__":
    unittest.main()
