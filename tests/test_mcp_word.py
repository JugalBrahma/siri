import os
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.tools import tool

import tools.terminal as terminal
import tools.word_tools as word_tools_module
from agents.document_writer import get_document_tools
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
        """A name that is not an installed app is refused before anything runs."""
        with patch("tools.terminal.sys.platform", "darwin"), patch("subprocess.run") as run:
            result = open_app.invoke({"app": "rm -rf ~"})
        self.assertIn("not installed", result)
        run.assert_not_called()

    def test_open_app_asks_before_every_app(self):
        """Every app needs the user's yes before it is launched."""
        installed = {"safari": "Safari", "blender": "Blender"}
        with patch("tools.terminal.sys.platform", "darwin"), \
             patch("tools.terminal.installed_apps", return_value=installed), \
             patch("tools.terminal.macos_windows.is_running", return_value=False), \
             patch("subprocess.run") as run:
            run.return_value.returncode = 0
            for app in ["safari", "blender"]:
                with patch("tools.terminal.interrupt", return_value="no") as ask:
                    self.assertIn("declined", open_app.invoke({"app": app}))
                ask.assert_called_once()
            run.assert_not_called()

            with patch("tools.terminal.interrupt", return_value="yes"):
                self.assertIn("Opened", open_app.invoke({"app": "blender"}))
            self.assertEqual(run.call_args.args[0], ["open", "-a", "Blender"])

    def test_open_app_uses_no_shell(self):
        """An approved app is launched with an argument list, never a shell string."""
        with patch("tools.terminal.sys.platform", "darwin"), \
             patch("tools.terminal.installed_apps", return_value={"safari": "Safari"}), \
             patch("tools.terminal.macos_windows.is_running", return_value=False), \
             patch("tools.terminal.interrupt", return_value="yes"), \
             patch("subprocess.run") as run:
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


class TestDocumentWriterTools(unittest.TestCase):
    def test_document_writer_tools_assembly(self):
        """The document writer gets file and Word tools, but no app or screen tools."""
        @tool
        def dummy_word_tool(doc: str) -> str:
            """Dummy word tool"""
            return doc

        with patch("agents.document_writer.get_word_tools", return_value=[dummy_word_tool]):
            names = [t.name for t in get_document_tools()]
        self.assertEqual(names, ["read_file", "modify_file", "ask_human", "dummy_word_tool"])


if __name__ == "__main__":
    unittest.main()
