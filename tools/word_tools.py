import os
import subprocess
import sys

import docx
from langchain_core.tools import tool

from tools.terminal import _resolve_in_workspace


def _workspace_docx(file_path: str) -> str:
    """Resolve a .docx path inside the workspace; raise ValueError otherwise."""
    if not file_path.lower().endswith(".docx"):
        file_path += ".docx"
    return _resolve_in_workspace(file_path)


def _prepare_path(file_path: str) -> str:
    """Return a workspace-confined absolute path and make sure its folder exists."""
    path = _workspace_docx(file_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _load(path: str):
    """Open the document if it exists, otherwise start a new one."""
    return docx.Document(path) if os.path.exists(path) else docx.Document()


def _launch(path: str) -> None:
    """Open a file with the default app on Windows, macOS or Linux."""
    if sys.platform == "win32":
        os.startfile(path)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen([opener, path])


@tool
def create_document(file_path: str) -> str:
    """Create a new empty Word (.docx) document at file_path."""
    try:
        path = _prepare_path(file_path)
        docx.Document().save(path)
        return f"Created '{path}'."
    except Exception as exc:
        return f"Failed to create document: {exc}"


@tool
def add_heading(file_path: str, text: str, level: int = 1) -> str:
    """Add a heading (level 1-9) to a Word (.docx) document. Creates the file if missing."""
    try:
        path = _prepare_path(file_path)
        doc = _load(path)
        doc.add_heading(text, level=min(max(level, 1), 9))
        doc.save(path)
        return f"Added heading '{text}' to '{path}'."
    except Exception as exc:
        return f"Failed to add heading: {exc}"


@tool
def add_paragraph(file_path: str, text: str) -> str:
    """Add a body paragraph to a Word (.docx) document. Creates the file if missing."""
    try:
        path = _prepare_path(file_path)
        doc = _load(path)
        doc.add_paragraph(text)
        doc.save(path)
        return f"Added paragraph to '{path}'."
    except Exception as exc:
        return f"Failed to add paragraph: {exc}"


@tool
def add_table(file_path: str, rows: int, cols: int, data: list[list[str]] | None = None) -> str:
    """Add a table (rows x cols) to a Word (.docx) document, optionally filled from a 2D list."""
    try:
        path = _prepare_path(file_path)
        doc = _load(path)
        table = doc.add_table(rows=rows, cols=cols)
        for r, row in enumerate(data or []):
            for c, value in enumerate(row):
                if r < rows and c < cols:
                    table.cell(r, c).text = str(value)
        doc.save(path)
        return f"Added {rows}x{cols} table to '{path}'."
    except Exception as exc:
        return f"Failed to add table: {exc}"


@tool
def open_word_document(file_path: str) -> str:
    """Open a Word (.docx) document on the user's desktop. Call this after creating or editing a document."""
    try:
        path = _workspace_docx(file_path)
        if not os.path.exists(path):
            return f"Error: '{file_path}' does not exist."
        _launch(path)
        return f"Opened '{os.path.basename(path)}'."
    except Exception as exc:
        return f"Failed to open document: {exc}"


@tool
def create_complete_word_document(file_path: str, title: str, content: str) -> str:
    """Create a full Word (.docx) document in one step and open it.

    Use this for articles, reports and research summaries. `content` can use
    markdown-style '#', '##', '###' headings and '- ' bullets.
    """
    try:
        path = _prepare_path(file_path)
        doc = docx.Document()
        doc.add_heading(title, level=0)

        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("### "):
                doc.add_heading(line[4:], level=2)
            elif line.startswith(("## ", "# ")):
                doc.add_heading(line.lstrip("#").strip(), level=1)
            elif line.startswith(("- ", "• ")):
                doc.add_paragraph(line[2:], style="List Bullet")
            else:
                doc.add_paragraph(line)

        doc.save(path)
        _launch(path)
        return f"Created and opened '{path}'."
    except Exception as exc:
        return f"Failed to create Word document: {exc}"


word_tools = [
    create_complete_word_document,
    create_document,
    add_heading,
    add_paragraph,
    add_table,
    open_word_document,
]

def get_word_tools():
    return word_tools