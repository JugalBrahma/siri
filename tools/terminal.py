import subprocess
import os
from langchain.tools import tool

# ---------------------------------------------------------
# 1. Run shell commands (open apps, run scripts, etc.)
# ---------------------------------------------------------
@tool
def run_terminal_command(command: str) -> str:
    """
    Executes a system shell command on the host machine and returns the text output.
    Use this to open applications (e.g., 'start chrome', 'start excel'), check files,
    or run system scripts.

    Args:
        command (str): The exact shell command string to execute.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=15
        )
        if result.returncode == 0:
            return result.stdout if result.stdout else "Command executed successfully with no output."
        else:
            return f"Error: {result.stderr}"
    except Exception as e:
        return f"Execution failed: {str(e)}"


# ---------------------------------------------------------
# 2. Read a local file (plain text / code)
# ---------------------------------------------------------
@tool
def read_file(file_path: str) -> str:
    """Reads and returns the full content of a local text file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Failed to read file: {str(e)}"


# ---------------------------------------------------------
# 3. Write / overwrite / append a local file (plain text / code)
# ---------------------------------------------------------
@tool
def modify_file(file_path: str, content: str, mode: str = "overwrite") -> str:
    """
    Writes content to a local text file.
    mode='overwrite' replaces the entire file content.
    mode='append' adds content to the end of the file.
    Always read the file first if you need to preserve existing content.
    """
    try:
        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        write_mode = "w" if mode == "overwrite" else "a"
        with open(file_path, write_mode, encoding="utf-8") as f:
            f.write(content)
        return f"Successfully {'wrote to' if mode == 'overwrite' else 'appended to'} {file_path}"
    except Exception as e:
        return f"Failed to edit file: {str(e)}"


# ---------------------------------------------------------
# 4. Create a brand new .docx file (no Word needed open/installed... well, python-docx needed)
# ---------------------------------------------------------
@tool
def write_to_word(file_path: str, content: str) -> str:
    """
    Creates or overwrites a Word document (.docx) with the given text content.
    Each line in 'content' becomes its own paragraph.
    Requires: pip install python-docx
    """
    try:
        # pyrefly: ignore [missing-import]
        from docx import Document
        doc = Document()
        for line in content.split("\n"):
            doc.add_paragraph(line)
        doc.save(file_path)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Failed to write Word file: {str(e)}"


# ---------------------------------------------------------
# 5. Type into an ALREADY OPEN Word window (live automation)
# ---------------------------------------------------------
@tool
def write_to_open_word(text: str) -> str:
    """
    Inserts text into the currently active (already open) Word document.
    Requires: pip install pywin32
    Requires Microsoft Word to be installed and currently running with a doc open.
    """
    try:
        import win32com.client
        word = win32com.client.GetObject(Class="Word.Application")
        doc = word.ActiveDocument
        doc.Content.InsertAfter(text)
        return "Text inserted into active Word document."
    except Exception as e:
        return f"Failed to interact with Word: {str(e)}"


# ---------------------------------------------------------
# Example: registering these tools with an agent (LangChain style)
# ---------------------------------------------------------
# tools = [run_terminal_command, read_file, modify_file, write_to_word, write_to_open_word]
# agent = create_react_agent(llm, tools, ...)  # wire into your existing agent setup