import os
import subprocess
import sys
from typing import Literal
from langchain.tools import tool

# ---------------------------------------------------------
# 1. Open an approved application (fixed list, no shell)
# ---------------------------------------------------------
# Friendly name -> (macOS app name, Windows executable, Linux command).
# Add new apps here; the model can never open anything outside this table.
ALLOWED_APPS = {
    "chrome": ("Google Chrome", "chrome", "google-chrome"),
    "safari": ("Safari", None, None),
    "firefox": ("Firefox", "firefox", "firefox"),
    "word": ("Microsoft Word", "winword", None),
    "excel": ("Microsoft Excel", "excel", None),
    "powerpoint": ("Microsoft PowerPoint", "powerpnt", None),
    "vscode": ("Visual Studio Code", "code", "code"),
    "notes": ("Notes", None, None),
    "calculator": ("Calculator", "calc", "gnome-calculator"),
    "terminal": ("Terminal", "cmd", "gnome-terminal"),
}

AppName = Literal[
    "chrome", "safari", "firefox", "word", "excel", "powerpoint",
    "vscode", "notes", "calculator", "terminal",
]


@tool
def open_app(app: AppName) -> str:
    """
    Opens one of the approved desktop applications by name.
    Allowed apps: chrome, safari, firefox, word, excel, powerpoint, vscode,
    notes, calculator, terminal. No other apps or commands can be run.
    """
    entry = ALLOWED_APPS.get(str(app).strip().lower())
    if entry is None:
        return f"'{app}' is not an allowed app. Allowed: {', '.join(ALLOWED_APPS)}"

    mac_name, win_cmd, linux_cmd = entry
    try:
        if sys.platform == "darwin":
            argv = ["open", "-a", mac_name]
        elif sys.platform == "win32":
            if not win_cmd:
                return f"'{app}' is not available on Windows."
            argv = ["cmd", "/c", "start", "", win_cmd]
        else:
            if not linux_cmd:
                return f"'{app}' is not available on Linux."
            argv = [linux_cmd]
        # List form, no shell=True: the model supplies only the app key.
        result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return f"Could not open {app}: {result.stderr.strip() or 'unknown error'}"
        return f"Opened {app}."
    except Exception as e:
        return f"Failed to open {app}: {str(e)}"


# ---------------------------------------------------------
# Workspace sandbox for file tools
# ---------------------------------------------------------
# All file reads/writes are confined to this folder. Override with SIRI_WORKSPACE.
WORKSPACE = os.path.realpath(
    os.path.expanduser(os.getenv("SIRI_WORKSPACE", "~/Documents/siri"))
)


def _resolve_in_workspace(file_path: str) -> str:
    """Resolve a path inside the workspace; raise ValueError if it escapes.

    Relative paths are taken relative to the workspace. Symlinks and '..' are
    resolved first, so neither can be used to reach files outside it.
    """
    candidate = os.path.expanduser(file_path)
    if not os.path.isabs(candidate):
        candidate = os.path.join(WORKSPACE, candidate)
    resolved = os.path.realpath(candidate)
    if os.path.commonpath([WORKSPACE, resolved]) != WORKSPACE:
        raise ValueError(f"Path is outside the allowed workspace ({WORKSPACE}).")
    return resolved


# ---------------------------------------------------------
# 2. Read a local file (plain text / code), workspace only
# ---------------------------------------------------------
@tool
def read_file(file_path: str) -> str:
    """Reads and returns the full content of a text file inside the workspace folder."""
    try:
        with open(_resolve_in_workspace(file_path), "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Failed to read file: {str(e)}"


# ---------------------------------------------------------
# 3. Write / overwrite / append a local file, workspace only
# ---------------------------------------------------------
@tool
def modify_file(file_path: str, content: str, mode: Literal["overwrite", "append"] = "overwrite") -> str:
    """
    Writes content to a text file inside the workspace folder.
    mode='overwrite' replaces the entire file content.
    mode='append' adds content to the end of the file.
    Always read the file first if you need to preserve existing content.
    """
    try:
        path = _resolve_in_workspace(file_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_mode = "w" if mode == "overwrite" else "a"
        with open(path, write_mode, encoding="utf-8") as f:
            f.write(content)
        return f"Successfully {'wrote to' if mode == 'overwrite' else 'appended to'} {path}"
    except Exception as e:
        return f"Failed to edit file: {str(e)}"


# ---------------------------------------------------------
# Exported terminal & file action tools
# ---------------------------------------------------------
terminal_tools = [open_app, read_file, modify_file]
