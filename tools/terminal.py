import os
import subprocess
import sys
from typing import Dict, List, Literal, Optional
from langchain.tools import tool
from langgraph.errors import GraphInterrupt
from langgraph.types import interrupt

from tools.human_tool import human_answer_text
from tools import macos_windows
from tools.screen_target import set_target_app

# ---------------------------------------------------------
# 1. Open any installed application, with the user's permission (no shell)
# ---------------------------------------------------------
# Folders scanned for installed macOS apps.
APP_DIRS = [
    "/Applications",
    "/Applications/Utilities",
    "/System/Applications",
    "/System/Applications/Utilities",
    os.path.expanduser("~/Applications"),
]


def installed_apps() -> Dict[str, str]:
    """Return {lowercase name: real app name} for every installed .app bundle."""
    apps: Dict[str, str] = {}
    for folder in APP_DIRS:
        try:
            entries = os.listdir(folder)
        except OSError:
            continue
        for entry in entries:
            if entry.endswith(".app"):
                name = entry[:-4]
                apps.setdefault(name.lower(), name)
    if os.path.isdir("/System/Library/CoreServices/Finder.app"):
        apps.setdefault("finder", "Finder")
    return apps


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _find_installed(app: str, apps: Dict[str, str]) -> Optional[str]:
    """Match a requested name to an installed app ('pages' -> 'Pages Creator Studio')."""
    wanted = _normalize(app)
    if not wanted:
        return None
    by_norm = {_normalize(lower): real for lower, real in apps.items()}
    if wanted in by_norm:
        return by_norm[wanted]
    matches = [real for norm, real in by_norm.items() if wanted in norm]
    return matches[0] if len(matches) == 1 else None


def _confirm_open(app_name: str) -> bool:
    """Pause the graph and ask the user before opening an app."""
    answer = human_answer_text(interrupt({
        "type": "ask_human",
        "question": f"Siri wants to open '{app_name}'. Allow? (yes/no)",
    }))
    return answer.strip().lower() in {"y", "yes", "ok", "okay", "sure", "allow", "yeah", "yep"}


def _launch(argv: List[str], label: str) -> str:
    # List form, no shell=True: nothing the model says is run as a command.
    result = subprocess.run(argv, capture_output=True, text=True, timeout=15)
    if result.returncode != 0:
        return f"Could not open {label}: {result.stderr.strip() or 'unknown error'}"
    return f"Opened {label}."


@tool
def list_apps() -> str:
    """
    Lists every application installed on this Mac. Use it to check an app's
    exact name before calling open_app, or when the user asks what apps exist.
    """
    if sys.platform != "darwin":
        return "Listing apps is only supported on macOS."
    names = sorted(installed_apps().values(), key=str.lower)
    return f"{len(names)} installed apps: " + ", ".join(names)


@tool
def open_app(app: str) -> str:
    """
    Opens any installed application on this Mac by name (e.g. "Safari",
    "Pages", "WhatsApp"). The user is always asked for permission first.
    Use list_apps to see what is installed. Only launches apps; never runs commands.
    """
    if sys.platform != "darwin":
        return "Opening apps is only supported on macOS."
    try:
        real_name = _find_installed(app, installed_apps())
        if real_name is None:
            return f"'{app}' is not installed. Call list_apps to see installed apps."
        if macos_windows.is_running(real_name) and macos_windows.focus_app(real_name):
            set_target_app(real_name)  # already open: nothing to launch, so no question
            return f"{real_name} was already open; brought it to the front."
        if not _confirm_open(real_name):
            return f"The user declined to open {real_name}."
        result = _launch(["open", "-a", real_name], real_name)
        if result.startswith("Opened"):
            set_target_app(real_name)  # a screen task now works in this app
        return result
    except GraphInterrupt:
        raise  # let the permission question reach the user
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
terminal_tools = [open_app, list_apps, read_file, modify_file]
