"""Find, focus and locate app windows on macOS (any display).

Positions are global screen points: (0, 0) is the top-left of the main
display, and other displays continue from there (they can be negative).
That is the same space pyautogui clicks in.
"""

import re
import subprocess
import time
from dataclasses import dataclass
from typing import List, Optional

# Ignore tiny windows (tooltips, toolbars) when looking for an app's window.
MIN_WINDOW_SIZE = 100
FOCUS_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class Window:
    id: int
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class RunningApp:
    name: str
    path: str
    pid: int


# `lsappinfo` asks the system directly, so its answers are always current and
# safe from any thread. (AppKit's NSWorkspace only refreshes on the main
# thread's event loop, and the API runs tools on worker threads.)
_APP_BLOCK = re.compile(r'^\s*\d+\) "(?P<name>[^"]*)".*?\n(?P<body>(?:\s{2,}.*\n?)*)', re.MULTILINE)
_PATH = re.compile(r'bundle path="(?P<path>[^"]+)"')
_PID = re.compile(r"\bpid = (?P<pid>\d+)")


def _lsappinfo(*args: str) -> str:
    return subprocess.run(["lsappinfo", *args], capture_output=True, text=True, timeout=5).stdout


def _running_apps() -> List[RunningApp]:
    apps = []
    for block in _APP_BLOCK.finditer(_lsappinfo("list")):
        path, pid = _PATH.search(block["body"]), _PID.search(block["body"])
        if path and pid:
            apps.append(RunningApp(block["name"], path["path"], int(pid["pid"])))
    return apps


def _running_app(name: str) -> Optional[RunningApp]:
    """The running app whose bundle is `name`.app (or whose name is `name`)."""
    for app in _running_apps():
        if app.path.endswith(f"/{name}.app") or app.name == name:
            return app
    return None


def _front_pid() -> Optional[int]:
    asn = _lsappinfo("front").strip()
    match = _PID.search(_lsappinfo("info", "-only", "pid", asn)) if asn else None
    return int(match["pid"]) if match else None


def is_running(name: str) -> bool:
    return _running_app(name) is not None


def is_frontmost(name: str) -> bool:
    app = _running_app(name)
    return app is not None and app.pid == _front_pid()


def focus_app(name: str) -> bool:
    """Bring an already-running app to the front. Never launches an app.

    Returns True once the app is in front.
    """
    if not is_running(name):
        return False
    if is_frontmost(name):
        return True
    subprocess.run(["open", "-a", name], capture_output=True, timeout=10)
    deadline = time.monotonic() + FOCUS_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if is_frontmost(name):
            time.sleep(0.2)  # let the window finish coming forward
            return True
        time.sleep(0.1)
    return False


def front_window(name: str) -> Optional[Window]:
    """The app's front-most visible window, on any display."""
    import Quartz

    app = _running_app(name)
    if app is None:
        return None
    pid = app.pid
    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    # The list is ordered front to back.
    for info in Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID):
        if info.get("kCGWindowOwnerPID") != pid or info.get("kCGWindowLayer") != 0:
            continue
        bounds = info["kCGWindowBounds"]
        width, height = int(bounds["Width"]), int(bounds["Height"])
        if width >= MIN_WINDOW_SIZE and height >= MIN_WINDOW_SIZE:
            return Window(int(info["kCGWindowNumber"]), int(bounds["X"]), int(bounds["Y"]), width, height)
    return None


def on_a_display(x: int, y: int) -> bool:
    """Whether a global point is on any connected display."""
    import Quartz

    _, displays, count = Quartz.CGGetActiveDisplayList(16, None, None)
    for display in displays[:count]:
        rect = Quartz.CGDisplayBounds(display)
        if (rect.origin.x <= x < rect.origin.x + rect.size.width
                and rect.origin.y <= y < rect.origin.y + rect.size.height):
            return True
    return False
