"""Screen and keyboard tools for the screen_reader / action_taker agents (macOS).

``screenshot`` is read-only: it captures the screen and lets a vision model
answer a question about it. ``type_text``, ``press_keys`` and ``click_on``
change things on screen.

Inside a screen task (see tools/screen_target.py) every tool first brings the
task's app to the front, so keys never land in another window, and screenshots
show only that app's window, on whichever display it is, at full Retina detail
so small icons (a filled vs an outlined Like) stay readable. Positions the
vision model reports are image pixels; click_on converts them to the screen.
Without a task app they use the whole main screen.

On/off buttons (Like, Subscribe, Follow, checkboxes ...) are clicked at most
once per task: a second click would undo the first.

pyautogui is imported lazily so the rest of the app (and the tests) work on
machines without it. Moving the mouse into a screen corner aborts it
(pyautogui's fail-safe).
"""

import base64
import io
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from prompts import load_prompt
from tools import macos_windows
from tools.screen_target import remember_toggle, target_app, was_toggled

# Seconds to wait after each action so the app can catch up (search results,
# opening a chat) before the next key arrives.
SETTLE_SECONDS = 0.6
# Seconds to wait after Cmd+V before restoring the clipboard. Too short and a
# slow app pastes the restored (old) clipboard instead.
PASTE_SECONDS = 1.0

# Longest side of a screenshot sent to the vision model, in pixels. Retina
# windows are captured at full detail up to this size.
MAX_IMAGE_SIDE = 2000

# Buttons that switch something on or off; clicking one twice undoes it.
# Matched in click_on's description, normalised to one name per kind.
TOGGLES = [
    (re.compile(r"\b(dislike|thumbs?[ -]?down)\b", re.I), "dislike"),
    (re.compile(r"\b(like|thumbs?[ -]?up)\b", re.I), "like"),
    (re.compile(r"\b(subscribe|follow|favou?rite|heart|star|bookmark|upvote|downvote)\b", re.I), None),
    (re.compile(r"\b(check ?box|toggle|switch)\b", re.I), None),
]

# Friendly key names -> pyautogui names.
KEY_ALIASES = {
    "cmd": "command", "⌘": "command", "control": "ctrl", "opt": "option", "alt": "option",
    "return": "enter", "esc": "escape", "del": "delete", "spacebar": "space",
}


def _gui():
    if sys.platform != "darwin":
        raise RuntimeError("Screen control is only supported on macOS.")
    import pyautogui
    return pyautogui


class NotInFront(Exception):
    """The task's app could not be brought to the front, so nothing was done."""


def _focus_target() -> None:
    """Bring the task's app to the front before acting; raise if that fails."""
    app = target_app()
    if app and not macos_windows.focus_app(app):
        state = "is not open" if not macos_windows.is_running(app) else "could not be brought to the front"
        raise NotInFront(f"{app} {state}, so nothing was done.")


@dataclass(frozen=True)
class Capture:
    image_b64: str
    width: int          # image size in pixels
    height: int
    origin_x: int       # where the image's top-left is on the screen, in points
    origin_y: int
    scale: float        # screen points per image pixel (0.5 on Retina)
    what: str           # "the Google Chrome window" or "the screen"

    def to_screen(self, x: int, y: int) -> tuple[int, int]:
        """An image-pixel position as a global screen point."""
        return self.origin_x + round(x * self.scale), self.origin_y + round(y * self.scale)


def _capture() -> Capture:
    """Screenshot the task app's window (any display), or the main screen."""
    from PIL import Image

    app = target_app()
    window = macos_windows.front_window(app) if app else None
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "capture.png")
        if window:
            command = ["screencapture", "-x", "-o", "-l", str(window.id), path]
            points_wide, origin, what = window.width, (window.x, window.y), f"the {app} window"
        else:
            command = ["screencapture", "-x", path]
            points_wide, origin, what = _gui().size()[0], (0, 0), "the screen"
        subprocess.run(command, check=True, timeout=15)
        image = Image.open(path).convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))  # keeps Retina detail, caps the size
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return Capture(base64.b64encode(buffer.getvalue()).decode("ascii"), image.width, image.height,
                   *origin, points_wide / image.width, what)


def _toggle_kind(target: str) -> str | None:
    """'like', 'subscribe', ... if the description names an on/off button."""
    for pattern, name in TOGGLES:
        match = pattern.search(target)
        if match:
            return name or match.group(1).lower()
    return None


def _ask_vision(question: str) -> tuple[str, Capture]:
    """Capture the task's window and ask the vision model about it."""
    from models.models import vision_llm

    capture = _capture()
    prompt = load_prompt("vision", what=capture.what, width=capture.width,
                         height=capture.height, question=question)
    response = vision_llm.invoke([HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{capture.image_b64}"}},
    ])])
    return str(response.content), capture


def _click_at(x: int, y: int, double: bool) -> str:
    """Click at a global screen point."""
    if not macos_windows.on_a_display(x, y):
        return f"Position {x},{y} is not on any screen."
    _gui().click(x, y, clicks=2 if double else 1)
    time.sleep(SETTLE_SECONDS)
    return f"{'Double-clicked' if double else 'Clicked'} at {x},{y}."


@tool
def screenshot(question: str) -> str:
    """
    Takes a screenshot of the app being worked in and answers a question
    about it, e.g. "Is a chat open? What is in the message box?"
    """
    try:
        _focus_target()
        answer, _ = _ask_vision(question)
        return answer
    except NotInFront as e:
        return str(e)
    except Exception as e:
        return f"Could not take a screenshot: {e}"


@tool
def type_text(text: str) -> str:
    """Types text where the cursor is, by pasting it (works with any language and emoji)."""
    try:
        _focus_target()
        gui = _gui()
        saved = subprocess.run(["pbpaste"], capture_output=True, text=True).stdout
        subprocess.run(["pbcopy"], input=text, text=True, check=True)
        gui.hotkey("command", "v")
        time.sleep(PASTE_SECONDS)
        subprocess.run(["pbcopy"], input=saved, text=True, check=True)
        return f"Typed {len(text)} characters."
    except NotInFront as e:
        return str(e)
    except Exception as e:
        return f"Failed to type text: {e}"


@tool
def press_keys(keys: str) -> str:
    """
    Presses a key or a shortcut, e.g. "enter", "tab", "escape", "cmd+f",
    "cmd+n", "cmd+shift+t". Keys joined with + are pressed together.
    """
    try:
        gui = _gui()
        parts = [KEY_ALIASES.get(k.strip().lower(), k.strip().lower()) for k in keys.split("+")]
        unknown = [k for k in parts if k not in gui.KEYBOARD_KEYS]
        if not parts or unknown:
            return f"Unknown key(s): {', '.join(unknown) or keys}."
        _focus_target()
        gui.hotkey(*parts)
        time.sleep(SETTLE_SECONDS)
        return f"Pressed {keys}."
    except NotInFront as e:
        return str(e)
    except Exception as e:
        return f"Failed to press {keys}: {e}"


@tool
def click_on(target: str, double: bool = False) -> str:
    """
    Finds something on screen from a description and clicks it, e.g.
    "the search result named Alex", "the message input box", "the Send button".
    double=True double-clicks. No coordinates needed. On/off buttons (Like,
    Subscribe, checkboxes ...) can be clicked only once per task.
    """
    toggle = _toggle_kind(target)
    if toggle and was_toggled(toggle):
        return (f"Not clicked: the {toggle} button was already clicked once in this task, and "
                "clicking it again would undo it. Check its state with a screenshot instead.")
    try:
        _focus_target()
        answer, capture = _ask_vision(
            f"Find this: {target}.\n"
            "Reply with ONLY its centre as x,y (for example 640,410), "
            "or NOT FOUND if it is not visible."
        )
    except NotInFront as e:
        return str(e)
    except Exception as e:
        return f"Could not take a screenshot: {e}"
    match = re.search(r"(\d+)\s*,\s*(\d+)", answer)
    if "NOT FOUND" in answer.upper() or not match:
        return f"Could not find {target} on screen."
    x, y = int(match.group(1)), int(match.group(2))
    if not (0 <= x < capture.width and 0 <= y < capture.height):
        return f"Could not find {target} on screen (position {x},{y} is outside {capture.what})."
    try:
        result = _click_at(*capture.to_screen(x, y), double)
    except Exception as e:
        return f"Failed to click: {e}"
    if toggle and result.startswith(("Clicked", "Double-clicked")):
        remember_toggle(toggle)
    return f"{result} ({target})"
