"""Which app the current screen task is working in.

The screen agents run their tools inside ``screen_target(app)``. The tools
read the app from here to bring it to the front and to screenshot only its
window. ``open_app`` sets it when it opens an app, so the agent reads it
back afterwards and keeps it for the next round.

Outside a ``screen_target`` block there is no target, and the tools fall back
to the whole main screen.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional, Set


@dataclass
class ScreenTarget:
    app: str = ""
    # On/off buttons already clicked in this task ("like", "subscribe", ...).
    toggled: Set[str] = field(default_factory=set)


_current: ContextVar[Optional[ScreenTarget]] = ContextVar("screen_target", default=None)


@contextmanager
def screen_target(app: str = "", toggled: Iterable[str] = ()) -> Iterator[ScreenTarget]:
    target = ScreenTarget(app, set(toggled))
    token = _current.set(target)
    try:
        yield target
    finally:
        _current.reset(token)


def target_app() -> str:
    target = _current.get()
    return target.app if target else ""


def was_toggled(kind: str) -> bool:
    target = _current.get()
    return target is not None and kind in target.toggled


def remember_toggle(kind: str) -> None:
    target = _current.get()
    if target is not None:
        target.toggled.add(kind)


def set_target_app(app: str) -> None:
    """Point the current screen task at `app` (no-op outside a screen task)."""
    target = _current.get()
    if target is not None:
        target.app = app
