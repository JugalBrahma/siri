"""Local chat history: only what the user saw.

Each finished turn stores two rows — the user's message and Siri's final
answer. Agent reasoning, routing decisions, and worker outputs are never
written here; in-progress run data lives in the LangGraph checkpoint tables
and is deleted once the turn finishes.
"""

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def default_db_path() -> Path:
    """The single local SQLite file used for checkpoints and chat history."""
    explicit = os.getenv("SIRI_DB_PATH")
    if explicit:
        return Path(explicit)
    return Path(os.getenv("SEMANTIC_MEMORY_DATA_DIR", "memory_data")) / "siri.db"


class ChatHistory:
    """Thread-safe store of user messages and final answers, keyed by thread_id."""

    def __init__(self, db_path: str | Path):
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )"""
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_messages_thread ON chat_messages(thread_id, id)"
            )

    def add_turn(self, thread_id: str, question: str, answer: str) -> None:
        """Store one finished turn: the user's message and Siri's answer."""
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (thread_id, "user", question, now),
            (thread_id, "assistant", answer, now),
        ]
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT INTO chat_messages (thread_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                rows,
            )

    def recent(self, thread_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """Return the last `limit` messages of a thread, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM chat_messages WHERE thread_id = ? ORDER BY id DESC LIMIT ?",
                (thread_id, limit),
            ).fetchall()
        return [{"role": role, "content": content} for role, content in reversed(rows)]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def clear(self, thread_id: str) -> int:
        """Delete a thread's history; returns how many messages were removed."""
        with self._lock, self._conn:
            cursor = self._conn.execute("DELETE FROM chat_messages WHERE thread_id = ?", (thread_id,))
        return cursor.rowcount
