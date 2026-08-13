"""Durable queue for completed-turn summaries waiting for distillation."""

import json
from pathlib import Path
from threading import RLock
from typing import Dict, List


class JsonEpisodeQueue:
    """Persist summaries before distillation so a restart does not lose them."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = RLock()

    def append(self, episode: Dict) -> None:
        with self._lock:
            episodes = self.peek_all()
            episodes.append(dict(episode))
            self._write(episodes)

    def peek_all(self) -> List[Dict]:
        with self._lock:
            if not self.path.exists():
                return []
            try:
                with self.path.open("r", encoding="utf-8") as file:
                    payload = json.load(file)
            except (OSError, json.JSONDecodeError):
                return []

            episodes = payload.get("episodes", []) if isinstance(payload, dict) else []
            return [dict(episode) for episode in episodes if isinstance(episode, dict)]

    def remove(self, episode_ids: List[str]) -> None:
        completed_ids = set(episode_ids)
        with self._lock:
            remaining = [
                episode
                for episode in self.peek_all()
                if episode.get("episode_id") not in completed_ids
            ]
            self._write(remaining)

    def _write(self, episodes: List[Dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump({"version": 1, "episodes": episodes}, file, ensure_ascii=False, indent=2)
        temporary_path.replace(self.path)
