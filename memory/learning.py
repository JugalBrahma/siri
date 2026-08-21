"""Asynchronous post-turn learning for semantic memory."""

from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

from memory.distiller import distil_episodes_to_facts, summarize_episode
from memory.episode_queue import JsonEpisodeQueue
from memory.observability import MemoryObserver


class MemoryLearningService:
    """Summarize completed turns, then distill queued summaries off the reply path."""

    def __init__(
        self,
        memory_store,
        user_id: str,
        queue: Optional[JsonEpisodeQueue] = None,
        observer: Optional[MemoryObserver] = None,
        batch_size: int = 2,
        executor: Optional[ThreadPoolExecutor] = None,
    ):
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2")

        self.memory_store = memory_store
        self.user_id = user_id
        data_directory = Path(os.getenv("SEMANTIC_MEMORY_DATA_DIR", "memory_data"))
        self.queue = queue or JsonEpisodeQueue(data_directory / "pending_episodes.json")
        self.observer = observer or MemoryObserver()
        self.batch_size = batch_size
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="semantic-memory",
        )

    def submit_completed_turn(self, messages: Iterable[Any]) -> Future:
        """Schedule learning after a reply; this method returns immediately."""
        snapshot = list(messages)
        print(f"[Memory Learning] Turn submitted ({len(snapshot)} messages) to background distillation worker.")
        return self._executor.submit(self._learn_from_turn, snapshot)

    def _learn_from_turn(self, messages: List[Any]) -> None:
        try:
            print("[Memory Learning] Generating episode summary for turn...")
            summary = summarize_episode(messages)
            if not summary:
                print("[Memory Learning] No summarizable content in turn.")
                return

            episode = {
                "episode_id": f"ep_{uuid4().hex}",
                "summary": summary,
                "date": datetime.now(timezone.utc).isoformat(),
            }
            self.queue.append(episode)
            queued_count = len(self.queue.peek_all())
            print(f"[Memory Learning] Episode queued: \"{summary}\" (Queue size: {queued_count}/{self.batch_size})")
            self._distill_if_ready()
        except Exception as error:
            print(f"[Memory Learning] Learning failed with error: {error}")
            self.observer.learning_failed(error)

    def _distill_if_ready(self) -> None:
        episodes = self.queue.peek_all()
        if len(episodes) < self.batch_size:
            print(f"[Memory Learning] Waiting for more episodes before distillation ({len(episodes)}/{self.batch_size} required).")
            return

        try:
            print(f"[Memory Learning] Batch size reached ({len(episodes)} episodes)! Distilling into semantic facts...")
            facts = distil_episodes_to_facts(episodes, user_id=self.user_id)
            print(f"[Memory Learning] Extracted {len(facts)} candidate facts: {[f.statement for f in facts]}")
            actions = Counter(self.memory_store.merge_fact(fact) for fact in facts)
            self.queue.remove([episode["episode_id"] for episode in episodes])
            print(f"[Memory Learning] Successfully merged {len(facts)} facts into store! Actions: {dict(actions)}")
            self.observer.learning_completed(len(episodes), len(facts), actions)
        except Exception as error:
            print(f"[Memory Learning] Distillation error: {error}")
            # Keep the episodes queued. A later completed turn can retry them.
            self.observer.learning_failed(error)

    def shutdown(self, wait: bool = False) -> None:
        """Release the worker when the host application is shutting down."""
        self._executor.shutdown(wait=wait, cancel_futures=False)
