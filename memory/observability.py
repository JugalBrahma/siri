"""Privacy-conscious metrics for semantic memory.

Events intentionally contain counts, sizes, timings, and exception *types*;
they never contain a user message, memory statement, fact ID, or API payload.
"""

import json
import logging
from typing import Mapping, Optional


class MemoryObserver:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger("siri.semantic_memory")

    def fetch_succeeded(self, elapsed_ms: float, profile: str, metrics: Mapping) -> None:
        self._emit(
            event="semantic_memory_fetch",
            status="success",
            latency_ms=round(elapsed_ms, 1),
            profile_characters=len(profile),
            fact_count=int(metrics.get("fact_count", 0)),
            confidence_min=metrics.get("confidence_min"),
            confidence_max=metrics.get("confidence_max"),
        )

    def fetch_failed(self, elapsed_ms: float, error: Exception) -> None:
        self._emit(
            event="semantic_memory_fetch",
            status="failure",
            latency_ms=round(elapsed_ms, 1),
            error_type=type(error).__name__,
        )

    def learning_completed(self, episode_count: int, fact_count: int, actions: Mapping[str, int]) -> None:
        self._emit(
            event="semantic_memory_distillation",
            status="success",
            episode_count=episode_count,
            fact_count=fact_count,
            actions=dict(actions),
        )

    def learning_failed(self, error: Exception) -> None:
        self._emit(
            event="semantic_memory_distillation",
            status="failure",
            error_type=type(error).__name__,
        )

    def _emit(self, **event: object) -> None:
        self.logger.info(json.dumps(event, sort_keys=True))
