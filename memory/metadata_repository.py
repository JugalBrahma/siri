"""Disk-backed metadata for semantic facts.

Pinecone stores vectors and light metadata used for similarity search. This
repository stores the complete fact record, including its confidence evolution
log and source episode IDs.
"""

import json
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional

from memory.data_model import FactCategory, SemanticFact


class JsonSemanticFactRepository:
    """A small, local metadata repository suitable for one Siri user/profile."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = RLock()

    def get(self, fact_id: str) -> Optional[SemanticFact]:
        with self._lock:
            raw_fact = self._read_facts().get(fact_id)
            return self._deserialize(raw_fact) if raw_fact else None

    def save(self, fact_id: str, fact: SemanticFact) -> None:
        with self._lock:
            facts = self._read_facts()
            facts[fact_id] = self._serialize(fact)
            self._write_facts(facts)

    def query_by_user(
        self,
        user_id: str,
        min_confidence: float,
        categories: Optional[List[str]] = None,
    ) -> List[SemanticFact]:
        requested_categories = set(categories) if categories else None
        with self._lock:
            facts = []
            for raw_fact in self._read_facts().values():
                try:
                    facts.append(self._deserialize(raw_fact))
                except (KeyError, TypeError, ValueError):
                    # A single invalid fact should not hide all valid facts.
                    continue

        results = [
            fact
            for fact in facts
            if fact.user_id == user_id
            and fact.confidence >= min_confidence
            and (
                requested_categories is None
                or str(fact.category.value) in requested_categories
            )
        ]
        return sorted(results, key=lambda fact: fact.confidence, reverse=True)

    def clear(self, user_id: Optional[str] = None) -> int:
        """Clear facts from repository. If user_id is provided, only clear for that user."""
        with self._lock:
            facts = self._read_facts()
            if user_id is None:
                count = len(facts)
                self._write_facts({})
                return count
            
            remaining = {
                fid: f for fid, f in facts.items()
                if f.get("user_id") != user_id
            }
            cleared_count = len(facts) - len(remaining)
            self._write_facts(remaining)
            return cleared_count

    def _read_facts(self) -> Dict[str, Dict]:
        if not self.path.exists():
            return {}

        try:
            with self.path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError):
            # A damaged local cache should not break an assistant turn. Keep
            # the file for investigation and begin with an empty repository.
            return {}

        facts = payload.get("facts", {}) if isinstance(payload, dict) else {}
        return facts if isinstance(facts, dict) else {}

    def _write_facts(self, facts: Dict[str, Dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        payload = {"version": 1, "facts": facts}
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        temporary_path.replace(self.path)

    @staticmethod
    def _serialize(fact: SemanticFact) -> Dict:
        category = fact.category.value if isinstance(fact.category, FactCategory) else str(fact.category)
        return {
            "fact_id": fact.fact_id,
            "user_id": fact.user_id,
            "statement": fact.statement,
            "category": category,
            "confidence": fact.confidence,
            "observation_count": fact.observation_count,
            "source_episode_ids": list(fact.source_episode_ids),
            "first_observed": fact.first_observed,
            "last_confirmed": fact.last_confirmed,
            "evolution_log": list(fact.evolution_log),
        }

    @staticmethod
    def _deserialize(raw_fact: Dict) -> SemanticFact:
        category_value = raw_fact.get("category", FactCategory.GENERAL.value)
        try:
            category = FactCategory(category_value)
        except ValueError:
            category = FactCategory.GENERAL

        return SemanticFact(
            fact_id=raw_fact["fact_id"],
            user_id=raw_fact["user_id"],
            statement=raw_fact["statement"],
            category=category,
            confidence=float(raw_fact["confidence"]),
            observation_count=int(raw_fact["observation_count"]),
            source_episode_ids=list(raw_fact.get("source_episode_ids", [])),
            first_observed=raw_fact["first_observed"],
            last_confirmed=raw_fact["last_confirmed"],
            evolution_log=list(raw_fact.get("evolution_log", [])),
        )
