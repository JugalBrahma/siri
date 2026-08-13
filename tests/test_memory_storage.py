"""Tests for local memory persistence; no Pinecone or LLM calls are made."""

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from memory.data_model import FactCategory, SemanticFact
from memory.episode_queue import JsonEpisodeQueue
from memory.metadata_repository import JsonSemanticFactRepository


class SemanticMemoryStorageTests(unittest.TestCase):
    def test_fact_repository_persists_and_filters_reliable_facts(self):
        with TemporaryDirectory() as directory:
            repository_path = Path(directory) / "semantic_facts.json"
            repository = JsonSemanticFactRepository(repository_path)
            now = datetime.now(timezone.utc).isoformat()
            fact = SemanticFact(
                fact_id="fact-1",
                user_id="user-1",
                statement="The user prefers concise answers.",
                category=FactCategory.PREFERENCE,
                confidence=0.8,
                observation_count=2,
                source_episode_ids=["ep-1", "ep-2"],
                first_observed=now,
                last_confirmed=now,
            )
            repository.save(fact.fact_id, fact)

            reloaded_repository = JsonSemanticFactRepository(repository_path)
            facts = reloaded_repository.query_by_user(
                user_id="user-1",
                min_confidence=0.65,
                categories=[FactCategory.PREFERENCE.value],
            )

            self.assertEqual(len(facts), 1)
            self.assertEqual(facts[0].statement, fact.statement)
            self.assertEqual(facts[0].format_for_injection(), "- The user prefers concise answers. [medium confidence]")

    def test_episode_queue_keeps_unprocessed_summaries(self):
        with TemporaryDirectory() as directory:
            queue = JsonEpisodeQueue(Path(directory) / "pending_episodes.json")
            queue.append({"episode_id": "ep-1", "summary": "First summary", "date": "2026-08-13"})
            queue.append({"episode_id": "ep-2", "summary": "Second summary", "date": "2026-08-13"})

            queue.remove(["ep-1"])

            self.assertEqual(queue.peek_all(), [
                {"episode_id": "ep-2", "summary": "Second summary", "date": "2026-08-13"},
            ])


if __name__ == "__main__":
    unittest.main()
