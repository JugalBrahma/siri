from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SIMILARITY_DUPLICATE = 0.92   # same fact restated -> strengthen, don't add
SIMILARITY_RELATED = 0.75     # related topic -> compare before deciding


class PineconeSemanticMemoryStore:
    def __init__(
        self,
        user_id: str,
        index,                 # pinecone Index, vectors live in namespace="semantic"
        metadata_db,           # .get(fact_id) / .save(fact_id, fact) / .query_by_user(...)
        embed_fn,              # text -> list[float]
        compare_fn,            # (old_statement, new_statement) -> "confirms" | "contradicts" | "unrelated"
        confidence_threshold: float = 0.65,
    ):
        self.user_id = user_id
        self.index = index
        self.metadata_db = metadata_db
        self.embed_fn = embed_fn
        self.compare_fn = compare_fn
        self.confidence_threshold = confidence_threshold
        self._last_retrieval_metrics: Dict[str, Optional[float]] = {
            "fact_count": 0,
            "confidence_min": None,
            "confidence_max": None,
        }

    # ---------- write path ----------

    def merge_fact(self, new_fact) -> str:
        """
        Vector-similarity dedup, replacing the old hash-based merge_facts.
        Returns the action taken: "added" | "strengthened" | "weakened"
        """
        embedding = self.embed_fn(new_fact.statement)
        match = self._find_closest(embedding)

        if match is None or match["score"] < SIMILARITY_RELATED:
            self._upsert(new_fact, embedding)
            return "added"

        existing = self.metadata_db.get(match["fact_id"])
        if existing is None:
            # The vector can outlive a local metadata file (for example after a
            # restore). Treat the new fact as authoritative and repair it.
            self._upsert(new_fact, embedding)
            return "added"

        if match["score"] >= SIMILARITY_DUPLICATE:
            # Near-certain restatement -- no need to ask the LLM.
            return self._strengthen(existing, new_fact)

        # Related but not identical: could be a refinement or a
        # contradiction -- worth a real comparison, not just the score.
        verdict = self.compare_fn(existing.statement, new_fact.statement)

        if verdict == "confirms":
            return self._strengthen(existing, new_fact)
        if verdict == "contradicts":
            existing.weaken(reason=f"contradicted by: {new_fact.statement}")
            self._save(existing)
            # The new statement may still be a real, separate fact.
            self._upsert(new_fact, embedding)
            return "weakened"

        # Related topic, genuinely different fact.
        self._upsert(new_fact, embedding)
        return "added"

    def _strengthen(self, existing, new_fact) -> str:
        episode_id = new_fact.source_episode_ids[-1] if new_fact.source_episode_ids else ""
        existing.strengthen(episode_id)
        self._save(existing)
        return "strengthened"

    def _find_closest(self, embedding: list[float]) -> Optional[dict]:
        result = self.index.query(
            vector=embedding,
            top_k=1,
            namespace="semantic",
            filter={"user_id": self.user_id},
            include_metadata=False,
        )
        if not result.matches:
            return None
        top = result.matches[0]
        return {"fact_id": top.id, "score": top.score}

    def _upsert(self, fact, embedding: list[float]) -> None:
        self.index.upsert(
            vectors=[{
                "id": fact.fact_id,
                "values": embedding,
                "metadata": {
                    "user_id": self.user_id,
                    "category": fact.category,
                    "confidence": fact.confidence,
                    "last_confirmed": fact.last_confirmed,
                },
            }],
            namespace="semantic",
        )
        self.metadata_db.save(fact.fact_id, fact)

    def _save(self, fact) -> None:
        """Confidence changed -- persist full record, refresh Pinecone's light copy too."""
        self.metadata_db.save(fact.fact_id, fact)
        self.index.update(
            id=fact.fact_id,
            set_metadata={"confidence": fact.confidence, "last_confirmed": fact.last_confirmed},
            namespace="semantic",
        )

    # ---------- read path ----------

    def get_reliable_facts(
        self,
        categories: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
    ) -> List:
        """
        'Give me everything above threshold' is a filter, not a similarity
        question -- this goes through metadata_db directly, same principle
        as session lookup bypassing the vector index earlier.
        """
        threshold = min_confidence if min_confidence is not None else self.confidence_threshold
        return self.metadata_db.query_by_user(
            user_id=self.user_id,
            min_confidence=threshold,
            categories=categories,
        )

    def format_for_injection(
        self,
        categories: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
    ) -> str:
        """Return reliable facts as one safe, turn-scoped prompt fragment.

        This method is deliberately called by ``GraphBuilder`` once before a
        graph turn starts. Workers receive the resulting string through their
        supervisor state and never query the store themselves.
        """
        facts = self.get_reliable_facts(
            categories=categories,
            min_confidence=min_confidence,
        )
        confidences = [fact.confidence for fact in facts if hasattr(fact, "confidence")]
        self._last_retrieval_metrics = {
            "fact_count": len(facts),
            "confidence_min": min(confidences) if confidences else None,
            "confidence_max": max(confidences) if confidences else None,
        }
        formatted_facts = [
            fact.format_for_injection()
            for fact in facts
            if hasattr(fact, "format_for_injection")
        ]
        return "\n".join(formatted_facts)

    def get_last_retrieval_metrics(self) -> Dict[str, Optional[float]]:
        """Return safe aggregate metrics for logging; never return fact text."""
        return dict(self._last_retrieval_metrics)

    def clear_memory(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Clear all facts for user from vector index and metadata database."""
        target_user = user_id or self.user_id
        cleared_facts = self.metadata_db.clear(user_id=target_user)
        try:
            self.index.delete(
                namespace="semantic",
                filter={"user_id": target_user},
            )
        except Exception:
            try:
                self.index.delete(delete_all=True, namespace="semantic")
            except Exception:
                pass

        self._last_retrieval_metrics = {
            "fact_count": 0,
            "confidence_min": None,
            "confidence_max": None,
        }
        return {"status": "success", "cleared_facts": cleared_facts, "user_id": target_user}
