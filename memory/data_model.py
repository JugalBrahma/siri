from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime, timezone
class FactCategory(str,Enum):
    """
    Categories for semantic facts.
    Enables targeted injection - only inject categories relevant to the current query.
    """
    BEHAVIOURAL = "behavioural"
    FINANCIAL = "financial"
    RISK= "risk"
    PREFERENCE ="preference"
    COMMUNICATION = "communication"
    GENERAL = "general"

class SemanticFact:
    """
    A Single distilled general truth about a user.
    Derived from patterns observed across multiple episodes.
    """

    def __init__(
        self,
        fact_id: str,
        user_id: str,
        statement: str,
        category: FactCategory,
        confidence: float,
        observation_count: int,
        source_episode_ids: List[str],
        first_observed: str,
        last_confirmed: str,
        evolution_log: Optional[List[Dict]] = None,
    ) -> None:
        # unique id for this distilled semantic fact
        self.fact_id = fact_id
        # the user this fact belongs to
        self.user_id = user_id
        # the distilled truth derived from multiple episodes
        self.statement = statement
        # category used for targeted injection and retrieval
        self.category = category
        # confidence score between 0.0 and 1.0
        self.confidence = confidence
        # number of episodes that support this fact
        self.observation_count = observation_count
        # episode ids where this fact was observed
        self.source_episode_ids = source_episode_ids
        # the first episode id where this fact was observed
        self.first_observed = first_observed
        # the last episode id where this fact was confirmed
        self.last_confirmed = last_confirmed
        # Log of every change to this fact:
        # [
        #    {"action": "created"|"confirmed"|"weakened"|"contradicted",
        #    "old_confidence": 0.7, "new_confidence": 0.8,
        #    "timestamp": "..." }
        # ]
        # Audit trail for fact evolution.
        self.evolution_log = evolution_log if evolution_log is not None else []

    def is_reliable(self, threshold: float = 0.7) -> bool:
        """Return True if this fact meets the confidence threshold for injection."""
        return self.confidence >= threshold

    def strengthen(self, episode_id: str):
        old_confidence = self.confidence
        self.confidence = self.confidence + 0.1
        if self.confidence > 1.0:
            self.confidence = 1.0
        self.observation_count += 1
        if episode_id and episode_id not in self.source_episode_ids:
            self.source_episode_ids.append(episode_id)
        self.last_confirmed = datetime.now(timezone.utc).isoformat()
        self.evolution_log.append({
            "action": "confirmed",
            "old_confidence": round(old_confidence, 3),
            "new_confidence": round(self.confidence, 3),
            "episode_id": episode_id,
            "timestamp": self.last_confirmed,
        })

    def weaken(self,reason: str = ""):
        old_confidence = self.confidence
        self.confidence = self.confidence - 0.2
        if self.confidence < 0.0:
            self.confidence = 0.0
        now = datetime.now(timezone.utc).isoformat()
        self.evolution_log.append({
            "action": "weakened",
            "old_confidence": round(old_confidence, 3),
            "new_confidence": round(self.confidence, 3),
            "reason": reason,
            "timestamp": now,
        })

    def format_for_injection(self) -> str:
        """Format as a single line for injection into the API context."""
        conf_label = (
            "[high confidence]" if self.confidence >= 0.85 else
            "[medium confidence]" if self.confidence >= 0.65 else
            "[low confidence]"
        )
        return f"- {self.statement} {conf_label}"
