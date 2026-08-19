from datetime import datetime, timezone
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional
from uuid import uuid4

from agents.actionagent import action_agent
from agents.guardrail_agent import guardrail_agent
from agents.output_sanitizer import output_sanitizer_agent
from agents.research_agent import research_agent
from agents.supervisor_agent import (
    sub_actionSuperVisorAgent,
    sub_infoSuperVisorAgent,
    superVisorAgent,
)
from agents.weather_agent import weather_agent
from graph_builder.graph import GraphBuilder
from memory.data_model import FactCategory, SemanticFact
from memory.observability import MemoryObserver
from memory.runtime import create_configured_memory_store
from services.token_tracker import LangSmithTokenTracker

LOGGER = logging.getLogger("siri.api.service")


class SiriAgentService:
    """Encapsulates the Siri multi-agent runtime for API requests."""

    def __init__(
        self,
        default_user_id: Optional[str] = None,
        memory_store: Optional[Any] = None,
        memory_observer: Optional[MemoryObserver] = None,
        learning_service: Optional[Any] = None,
        builder: Optional[GraphBuilder] = None,
    ):
        self.default_user_id = default_user_id or os.getenv("SIRI_USER_ID", "siri_user")
        self.memory_observer = memory_observer or MemoryObserver()
        self.memory_store = memory_store
        self.learning_service = learning_service
        self.builder = builder
        self._is_initialized = False

    def initialize(self) -> None:
        """Initialize semantic memory, learning workers, and compile LangGraph."""
        if self._is_initialized:
            return

        # Initialize real memory store if not explicitly injected
        if self.memory_store is None:
            self.memory_store = create_configured_memory_store(user_id=self.default_user_id)

        # Initialize background learning service if memory store is active
        if self.memory_store is not None and self.learning_service is None:
            try:
                from memory.learning import MemoryLearningService

                self.learning_service = MemoryLearningService(
                    memory_store=self.memory_store,
                    user_id=self.default_user_id,
                    observer=self.memory_observer,
                )
                LOGGER.info("Semantic memory learning service successfully initialized.")
            except Exception as exc:
                LOGGER.warning("Could not initialize MemoryLearningService: %s", exc)

        # Initialize and build LangGraph multi-agent workflow
        if self.builder is None:
            self.builder = GraphBuilder(
                guardrail=guardrail_agent,
                supervisor=superVisorAgent,
                sub_actionsupervisor=sub_actionSuperVisorAgent,
                sub_infosupervisor=sub_infoSuperVisorAgent,
                action=action_agent,
                researcher=research_agent,
                weather=weather_agent,
                output_sanitizer=output_sanitizer_agent,
                memory_store=self.memory_store,
                memory_observer=self.memory_observer,
            )

        if self.builder.graph is None:
            self.builder.build()

        self._is_initialized = True
        LOGGER.info("SiriAgentService initialization complete.")

    def run_turn(
        self,
        messages: List[Dict[str, Any]],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a conversation turn through the compiled graph.
        
        Args:
            messages: List of conversation message dictionaries.
            user_id: Optional user identifier for memory scoping.
            
        Returns:
            Dict containing final output, hop count, user ID, and exact token usage.
        """
        if not self._is_initialized:
            self.initialize()

        active_user_id = user_id or self.default_user_id
        initial_state = self.builder.create_turn_state(messages)

        tracker = LangSmithTokenTracker()
        config = {"callbacks": [tracker]}

        hop_count = 0
        final_state = initial_state
        for state in self.builder.graph.stream(initial_state, stream_mode="values", config=config):
            final_state = state
            hop_count += 1
            current_node = state.get("next", "")
            tracker.set_current_node(current_node)

        hop_count = max(0, hop_count - 1)

        final_output = ""
        if "messages" in final_state and len(final_state["messages"]) > 0:
            final_output = final_state["messages"][-1].content

        # Schedule post-turn memory distillation off the reply path
        if self.learning_service is not None:
            self.learning_service.submit_completed_turn(final_state.get("messages", []))

        return {
            "response": final_output,
            "hop_count": hop_count,
            "user_id": active_user_id,
            "status": "success",
            "token_usage": tracker.get_usage(),
        }

    def get_memory_info(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve stored semantic memory profile for a user."""
        if not self._is_initialized:
            self.initialize()

        target_user = user_id or self.default_user_id
        if self.memory_store is None:
            return {
                "user_id": target_user,
                "fact_count": 0,
                "facts": [],
                "formatted_profile": "",
            }

        formatted = self.memory_store.format_for_injection(user_id=target_user)
        raw_facts = []
        if hasattr(self.memory_store, "get_reliable_facts"):
            raw_facts = self.memory_store.get_reliable_facts(min_confidence=0.0, user_id=target_user)

        facts = []
        for f in raw_facts:
            cat_val = f.category.value if hasattr(f.category, "value") else str(f.category)
            facts.append({
                "fact_id": f.fact_id,
                "user_id": f.user_id,
                "statement": f.statement,
                "content": f.statement,  # Compatible with UI components expecting .content
                "category": cat_val,
                "confidence": round(float(f.confidence), 2),
                "observation_count": getattr(f, "observation_count", 1),
                "first_observed": getattr(f, "first_observed", ""),
                "last_confirmed": getattr(f, "last_confirmed", ""),
                "created_at": getattr(f, "first_observed", ""),
                "evolution_log": getattr(f, "evolution_log", []),
            })

        return {
            "user_id": target_user,
            "fact_count": len(facts),
            "facts": facts,
            "formatted_profile": formatted,
        }

    def add_user_fact(
        self,
        user_id: Optional[str] = None,
        statement: str = "",
        category: str = "preference",
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """Manually store or merge a semantic fact into memory store."""
        if not self._is_initialized:
            self.initialize()

        target_user = user_id or self.default_user_id
        if self.memory_store is None:
            raise RuntimeError("Semantic memory store is not enabled or credentials are missing.")

        cat_enum = FactCategory.GENERAL
        for enum_item in FactCategory:
            if enum_item.value.lower() == category.lower():
                cat_enum = enum_item
                break

        now = datetime.now(timezone.utc).isoformat()
        fact = SemanticFact(
            fact_id=f"fact_{uuid4().hex[:12]}",
            user_id=target_user,
            statement=statement.strip(),
            category=cat_enum,
            confidence=max(0.0, min(1.0, float(confidence))),
            observation_count=1,
            source_episode_ids=[],
            first_observed=now,
            last_confirmed=now,
            evolution_log=[{
                "action": "created_manually",
                "old_confidence": 0.0,
                "new_confidence": round(float(confidence), 2),
                "timestamp": now,
            }],
        )

        action = self.memory_store.merge_fact(fact)
        return {
            "status": "success",
            "action": action,
            "fact_id": fact.fact_id,
            "statement": fact.statement,
            "content": fact.statement,
            "category": cat_enum.value,
            "confidence": fact.confidence,
            "user_id": target_user,
        }

    def delete_user_fact(self, user_id: Optional[str] = None, fact_id: str = "") -> Dict[str, Any]:
        """Delete an individual fact by its fact_id."""
        if not self._is_initialized:
            self.initialize()

        target_user = user_id or self.default_user_id
        if self.memory_store is None:
            raise RuntimeError("Semantic memory store is not enabled.")

        deleted = False
        if hasattr(self.memory_store, "delete_fact"):
            deleted = self.memory_store.delete_fact(fact_id)

        return {
            "status": "success" if deleted else "not_found",
            "fact_id": fact_id,
            "user_id": target_user,
            "deleted": deleted,
        }

    def clear_user_memory(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Clear memory facts and pending queues for a given user."""
        target_user = user_id or self.default_user_id
        cleared_facts = 0
        if self.memory_store is not None and hasattr(self.memory_store, "clear_memory"):
            res = self.memory_store.clear_memory(user_id=target_user)
            cleared_facts = res.get("cleared_facts", 0)

        if self.learning_service is not None and hasattr(self.learning_service, "queue"):
            self.learning_service.queue.clear()

        print(f"🗑️ [Memory Service] Cleared memory for user '{target_user}' ({cleared_facts} facts removed).")
        return {
            "status": "success",
            "message": f"Memory successfully cleared for user '{target_user}'",
            "cleared_facts": cleared_facts,
            "user_id": target_user,
        }

    def shutdown(self) -> None:
        """Gracefully shut down background threads and resources."""
        if self.learning_service is not None:
            LOGGER.info("Shutting down semantic memory background workers...")
            self.learning_service.shutdown(wait=True)
