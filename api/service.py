"""Service layer coordinating GraphBuilder, semantic memory, and background learning."""

import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

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
from memory.observability import MemoryObserver
from memory.runtime import create_configured_memory_store

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
            Dict containing final output, hop count, and user ID.
        """
        if not self._is_initialized:
            self.initialize()

        active_user_id = user_id or self.default_user_id
        initial_state = self.builder.create_turn_state(messages)

        hop_count = 0
        final_state = initial_state
        for state in self.builder.graph.stream(initial_state, stream_mode="values"):
            final_state = state
            hop_count += 1

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
        }

    def get_memory_info(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve stored semantic memory profile for a user."""
        target_user = user_id or self.default_user_id
        if self.memory_store is None:
            return {
                "user_id": target_user,
                "fact_count": 0,
                "facts": [],
                "formatted_profile": "",
            }

        formatted = self.memory_store.format_for_injection()
        facts = []
        if hasattr(self.memory_store, "get_reliable_facts"):
            raw_facts = self.memory_store.get_reliable_facts(min_confidence=0.0)
            facts = [
                f.model_dump() if hasattr(f, "model_dump") else (
                    f.__dict__ if hasattr(f, "__dict__") else dict(f)
                )
                for f in raw_facts
            ]

        return {
            "user_id": target_user,
            "fact_count": len(facts),
            "facts": facts,
            "formatted_profile": formatted,
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
