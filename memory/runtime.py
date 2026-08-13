"""Build the real semantic-memory store from environment configuration."""

import logging
import os
from pathlib import Path
from typing import Optional

from memory.metadata_repository import JsonSemanticFactRepository
from memory.store import PineconeSemanticMemoryStore


LOGGER = logging.getLogger("siri.semantic_memory")


def create_configured_memory_store(
    user_id: Optional[str] = None,
    data_directory: Optional[str | Path] = None,
) -> Optional[PineconeSemanticMemoryStore]:
    """Create the real store, or return ``None`` when memory is unconfigured.

    The implementation uses OpenAI embeddings, so both Pinecone and OpenAI
    credentials are required. Siri continues without semantic memory if either
    is missing; it never falls back to a fake store for real user queries.
    """
    if not os.getenv("PINECONE_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        LOGGER.info(
            "semantic memory disabled: PINECONE_API_KEY and OPENAI_API_KEY are required",
        )
        return None

    try:
        from memory.client import get_embedding, pinecone_index
        from memory.comparison import compare_semantic_facts
    except Exception as error:
        LOGGER.warning("semantic memory disabled: client initialization failed (%s)", type(error).__name__)
        return None

    if pinecone_index is None:
        LOGGER.warning("semantic memory disabled: Pinecone index is unavailable")
        return None

    resolved_user_id = user_id or os.getenv("SIRI_USER_ID", "siri_user")
    directory = Path(data_directory or os.getenv("SEMANTIC_MEMORY_DATA_DIR", "memory_data"))
    repository = JsonSemanticFactRepository(directory / "semantic_facts.json")

    return PineconeSemanticMemoryStore(
        user_id=resolved_user_id,
        index=pinecone_index,
        metadata_db=repository,
        embed_fn=get_embedding,
        compare_fn=compare_semantic_facts,
        confidence_threshold=0.65,
    )
