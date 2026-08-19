"""Pydantic schemas for Siri FastAPI request/response contracts."""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


class ChatMessage(BaseModel):
    """Single chat message representation."""
    role: Literal["user", "human", "assistant", "ai", "system"] = Field(
        ...,
        description="The role of the message sender.",
        examples=["user", "assistant"],
    )
    content: str = Field(
        ...,
        description="The text content of the message.",
        examples=["What is the weather in Tokyo?"],
    )


class ChatRequest(BaseModel):
    """Request schema for /chat endpoint.
    
    Accepts either a single `message` string or a list of `messages`.
    """
    message: Optional[str] = Field(
        default=None,
        description="A single user query message string.",
        examples=["What is the weather in Tokyo?"],
    )
    messages: Optional[List[ChatMessage]] = Field(
        default=None,
        description="List of prior conversation messages including current query.",
    )
    user_id: Optional[str] = Field(
        default="siri_user",
        description="Identifier for the user (used for personalized semantic memory).",
        examples=["siri_user"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "What is the weather in Tokyo?",
                "user_id": "siri_user",
            }
        }
    }

    @model_validator(mode="after")
    def validate_message_presence(self) -> "ChatRequest":
        if not self.message and not self.messages:
            raise ValueError("Either 'message' or 'messages' must be provided.")
        return self

    def to_message_list(self) -> List[Dict[str, str]]:
        """Normalize input to standard message dict format for GraphBuilder."""
        if self.messages:
            return [{"role": m.role, "content": m.content} for m in self.messages]
        return [{"role": "user", "content": self.message or ""}]


class ChatResponse(BaseModel):
    """Response schema for /chat endpoint."""
    response: str = Field(
        ...,
        description="The final sanitized text response from the multi-agent system.",
    )
    hop_count: int = Field(
        default=0,
        description="Number of graph hops / nodes traversed during agent execution.",
    )
    user_id: str = Field(
        default="siri_user",
        description="User ID for which memory was scoped.",
    )
    status: str = Field(
        default="success",
        description="Execution status.",
    )
    token_usage: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Exact token usage and node breakdown from LangSmith/LangChain.",
    )


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""
    status: str = "ok"
    version: str = "1.0.0"
    semantic_memory_enabled: bool = False
    default_user_id: str = "siri_user"


class MemoryFactItem(BaseModel):
    """Representation of an individual semantic memory fact."""
    fact_id: str
    content: Optional[str] = None
    statement: Optional[str] = None
    confidence: Optional[float] = None
    category: Optional[str] = None


class AddFactRequest(BaseModel):
    """Payload for manually injecting a semantic fact."""
    statement: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = "preference"
    confidence: Optional[float] = 0.95


class MemoryResponse(BaseModel):
    """Response schema for semantic memory inspection."""
    user_id: str
    fact_count: int
    facts: List[Dict[str, Any]] = []
    formatted_profile: Optional[str] = None
