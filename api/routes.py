import asyncio
import json
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from api.schemas import (
    AddFactRequest,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MemoryResponse,
)
from api.service import SiriAgentService
from services.token_tracker import LangSmithTokenTracker

router = APIRouter(tags=["Siri Agent API"])


def get_agent_service(request: Request) -> SiriAgentService:
    """Dependency that extracts the SiriAgentService from application state."""
    service = getattr(request.app.state, "agent_service", None)
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Siri Agent Service is not initialized.",
        )
    return service


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the operational health of the Siri API and memory backend.",
)
async def health_check(
    service: SiriAgentService = Depends(get_agent_service),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version="1.0.0",
        semantic_memory_enabled=service.memory_store is not None,
        default_user_id=service.default_user_id,
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat with Siri multi-agent",
    description="Send a message or conversation history to Siri and receive the multi-agent response.",
)
async def chat_endpoint(
    request_data: ChatRequest,
    service: SiriAgentService = Depends(get_agent_service),
) -> ChatResponse:
    try:
        messages = request_data.to_message_list()
        result = await asyncio.to_thread(
            service.run_turn,
            messages=messages,
            user_id=request_data.user_id,
        )
        return ChatResponse(
            response=result["response"],
            hop_count=result["hop_count"],
            user_id=result["user_id"],
            status=result["status"],
            token_usage=result.get("token_usage"),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution error: {str(exc)}",
        ) from exc


@router.post(
    "/chat/stream",
    summary="Stream conversation execution",
    description="Stream real-time graph node transitions and final response via Server-Sent Events (SSE).",
)
async def chat_stream_endpoint(
    request_data: ChatRequest,
    service: SiriAgentService = Depends(get_agent_service),
):
    """Stream execution progress using SSE format with exact LangSmith token metrics."""
    messages = request_data.to_message_list()
    active_user_id = request_data.user_id or service.default_user_id

    async def event_generator():
        try:
            if not service._is_initialized:
                service.initialize()

            initial_state = service.builder.create_turn_state(messages)
            tracker = LangSmithTokenTracker()
            config = {"callbacks": [tracker]}

            hop_count = 0
            final_state = initial_state

            yield f"data: {json.dumps({'event': 'start', 'user_id': active_user_id})}\n\n"

            for state in service.builder.graph.stream(initial_state, stream_mode="values", config=config):
                final_state = state
                hop_count += 1
                current_node = state.get("next", "")
                tracker.set_current_node(current_node)
                yield f"data: {json.dumps({'event': 'node_update', 'hop': hop_count, 'next': current_node})}\n\n"

            final_output = ""
            if "messages" in final_state and len(final_state["messages"]) > 0:
                final_output = final_state["messages"][-1].content

            if service.learning_service is not None:
                service.learning_service.submit_completed_turn(final_state.get("messages", []))

            usage_metrics = tracker.get_usage()
            yield f"data: {json.dumps({'event': 'complete', 'response': final_output, 'hops': max(0, hop_count - 1), 'token_usage': usage_metrics})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'event': 'error', 'detail': str(exc)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get(
    "/memory/{user_id}",
    response_model=MemoryResponse,
    summary="Inspect user semantic memory",
    description="Fetch active semantic memory facts and formatted injection profile for a given user.",
)
async def get_user_memory(
    user_id: str,
    service: SiriAgentService = Depends(get_agent_service),
) -> MemoryResponse:
    info = service.get_memory_info(user_id=user_id)
    return MemoryResponse(
        user_id=info["user_id"],
        fact_count=info["fact_count"],
        facts=info["facts"],
        formatted_profile=info["formatted_profile"],
    )


@router.post(
    "/memory/{user_id}",
    summary="Add fact to user semantic memory",
    description="Store or merge an individual semantic fact for a given user.",
)
@router.post(
    "/memory/{user_id}/facts",
    summary="Add fact to user semantic memory",
    description="Store or merge an individual semantic fact for a given user.",
)
async def add_user_memory_fact(
    user_id: str,
    payload: AddFactRequest,
    service: SiriAgentService = Depends(get_agent_service),
):
    text = payload.statement or payload.content or ""
    if not text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Fact statement or content is required.")
    return service.add_user_fact(
        user_id=user_id,
        statement=text.strip(),
        category=payload.category or "preference",
        confidence=payload.confidence if payload.confidence is not None else 0.95,
    )


@router.delete(
    "/memory/{user_id}/facts/{fact_id}",
    summary="Delete single memory fact",
    description="Delete a specific semantic fact by its fact_id.",
)
async def delete_single_fact(
    user_id: str,
    fact_id: str,
    service: SiriAgentService = Depends(get_agent_service),
):
    res = service.delete_user_fact(user_id=user_id, fact_id=fact_id)
    if not res.get("deleted"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Fact with ID '{fact_id}' not found.")
    return res


@router.delete(
    "/memory/{user_id}",
    summary="Clear user semantic memory",
    description="Clear all stored semantic facts and queued learning episodes for a given user.",
)
async def clear_user_memory(
    user_id: str,
    service: SiriAgentService = Depends(get_agent_service),
):
    return service.clear_user_memory(user_id=user_id)


@router.delete(
    "/memory",
    summary="Clear default semantic memory",
    description="Clear all stored semantic facts and queued learning episodes for the default user.",
)
async def clear_default_memory(
    service: SiriAgentService = Depends(get_agent_service),
):
    return service.clear_user_memory(user_id=service.default_user_id)
