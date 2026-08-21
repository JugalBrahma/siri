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
    InterruptInfo,
    MemoryResponse,
    ResumeChatRequest,
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
            interrupt=result.get("interrupt"),
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
            config = {
                "callbacks": [tracker],
                "configurable": {"thread_id": str(active_user_id)},
            }

            hop_count = 0
            final_state = initial_state

            yield f"data: {json.dumps({'event': 'start', 'user_id': active_user_id})}\n\n"

            for state in service.builder.graph.stream(initial_state, stream_mode="values", config=config):
                final_state = state
                hop_count += 1
                current_node = state.get("next", "")
                tracker.set_current_node(current_node)
                yield f"data: {json.dumps({'event': 'node_update', 'hop': hop_count, 'next': current_node})}\n\n"

            # Check if execution paused on an interrupt
            current_graph_state = getattr(service.builder.graph, "get_state", lambda c: None)(config)
            interrupts = []
            if current_graph_state and hasattr(current_graph_state, "tasks"):
                for t in current_graph_state.tasks:
                    for i in getattr(t, "interrupts", []):
                        interrupts.append(getattr(i, "value", i))

            if interrupts:
                interrupt_payload = interrupts[0] if isinstance(interrupts[0], dict) else {"type": "ask_human", "question": str(interrupts[0])}
                usage_metrics = tracker.get_usage()
                yield f"data: {json.dumps({'event': 'interrupt', 'user_id': active_user_id, 'interrupt': interrupt_payload, 'hop': hop_count, 'token_usage': usage_metrics})}\n\n"
                return

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


@router.post(
    "/chat/resume",
    response_model=ChatResponse,
    summary="Resume interrupted conversation",
    description="Provide human input/clarification to resume a paused LangGraph workflow.",
)
async def chat_resume_endpoint(
    request_data: ResumeChatRequest,
    service: SiriAgentService = Depends(get_agent_service),
) -> ChatResponse:
    try:
        result = await asyncio.to_thread(
            service.resume_turn,
            user_id=request_data.user_id,
            human_response=request_data.response,
        )
        return ChatResponse(
            response=result["response"],
            hop_count=result["hop_count"],
            user_id=result["user_id"],
            status=result["status"],
            interrupt=result.get("interrupt"),
            token_usage=result.get("token_usage"),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent resume execution error: {str(exc)}",
        ) from exc


@router.post(
    "/chat/resume/stream",
    summary="Stream resumed conversation execution",
    description="Stream real-time graph execution after human input via Server-Sent Events (SSE).",
)
async def chat_resume_stream_endpoint(
    request_data: ResumeChatRequest,
    service: SiriAgentService = Depends(get_agent_service),
):
    """Stream execution progress after resuming from an interrupt."""
    active_user_id = request_data.user_id or service.default_user_id
    human_response = request_data.response

    async def event_generator():
        try:
            if not service._is_initialized:
                service.initialize()

            from langgraph.types import Command
            resume_command = Command(resume=human_response)

            tracker = LangSmithTokenTracker()
            config = {
                "callbacks": [tracker],
                "configurable": {"thread_id": str(active_user_id)},
            }

            hop_count = 0
            final_state = {}

            yield f"data: {json.dumps({'event': 'start', 'user_id': active_user_id})}\n\n"

            for state in service.builder.graph.stream(resume_command, stream_mode="values", config=config):
                final_state = state
                hop_count += 1
                current_node = state.get("next", "")
                tracker.set_current_node(current_node)
                yield f"data: {json.dumps({'event': 'node_update', 'hop': hop_count, 'next': current_node})}\n\n"

            current_graph_state = getattr(service.builder.graph, "get_state", lambda c: None)(config)
            interrupts = []
            if current_graph_state and hasattr(current_graph_state, "tasks"):
                for t in current_graph_state.tasks:
                    for i in getattr(t, "interrupts", []):
                        interrupts.append(getattr(i, "value", i))

            if interrupts:
                interrupt_payload = interrupts[0] if isinstance(interrupts[0], dict) else {"type": "ask_human", "question": str(interrupts[0])}
                usage_metrics = tracker.get_usage()
                yield f"data: {json.dumps({'event': 'interrupt', 'user_id': active_user_id, 'interrupt': interrupt_payload, 'hop': hop_count, 'token_usage': usage_metrics})}\n\n"
                return

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
