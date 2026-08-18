"""FastAPI Application for Siri Multi-Agent Assistant."""

from contextlib import asynccontextmanager
import logging
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import config.config  # Initialize tracing and environment config
from api.routes import router
from api.service import SiriAgentService

LOGGER = logging.getLogger("siri.api")


def create_app(agent_service: SiriAgentService | None = None) -> FastAPI:
    """Application factory for Siri FastAPI backend."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: initialize the multi-agent service
        service = agent_service or SiriAgentService()
        try:
            service.initialize()
            LOGGER.info("Siri Multi-Agent API initialized successfully.")
        except Exception as exc:
            LOGGER.error("Failed to initialize Siri Agent Service during startup: %s", exc)

        app.state.agent_service = service
        yield

        # Shutdown: release workers and executors
        LOGGER.info("Shutting down Siri Multi-Agent API...")
        service.shutdown()

    app = FastAPI(
        title="Siri Multi-Agent API",
        description="RESTful and streaming backend for the Siri LangGraph multi-agent system with semantic memory.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS configuration for Flutter, Mobile, and Web clients
    cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(router)

    @app.get("/", tags=["Root"])
    async def root():
        return {
            "name": "Siri Multi-Agent API",
            "version": "1.0.0",
            "docs_url": "/docs",
            "health_url": "/health",
        }

    return app


app = create_app()


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    uvicorn.run("api.app:app", host=host, port=port, reload=True, log_level=log_level)
