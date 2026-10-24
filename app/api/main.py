"""
FastAPI application entry point.
Provides documented API for document management and RAG queries.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes.documents import router as documents_router
from app.api.routes.health import router as health_router
from app.api.routes.query import router as query_router
from app.config.logging_config import setup_logging
from app.config.settings import settings


# Configure logging before application startup.
setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""

    logger.info("Starting %s", settings.app_name)

    capabilities = settings.report_capabilities()

    for key, value in capabilities.items():
        logger.info("  %s: %s", key, value)

    yield

    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    description=(
        "Multimodal RAG Knowledge Assistant API. "
        "Supports document ingestion, hybrid retrieval, and grounded answer generation."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# Development configuration: allow any origin, without credentialed requests.
# Restrict origins before deploying to production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Return a sanitized response for unexpected server errors."""

    logger.error(
        "Unhandled exception on %s",
        request.url.path,
        exc_info=True,
    )

    detail_message = (
        str(exc)
        if settings.debug
        else "An unexpected internal error occurred."
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": detail_message,
        },
    )


# Register API routes.
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(query_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )