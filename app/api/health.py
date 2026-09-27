from __future__ import annotations

from fastapi import APIRouter

from app.core.config import settings
from app.core.logging import get_logger
from app.models.responses import HealthResponse


router = APIRouter(
    prefix="/health",
    tags=["Health"],
)

logger = get_logger(__name__)


@router.get(
    "/",
    response_model=HealthResponse,
)
async def health_check() -> HealthResponse:
    """
    Basic application health check.

    This endpoint is intentionally lightweight so it can be
    used by Docker, nginx, AWS, or other infrastructure.
    """

    return HealthResponse(
        status="healthy",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
)
async def readiness_check() -> HealthResponse:
    """
    Readiness check.

    Indicates that the API process is running and ready
    to receive requests.

    Dependency-specific checks can be added later for
    ChromaDB, the LLM, embeddings, etc.
    """

    return HealthResponse(
        status="ready",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )