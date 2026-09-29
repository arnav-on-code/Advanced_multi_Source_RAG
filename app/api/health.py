from fastapi import APIRouter

from app.core.config import settings
from app.models.responses import HealthResponse


router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get(
    "",
    response_model=HealthResponse,
)
async def health_check() -> HealthResponse:
    """
    Liveness check.

    Confirms that the API process is running.
    Designed for Docker, nginx, AWS, and orchestration
    health checks.
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

    Confirms that the application is initialized and
    able to accept requests.

    Dependency-specific checks can be added here when
    ChromaDB, embedding models, and the LLM are initialized
    as managed application services.
    """

    return HealthResponse(
        status="ready",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )