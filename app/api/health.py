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
    """Return the liveness status of the API."""

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
    """Return the readiness status of the API."""

    return HealthResponse(
        status="ready",
        service=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )