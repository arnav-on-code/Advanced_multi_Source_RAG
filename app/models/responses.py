from typing import Any, Literal

from pydantic import BaseModel, Field

from app.models.requests import SourceType


class SourceCitation(BaseModel):
    """Structured citation information for a retrieved source."""

    source_type: SourceType

    source: str | None = None

    file_name: str | None = None

    page: int | None = Field(
        default=None,
        ge=1,
    )

    chunk_id: str | None = None

    citation: dict[str, Any] | None = None


class IngestResponse(BaseModel):
    """Response returned after source ingestion."""

    status: Literal["success"] = "success"

    source_type: SourceType

    documents: int = Field(
        ge=0,
    )

    chunks: int = Field(
        ge=0,
    )

    message: str


class QuerySource(BaseModel):
    """Evidence source returned with a RAG answer."""

    rank: int = Field(
        ge=1,
    )

    rerank_score: float

    original_rank: int = Field(
        ge=1,
    )

    original_score: float

    source_type: SourceType | None = None

    source: str | None = None

    file_name: str | None = None

    page: int | None = Field(
        default=None,
        ge=1,
    )

    citation: dict[str, Any] | None = None

    chunk_id: str | None = None


class QueryResponse(BaseModel):
    """Final response returned by the RAG query endpoint."""

    answer: str

    sources: list[QuerySource] = Field(
        default_factory=list,
    )

    retrieved_documents: int = Field(
        ge=0,
    )


class HealthResponse(BaseModel):
    """API health and readiness response."""

    status: Literal[
        "healthy",
        "ready",
        "degraded",
        "unhealthy",
    ]

    service: str

    version: str

    environment: str

    dependencies: dict[str, str] | None = None