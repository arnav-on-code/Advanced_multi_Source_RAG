from typing import Literal

from pydantic import BaseModel, Field, field_validator


SourceType = Literal[
    "pdf",
    "url",
    "youtube",
    "docx",
    "csv",
    "txt",
    "json",
]


class SourceFilter(BaseModel):
    """Optional metadata filters for retrieval."""

    source_type: SourceType | None = Field(
        default=None,
        description="Filter results by source type.",
    )

    source: str | None = Field(
        default=None,
        min_length=1,
        description="Filter by source path or URL.",
    )

    file_name: str | None = Field(
        default=None,
        min_length=1,
        description="Filter by file name.",
    )

    page: int | None = Field(
        default=None,
        ge=1,
        description="Filter PDF results by page number.",
    )

    @field_validator(
        "source",
        "file_name",
    )
    @classmethod
    def strip_values(
        cls,
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        value = value.strip()

        if not value:
            raise ValueError(
                "Filter value cannot be empty."
            )

        return value


class QueryRequest(BaseModel):
    """Request body for the RAG query endpoint."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Question to ask the RAG system.",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of final retrieved results.",
    )

    use_reranking: bool = Field(
        default=True,
        description="Whether to apply cross-encoder re-ranking.",
    )

    metadata_filter: SourceFilter | None = Field(
        default=None,
        description="Optional source metadata filter.",
    )

    @field_validator("query")
    @classmethod
    def validate_query(
        cls,
        value: str,
    ) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "Query cannot contain only whitespace."
            )

        return value


class IngestRequest(BaseModel):
    """Request body for source ingestion."""

    source_type: SourceType = Field(
        ...,
        description="Type of source to ingest.",
    )

    source: str = Field(
        ...,
        min_length=1,
        description="File path, URL, or source identifier.",
    )

    @field_validator("source")
    @classmethod
    def validate_source(
        cls,
        value: str,
    ) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "Source cannot be empty."
            )

        return value


class HealthCheckRequest(BaseModel):
    """
    Optional health-check configuration.

    Can be removed until dependency-aware health
    checks are implemented.
    """

    include_dependencies: bool = False