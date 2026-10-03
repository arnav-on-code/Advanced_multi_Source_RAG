from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.requests import (
    IngestRequest,
    QueryRequest,
    SourceFilter,
)
from app.models.responses import (
    IngestResponse,
    QueryResponse,
    QuerySource,
)


SUPPORTED_SOURCES = [
    "pdf",
    "url",
    "youtube",
    "docx",
    "csv",
    "txt",
    "json",
]


# =========================================================
# QUERY REQUEST
# =========================================================


def test_query_request():
    request = QueryRequest(
        query="What is RAG?",
    )

    assert request.query == "What is RAG?"
    assert request.top_k == 5
    assert request.use_reranking is True
    assert request.metadata_filter is None


def test_query_request_with_options():
    request = QueryRequest(
        query="What is FastAPI?",
        top_k=10,
        use_reranking=False,
    )

    assert request.query == "What is FastAPI?"
    assert request.top_k == 10
    assert request.use_reranking is False


def test_query_request_with_metadata_filter():
    request = QueryRequest(
        query="What is RAG?",
        metadata_filter=SourceFilter(
            source_type="pdf",
            file_name="guide.pdf",
            page=2,
        ),
    )

    assert request.metadata_filter is not None
    assert request.metadata_filter.source_type == "pdf"
    assert request.metadata_filter.file_name == "guide.pdf"
    assert request.metadata_filter.page == 2


@pytest.mark.parametrize(
    "query",
    ["", " ", "   ", "\n", "\t"],
)
def test_query_request_rejects_empty_query(query):
    with pytest.raises(ValidationError):
        QueryRequest(query=query)


@pytest.mark.parametrize(
    "top_k",
    [0, -1, 101],
)
def test_query_request_rejects_invalid_top_k(top_k):
    with pytest.raises(ValidationError):
        QueryRequest(
            query="What is RAG?",
            top_k=top_k,
        )


def test_query_request_invalid_source_filter():
    with pytest.raises(ValidationError):
        QueryRequest(
            query="What is RAG?",
            metadata_filter="invalid",
        )


def test_query_request_invalid_source_type():
    with pytest.raises(ValidationError):
        QueryRequest(
            query="What is RAG?",
            metadata_filter={
                "source_type": "audio",
            },
        )


def test_query_request_invalid_page():
    with pytest.raises(ValidationError):
        QueryRequest(
            query="What is RAG?",
            metadata_filter={
                "page": 0,
            },
        )


# =========================================================
# SOURCE FILTER
# =========================================================


def test_source_filter():
    source_filter = SourceFilter(
        source_type="pdf",
        source="guide.pdf",
        file_name="guide.pdf",
        page=3,
    )

    assert source_filter.source_type == "pdf"
    assert source_filter.source == "guide.pdf"
    assert source_filter.file_name == "guide.pdf"
    assert source_filter.page == 3


def test_source_filter_strips_values():
    source_filter = SourceFilter(
        source=" guide.pdf ",
        file_name=" guide.pdf ",
    )

    assert source_filter.source == "guide.pdf"
    assert source_filter.file_name == "guide.pdf"


def test_source_filter_rejects_empty_source():
    with pytest.raises(ValidationError):
        SourceFilter(source="   ")


def test_source_filter_rejects_empty_file_name():
    with pytest.raises(ValidationError):
        SourceFilter(file_name="   ")


# =========================================================
# INGEST REQUEST
# =========================================================


def test_ingest_request():
    request = IngestRequest(
        source_type="url",
        source="https://example.com",
    )

    assert request.source_type == "url"
    assert request.source == "https://example.com"


@pytest.mark.parametrize(
    "source_type",
    SUPPORTED_SOURCES,
)
def test_ingest_request_supported_sources(source_type):
    request = IngestRequest(
        source_type=source_type,
        source="sample",
    )

    assert request.source_type == source_type


def test_ingest_request_invalid_source_type():
    with pytest.raises(ValidationError):
        IngestRequest(
            source_type="audio",
            source="sample.mp3",
        )


@pytest.mark.parametrize(
    "source",
    ["", " ", "   ", "\n"],
)
def test_ingest_request_rejects_empty_source(source):
    with pytest.raises(ValidationError):
        IngestRequest(
            source_type="txt",
            source=source,
        )


def test_ingest_request_strips_source():
    request = IngestRequest(
        source_type="txt",
        source=" sample.txt ",
    )

    assert request.source == "sample.txt"


# =========================================================
# QUERY SOURCE
# =========================================================


def test_query_source():
    source = QuerySource(
        rank=1,
        rerank_score=0.95,
        original_rank=1,
        original_score=0.85,
        source_type="pdf",
        source="document.pdf",
        file_name="document.pdf",
        page=5,
        chunk_id="chunk-1",
    )

    assert source.rank == 1
    assert source.rerank_score == 0.95
    assert source.original_rank == 1
    assert source.original_score == 0.85
    assert source.source_type == "pdf"
    assert source.source == "document.pdf"
    assert source.file_name == "document.pdf"
    assert source.page == 5
    assert source.chunk_id == "chunk-1"


def test_query_source_minimal():
    source = QuerySource(
        rank=1,
        rerank_score=0.9,
        original_rank=1,
        original_score=0.8,
    )

    assert source.rank == 1
    assert source.source_type is None
    assert source.source is None


# =========================================================
# QUERY RESPONSE
# =========================================================


def test_query_response():
    response = QueryResponse(
        answer=(
            "RAG stands for "
            "Retrieval-Augmented Generation."
        ),
        sources=[],
        retrieved_documents=0,
    )

    assert response.answer.startswith("RAG")
    assert response.sources == []
    assert response.retrieved_documents == 0


def test_query_response_with_source():
    response = QueryResponse(
        answer="FastAPI is a Python web framework.",
        sources=[
            QuerySource(
                rank=1,
                rerank_score=0.95,
                original_rank=1,
                original_score=0.85,
                source_type="txt",
                source="fastapi.txt",
            )
        ],
        retrieved_documents=1,
    )

    assert len(response.sources) == 1
    assert response.retrieved_documents == 1
    assert response.sources[0].source_type == "txt"


# =========================================================
# INGEST RESPONSE
# =========================================================


def test_ingest_response():
    response = IngestResponse(
        status="success",
        source_type="txt",
        documents=2,
        chunks=5,
        message="Source loaded successfully.",
    )

    assert response.status == "success"
    assert response.source_type == "txt"
    assert response.documents == 2
    assert response.chunks == 5
    assert response.message == (
        "Source loaded successfully."
    )


def test_ingest_response_rejects_negative_counts():
    with pytest.raises(ValidationError):
        IngestResponse(
            source_type="txt",
            documents=-1,
            chunks=2,
            message="Failed",
        )


def test_ingest_response_rejects_invalid_source_type():
    with pytest.raises(ValidationError):
        IngestResponse(
            source_type="audio",
            documents=1,
            chunks=1,
            message="Failed",
        )