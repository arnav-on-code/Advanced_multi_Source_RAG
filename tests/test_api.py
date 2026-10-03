from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.api.query import get_rag_pipeline
from app.pipeline.rag_pipeline import RAGResponse


client = TestClient(app)


class MockRAGPipeline:
    def invoke(
        self,
        query,
        top_k=None,
        use_reranking=True,
        metadata_filter=None,
    ):
        assert query == "What is RAG?"
        assert top_k == 3
        assert use_reranking is True
        assert metadata_filter is None

        return RAGResponse(
            answer="RAG combines retrieval with generation.",
            sources=[
                {
                    "rank": 1,
                    "rerank_score": 0.95,
                    "original_rank": 1,
                    "original_score": 0.85,
                    "source_type": "txt",
                    "source": "rag.txt",
                    "file_name": "rag.txt",
                    "chunk_id": "chunk-1",
                }
            ],
            retrieved_documents=1,
        )


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["project"] == (
        "Advanced Multi-Source RAG API"
    )
    assert data["status"] == "running"
    assert data["version"] == "1.0.0"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"]
    assert data["version"]
    assert data["environment"]


def test_ready_endpoint():
    response = client.get("/health/ready")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ready"
    assert data["service"]
    assert data["version"]
    assert data["environment"]


def test_query_endpoint_success():
    app.dependency_overrides[
        get_rag_pipeline
    ] = lambda: MockRAGPipeline()

    try:
        response = client.post(
            "/query",
            json={
                "query": "What is RAG?",
                "top_k": 3,
                "use_reranking": True,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["answer"] == (
            "RAG combines retrieval with generation."
        )

        assert data["retrieved_documents"] == 1
        assert len(data["sources"]) == 1

        assert data["sources"][0]["source_type"] == "txt"
        assert data["sources"][0]["file_name"] == "rag.txt"

    finally:
        app.dependency_overrides.clear()


def test_query_endpoint_validation():
    response = client.post(
        "/query",
        json={
            "query": "",
        },
    )

    assert response.status_code == 422


def test_query_endpoint_missing_body():
    response = client.post("/query")

    assert response.status_code == 422


def test_query_endpoint_invalid_top_k():
    response = client.post(
        "/query",
        json={
            "query": "What is RAG?",
            "top_k": 0,
        },
    )

    assert response.status_code == 422


def test_query_endpoint_top_k_too_large():
    response = client.post(
        "/query",
        json={
            "query": "What is RAG?",
            "top_k": 101,
        },
    )

    assert response.status_code == 422


def test_query_endpoint_whitespace_query():
    response = client.post(
        "/query",
        json={
            "query": "   ",
        },
    )

    assert response.status_code == 422


def test_query_endpoint_invalid_source_type():
    response = client.post(
        "/query",
        json={
            "query": "What is RAG?",
            "metadata_filter": {
                "source_type": "invalid",
            },
        },
    )

    assert response.status_code == 422


def test_query_endpoint_invalid_page():
    response = client.post(
        "/query",
        json={
            "query": "What is RAG?",
            "metadata_filter": {
                "page": 0,
            },
        },
    )

    assert response.status_code == 422


def test_docs_info_endpoint():
    response = client.get("/docs-info")

    assert response.status_code == 200

    data = response.json()

    assert "endpoints" in data
    assert "GET /health" in data["endpoints"]
    assert "POST /query" in data["endpoints"]