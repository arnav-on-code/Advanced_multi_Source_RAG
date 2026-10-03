from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.embeddings.embedder import (
    DEFAULT_EMBEDDING_MODEL,
    embed_documents,
    embed_query,
)


# =========================================================
# CONFIGURATION
# =========================================================


def test_default_embedding_model():
    """Default embedding model should remain stable."""

    assert (
        DEFAULT_EMBEDDING_MODEL
        == "BAAI/bge-small-en-v1.5"
    )


# =========================================================
# DOCUMENT EMBEDDINGS
# =========================================================


def test_embed_documents_empty_input():
    """Empty document input should return an empty list."""

    result = embed_documents([])

    assert result == []


def test_embed_documents(
    monkeypatch: pytest.MonkeyPatch,
):
    """Documents should be converted into embeddings."""

    expected_embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    class MockEmbeddingModel:
        def embed_documents(
            self,
            texts: list[str],
        ):
            assert texts == [
                "Hello world",
                "FastAPI RAG",
            ]

            return expected_embeddings

    monkeypatch.setattr(
        "app.embeddings.embedder.get_embedding_model",
        lambda model_name=DEFAULT_EMBEDDING_MODEL:
            MockEmbeddingModel(),
    )

    documents = [
        Document(
            page_content="Hello world",
        ),
        Document(
            page_content="FastAPI RAG",
        ),
    ]

    embeddings = embed_documents(documents)

    assert embeddings == expected_embeddings
    assert len(embeddings) == len(documents)


def test_embed_documents_uses_custom_model(
    monkeypatch: pytest.MonkeyPatch,
):
    """Custom embedding model names should be forwarded correctly."""

    requested_model = "custom-embedding-model"

    class MockEmbeddingModel:
        def embed_documents(
            self,
            texts: list[str],
        ):
            return [[1.0, 2.0]]

    def mock_get_embedding_model(
        model_name=DEFAULT_EMBEDDING_MODEL,
    ):
        assert model_name == requested_model
        return MockEmbeddingModel()

    monkeypatch.setattr(
        "app.embeddings.embedder.get_embedding_model",
        mock_get_embedding_model,
    )

    documents = [
        Document(
            page_content="Test document",
        )
    ]

    result = embed_documents(
        documents,
        model_name=requested_model,
    )

    assert result == [[1.0, 2.0]]


def test_embed_documents_rejects_empty_content():
    """Documents containing empty text should be rejected."""

    documents = [
        Document(
            page_content="Valid document",
        ),
        Document(
            page_content="   ",
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Documents must contain non-empty text",
    ):
        embed_documents(documents)


def test_embed_documents_model_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    """Embedding model failures should propagate."""

    class MockEmbeddingModel:
        def embed_documents(
            self,
            texts: list[str],
        ):
            raise RuntimeError(
                "Embedding generation failed"
            )

    monkeypatch.setattr(
        "app.embeddings.embedder.get_embedding_model",
        lambda model_name=DEFAULT_EMBEDDING_MODEL:
            MockEmbeddingModel(),
    )

    documents = [
        Document(
            page_content="Test document",
        )
    ]

    with pytest.raises(
        RuntimeError,
        match="Embedding generation failed",
    ):
        embed_documents(documents)


# =========================================================
# QUERY EMBEDDINGS
# =========================================================


def test_embed_query(
    monkeypatch: pytest.MonkeyPatch,
):
    """A valid query should be converted into an embedding."""

    class MockEmbeddingModel:
        def embed_query(
            self,
            query: str,
        ):
            assert query == "What is RAG?"
            return [0.1, 0.2, 0.3]

    monkeypatch.setattr(
        "app.embeddings.embedder.get_embedding_model",
        lambda model_name=DEFAULT_EMBEDDING_MODEL:
            MockEmbeddingModel(),
    )

    embedding = embed_query(
        "What is RAG?"
    )

    assert embedding == [
        0.1,
        0.2,
        0.3,
    ]


def test_embed_query_strips_whitespace(
    monkeypatch: pytest.MonkeyPatch,
):
    """Query whitespace should be removed before embedding."""

    class MockEmbeddingModel:
        def embed_query(
            self,
            query: str,
        ):
            assert query == "What is RAG?"
            return [0.1, 0.2, 0.3]

    monkeypatch.setattr(
        "app.embeddings.embedder.get_embedding_model",
        lambda model_name=DEFAULT_EMBEDDING_MODEL:
            MockEmbeddingModel(),
    )

    result = embed_query(
        "   What is RAG?   "
    )

    assert result == [
        0.1,
        0.2,
        0.3,
    ]


@pytest.mark.parametrize(
    "query",
    [
        "",
        " ",
        "   ",
        "\n",
        "\t",
    ],
)
def test_embed_query_empty_query(
    query: str,
):
    """Empty or whitespace-only queries should be rejected."""

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        embed_query(query)


def test_embed_query_uses_custom_model(
    monkeypatch: pytest.MonkeyPatch,
):
    """Custom model names should be forwarded to the embedding factory."""

    requested_model = "custom-query-model"

    class MockEmbeddingModel:
        def embed_query(
            self,
            query: str,
        ):
            return [1.0, 2.0]

    def mock_get_embedding_model(
        model_name=DEFAULT_EMBEDDING_MODEL,
    ):
        assert model_name == requested_model
        return MockEmbeddingModel()

    monkeypatch.setattr(
        "app.embeddings.embedder.get_embedding_model",
        mock_get_embedding_model,
    )

    result = embed_query(
        "test query",
        model_name=requested_model,
    )

    assert result == [
        1.0,
        2.0,
    ]


def test_embed_query_model_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    """Embedding model failures should propagate."""

    class MockEmbeddingModel:
        def embed_query(
            self,
            query: str,
        ):
            raise RuntimeError(
                "Query embedding failed"
            )

    monkeypatch.setattr(
        "app.embeddings.embedder.get_embedding_model",
        lambda model_name=DEFAULT_EMBEDDING_MODEL:
            MockEmbeddingModel(),
    )

    with pytest.raises(
        RuntimeError,
        match="Query embedding failed",
    ):
        embed_query("What is RAG?")