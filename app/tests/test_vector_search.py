from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.retrieval.vector_search import VectorSearcher


# =========================================================
# SEARCH
# =========================================================


def test_vector_search(
    monkeypatch: pytest.MonkeyPatch,
):
    """Vector search should return Chroma results unchanged."""

    documents = [
        Document(
            page_content="FastAPI is a Python web framework.",
            metadata={"source_type": "txt"},
        ),
        Document(
            page_content="ChromaDB is a vector database.",
            metadata={"source_type": "txt"},
        ),
    ]

    class MockVectorStore:
        def similarity_search_with_score(
            self,
            query,
            k=5,
            filter=None,
        ):
            assert query == "What is FastAPI?"
            assert k == 2
            assert filter is None

            return [
                (documents[0], 0.10),
                (documents[1], 0.80),
            ]

    class MockEmbeddingModel:
        pass

    monkeypatch.setattr(
        "app.retrieval.vector_search.get_embedding_model",
        lambda model_name: MockEmbeddingModel(),
    )

    monkeypatch.setattr(
        "app.retrieval.vector_search.ChromaVectorStore",
        lambda **kwargs: MockVectorStore(),
    )

    searcher = VectorSearcher()

    results = searcher.search(
        query="What is FastAPI?",
        top_k=2,
    )

    assert len(results) == 2

    assert results[0].document.page_content == (
        "FastAPI is a Python web framework."
    )

    assert results[0].score == 0.10

    assert results[1].document.page_content == (
        "ChromaDB is a vector database."
    )

    assert results[1].score == 0.80


# =========================================================
# METADATA FILTER
# =========================================================


def test_vector_search_metadata_filter(
    monkeypatch: pytest.MonkeyPatch,
):
    """Metadata filters should be passed to Chroma."""

    documents = [
        Document(
            page_content="FastAPI documentation",
            metadata={"source_type": "pdf"},
        )
    ]

    metadata_filter = {
        "source_type": "pdf",
    }

    class MockVectorStore:
        def similarity_search_with_score(
            self,
            query,
            k=5,
            filter=None,
        ):
            assert query == "FastAPI"
            assert k == 5
            assert filter == metadata_filter

            return [
                (documents[0], 0.15),
            ]

    monkeypatch.setattr(
        "app.retrieval.vector_search.get_embedding_model",
        lambda model_name: object(),
    )

    monkeypatch.setattr(
        "app.retrieval.vector_search.ChromaVectorStore",
        lambda **kwargs: MockVectorStore(),
    )

    searcher = VectorSearcher()

    results = searcher.search(
        query="FastAPI",
        metadata_filter=metadata_filter,
    )

    assert len(results) == 1
    assert results[0].document == documents[0]
    assert results[0].score == 0.15


# =========================================================
# EMPTY QUERY
# =========================================================


def test_vector_search_empty_query(
    monkeypatch: pytest.MonkeyPatch,
):
    """Empty queries should raise ValueError."""

    monkeypatch.setattr(
        "app.retrieval.vector_search.get_embedding_model",
        lambda model_name: object(),
    )

    monkeypatch.setattr(
        "app.retrieval.vector_search.ChromaVectorStore",
        lambda **kwargs: object(),
    )

    searcher = VectorSearcher()

    with pytest.raises(
        ValueError,
        match="Search query cannot be empty",
    ):
        searcher.search(
            query="",
            top_k=5,
        )


@pytest.mark.parametrize(
    "query",
    [
        " ",
        "   ",
        "\n",
        "\t",
    ],
)
def test_vector_search_whitespace_query(
    query: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """Whitespace-only queries should be rejected."""

    monkeypatch.setattr(
        "app.retrieval.vector_search.get_embedding_model",
        lambda model_name: object(),
    )

    monkeypatch.setattr(
        "app.retrieval.vector_search.ChromaVectorStore",
        lambda **kwargs: object(),
    )

    searcher = VectorSearcher()

    with pytest.raises(
        ValueError,
        match="Search query cannot be empty",
    ):
        searcher.search(
            query=query,
            top_k=5,
        )


# =========================================================
# TOP-K VALIDATION
# =========================================================


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
        101,
    ],
)
def test_vector_search_invalid_top_k(
    top_k: int,
    monkeypatch: pytest.MonkeyPatch,
):
    """top_k outside the supported range should be rejected."""

    monkeypatch.setattr(
        "app.retrieval.vector_search.get_embedding_model",
        lambda model_name: object(),
    )

    monkeypatch.setattr(
        "app.retrieval.vector_search.ChromaVectorStore",
        lambda **kwargs: object(),
    )

    searcher = VectorSearcher()

    with pytest.raises(
        ValueError,
        match="top_k must be between 1 and 100",
    ):
        searcher.search(
            query="FastAPI",
            top_k=top_k,
        )


# =========================================================
# SEARCH DOCUMENTS
# =========================================================


def test_search_documents(
    monkeypatch: pytest.MonkeyPatch,
):
    """search_documents() should return only Documents."""

    documents = [
        Document(
            page_content="FastAPI content",
            metadata={},
        ),
        Document(
            page_content="Chroma content",
            metadata={},
        ),
    ]

    class MockVectorStore:
        def similarity_search_with_score(
            self,
            query,
            k=5,
            filter=None,
        ):
            return [
                (documents[0], 0.1),
                (documents[1], 0.2),
            ]

    monkeypatch.setattr(
        "app.retrieval.vector_search.get_embedding_model",
        lambda model_name: object(),
    )

    monkeypatch.setattr(
        "app.retrieval.vector_search.ChromaVectorStore",
        lambda **kwargs: MockVectorStore(),
    )

    searcher = VectorSearcher()

    results = searcher.search_documents(
        query="FastAPI",
        top_k=2,
    )

    assert results == documents
    assert all(
        isinstance(document, Document)
        for document in results
    )


# =========================================================
# RETRIEVER
# =========================================================


def test_get_retriever(
    monkeypatch: pytest.MonkeyPatch,
):
    """VectorSearcher should expose a Chroma retriever."""

    mock_retriever = object()

    class MockVectorStore:
        def get_retriever(
            self,
            k=5,
            filter=None,
        ):
            assert k == 3
            assert filter == {
                "source_type": "pdf",
            }

            return mock_retriever

    monkeypatch.setattr(
        "app.retrieval.vector_search.get_embedding_model",
        lambda model_name: object(),
    )

    monkeypatch.setattr(
        "app.retrieval.vector_search.ChromaVectorStore",
        lambda **kwargs: MockVectorStore(),
    )

    searcher = VectorSearcher()

    retriever = searcher.get_retriever(
        top_k=3,
        metadata_filter={
            "source_type": "pdf",
        },
    )

    assert retriever is mock_retriever


# =========================================================
# COUNT
# =========================================================


def test_vector_search_count(
    monkeypatch: pytest.MonkeyPatch,
):
    """count() should return the number of stored vectors."""

    class MockVectorStore:
        def count(self):
            return 10

    monkeypatch.setattr(
        "app.retrieval.vector_search.get_embedding_model",
        lambda model_name: object(),
    )

    monkeypatch.setattr(
        "app.retrieval.vector_search.ChromaVectorStore",
        lambda **kwargs: MockVectorStore(),
    )

    searcher = VectorSearcher()

    assert searcher.count() == 10