from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.retrieval.bm25 import BM25SearchResult
from app.retrieval.hybrid_search import HybridSearchResult, HybridSearcher
from app.retrieval.vector_search import VectorSearchResult


# =========================================================
# BASIC HYBRID SEARCH
# =========================================================


def test_hybrid_search():
    """Hybrid search should combine vector and BM25 results."""

    documents = [
        Document(
            page_content="FastAPI is a Python web framework.",
            metadata={"chunk_id": "chunk-1"},
        ),
        Document(
            page_content="ChromaDB is a vector database.",
            metadata={"chunk_id": "chunk-2"},
        ),
    ]

    class MockVectorSearcher:
        def search(
            self,
            query,
            top_k=10,
            metadata_filter=None,
        ):
            assert query == "FastAPI framework"
            assert top_k == 10
            assert metadata_filter is None

            return [
                VectorSearchResult(
                    document=documents[0],
                    score=0.9,
                ),
                VectorSearchResult(
                    document=documents[1],
                    score=0.5,
                ),
            ]

    class MockBM25Retriever:
        def search(
            self,
            query,
            top_k=10,
        ):
            assert query == "FastAPI framework"
            assert top_k == 10

            return [
                BM25SearchResult(
                    document=documents[1],
                    score=1.0,
                ),
                BM25SearchResult(
                    document=documents[0],
                    score=0.5,
                ),
            ]

    searcher = HybridSearcher(
        vector_searcher=MockVectorSearcher(),
        bm25_retriever=MockBM25Retriever(),
        vector_weight=0.6,
        bm25_weight=0.4,
    )

    results = searcher.search(
        query="FastAPI framework",
        top_k=2,
    )

    assert len(results) == 2

    assert all(
        isinstance(result, HybridSearchResult)
        for result in results
    )

    assert all(
        result.document is not None
        for result in results
    )

    assert all(
        result.rank > 0
        for result in results
    )

    assert all(
        0.0 <= result.score <= 1.0
        for result in results
    )

    assert results[0].score >= results[1].score


# =========================================================
# SCORE NORMALIZATION
# =========================================================


def test_vector_distance_normalization():
    """Lower Chroma distances should receive higher normalized scores."""

    distances = [
        0.1,
        0.5,
        0.9,
    ]

    normalized = (
        HybridSearcher._normalize_vector_distances(
            distances
        )
    )

    assert normalized[0] == pytest.approx(1.0)
    assert normalized[1] == pytest.approx(0.5)
    assert normalized[2] == pytest.approx(0.0)


def test_bm25_score_normalization():
    """Higher BM25 scores should receive higher normalized scores."""

    scores = [
        1.0,
        3.0,
        5.0,
    ]

    normalized = (
        HybridSearcher._normalize_scores(
            scores
        )
    )

    assert normalized[0] == pytest.approx(0.0)
    assert normalized[1] == pytest.approx(0.5)
    assert normalized[2] == pytest.approx(1.0)


def test_constant_vector_distances():
    """Equal vector distances should produce equal scores."""

    normalized = (
        HybridSearcher._normalize_vector_distances(
            [0.5, 0.5, 0.5]
        )
    )

    assert normalized == [
        1.0,
        1.0,
        1.0,
    ]


def test_constant_bm25_scores():
    """Equal BM25 scores should produce equal scores."""

    normalized = (
        HybridSearcher._normalize_scores(
            [2.0, 2.0, 2.0]
        )
    )

    assert normalized == [
        1.0,
        1.0,
        1.0,
    ]


def test_empty_score_normalization():
    """Empty score lists should remain empty."""

    assert (
        HybridSearcher._normalize_scores([])
        == []
    )

    assert (
        HybridSearcher._normalize_vector_distances([])
        == []
    )


# =========================================================
# RESULT DEDUPLICATION / FUSION
# =========================================================


def test_hybrid_search_fuses_same_document():
    """A document appearing in both retrievers should become one result."""

    document = Document(
        page_content="FastAPI framework",
        metadata={"chunk_id": "chunk-1"},
    )

    class MockVectorSearcher:
        def search(
            self,
            query,
            top_k=10,
            metadata_filter=None,
        ):
            return [
                VectorSearchResult(
                    document=document,
                    score=0.2,
                )
            ]

    class MockBM25Retriever:
        def search(
            self,
            query,
            top_k=10,
        ):
            return [
                BM25SearchResult(
                    document=document,
                    score=5.0,
                )
            ]

    searcher = HybridSearcher(
        vector_searcher=MockVectorSearcher(),
        bm25_retriever=MockBM25Retriever(),
    )

    results = searcher.search(
        query="FastAPI",
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].document is document


# =========================================================
# RESULT COUNT
# =========================================================


def test_hybrid_search_result_count():
    """top_k should control the final number of results."""

    documents = [
        Document(
            page_content="Python programming",
            metadata={"chunk_id": "chunk-1"},
        ),
        Document(
            page_content="FastAPI framework",
            metadata={"chunk_id": "chunk-2"},
        ),
    ]

    class MockVectorSearcher:
        def search(
            self,
            query,
            top_k=10,
            metadata_filter=None,
        ):
            return [
                VectorSearchResult(
                    documents[0],
                    0.8,
                ),
                VectorSearchResult(
                    documents[1],
                    0.7,
                ),
            ]

    class MockBM25Retriever:
        def search(
            self,
            query,
            top_k=10,
        ):
            return [
                BM25SearchResult(
                    documents[0],
                    0.9,
                ),
                BM25SearchResult(
                    documents[1],
                    0.6,
                ),
            ]

    searcher = HybridSearcher(
        vector_searcher=MockVectorSearcher(),
        bm25_retriever=MockBM25Retriever(),
    )

    results = searcher.search(
        query="Python",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].rank == 1


# =========================================================
# EMPTY QUERY
# =========================================================


def test_hybrid_search_empty_query():
    """Empty queries should raise ValueError."""

    class MockVectorSearcher:
        def search(
            self,
            query,
            top_k=10,
            metadata_filter=None,
        ):
            return []

    class MockBM25Retriever:
        def search(
            self,
            query,
            top_k=10,
        ):
            return []

    searcher = HybridSearcher(
        vector_searcher=MockVectorSearcher(),
        bm25_retriever=MockBM25Retriever(),
    )

    with pytest.raises(
        ValueError,
        match="Search query cannot be empty",
    ):
        searcher.search(
            query="",
            top_k=5,
        )


# =========================================================
# INVALID PARAMETERS
# =========================================================


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
        101,
    ],
)
def test_hybrid_search_invalid_top_k(
    top_k: int,
):
    """Invalid top_k values should be rejected."""

    class MockVectorSearcher:
        def search(self, *args, **kwargs):
            return []

    class MockBM25Retriever:
        def search(self, *args, **kwargs):
            return []

    searcher = HybridSearcher(
        vector_searcher=MockVectorSearcher(),
        bm25_retriever=MockBM25Retriever(),
    )

    with pytest.raises(
        ValueError,
        match="top_k must be between 1 and 100",
    ):
        searcher.search(
            query="FastAPI",
            top_k=top_k,
        )


# =========================================================
# METADATA FILTER
# =========================================================


def test_hybrid_search_metadata_filter():
    """
    Metadata filters should be forwarded to vector search.

    BM25 filtering is currently not supported by the
    BM25Retriever interface.
    """

    metadata_filter = {
        "source_type": "pdf",
    }

    class MockVectorSearcher:
        def search(
            self,
            query,
            top_k=10,
            metadata_filter=None,
        ):
            assert metadata_filter == {
                "source_type": "pdf",
            }

            return []

    class MockBM25Retriever:
        def search(
            self,
            query,
            top_k=10,
        ):
            return []

    searcher = HybridSearcher(
        vector_searcher=MockVectorSearcher(),
        bm25_retriever=MockBM25Retriever(),
    )

    results = searcher.search(
        query="FastAPI",
        top_k=5,
        metadata_filter=metadata_filter,
    )

    assert results == []