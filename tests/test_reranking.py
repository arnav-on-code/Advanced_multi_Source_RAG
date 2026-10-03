from __future__ import annotations

import pytest

from langchain_core.documents import Document

from app.retrieval.hybrid_search import HybridSearchResult
from app.retrieval.reranker import (
    DEFAULT_RERANKER_MODEL,
    Reranker,
)


def test_reranking(monkeypatch):
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

    class MockCrossEncoder:
        def __init__(
            self,
            model_name,
            max_length,
            device,
        ):
            assert model_name == DEFAULT_RERANKER_MODEL
            assert max_length == 512
            assert device == "cpu"

        def predict(
            self,
            pairs,
            batch_size,
            show_progress_bar,
        ):
            assert len(pairs) == 2
            assert pairs[0][0] == "What is FastAPI?"
            assert pairs[0][1] == (
                "FastAPI is a Python web framework."
            )
            assert pairs[1][1] == (
                "ChromaDB is a vector database."
            )
            assert batch_size == 16
            assert show_progress_bar is False

            return [0.95, 0.35]

    monkeypatch.setattr(
        "app.retrieval.reranker.CrossEncoder",
        MockCrossEncoder,
    )

    reranker = Reranker()

    candidates = [
        HybridSearchResult(
            document=documents[0],
            score=0.8,
            vector_score=0.9,
            bm25_score=0.7,
            rank=1,
        ),
        HybridSearchResult(
            document=documents[1],
            score=0.6,
            vector_score=0.5,
            bm25_score=0.7,
            rank=2,
        ),
    ]

    results = reranker.rerank(
        query="What is FastAPI?",
        results=candidates,
        top_k=2,
    )

    assert len(results) == 2

    assert results[0].document.page_content == (
        "FastAPI is a Python web framework."
    )
    assert results[0].score == 0.95
    assert results[0].original_score == 0.8
    assert results[0].original_rank == 1
    assert results[0].rank == 1

    assert results[1].document.page_content == (
        "ChromaDB is a vector database."
    )
    assert results[1].score == 0.35
    assert results[1].original_score == 0.6
    assert results[1].original_rank == 2
    assert results[1].rank == 2


def test_reranking_empty_results(monkeypatch):
    class MockCrossEncoder:
        def __init__(
            self,
            model_name,
            max_length,
            device,
        ):
            pass

    monkeypatch.setattr(
        "app.retrieval.reranker.CrossEncoder",
        MockCrossEncoder,
    )

    reranker = Reranker()

    results = reranker.rerank(
        query="FastAPI",
        results=[],
        top_k=5,
    )

    assert results == []


def test_reranking_respects_top_k(monkeypatch):
    documents = [
        Document(page_content="Document 1"),
        Document(page_content="Document 2"),
        Document(page_content="Document 3"),
    ]

    class MockCrossEncoder:
        def __init__(
            self,
            model_name,
            max_length,
            device,
        ):
            pass

        def predict(
            self,
            pairs,
            batch_size,
            show_progress_bar,
        ):
            assert len(pairs) == 3
            return [0.9, 0.8, 0.7]

    monkeypatch.setattr(
        "app.retrieval.reranker.CrossEncoder",
        MockCrossEncoder,
    )

    reranker = Reranker()

    candidates = [
        HybridSearchResult(
            documents[0],
            0.9,
            0.9,
            0.8,
            1,
        ),
        HybridSearchResult(
            documents[1],
            0.8,
            0.8,
            0.7,
            2,
        ),
        HybridSearchResult(
            documents[2],
            0.7,
            0.7,
            0.6,
            3,
        ),
    ]

    results = reranker.rerank(
        query="test query",
        results=candidates,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].score == 0.9
    assert results[1].score == 0.8
    assert results[0].rank == 1
    assert results[1].rank == 2


@pytest.mark.parametrize(
    "query",
    ["", " ", "   ", "\n", "\t"],
)
def test_reranking_rejects_empty_query(
    monkeypatch,
    query,
):
    class MockCrossEncoder:
        def __init__(
            self,
            model_name,
            max_length,
            device,
        ):
            pass

    monkeypatch.setattr(
        "app.retrieval.reranker.CrossEncoder",
        MockCrossEncoder,
    )

    reranker = Reranker()

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        reranker.rerank(
            query=query,
            results=[],
            top_k=5,
        )


@pytest.mark.parametrize(
    "top_k",
    [0, -1, 101],
)
def test_reranking_rejects_invalid_top_k(
    monkeypatch,
    top_k,
):
    class MockCrossEncoder:
        def __init__(
            self,
            model_name,
            max_length,
            device,
        ):
            pass

    monkeypatch.setattr(
        "app.retrieval.reranker.CrossEncoder",
        MockCrossEncoder,
    )

    reranker = Reranker()

    with pytest.raises(
        ValueError,
        match="top_k must be between 1 and 100",
    ):
        reranker.rerank(
            query="FastAPI",
            results=[],
            top_k=top_k,
        )