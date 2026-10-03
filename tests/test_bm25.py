from __future__ import annotations

import pytest

from langchain_core.documents import Document

from app.retrieval.bm25 import (
    BM25Retriever,
    BM25SearchResult,
)


@pytest.fixture
def documents():
    return [
        Document(
            page_content="FastAPI is a Python web framework.",
            metadata={"source_type": "txt"},
        ),
        Document(
            page_content="ChromaDB is a vector database.",
            metadata={"source_type": "txt"},
        ),
    ]


def test_bm25_fit(documents):
    retriever = BM25Retriever()

    retriever.fit(documents)

    assert retriever.count() == 2
    assert retriever.bm25 is not None


def test_bm25_search(documents):
    retriever = BM25Retriever(documents)

    results = retriever.search(
        query="FastAPI framework",
        top_k=1,
    )

    assert len(results) == 1
    assert isinstance(results[0], BM25SearchResult)

    assert (
        "FastAPI"
        in results[0].document.page_content
    )

    assert results[0].score >= 0


def test_bm25_search_result_contains_document_and_score(
    documents,
):
    retriever = BM25Retriever(documents)

    results = retriever.search(
        query="FastAPI",
        top_k=1,
    )

    result = results[0]

    assert isinstance(result.document, Document)
    assert isinstance(result.score, float)


def test_bm25_fit_rejects_empty_documents():
    retriever = BM25Retriever()

    with pytest.raises(
        ValueError,
        match="Cannot build BM25 index from empty documents",
    ):
        retriever.fit([])


def test_bm25_fit_rejects_only_empty_documents():
    documents = [
        Document(page_content=""),
        Document(page_content="   "),
        Document(page_content="\n"),
    ]

    retriever = BM25Retriever()

    with pytest.raises(
        ValueError,
        match="Cannot build BM25 index from empty documents",
    ):
        retriever.fit(documents)


def test_bm25_fit_filters_empty_documents():
    documents = [
        Document(page_content="FastAPI framework"),
        Document(page_content=""),
        Document(page_content="   "),
        Document(page_content="Python programming"),
    ]

    retriever = BM25Retriever()

    retriever.fit(documents)

    assert retriever.count() == 2


def test_bm25_search_before_fit():
    retriever = BM25Retriever()

    with pytest.raises(
        RuntimeError,
        match="BM25 index has not been built",
    ):
        retriever.search(
            query="FastAPI",
            top_k=5,
        )


@pytest.mark.parametrize(
    "query",
    ["", " ", "   ", "\n", "\t"],
)
def test_bm25_search_rejects_empty_query(
    documents,
    query,
):
    retriever = BM25Retriever(documents)

    with pytest.raises(
        ValueError,
        match="Search query cannot be empty",
    ):
        retriever.search(
            query=query,
            top_k=5,
        )


@pytest.mark.parametrize(
    "top_k",
    [0, -1, 101],
)
def test_bm25_search_rejects_invalid_top_k(
    documents,
    top_k,
):
    retriever = BM25Retriever(documents)

    with pytest.raises(
        ValueError,
        match="top_k must be between 1 and 100",
    ):
        retriever.search(
            query="FastAPI",
            top_k=top_k,
        )


def test_bm25_respects_top_k():
    documents = [
        Document(page_content="Python programming"),
        Document(page_content="Python FastAPI"),
        Document(page_content="Python machine learning"),
    ]

    retriever = BM25Retriever(documents)

    results = retriever.search(
        query="Python",
        top_k=2,
    )

    assert len(results) == 2


def test_bm25_search_returns_ranked_results():
    documents = [
        Document(page_content="Python programming"),
        Document(page_content="FastAPI is a Python framework"),
        Document(page_content="Machine learning with Python"),
    ]

    retriever = BM25Retriever(documents)

    results = retriever.search(
        query="FastAPI Python framework",
        top_k=3,
    )

    assert len(results) == 3

    assert (
        results[0].document.page_content
        == "FastAPI is a Python framework"
    )

    assert results[0].score >= results[1].score


def test_bm25_count_after_refit():
    first_documents = [
        Document(page_content="FastAPI"),
        Document(page_content="Python"),
    ]

    second_documents = [
        Document(page_content="ChromaDB"),
    ]

    retriever = BM25Retriever(first_documents)

    assert retriever.count() == 2

    retriever.fit(second_documents)

    assert retriever.count() == 1