from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.pipeline.rag_pipeline import RAGPipeline
from app.retrieval.hybrid_search import HybridSearchResult
from app.retrieval.reranker import RerankedResult


# =========================================================
# BASIC PIPELINE
# =========================================================


def test_rag_pipeline():
    """RAG pipeline should retrieve, rerank, and generate an answer."""

    document = Document(
        page_content="FastAPI is a Python web framework.",
        metadata={
            "source_type": "txt",
            "source": "sample.txt",
            "page": 1,
        },
    )

    class MockHybridSearcher:
        def search(
            self,
            query,
            top_k=10,
            candidate_k=None,
            metadata_filter=None,
        ):
            assert query == "What is FastAPI?"
            assert top_k == 10
            assert candidate_k == 10
            assert metadata_filter is None

            return [
                HybridSearchResult(
                    document=document,
                    score=0.9,
                    vector_score=0.8,
                    bm25_score=0.7,
                    rank=1,
                )
            ]

    class MockReranker:
        def rerank(
            self,
            query,
            results,
            top_k=5,
        ):
            assert query == "What is FastAPI?"
            assert len(results) == 1
            assert top_k == 1

            return [
                RerankedResult(
                    document=document,
                    score=0.95,
                    original_score=0.9,
                    original_rank=1,
                    rank=1,
                )
            ]

    class MockLLMGenerator:
        def generate(
            self,
            query,
            documents,
        ):
            assert query == "What is FastAPI?"
            assert documents == [document]

            return (
                "FastAPI is a Python web framework."
            )

    pipeline = RAGPipeline(
        hybrid_searcher=MockHybridSearcher(),
        reranker=MockReranker(),
        llm_generator=MockLLMGenerator(),
        retrieval_top_k=5,
        candidate_k=10,
        reranker_top_k=5,
    )

    response = pipeline.invoke(
        "What is FastAPI?"
    )

    assert response.answer == (
        "FastAPI is a Python web framework."
    )

    assert len(response.sources) == 1
    assert response.retrieved_documents == 1

    source = response.sources[0]

    assert source["source_type"] == "txt"
    assert source["source"] == "sample.txt"
    assert source["page"] == 1
    assert source["rank"] == 1
    assert source["rerank_score"] == 0.95


# =========================================================
# QUERY ALIAS
# =========================================================


def test_rag_pipeline_query_alias():
    """query() should behave as an alias for invoke()."""

    document = Document(
        page_content="ChromaDB is a vector database.",
        metadata={
            "source_type": "txt",
            "source": "chroma.txt",
        },
    )

    class MockHybridSearcher:
        def search(
            self,
            query,
            top_k=10,
            candidate_k=None,
            metadata_filter=None,
        ):
            return [
                HybridSearchResult(
                    document=document,
                    score=0.8,
                    vector_score=0.7,
                    bm25_score=0.6,
                    rank=1,
                )
            ]

    class MockReranker:
        def rerank(
            self,
            query,
            results,
            top_k=5,
        ):
            return [
                RerankedResult(
                    document=document,
                    score=0.9,
                    original_score=0.8,
                    original_rank=1,
                    rank=1,
                )
            ]

    class MockLLMGenerator:
        def generate(
            self,
            query,
            documents,
        ):
            return (
                "ChromaDB is a vector database."
            )

    pipeline = RAGPipeline(
        hybrid_searcher=MockHybridSearcher(),
        reranker=MockReranker(),
        llm_generator=MockLLMGenerator(),
    )

    response = pipeline.query(
        "What is ChromaDB?"
    )

    assert response.answer == (
        "ChromaDB is a vector database."
    )


# =========================================================
# NO RETRIEVED DOCUMENTS
# =========================================================


def test_rag_pipeline_no_documents():
    """Pipeline should return a safe empty response."""

    class MockHybridSearcher:
        def search(
            self,
            query,
            top_k=10,
            candidate_k=None,
            metadata_filter=None,
        ):
            return []

    class MockReranker:
        def rerank(
            self,
            query,
            results,
            top_k=5,
        ):
            pytest.fail(
                "Reranker should not be called "
                "when no documents are retrieved."
            )

    class MockLLMGenerator:
        def generate(
            self,
            query,
            documents,
        ):
            pytest.fail(
                "LLM should not be called "
                "when no documents are retrieved."
            )

    pipeline = RAGPipeline(
        hybrid_searcher=MockHybridSearcher(),
        reranker=MockReranker(),
        llm_generator=MockLLMGenerator(),
    )

    response = pipeline.invoke(
        "Unknown question"
    )

    assert response.answer == (
        "I could not find relevant information "
        "in the available sources."
    )

    assert response.sources == []
    assert response.retrieved_documents == 0


# =========================================================
# EMPTY QUERY
# =========================================================


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
def test_rag_pipeline_empty_query(
    query: str,
):
    """Empty or whitespace-only queries should be rejected."""

    class MockHybridSearcher:
        def search(self, *args, **kwargs):
            pytest.fail(
                "Hybrid search should not be called."
            )

    pipeline = RAGPipeline(
        hybrid_searcher=MockHybridSearcher(),
        reranker=None,
        llm_generator=None,
    )

    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        pipeline.invoke(query)


# =========================================================
# METADATA FILTER
# =========================================================


def test_rag_pipeline_metadata_filter():
    """Metadata filters should reach hybrid retrieval."""

    metadata_filter = {
        "source_type": "pdf",
        "file_name": "guide.pdf",
    }

    class MockHybridSearcher:
        def search(
            self,
            query,
            top_k=10,
            candidate_k=None,
            metadata_filter=None,
        ):
            assert metadata_filter == {
                "source_type": "pdf",
                "file_name": "guide.pdf",
            }

            return []

    pipeline = RAGPipeline(
        hybrid_searcher=MockHybridSearcher(),
        reranker=None,
        llm_generator=None,
    )

    response = pipeline.invoke(
        "FastAPI",
        metadata_filter=metadata_filter,
    )

    assert response.sources == []
    assert response.retrieved_documents == 0


# =========================================================
# CONFIGURATION VALIDATION
# =========================================================


def test_rag_pipeline_invalid_retrieval_top_k():
    with pytest.raises(
        ValueError,
        match="retrieval_top_k must be between 1 and 100",
    ):
        RAGPipeline(
            hybrid_searcher=object(),
            reranker=object(),
            llm_generator=object(),
            retrieval_top_k=0,
        )


def test_rag_pipeline_invalid_candidate_k():
    with pytest.raises(
        ValueError,
        match="candidate_k must be greater than or equal",
    ):
        RAGPipeline(
            hybrid_searcher=object(),
            reranker=object(),
            llm_generator=object(),
            retrieval_top_k=10,
            candidate_k=5,
        )


def test_rag_pipeline_invalid_reranker_top_k():
    with pytest.raises(
        ValueError,
        match="reranker_top_k cannot exceed",
    ):
        RAGPipeline(
            hybrid_searcher=object(),
            reranker=object(),
            llm_generator=object(),
            retrieval_top_k=5,
            candidate_k=10,
            reranker_top_k=10,
        )