from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.llm.generator import LLMGenerator
from app.retrieval.hybrid_search import HybridSearchResult
from app.retrieval.hybrid_search import HybridSearcher
from app.retrieval.reranker import RerankedResult
from app.retrieval.reranker import Reranker


@dataclass
class RAGResponse:
    answer: str
    sources: list[dict[str, Any]]
    retrieved_documents: int


class RAGPipeline:
    def __init__(
        self,
        hybrid_searcher: HybridSearcher,
        reranker: Reranker,
        llm_generator: LLMGenerator,
        retrieval_top_k: int = 5,
        candidate_k: int | None = None,
        reranker_top_k: int = 5,
    ) -> None:

        if not 1 <= retrieval_top_k <= 100:
            raise ValueError(
                "retrieval_top_k must be between 1 and 100."
            )

        if candidate_k is not None:
            if not 1 <= candidate_k <= 200:
                raise ValueError(
                    "candidate_k must be between 1 and 200."
                )

            if candidate_k < retrieval_top_k:
                raise ValueError(
                    "candidate_k must be greater than or equal "
                    "to retrieval_top_k."
                )

        if not 1 <= reranker_top_k <= 100:
            raise ValueError(
                "reranker_top_k must be between 1 and 100."
            )

        if reranker_top_k > retrieval_top_k:
            raise ValueError(
                "reranker_top_k cannot exceed retrieval_top_k."
            )

        self.hybrid_searcher = hybrid_searcher
        self.reranker = reranker
        self.llm_generator = llm_generator

        self.retrieval_top_k = retrieval_top_k
        self.candidate_k = candidate_k
        self.reranker_top_k = reranker_top_k

    def invoke(
        self,
        query: str,
        top_k: int | None = None,
        use_reranking: bool = True,
        metadata_filter: dict[str, Any] | None = None,
    ) -> RAGResponse:

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        resolved_top_k = (
            self.retrieval_top_k
            if top_k is None
            else top_k
        )

        if not 1 <= resolved_top_k <= 100:
            raise ValueError(
                "top_k must be between 1 and 100."
            )

        candidate_k = self.candidate_k

        if candidate_k is None:
            candidate_k = max(
                resolved_top_k * 3,
                10,
            )

        if candidate_k < resolved_top_k:
            candidate_k = resolved_top_k

        hybrid_results = self.hybrid_searcher.search(
            query=query,
            top_k=candidate_k,
            candidate_k=candidate_k,
            metadata_filter=metadata_filter,
        )

        if not hybrid_results:
            return self._empty_response()

        if use_reranking:
            reranked_results = self.reranker.rerank(
                query=query,
                results=hybrid_results,
                top_k=min(
                    resolved_top_k,
                    self.reranker_top_k,
                    len(hybrid_results),
                ),
            )

            if not reranked_results:
                return self._empty_response()

            documents = [
                result.document
                for result in reranked_results
            ]

            sources = self._build_sources(
                reranked_results
            )

        else:
            selected_results = hybrid_results[
                :resolved_top_k
            ]

            documents = [
                result.document
                for result in selected_results
            ]

            sources = self._build_hybrid_sources(
                selected_results
            )

        answer = self.llm_generator.generate(
            query=query,
            documents=documents,
        )

        return RAGResponse(
            answer=answer,
            sources=sources,
            retrieved_documents=len(documents),
        )

    def query(
        self,
        query: str,
        top_k: int | None = None,
        use_reranking: bool = True,
        metadata_filter: dict[str, Any] | None = None,
    ) -> RAGResponse:

        return self.invoke(
            query=query,
            top_k=top_k,
            use_reranking=use_reranking,
            metadata_filter=metadata_filter,
        )

    @staticmethod
    def _build_sources(
        results: list[RerankedResult],
    ) -> list[dict[str, Any]]:

        sources: list[dict[str, Any]] = []

        for result in results:
            metadata = result.document.metadata

            source = {
                "rank": result.rank,
                "rerank_score": result.score,
                "original_rank": result.original_rank,
                "original_score": result.original_score,
                "source_type": metadata.get("source_type"),
                "source": metadata.get("source"),
                "file_name": metadata.get("file_name"),
                "page": metadata.get("page"),
                "citation": metadata.get("citation"),
                "chunk_id": metadata.get("chunk_id"),
            }

            sources.append(
                {
                    key: value
                    for key, value in source.items()
                    if value is not None
                }
            )

        return sources

    @staticmethod
    def _build_hybrid_sources(
        results: list[HybridSearchResult],
    ) -> list[dict[str, Any]]:

        sources: list[dict[str, Any]] = []

        for result in results:
            metadata = result.document.metadata

            source = {
                "rank": result.rank,
                "rerank_score": result.score,
                "original_rank": result.rank,
                "original_score": result.score,
                "source_type": metadata.get("source_type"),
                "source": metadata.get("source"),
                "file_name": metadata.get("file_name"),
                "page": metadata.get("page"),
                "citation": metadata.get("citation"),
                "chunk_id": metadata.get("chunk_id"),
            }

            sources.append(
                {
                    key: value
                    for key, value in source.items()
                    if value is not None
                }
            )

        return sources

    @staticmethod
    def _empty_response() -> RAGResponse:
        return RAGResponse(
            answer=(
                "I could not find relevant information "
                "in the available sources."
            ),
            sources=[],
            retrieved_documents=0,
        )