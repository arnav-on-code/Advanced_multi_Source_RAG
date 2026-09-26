from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.vector_search import VectorSearcher


@dataclass
class HybridSearchResult:
    """Result returned by hybrid retrieval."""

    document: Document
    score: float
    vector_score: float
    bm25_score: float
    rank: int


class HybridSearcher:
    """
    Hybrid retriever combining:

        1. Semantic vector search
        2. BM25 lexical search

    Scores are normalized and combined using
    weighted score fusion.
    """

    def __init__(
        self,
        vector_searcher: VectorSearcher,
        bm25_retriever: BM25Retriever,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
    ) -> None:

        if vector_weight < 0 or bm25_weight < 0:
            raise ValueError(
                "Search weights cannot be negative."
            )

        total_weight = vector_weight + bm25_weight

        if total_weight <= 0:
            raise ValueError(
                "At least one search weight must be greater than zero."
            )

        self.vector_searcher = vector_searcher
        self.bm25_retriever = bm25_retriever

        self.vector_weight = vector_weight / total_weight
        self.bm25_weight = bm25_weight / total_weight

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[HybridSearchResult]:
        """Perform hybrid vector + BM25 retrieval."""

        query = query.strip()

        if not query:
            raise ValueError(
                "Search query cannot be empty."
            )

        if not 1 <= top_k <= 100:
            raise ValueError(
                "top_k must be between 1 and 100."
            )

        if candidate_k is None:
            candidate_k = max(top_k * 3, 10)

        if not 1 <= candidate_k <= 200:
            raise ValueError(
                "candidate_k must be between 1 and 200."
            )

        vector_results = self.vector_searcher.search(
            query=query,
            top_k=candidate_k,
            metadata_filter=metadata_filter,
        )

        bm25_results = self.bm25_retriever.search(
            query=query,
            top_k=candidate_k,
        )

        # Chroma similarity_search_with_score() normally
        # returns distance values: lower distance = better.
        vector_distances = [
            result.score
            for result in vector_results
        ]

        bm25_scores = [
            result.score
            for result in bm25_results
        ]

        vector_scores = self._normalize_vector_distances(
            vector_distances
        )

        bm25_scores = self._normalize_scores(
            bm25_scores
        )

        fused: dict[str, dict[str, Any]] = {}

        for index, result in enumerate(vector_results):
            document_id = self._document_key(
                result.document
            )

            item = fused.setdefault(
                document_id,
                {
                    "document": result.document,
                    "vector_score": 0.0,
                    "bm25_score": 0.0,
                },
            )

            item["vector_score"] = vector_scores[index]

        for index, result in enumerate(bm25_results):
            document_id = self._document_key(
                result.document
            )

            item = fused.setdefault(
                document_id,
                {
                    "document": result.document,
                    "vector_score": 0.0,
                    "bm25_score": 0.0,
                },
            )

            item["bm25_score"] = bm25_scores[index]

        results = []

        for item in fused.values():
            vector_score = item["vector_score"]
            bm25_score = item["bm25_score"]

            final_score = (
                self.vector_weight * vector_score
                + self.bm25_weight * bm25_score
            )

            results.append(
                HybridSearchResult(
                    document=item["document"],
                    score=final_score,
                    vector_score=vector_score,
                    bm25_score=bm25_score,
                    rank=0,
                )
            )

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        final_results = results[:top_k]

        for rank, result in enumerate(
            final_results,
            start=1,
        ):
            result.rank = rank

        return final_results

    @staticmethod
    def _normalize_vector_distances(
        distances: list[float],
    ) -> list[float]:
        """
        Convert Chroma distances into similarity scores [0, 1].

        Lower distance means better similarity.
        """

        if not distances:
            return []

        minimum = min(distances)
        maximum = max(distances)

        if maximum == minimum:
            return [1.0] * len(distances)

        return [
            (maximum - distance)
            / (maximum - minimum)
            for distance in distances
        ]

    @staticmethod
    def _normalize_scores(
        scores: list[float],
    ) -> list[float]:
        """Min-max normalize scores to [0, 1]."""

        if not scores:
            return []

        minimum = min(scores)
        maximum = max(scores)

        if maximum == minimum:
            return [1.0] * len(scores)

        return [
            (score - minimum)
            / (maximum - minimum)
            for score in scores
        ]

    @staticmethod
    def _document_key(
        document: Document,
    ) -> str:
        """Generate a stable key for merging retrieval results."""

        metadata = document.metadata

        chunk_id = metadata.get("chunk_id")
        if chunk_id:
            return str(chunk_id)

        document_id = metadata.get("document_id")
        if document_id:
            return str(document_id)

        return (
            f"{metadata.get('source', '')}|"
            f"{document.page_content}"
        )