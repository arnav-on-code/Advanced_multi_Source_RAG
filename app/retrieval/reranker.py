from dataclasses import dataclass

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.retrieval.hybrid_search import HybridSearchResult


DEFAULT_RERANKER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


@dataclass
class RerankedResult:
    """Result returned after cross-encoder re-ranking."""

    document: Document
    score: float
    original_score: float
    original_rank: int
    rank: int


class Reranker:

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:

        if max_length <= 0:
            raise ValueError(
                "max_length must be greater than zero."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        self.model_name = model_name
        self.batch_size = batch_size

        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=device,
        )

    def rerank(
        self,
        query: str,
        results: list[HybridSearchResult],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        """Re-rank hybrid-search candidates."""

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        if not 1 <= top_k <= 100:
            raise ValueError(
                "top_k must be between 1 and 100."
            )

        if not results:
            return []

        pairs = [
            (
                query,
                result.document.page_content,
            )
            for result in results
        ]

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        reranked = [
            RerankedResult(
                document=result.document,
                score=float(score),
                original_score=result.score,
                original_rank=result.rank,
                rank=0,
            )
            for result, score in zip(
                results,
                scores,
            )
        ]

        reranked.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        final_results = reranked[:top_k]

        for rank, result in enumerate(
            final_results,
            start=1,
        ):
            result.rank = rank

        return final_results