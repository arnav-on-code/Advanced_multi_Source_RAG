from dataclasses import dataclass
import re
from heapq import nlargest

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


@dataclass
class BM25SearchResult:
    """Result returned by BM25 lexical search."""

    document: Document
    score: float


class BM25Retriever:
    """
    BM25 lexical retriever.

    Useful for:
        - Exact terms
        - Names
        - Identifiers
        - Technical terminology
        - Keyword-heavy queries
    """

    def __init__(
        self,
        documents: list[Document] | None = None,
    ) -> None:

        self.documents: list[Document] = []
        self.bm25: BM25Okapi | None = None

        if documents:
            self.fit(documents)

    def fit(
        self,
        documents: list[Document],
    ) -> None:
        """Build the BM25 index from documents."""

        valid_documents = [
            document
            for document in documents
            if document.page_content.strip()
        ]

        if not valid_documents:
            raise ValueError(
                "Cannot build BM25 index from empty documents."
            )

        self.documents = valid_documents

        tokenized_documents = [
            self._tokenize(document.page_content)
            for document in self.documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[BM25SearchResult]:
        """Search the BM25 index."""

        query = query.strip()

        if not query:
            raise ValueError(
                "Search query cannot be empty."
            )

        if not 1 <= top_k <= 100:
            raise ValueError(
                "top_k must be between 1 and 100."
            )

        if self.bm25 is None:
            raise RuntimeError(
                "BM25 index has not been built. "
                "Call fit() first."
            )

        tokenized_query = self._tokenize(query)

        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = nlargest(
            top_k,
            range(len(scores)),
            key=scores.__getitem__,
        )

        return [
            BM25SearchResult(
                document=self.documents[index],
                score=float(scores[index]),
            )
            for index in ranked_indices
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        Tokenize text for BM25.

        Lowercases text and removes punctuation while
        preserving useful technical terms.
        """

        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )

    def count(self) -> int:
        """Return the number of indexed documents."""

        return len(self.documents)