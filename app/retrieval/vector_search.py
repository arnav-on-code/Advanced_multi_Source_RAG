from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document

from app.embeddings.embedder import (
    DEFAULT_EMBEDDING_MODEL,
    get_embedding_model,
)
from app.vectorstore.chroma import (
    ChromaVectorStore,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIRECTORY,
)


@dataclass
class VectorSearchResult:
    """Result returned by semantic vector search."""

    document: Document
    score: float


class VectorSearcher:
    """
    Semantic retrieval layer using ChromaDB.

    Responsibilities:
        - Query validation
        - Vector similarity search
        - Score handling
        - Metadata filtering
    """

    def __init__(
        self,
        persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:

        embeddings = get_embedding_model(
            embedding_model
        )

        self.vectorstore = ChromaVectorStore(
            embedding_function=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """
        Perform semantic similarity search.

        Args:
            query: User's natural-language query.
            top_k: Maximum number of results.
            metadata_filter: Optional Chroma metadata filter.

        Returns:
            Ranked semantic search results.
        """

        if not query or not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        results = (
            self.vectorstore
            .similarity_search_with_score(
                query=query.strip(),
                k=top_k,
                filter=metadata_filter,
            )
        )

        return [
            VectorSearchResult(
                document=document,
                score=float(score),
            )
            for document, score in results
        ]

    def search_documents(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Return only documents from semantic search.
        """

        results = self.search(
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        return [
            result.document
            for result in results
        ]

    def get_retriever(
        self,
        top_k: int = 5,
        metadata_filter: dict[str, Any] | None = None,
    ):
        """
        Return a LangChain retriever for use in LCEL pipelines.
        """

        return self.vectorstore.get_retriever(
            k=top_k,
            filter=metadata_filter,
        )

    def count(self) -> int:
        """Return the number of chunks stored in ChromaDB."""

        return self.vectorstore.count()