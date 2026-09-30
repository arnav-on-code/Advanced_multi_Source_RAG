from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from app.core.exceptions import VectorStoreError


DEFAULT_COLLECTION_NAME = "p1_rag_documents"
DEFAULT_PERSIST_DIRECTORY = "data/chroma"

DEFAULT_SEARCH_K = 5
MAX_SEARCH_K = 100
DEFAULT_BATCH_SIZE = 100


class ChromaVectorStore:
    """
    Production-ready ChromaDB vector-store wrapper.

    Responsibilities:
        - Persistent vector storage
        - Document/chunk insertion
        - Idempotent upsert
        - Semantic similarity search
        - Metadata-filtered search
        - LangChain retriever creation
        - Document deletion
        - Source-level deletion
        - Collection statistics
        - Collection reset
    """

    def __init__(
        self,
        embedding_function: Embeddings,
        persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:

        if embedding_function is None:
            raise ValueError(
                "embedding_function cannot be None."
            )

        collection_name = collection_name.strip()

        if not collection_name:
            raise ValueError(
                "collection_name cannot be empty."
            )

        persist_path = Path(persist_directory)

        try:
            persist_path.mkdir(
                parents=True,
                exist_ok=True,
            )
        except OSError as exc:
            raise VectorStoreError(
                f"Unable to create Chroma directory: {persist_path}"
            ) from exc

        self.persist_directory = persist_path
        self._collection_name = collection_name

        try:
            self.vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=embedding_function,
                persist_directory=str(persist_path),
            )
        except Exception as exc:
            raise VectorStoreError(
                "Failed to initialize ChromaDB."
            ) from exc

    # =========================================================
    # INSERT / UPSERT
    # =========================================================

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> list[str]:
        """
        Add documents/chunks to ChromaDB.

        Existing IDs may cause conflicts. Use upsert_documents()
        when ingestion must be idempotent.
        """

        if not documents:
            return []

        self._validate_batch_size(batch_size)

        resolved_ids = (
            ids
            if ids is not None
            else self._generate_ids(documents)
        )

        self._validate_ids(
            resolved_ids,
            len(documents),
        )

        try:
            for start in range(
                0,
                len(documents),
                batch_size,
            ):
                end = start + batch_size

                self.vectorstore.add_documents(
                    documents=documents[start:end],
                    ids=resolved_ids[start:end],
                )

        except Exception as exc:
            raise VectorStoreError(
                "Failed to add documents to ChromaDB."
            ) from exc

        return resolved_ids

    def upsert_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> list[str]:
        """
        Insert or update documents using deterministic IDs.

        Recommended for the RAG ingestion pipeline because
        repeated ingestion of the same chunks remains idempotent.
        """

        if not documents:
            return []

        self._validate_batch_size(batch_size)

        resolved_ids = (
            ids
            if ids is not None
            else self._generate_ids(documents)
        )

        self._validate_ids(
            resolved_ids,
            len(documents),
        )

        try:
            for start in range(
                0,
                len(documents),
                batch_size,
            ):
                end = start + batch_size

                self.vectorstore.update_documents(
                    ids=resolved_ids[start:end],
                    documents=documents[start:end],
                )

        except Exception:
            # Chroma update_documents requires existing IDs.
            # Fall back to add for IDs that are not already present.
            try:
                for start in range(
                    0,
                    len(documents),
                    batch_size,
                ):
                    end = start + batch_size

                    self.vectorstore.add_documents(
                        documents=documents[start:end],
                        ids=resolved_ids[start:end],
                    )

            except Exception as exc:
                raise VectorStoreError(
                    "Failed to upsert documents into ChromaDB."
                ) from exc

        return resolved_ids

    # =========================================================
    # SEARCH
    # =========================================================

    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_SEARCH_K,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Perform semantic similarity search.
        """

        self._validate_query(query)
        self._validate_k(k)

        try:
            return self.vectorstore.similarity_search(
                query=query.strip(),
                k=k,
                filter=metadata_filter,
            )
        except Exception as exc:
            raise VectorStoreError(
                "Chroma similarity search failed."
            ) from exc

    def similarity_search_with_score(
        self,
        query: str,
        k: int = DEFAULT_SEARCH_K,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Perform semantic search with Chroma distance scores.

        Important:
            Chroma generally returns distances, where lower
            values indicate greater similarity.
        """

        self._validate_query(query)
        self._validate_k(k)

        try:
            return self.vectorstore.similarity_search_with_score(
                query=query.strip(),
                k=k,
                filter=metadata_filter,
            )
        except Exception as exc:
            raise VectorStoreError(
                "Chroma similarity search with score failed."
            ) from exc

    # =========================================================
    # RETRIEVER
    # =========================================================

    def get_retriever(
        self,
        k: int = DEFAULT_SEARCH_K,
        metadata_filter: dict[str, Any] | None = None,
    ) -> BaseRetriever:
        """
        Create a LangChain retriever backed by ChromaDB.
        """

        self._validate_k(k)

        search_kwargs: dict[str, Any] = {
            "k": k,
        }

        if metadata_filter:
            search_kwargs["filter"] = metadata_filter

        try:
            return self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs=search_kwargs,
            )
        except Exception as exc:
            raise VectorStoreError(
                "Failed to create Chroma retriever."
            ) from exc

    # =========================================================
    # DELETE
    # =========================================================

    def delete(
        self,
        ids: list[str],
    ) -> None:
        """
        Delete specific document/chunk IDs.
        """

        if not ids:
            return

        ids = self._clean_ids(ids)

        if not ids:
            return

        try:
            self.vectorstore.delete(
                ids=ids,
            )
        except Exception as exc:
            raise VectorStoreError(
                "Failed to delete documents from ChromaDB."
            ) from exc

    def delete_by_source(
        self,
        source: str,
    ) -> None:
        """
        Delete all chunks belonging to a source.

        Useful when re-ingesting an updated PDF, DOCX,
        URL, CSV, TXT, or JSON source.
        """

        source = source.strip()

        if not source:
            raise ValueError(
                "source cannot be empty."
            )

        try:
            collection = self.vectorstore._collection

            collection.delete(
                where={
                    "source": source,
                }
            )

        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete source: {source}"
            ) from exc

    # =========================================================
    # COLLECTION INFORMATION
    # =========================================================

    def count(self) -> int:
        """
        Return the number of vectors stored.
        """

        try:
            return self.vectorstore._collection.count()
        except Exception as exc:
            raise VectorStoreError(
                "Failed to retrieve Chroma collection count."
            ) from exc

    @property
    def collection_name(self) -> str:
        """Return the active collection name."""

        return self._collection_name

    # =========================================================
    # RESET
    # =========================================================

    def reset(self) -> None:
        """
        Delete the complete Chroma collection.

        WARNING:
            This permanently removes all vectors.
        """

        try:
            self.vectorstore.delete_collection()

            # Re-create the collection so the same instance
            # remains usable after reset.
            self.vectorstore = Chroma(
                collection_name=self._collection_name,
                embedding_function=self.vectorstore._embedding_function,
                persist_directory=str(
                    self.persist_directory
                ),
            )

        except Exception as exc:
            raise VectorStoreError(
                "Failed to reset Chroma collection."
            ) from exc

    # =========================================================
    # INTERNAL HELPERS
    # =========================================================

    @staticmethod
    def _generate_ids(
        documents: list[Document],
    ) -> list[str]:
        """
        Generate deterministic IDs.

        Priority:
            1. chunk_id
            2. document_id
            3. SHA-256(source + content)
        """

        ids: list[str] = []
        used_ids: set[str] = set()

        for document in documents:
            metadata = document.metadata

            existing_id = (
                metadata.get("chunk_id")
                or metadata.get("document_id")
            )

            if existing_id:
                base_id = str(existing_id)

            else:
                source = str(
                    metadata.get(
                        "source",
                        "unknown",
                    )
                )

                content = document.page_content

                raw_id = (
                    f"{source}|{content}"
                )

                base_id = hashlib.sha256(
                    raw_id.encode("utf-8")
                ).hexdigest()[:32]

            candidate = base_id
            counter = 1

            while candidate in used_ids:
                candidate = (
                    f"{base_id}_{counter}"
                )
                counter += 1

            used_ids.add(candidate)
            ids.append(candidate)

        return ids

    @staticmethod
    def _validate_ids(
        ids: list[str],
        expected_length: int,
    ) -> None:

        if len(ids) != expected_length:
            raise ValueError(
                "Number of IDs must match "
                "number of documents."
            )

        cleaned_ids = [
            str(value).strip()
            for value in ids
        ]

        if any(not value for value in cleaned_ids):
            raise ValueError(
                "Document IDs cannot be empty."
            )

        if len(set(cleaned_ids)) != len(cleaned_ids):
            raise ValueError(
                "Document IDs must be unique."
            )

    @staticmethod
    def _clean_ids(
        ids: list[str],
    ) -> list[str]:

        return list(
            dict.fromkeys(
                str(value).strip()
                for value in ids
                if str(value).strip()
            )
        )

    @staticmethod
    def _validate_query(
        query: str,
    ) -> None:

        if not isinstance(query, str):
            raise ValueError(
                "Search query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "Search query cannot be empty."
            )

    @staticmethod
    def _validate_k(
        k: int,
    ) -> None:

        if not isinstance(k, int):
            raise ValueError(
                "k must be an integer."
            )

        if not 1 <= k <= MAX_SEARCH_K:
            raise ValueError(
                f"k must be between 1 and {MAX_SEARCH_K}."
            )

    @staticmethod
    def _validate_batch_size(
        batch_size: int,
    ) -> None:

        if not isinstance(batch_size, int):
            raise ValueError(
                "batch_size must be an integer."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )