from __future__ import annotations

import hashlib
import json
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
    """Persistent ChromaDB wrapper for the RAG pipeline."""

    def __init__(
        self,
        embedding_function: Embeddings,
        persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        if embedding_function is None:
            raise ValueError("embedding_function cannot be None.")

        collection_name = str(collection_name).strip()
        if not collection_name:
            raise ValueError("collection_name cannot be empty.")

        persist_path = Path(persist_directory)

        try:
            persist_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise VectorStoreError(
                f"Unable to create Chroma directory: {persist_path}"
            ) from exc

        self.persist_directory = persist_path
        self._collection_name = collection_name
        self._embedding_function = embedding_function

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

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> list[str]:
        """Add documents. Existing IDs are not overwritten."""
        if not documents:
            return []

        self._validate_batch_size(batch_size)
        prepared = self._prepare_documents(documents)

        resolved_ids = (
            self._generate_ids(prepared) if ids is None
            else self._normalize_ids(ids, len(prepared))
        )

        try:
            for start in range(0, len(prepared), batch_size):
                end = start + batch_size
                self.vectorstore.add_documents(
                    documents=prepared[start:end],
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
        Insert new documents and update existing IDs.

        This uses LangChain's public Chroma methods so the configured
        embedding function is always used for both inserts and updates.
        """
        if not documents:
            return []

        self._validate_batch_size(batch_size)
        prepared = self._prepare_documents(documents)

        resolved_ids = (
            self._generate_ids(prepared) if ids is None
            else self._normalize_ids(ids, len(prepared))
        )

        try:
            for start in range(0, len(prepared), batch_size):
                end = start + batch_size
                batch_documents = prepared[start:end]
                batch_ids = resolved_ids[start:end]

                existing_data = self.vectorstore._collection.get(
                    ids=batch_ids,
                    include=[],
                )
                existing_ids = set(existing_data.get("ids") or [])

                update_documents: list[Document] = []
                update_ids: list[str] = []
                add_documents: list[Document] = []
                add_ids: list[str] = []

                for document, document_id in zip(
                    batch_documents,
                    batch_ids,
                ):
                    if document_id in existing_ids:
                        update_documents.append(document)
                        update_ids.append(document_id)
                    else:
                        add_documents.append(document)
                        add_ids.append(document_id)

                if update_documents:
                    self.vectorstore.update_documents(
                        ids=update_ids,
                        documents=update_documents,
                    )

                if add_documents:
                    self.vectorstore.add_documents(
                        documents=add_documents,
                        ids=add_ids,
                    )

        except Exception as exc:
            raise VectorStoreError(
                "Failed to upsert documents into ChromaDB."
            ) from exc

        return resolved_ids

    def get_documents(self) -> list[Document]:
        """Return all stored documents for lexical retrieval."""
        try:
            data = self.vectorstore._collection.get(
                include=["documents", "metadatas"]
            )

            documents = data.get("documents") or []
            metadatas = data.get("metadatas") or []

            return [
                Document(
                    page_content=str(content),
                    metadata=self._sanitize_metadata(metadata or {}),
                )
                for content, metadata in zip(documents, metadatas)
                if content is not None
            ]
        except Exception as exc:
            raise VectorStoreError(
                "Failed to retrieve documents from ChromaDB."
            ) from exc

    def similarity_search(
        self,
        query: str,
        k: int = DEFAULT_SEARCH_K,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Perform semantic similarity search."""

        self._validate_query(query)
        self._validate_k(k)

        try:
            if metadata_filter:
                embedding = self._embedding_function.embed_query(
                    query.strip()
                )

                results = self.vectorstore._collection.query(
                    query_embeddings=[embedding],
                    n_results=k,
                    where=metadata_filter,
                )

                documents = results.get("documents") or [[]]
                metadatas = results.get("metadatas") or [[]]

                return [
                    Document(
                        page_content=str(content),
                        metadata=self._sanitize_metadata(metadata or {}),
                    )
                    for content, metadata in zip(
                        documents[0],
                        metadatas[0],
                    )
                    if content is not None
                ]

            return self.vectorstore.similarity_search(
                query=query.strip(),
                k=k,
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
        """Perform semantic similarity search and return distances."""

        self._validate_query(query)
        self._validate_k(k)

        try:
            if metadata_filter:
                embedding = self._embedding_function.embed_query(
                    query.strip()
                )

                results = self.vectorstore._collection.query(
                    query_embeddings=[embedding],
                    n_results=k,
                    where=metadata_filter,
                    include=[
                        "documents",
                        "metadatas",
                        "distances",
                    ],
                )

                documents = results.get("documents") or [[]]
                metadatas = results.get("metadatas") or [[]]
                distances = results.get("distances") or [[]]

                return [
                    (
                        Document(
                            page_content=str(content),
                            metadata=self._sanitize_metadata(metadata or {}),
                        ),
                        float(distance),
                    )
                    for content, metadata, distance in zip(
                        documents[0],
                        metadatas[0],
                        distances[0],
                    )
                    if content is not None
                ]

            return self.vectorstore.similarity_search_with_score(
                query=query.strip(),
                k=k,
            )

        except Exception as exc:
            raise VectorStoreError(
                "Chroma similarity search with score failed."
            ) from exc

    def get_retriever(
        self,
        k: int = DEFAULT_SEARCH_K,
        metadata_filter: dict[str, Any] | None = None,
    ) -> BaseRetriever:
        self._validate_k(k)

        search_kwargs: dict[str, Any] = {"k": k}
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

    def delete(self, ids: list[str]) -> None:
        """Delete specific document/chunk IDs."""
        cleaned_ids = self._clean_ids(ids)
        if not cleaned_ids:
            return

        try:
            self.vectorstore.delete(ids=cleaned_ids)
        except Exception as exc:
            raise VectorStoreError(
                "Failed to delete documents from ChromaDB."
            ) from exc

    def delete_by_source(self, source: str) -> None:
        """Delete every chunk whose source metadata matches exactly."""
        source = str(source).strip()
        if not source:
            raise ValueError("source cannot be empty.")

        try:
            collection = self.vectorstore._collection
            data = collection.get(
                where={"source": source},
                include=[],
            )
            ids = data.get("ids") or []
            if ids:
                collection.delete(ids=ids)
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete source: {source}"
            ) from exc

    def count(self) -> int:
        """Return the number of stored vectors."""
        try:
            return int(self.vectorstore._collection.count())
        except Exception as exc:
            raise VectorStoreError(
                "Failed to retrieve Chroma collection count."
            ) from exc

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def reset(self) -> None:
        """Delete and recreate the complete collection."""
        try:
            self.vectorstore.delete_collection()
            self.vectorstore = Chroma(
                collection_name=self._collection_name,
                embedding_function=self._embedding_function,
                persist_directory=str(self.persist_directory),
            )
        except Exception as exc:
            raise VectorStoreError(
                "Failed to reset Chroma collection."
            ) from exc

    @staticmethod
    def _prepare_documents(documents: list[Document]) -> list[Document]:
        """Return copies with Chroma-safe metadata."""
        prepared: list[Document] = []

        for document in documents:
            if not isinstance(document, Document):
                raise TypeError("All items must be LangChain Document objects.")

            prepared.append(
                Document(
                    page_content=str(document.page_content),
                    metadata=ChromaVectorStore._sanitize_metadata(
                        document.metadata
                    ),
                )
            )

        return prepared

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """
        Convert metadata to Chroma-supported values.

        Chroma does not accept nested dictionaries or arbitrary objects
        as metadata values.
        """
        if not metadata:
            return {}

        sanitized: dict[str, Any] = {}

        for key, value in metadata.items():
            clean_key = str(key)

            if value is None:
                sanitized[clean_key] = None
            elif isinstance(value, (str, int, float, bool)):
                sanitized[clean_key] = value
            elif isinstance(value, (list, tuple, dict)):
                sanitized[clean_key] = json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
            else:
                sanitized[clean_key] = str(value)

        return sanitized

    @staticmethod
    def _generate_ids(documents: list[Document]) -> list[str]:
        """
        Generate stable IDs.

        Priority:
            1. chunk_id
            2. document_id + chunk position
            3. SHA-256(source + content)
        """
        ids: list[str] = []
        used_ids: set[str] = set()

        for index, document in enumerate(documents):
            metadata = document.metadata
            chunk_id = metadata.get("chunk_id")

            if chunk_id:
                base_id = str(chunk_id).strip()
            else:
                document_id = metadata.get("document_id")

                if document_id:
                    raw_id = (
                        f"{document_id}|{index}|"
                        f"{document.page_content}"
                    )
                else:
                    source = str(metadata.get("source", "unknown"))
                    raw_id = f"{source}|{document.page_content}"

                base_id = hashlib.sha256(
                    raw_id.encode("utf-8")
                ).hexdigest()[:32]

            if not base_id:
                raise ValueError("Generated document ID cannot be empty.")

            candidate = base_id
            counter = 1
            while candidate in used_ids:
                candidate = f"{base_id}_{counter}"
                counter += 1

            used_ids.add(candidate)
            ids.append(candidate)

        return ids

    @staticmethod
    def _normalize_ids(
        ids: list[str],
        expected_length: int,
    ) -> list[str]:
        if len(ids) != expected_length:
            raise ValueError(
                "Number of IDs must match number of documents."
            )

        cleaned = [str(value).strip() for value in ids]

        if any(not value for value in cleaned):
            raise ValueError("Document IDs cannot be empty.")

        if len(set(cleaned)) != len(cleaned):
            raise ValueError("Document IDs must be unique.")

        return cleaned

    @staticmethod
    def _clean_ids(ids: list[str]) -> list[str]:
        return list(
            dict.fromkeys(
                str(value).strip()
                for value in ids
                if str(value).strip()
            )
        )

    @staticmethod
    def _validate_query(query: str) -> None:
        if not isinstance(query, str):
            raise ValueError("Search query must be a string.")
        if not query.strip():
            raise ValueError("Search query cannot be empty.")

    @staticmethod
    def _validate_k(k: int) -> None:
        if isinstance(k, bool) or not isinstance(k, int):
            raise ValueError("k must be an integer.")
        if not 1 <= k <= MAX_SEARCH_K:
            raise ValueError(
                f"k must be between 1 and {MAX_SEARCH_K}."
            )

    @staticmethod
    def _validate_batch_size(batch_size: int) -> None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise ValueError("batch_size must be an integer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
