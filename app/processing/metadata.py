from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


SUPPORTED_SOURCE_TYPES = {
    "pdf",
    "url",
    "youtube",
    "docx",
    "csv",
    "txt",
    "json",
}

DEFAULT_ID_LENGTH = 24


def _stable_hash(
    value: str,
    length: int = DEFAULT_ID_LENGTH,
) -> str:
    """Generate a deterministic SHA-256 identifier."""

    if length <= 0:
        raise ValueError(
            "Hash length must be greater than zero."
        )

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()[:length]


def _normalize_value(value: Any) -> Any:
    """
    Convert metadata values into JSON-safe values.
    """

    if value is None:
        return None

    if isinstance(value, bytes):
        return None

    if isinstance(value, Path):
        return str(value)

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(value, (list, tuple)):
        return [
            _normalize_value(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            str(key): _normalize_value(item)
            for key, item in value.items()
        }

    try:
        json.dumps(value)
        return value

    except (TypeError, ValueError):
        return str(value)


def create_document_id(
    document: Document,
) -> str:
    """
    Create a deterministic ID for a source document.

    The ID is based on:
        - source type
        - source
        - document content
    """

    if not isinstance(document, Document):
        raise TypeError(
            "document must be a LangChain Document."
        )

    metadata = document.metadata

    source_type = str(
        metadata.get(
            "source_type",
            "unknown",
        )
    ).strip().lower()

    source = str(
        metadata.get(
            "source",
            "",
        )
    ).strip()

    content = document.page_content or ""

    content_hash = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

    return _stable_hash(
        f"{source_type}|{source}|{content_hash}"
    )


def create_chunk_id(
    document_id: str,
    chunk_index: int,
) -> str:
    """
    Create a deterministic ID for a document chunk.
    """

    document_id = str(document_id).strip()

    if not document_id:
        raise ValueError(
            "document_id cannot be empty."
        )

    if chunk_index < 0:
        raise ValueError(
            "chunk_index cannot be negative."
        )

    return _stable_hash(
        f"{document_id}|chunk|{chunk_index}"
    )


def build_citation(
    document: Document,
) -> dict[str, Any]:
    """
    Build structured citation metadata for a document.
    """

    if not isinstance(document, Document):
        raise TypeError(
            "document must be a LangChain Document."
        )

    metadata = document.metadata

    source_type = str(
        metadata.get(
            "source_type",
            "unknown",
        )
    ).strip().lower()

    source = str(
        metadata.get(
            "source",
            "",
        )
    ).strip()

    citation: dict[str, Any] = {
        "source_type": source_type,
        "source": source,
    }

    # -----------------------------------------------------
    # PDF
    # -----------------------------------------------------

    if source_type == "pdf":

        if metadata.get("page") is not None:
            citation["page"] = metadata["page"]

        if metadata.get("file_name"):
            citation["file_name"] = metadata[
                "file_name"
            ]

        if metadata.get("content_type"):
            citation["content_type"] = metadata[
                "content_type"
            ]

    # -----------------------------------------------------
    # YOUTUBE
    # -----------------------------------------------------

    elif source_type == "youtube":

        if metadata.get("title"):
            citation["title"] = metadata["title"]

        if metadata.get("video_id"):
            citation["video_id"] = metadata[
                "video_id"
            ]

        if metadata.get("url"):
            citation["url"] = metadata["url"]

        elif source:
            citation["url"] = source

    # -----------------------------------------------------
    # URL
    # -----------------------------------------------------

    elif source_type == "url":

        citation["url"] = (
            metadata.get("url")
            or source
        )

        if metadata.get("title"):
            citation["title"] = metadata["title"]

    # -----------------------------------------------------
    # FILE-BASED SOURCES
    # -----------------------------------------------------

    elif source_type in {
        "docx",
        "csv",
        "txt",
        "json",
    }:

        if metadata.get("file_name"):
            citation["file_name"] = metadata[
                "file_name"
            ]

        if metadata.get("row") is not None:
            citation["row"] = metadata["row"]

    return citation


def enrich_document(
    document: Document,
) -> Document:
    """
    Normalize and enrich one document.

    Adds:
        - normalized metadata
        - source_type
        - source
        - deterministic document_id
        - citation

    This function should run before chunking.

    A new Document is returned so the original document
    is not modified.
    """

    if not isinstance(document, Document):
        raise TypeError(
            "document must be a LangChain Document."
        )

    metadata = {
        str(key): _normalize_value(value)
        for key, value in document.metadata.items()
    }

    source_type = str(
        metadata.get(
            "source_type",
            "unknown",
        )
    ).strip().lower()

    if source_type not in SUPPORTED_SOURCE_TYPES:
        source_type = "unknown"

    source = str(
        metadata.get(
            "source",
            "",
        )
    ).strip()

    metadata["source_type"] = source_type
    metadata["source"] = source

    enriched = Document(
        page_content=document.page_content,
        metadata=metadata,
    )

    enriched.metadata["document_id"] = (
        create_document_id(enriched)
    )

    enriched.metadata["citation"] = (
        build_citation(enriched)
    )

    return enriched


def enrich_metadata(
    document: Document,
) -> Document:
    """
    Backward-compatible alias for enrich_document().

    Prefer enrich_document() in new code.
    """

    return enrich_document(document)


def enrich_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Normalize and enrich multiple documents while
    preserving their order.
    """

    if not documents:
        return []

    return [
        enrich_document(document)
        for document in documents
    ]