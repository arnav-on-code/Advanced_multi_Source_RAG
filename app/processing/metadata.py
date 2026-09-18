from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


SUPPORTED_SOURCE_TYPES = {
    "pdf",
    "pdf_visual",
    "url",
    "youtube",
    "docx",
    "csv",
    "txt",
    "json",
}


def _stable_hash(value: str, length: int = 16) -> str:
    """Generate a stable SHA-256 based identifier."""
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()[:length]


def _normalize_value(value: Any) -> Any:
    """Convert metadata values into JSON-safe values."""

    if value is None:
        return None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (str, int, float, bool)):
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

    # Avoid storing raw binary data such as extracted image bytes
    # directly in Chroma metadata.
    if isinstance(value, bytes):
        return None

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

    The same source and content produce the same ID,
    which helps with deduplication.
    """

    metadata = document.metadata

    source = str(
        metadata.get("source", "")
    )

    source_type = str(
        metadata.get("source_type", "")
    )

    content_hash = hashlib.sha256(
        document.page_content.encode("utf-8")
    ).hexdigest()

    raw_id = (
        f"{source_type}|"
        f"{source}|"
        f"{content_hash}"
    )

    return _stable_hash(raw_id, length=24)


def create_chunk_id(
    document_id: str,
    chunk_index: int,
) -> str:
    """Create a deterministic ID for a document chunk."""

    return _stable_hash(
        f"{document_id}|chunk|{chunk_index}",
        length=24,
    )


def enrich_metadata(
    document: Document,
    *,
    chunk_index: int | None = None,
) -> Document:
    """
    Normalize and enrich metadata for downstream RAG processing.

    Adds:
        - source_type
        - source
        - document_id
        - chunk_id
        - has_text
        - citation information
    """

    metadata = {
        key: _normalize_value(value)
        for key, value in document.metadata.items()
    }

    source_type = str(
        metadata.get("source_type", "unknown")
    ).lower()

    if source_type not in SUPPORTED_SOURCE_TYPES:
        source_type = "unknown"

    source = str(
        metadata.get("source", "")
    ).strip()

    document.metadata = metadata

    document.metadata["source_type"] = source_type
    document.metadata["source"] = source
    document.metadata["has_text"] = bool(
        document.page_content.strip()
    )

    document_id = create_document_id(document)

    document.metadata["document_id"] = document_id

    if chunk_index is not None:
        document.metadata["chunk_index"] = chunk_index
        document.metadata["chunk_id"] = create_chunk_id(
            document_id,
            chunk_index,
        )

    document.metadata["citation"] = build_citation(
        document
    )

    return document


def build_citation(
    document: Document,
) -> dict[str, Any]:
    """
    Build structured citation metadata based on source type.

    This metadata can later be returned by the RAG API
    alongside the generated answer.
    """

    metadata = document.metadata

    source_type = metadata.get(
        "source_type",
        "unknown",
    )

    source = metadata.get(
        "source",
        "",
    )

    citation: dict[str, Any] = {
        "source_type": source_type,
        "source": source,
    }

    if source_type in {"pdf", "pdf_visual"}:
        if metadata.get("page") is not None:
            citation["page"] = metadata["page"]

        if metadata.get("file_name"):
            citation["file_name"] = metadata["file_name"]

    elif source_type == "youtube":
        if metadata.get("title"):
            citation["title"] = metadata["title"]

        if metadata.get("video_id"):
            citation["video_id"] = metadata["video_id"]

        if metadata.get("url"):
            citation["url"] = metadata["url"]

    elif source_type == "url":
        citation["url"] = metadata.get(
            "url",
            source,
        )

        if metadata.get("title"):
            citation["title"] = metadata["title"]

    elif source_type in {
        "docx",
        "csv",
        "txt",
        "json",
    }:
        if metadata.get("file_name"):
            citation["file_name"] = metadata["file_name"]

    return citation


def enrich_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Enrich a list of documents with standardized metadata.

    Chunk IDs are intentionally not created here because
    chunking happens later.
    """

    enriched_documents = []

    for document in documents:
        enriched_documents.append(
            enrich_metadata(document)
        )

    return enriched_documents