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


def _stable_hash(value: str, length: int = 24) -> str:
    """Generate a deterministic SHA-256 identifier."""

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()[:length]


def _normalize_value(value: Any) -> Any:
    """Convert metadata values into JSON-safe values."""

    if value is None:
        return None

    if isinstance(value, bytes):
        return None

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]

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


def create_document_id(document: Document) -> str:
    """
    Create a deterministic ID for the complete source document.

    The ID is based on source type, source, and document content.
    """

    metadata = document.metadata

    source_type = str(
        metadata.get("source_type", "unknown")
    ).lower()

    source = str(
        metadata.get("source", "")
    ).strip()

    content_hash = hashlib.sha256(
        document.page_content.encode("utf-8")
    ).hexdigest()

    return _stable_hash(
        f"{source_type}|{source}|{content_hash}"
    )


def create_chunk_id(
    document_id: str,
    chunk_index: int,
) -> str:
    """Create a deterministic ID for a document chunk."""

    return _stable_hash(
        f"{document_id}|chunk|{chunk_index}"
    )


def build_citation(document: Document) -> dict[str, Any]:
    """Build structured citation metadata for a document."""

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

    if source_type == "pdf":
        if metadata.get("page") is not None:
            citation["page"] = metadata["page"]

        if metadata.get("file_name"):
            citation["file_name"] = metadata["file_name"]

        if metadata.get("content_type"):
            citation["content_type"] = metadata["content_type"]

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

        if metadata.get("row") is not None:
            citation["row"] = metadata["row"]

    return citation


def enrich_metadata(document: Document) -> Document:
    """
    Normalize document metadata and assign a deterministic
    document ID and citation information.

    This function should run before chunking.
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

    metadata["source_type"] = source_type
    metadata["source"] = str(
        metadata.get("source", "")
    ).strip()

    document.metadata = metadata

    document.metadata["document_id"] = create_document_id(
        document
    )

    document.metadata["citation"] = build_citation(
        document
    )

    return document


def enrich_documents(
    documents: list[Document],
) -> list[Document]:
    """Normalize and enrich a list of documents."""

    return [
        enrich_metadata(document)
        for document in documents
    ]