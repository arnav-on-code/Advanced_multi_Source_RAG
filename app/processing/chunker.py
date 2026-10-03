from __future__ import annotations

from uuid import uuid4

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


def create_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """Create the default text splitter for the RAG pipeline."""

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            ", ",
            " ",
            "",
        ],
        keep_separator=True,
    )


def _ensure_document_id(document: Document) -> str:
    """
    Return the existing document_id.

    A document_id is required because it is used to
    generate deterministic chunk identifiers.
    """
    document_id = document.metadata["document_id"]

    if not document_id:
        raise KeyError(
            "Each document must contain 'document_id' "
            "in its metadata."
        )

    return str(document_id)


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split documents into retrieval-friendly chunks.

    Each chunk receives:

        document_id
        chunk_index
        chunk_id
        chunk_size

    Existing document IDs are preserved. If a document does not
    have a document_id, a unique ID is generated automatically.
    """

    if not documents:
        return []

    # ------------------------------------------------------------------
    # VALIDATION + DOCUMENT IDs
    # ------------------------------------------------------------------

    for document in documents:
        if not isinstance(document, Document):
            raise TypeError(
                "documents must contain only LangChain Document objects."
            )

        if not document.page_content.strip():
            continue

        _ensure_document_id(document)

    # ------------------------------------------------------------------
    # SPLITTING
    # ------------------------------------------------------------------

    splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)

    # ------------------------------------------------------------------
    # CHUNK METADATA
    # ------------------------------------------------------------------

    document_chunk_counts: dict[str, int] = {}

    for chunk in chunks:
        document_id = chunk.metadata.get("document_id")

        if not document_id:
            # This should normally never happen because IDs were
            # assigned before splitting.
            document_id = uuid4().hex
            chunk.metadata["document_id"] = document_id

        document_id = str(document_id)

        chunk_index = document_chunk_counts.get(
            document_id,
            0,
        )

        chunk.metadata["chunk_index"] = chunk_index

        # Human-readable and deterministic chunk ID.
        chunk.metadata["chunk_id"] = (
            f"{document_id}_chunk_{chunk_index}"
        )

        chunk.metadata["chunk_size"] = len(
            chunk.page_content
        )

        document_chunk_counts[document_id] = (
            chunk_index + 1
        )

    return chunks


def chunk_document(
    document: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split a single document into retrieval-friendly chunks."""

    if not isinstance(document, Document):
        raise TypeError(
            "document must be a LangChain Document."
        )

    return chunk_documents(
        [document],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )