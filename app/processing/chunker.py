from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.processing.metadata import create_chunk_id


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


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split documents into retrieval-friendly chunks.

    Each document must contain a document_id in its metadata.
    Chunk IDs are deterministic and derived from document_id
    and chunk index.
    """

    if not documents:
        return []

    # Validate before creating the splitter.
    for document in documents:
        if not isinstance(document, Document):
            raise TypeError(
                "documents must contain only LangChain Document objects."
            )

        if not document.page_content.strip():
            continue

        document_id = document.metadata.get("document_id")

        if not document_id:
            raise ValueError(
                "Each non-empty document must contain "
                "'document_id' in metadata before chunking."
            )

    splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)

    document_chunk_counts: dict[str, int] = {}

    for chunk in chunks:
        document_id = chunk.metadata.get("document_id")

        if not document_id:
            raise ValueError(
                "Chunk is missing 'document_id' metadata."
            )

        document_id = str(document_id)

        chunk_index = document_chunk_counts.get(
            document_id,
            0,
        )

        chunk.metadata["chunk_index"] = chunk_index

        chunk.metadata["chunk_id"] = create_chunk_id(
            document_id=document_id,
            chunk_index=chunk_index,
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