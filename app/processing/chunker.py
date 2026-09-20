from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


def create_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    """
    Create the default text splitter for the RAG pipeline.

    RecursiveCharacterTextSplitter attempts to preserve
    natural text boundaries before splitting into smaller chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

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
        length_function=len,
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

    Metadata from the original document is preserved and
    chunk-specific metadata is added.

    Args:
        documents: Documents produced by the loaders/processing layer.
        chunk_size: Maximum target chunk size.
        chunk_overlap: Number of overlapping characters.
    
    Returns:
        A list of chunked LangChain Documents.
    """

    if not documents:
        return []

    splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)

    for chunk_index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = chunk_index
        chunk.metadata["chunk_size"] = len(
            chunk.page_content
        )
        chunk.metadata["has_text"] = bool(
            chunk.page_content.strip()
        )

        document_id = chunk.metadata.get(
            "document_id"
        )

        if document_id:
            chunk.metadata["chunk_id"] = (
                f"{document_id}_chunk_{chunk_index}"
            )

    return chunks


def chunk_document(
    document: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Convenience function for chunking a single document.
    """

    return chunk_documents(
        [document],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )