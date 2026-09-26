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
        raise ValueError("chunk_size must be greater than 0.")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and smaller than chunk_size."
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

    Original metadata is preserved and each chunk receives
    its own document ID, chunk index, chunk ID, and size.
    """

    if not documents:
        return []

    for document in documents:
        if not document.page_content.strip():
            continue

        document.metadata.setdefault(
            "document_id",
            uuid4().hex,
        )

    splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)

    document_chunk_counts: dict[str, int] = {}

    for chunk in chunks:
        document_id = chunk.metadata["document_id"]

        chunk_index = document_chunk_counts.get(
            document_id,
            0,
        )

        chunk.metadata["chunk_index"] = chunk_index

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

    return chunk_documents(
        [document],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )