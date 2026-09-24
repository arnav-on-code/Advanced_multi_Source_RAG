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

    Original document metadata is preserved automatically.
    Each chunk receives chunk-specific metadata.
    """

    if not documents:
        return []

    splitter = create_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = splitter.split_documents(documents)

    document_chunk_counts: dict[str, int] = {}

    for chunk in chunks:
        document_id = chunk.metadata.get("document_id")

        if document_id:
            chunk_index = document_chunk_counts.get(
                document_id,
                0,
            )

            chunk.metadata["chunk_index"] = chunk_index
            chunk.metadata["chunk_id"] = (
                f"{document_id}_chunk_{chunk_index}"
            )

            document_chunk_counts[document_id] = (
                chunk_index + 1
            )

        else:
            chunk.metadata["chunk_index"] = 0

        chunk.metadata["chunk_size"] = len(
            chunk.page_content
        )

    return chunks


def chunk_document(
    document: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:

    return chunk_documents(
        [document],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )