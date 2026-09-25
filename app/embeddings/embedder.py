from functools import lru_cache

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def get_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> HuggingFaceEmbeddings:
    """Create and cache the local embedding model."""

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


def embed_documents(
    documents: list[Document],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> list[list[float]]:
    """Generate embeddings for documents."""

    if not documents:
        return []

    texts = [
        document.page_content
        for document in documents
    ]

    if any(not text.strip() for text in texts):
        raise ValueError(
            "Documents must contain non-empty text."
        )

    return get_embedding_model(
        model_name
    ).embed_documents(texts)


def embed_query(
    query: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> list[float]:
    """Generate an embedding for a user query."""

    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    return get_embedding_model(
        model_name
    ).embed_query(query)