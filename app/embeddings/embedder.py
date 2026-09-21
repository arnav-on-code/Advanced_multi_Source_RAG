from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


@lru_cache(maxsize=1)
def get_embedding_model(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> HuggingFaceEmbeddings:
    """
    Create and cache the embedding model.

    The model runs locally and does not require a paid API.
    """

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
    """
    Generate embeddings for a list of documents.
    """

    if not documents:
        return []

    texts = [
        document.page_content
        for document in documents
    ]

    embedding_model = get_embedding_model(model_name)

    return embedding_model.embed_documents(texts)


def embed_query(
    query: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> list[float]:
    """
    Generate an embedding for a user query.
    """

    if not query or not query.strip():
        raise ValueError("Query cannot be empty.")

    embedding_model = get_embedding_model(model_name)

    return embedding_model.embed_query(query.strip())