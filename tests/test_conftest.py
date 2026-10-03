from __future__ import annotations

import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_document() -> Document:
    return Document(
        page_content=(
            "FastAPI is a modern Python web framework "
            "for building APIs."
        ),
        metadata={
            "source_type": "txt",
            "source": "sample.txt",
            "file_name": "sample.txt",
        },
    )


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(
            page_content=(
                "FastAPI is a Python web framework."
            ),
            metadata={
                "source_type": "txt",
                "source": "fastapi.txt",
                "file_name": "fastapi.txt",
            },
        ),
        Document(
            page_content=(
                "ChromaDB is a vector database."
            ),
            metadata={
                "source_type": "txt",
                "source": "chroma.txt",
                "file_name": "chroma.txt",
            },
        ),
        Document(
            page_content=(
                "RAG combines retrieval with generation."
            ),
            metadata={
                "source_type": "txt",
                "source": "rag.txt",
                "file_name": "rag.txt",
            },
        ),
    ]


@pytest.fixture
def sample_query() -> str:
    return "What is FastAPI?"


@pytest.fixture
def sample_metadata() -> dict[str, str]:
    return {
        "source_type": "txt",
        "source": "sample.txt",
        "file_name": "sample.txt",
    }


@pytest.fixture
def long_document() -> Document:
    return Document(
        page_content=(
            "FastAPI is a modern Python web framework "
            "for building APIs. "
        )
        * 100,
        metadata={
            "source_type": "txt",
            "source": "long_sample.txt",
            "file_name": "long_sample.txt",
        },
    )


@pytest.fixture
def empty_document() -> Document:
    return Document(
        page_content="",
        metadata={
            "source_type": "txt",
            "source": "empty.txt",
            "file_name": "empty.txt",
        },
    )