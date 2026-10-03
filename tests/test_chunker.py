from __future__ import annotations

import pytest

from langchain_core.documents import Document

from app.processing.chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_document,
    chunk_documents,
    create_text_splitter,
)


def test_create_text_splitter_defaults():
    splitter = create_text_splitter()

    assert splitter._chunk_size == DEFAULT_CHUNK_SIZE
    assert splitter._chunk_overlap == DEFAULT_CHUNK_OVERLAP


def test_create_text_splitter_custom_values():
    splitter = create_text_splitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    assert splitter._chunk_size == 500
    assert splitter._chunk_overlap == 50


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (-1, 0),
        (500, 500),
        (500, 600),
        (100, 100),
    ],
)
def test_create_text_splitter_rejects_invalid_values(
    chunk_size,
    chunk_overlap,
):
    with pytest.raises(ValueError):
        create_text_splitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def test_chunk_documents_empty_input():
    assert chunk_documents([]) == []


def test_chunk_documents_creates_chunks():
    document = Document(
        page_content=(
            "FastAPI is a modern Python web framework. "
            "It is commonly used for building APIs. "
            "FastAPI provides automatic API documentation."
        ),
        metadata={
            "document_id": "doc-1",
            "source_type": "txt",
            "source": "sample.txt",
        },
    )

    chunks = chunk_documents(
        [document],
        chunk_size=50,
        chunk_overlap=10,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert isinstance(chunk, Document)
        assert chunk.page_content.strip()


def test_chunk_documents_preserves_metadata():
    document = Document(
        page_content="FastAPI is a Python framework.",
        metadata={
            "document_id": "doc-123",
            "source_type": "txt",
            "source": "sample.txt",
            "custom_field": "test-value",
        },
    )

    chunks = chunk_documents(
        [document],
        chunk_size=20,
        chunk_overlap=5,
    )

    assert chunks

    for chunk in chunks:
        assert chunk.metadata["document_id"] == "doc-123"
        assert chunk.metadata["source_type"] == "txt"
        assert chunk.metadata["source"] == "sample.txt"
        assert chunk.metadata["custom_field"] == "test-value"


def test_chunk_documents_assigns_chunk_index():
    document = Document(
        page_content=(
            "This is a long document that needs "
            "to be divided into multiple chunks "
            "for retrieval."
        ),
        metadata={
            "document_id": "doc-1",
        },
    )

    chunks = chunk_documents(
        [document],
        chunk_size=30,
        chunk_overlap=5,
    )

    assert len(chunks) > 1

    assert [
        chunk.metadata["chunk_index"]
        for chunk in chunks
    ] == list(range(len(chunks)))


def test_chunk_documents_assigns_chunk_id():
    document = Document(
        page_content=(
            "FastAPI is a Python framework "
            "used for API development."
        ),
        metadata={
            "document_id": "doc-123",
        },
    )

    chunks = chunk_documents(
        [document],
        chunk_size=25,
        chunk_overlap=5,
    )

    assert chunks

    for index, chunk in enumerate(chunks):
        assert chunk.metadata["chunk_id"] == (
            f"doc-123_chunk_{index}"
        )


def test_chunk_documents_assigns_chunk_size():
    document = Document(
        page_content=(
            "FastAPI is a Python framework "
            "used for building APIs."
        ),
        metadata={
            "document_id": "doc-1",
        },
    )

    chunks = chunk_documents(
        [document],
        chunk_size=30,
        chunk_overlap=5,
    )

    assert chunks

    for chunk in chunks:
        assert chunk.metadata["chunk_size"] == len(
            chunk.page_content
        )


def test_chunk_documents_supports_multiple_documents():
    documents = [
        Document(
            page_content="Document one contains FastAPI.",
            metadata={"document_id": "doc-1"},
        ),
        Document(
            page_content="Document two contains ChromaDB.",
            metadata={"document_id": "doc-2"},
        ),
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=20,
        chunk_overlap=5,
    )

    assert chunks

    document_ids = {
        chunk.metadata["document_id"]
        for chunk in chunks
    }

    assert document_ids == {"doc-1", "doc-2"}


def test_chunk_documents_resets_chunk_index_per_document():
    documents = [
        Document(
            page_content=(
                "First document with enough content "
                "to produce multiple chunks."
            ),
            metadata={"document_id": "doc-1"},
        ),
        Document(
            page_content=(
                "Second document with enough content "
                "to produce multiple chunks."
            ),
            metadata={"document_id": "doc-2"},
        ),
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=25,
        chunk_overlap=5,
    )

    for document_id in {"doc-1", "doc-2"}:
        document_chunks = [
            chunk
            for chunk in chunks
            if chunk.metadata["document_id"] == document_id
        ]

        assert document_chunks
        assert [
            chunk.metadata["chunk_index"]
            for chunk in document_chunks
        ] == list(range(len(document_chunks)))


def test_chunk_document_single_document_wrapper():
    document = Document(
        page_content="FastAPI is a Python framework.",
        metadata={"document_id": "doc-1"},
    )

    chunks = chunk_document(
        document,
        chunk_size=20,
        chunk_overlap=5,
    )

    assert chunks
    assert all(
        isinstance(chunk, Document)
        for chunk in chunks
    )


def test_chunk_documents_requires_document_id():
    document = Document(
        page_content="FastAPI is a Python framework.",
        metadata={},
    )

    with pytest.raises(KeyError):
        chunk_documents(
            [document],
            chunk_size=20,
            chunk_overlap=5,
        )