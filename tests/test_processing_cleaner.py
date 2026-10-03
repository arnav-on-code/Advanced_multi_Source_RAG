from __future__ import annotations

from langchain_core.documents import Document

from app.processing.cleaner import (
    clean_document,
    clean_documents,
)


def test_clean_document_removes_extra_whitespace():
    document = Document(
        page_content="  FastAPI   is a Python   framework.  ",
        metadata={
            "source_type": "txt",
            "source": "sample.txt",
        },
    )

    cleaned = clean_document(document)

    assert cleaned.page_content == "FastAPI is a Python framework."


def test_clean_document_normalizes_newlines():
    document = Document(
        page_content=(
            "FastAPI is a framework.\n\n\n\n"
            "It is used to build APIs."
        ),
        metadata={"source_type": "txt"},
    )

    cleaned = clean_document(document)

    assert "\n\n\n" not in cleaned.page_content
    assert cleaned.page_content


def test_clean_document_strips_leading_and_trailing_whitespace():
    document = Document(
        page_content=" \n FastAPI framework. \n ",
        metadata={},
    )

    cleaned = clean_document(document)

    assert cleaned.page_content == "FastAPI framework."


def test_clean_document_empty_content():
    document = Document(
        page_content="   ",
        metadata={"source_type": "txt"},
    )

    cleaned = clean_document(document)

    assert cleaned.page_content == ""


def test_clean_document_preserves_metadata():
    metadata = {
        "source_type": "pdf",
        "source": "document.pdf",
        "page": 3,
        "file_name": "document.pdf",
    }

    document = Document(
        page_content="  PDF content here.  ",
        metadata=metadata,
    )

    cleaned = clean_document(document)

    assert cleaned.metadata == metadata


def test_clean_document_does_not_modify_original():
    document = Document(
        page_content="  FastAPI   framework.  ",
        metadata={"source_type": "txt"},
    )

    original_content = document.page_content
    original_metadata = document.metadata.copy()

    cleaned = clean_document(document)

    assert document.page_content == original_content
    assert document.metadata == original_metadata
    assert cleaned.page_content != document.page_content


def test_clean_documents():
    documents = [
        Document(
            page_content="  FastAPI   framework.  ",
            metadata={"source_type": "txt"},
        ),
        Document(
            page_content="  ChromaDB   vector database.  ",
            metadata={"source_type": "txt"},
        ),
    ]

    cleaned_documents = clean_documents(documents)

    assert len(cleaned_documents) == 2

    assert cleaned_documents[0].page_content == (
        "FastAPI framework."
    )

    assert cleaned_documents[1].page_content == (
        "ChromaDB vector database."
    )


def test_clean_documents_empty_input():
    cleaned_documents = clean_documents([])

    assert cleaned_documents == []


def test_clean_documents_preserves_order():
    documents = [
        Document(page_content="  First document.  ", metadata={}),
        Document(page_content="  Second document.  ", metadata={}),
        Document(page_content="  Third document.  ", metadata={}),
    ]

    cleaned_documents = clean_documents(documents)

    assert [
        document.page_content
        for document in cleaned_documents
    ] == [
        "First document.",
        "Second document.",
        "Third document.",
    ]