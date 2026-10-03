from __future__ import annotations

from langchain_core.documents import Document

from app.processing.cleaner import (
    clean_document,
    clean_documents,
)


def test_clean_document_strips_whitespace():
    document = Document(
        page_content="   FastAPI is a framework.   ",
        metadata={"source_type": "txt"},
    )

    result = clean_document(document)

    assert result.page_content == (
        "FastAPI is a framework."
    )


def test_clean_document_normalizes_line_breaks():
    document = Document(
        page_content=(
            "FastAPI\r\n"
            "is a Python\r"
            "framework."
        ),
        metadata={},
    )

    result = clean_document(document)

    assert "\r" not in result.page_content


def test_clean_document_preserves_metadata():
    document = Document(
        page_content="  FastAPI  ",
        metadata={
            "source_type": "txt",
            "source": "sample.txt",
            "document_id": "doc-1",
        },
    )

    result = clean_document(document)

    assert result.metadata == {
        "source_type": "txt",
        "source": "sample.txt",
        "document_id": "doc-1",
    }


def test_clean_document_empty_content():
    document = Document(
        page_content="",
        metadata={"source_type": "txt"},
    )

    result = clean_document(document)

    assert result.page_content == ""


def test_clean_document_whitespace_only():
    document = Document(
        page_content="   \n\t   ",
        metadata={"source_type": "txt"},
    )

    result = clean_document(document)

    assert result.page_content == ""


def test_clean_documents_empty_list():
    assert clean_documents([]) == []


def test_clean_documents_multiple_documents():
    documents = [
        Document(
            page_content="  FastAPI  ",
            metadata={"id": "1"},
        ),
        Document(
            page_content="\n ChromaDB \n",
            metadata={"id": "2"},
        ),
    ]

    results = clean_documents(documents)

    assert len(results) == 2
    assert results[0].page_content == "FastAPI"
    assert results[1].page_content == "ChromaDB"


def test_clean_documents_preserves_document_order():
    documents = [
        Document(
            page_content="  First  ",
            metadata={"id": "1"},
        ),
        Document(
            page_content="  Second  ",
            metadata={"id": "2"},
        ),
        Document(
            page_content="  Third  ",
            metadata={"id": "3"},
        ),
    ]

    results = clean_documents(documents)

    assert [
        document.metadata["id"]
        for document in results
    ] == ["1", "2", "3"]


def test_clean_documents_does_not_modify_original_list():
    documents = [
        Document(
            page_content="  FastAPI  ",
            metadata={"id": "1"},
        )
    ]

    original_documents = list(documents)

    results = clean_documents(documents)

    assert documents == original_documents
    assert results is not documents