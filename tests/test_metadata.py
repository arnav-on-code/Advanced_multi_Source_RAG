from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from app.processing.metadata import (
    SUPPORTED_SOURCE_TYPES,
    build_citation,
    create_chunk_id,
    create_document_id,
    enrich_document,
    enrich_documents,
)


def test_supported_source_types():
    assert SUPPORTED_SOURCE_TYPES == {
        "pdf",
        "url",
        "youtube",
        "docx",
        "csv",
        "txt",
        "json",
    }


def test_create_document_id_is_deterministic():
    document = Document(
        page_content="FastAPI is a Python framework.",
        metadata={
            "source_type": "txt",
            "source": "sample.txt",
        },
    )

    first_id = create_document_id(document)
    second_id = create_document_id(document)

    assert first_id == second_id
    assert isinstance(first_id, str)
    assert len(first_id) == 24


def test_create_document_id_changes_when_content_changes():
    document_one = Document(
        page_content="FastAPI",
        metadata={
            "source_type": "txt",
            "source": "sample.txt",
        },
    )

    document_two = Document(
        page_content="ChromaDB",
        metadata={
            "source_type": "txt",
            "source": "sample.txt",
        },
    )

    assert create_document_id(
        document_one
    ) != create_document_id(
        document_two
    )


def test_create_document_id_changes_when_source_changes():
    document_one = Document(
        page_content="FastAPI",
        metadata={
            "source_type": "txt",
            "source": "one.txt",
        },
    )

    document_two = Document(
        page_content="FastAPI",
        metadata={
            "source_type": "txt",
            "source": "two.txt",
        },
    )

    assert create_document_id(
        document_one
    ) != create_document_id(
        document_two
    )


def test_create_document_id_changes_when_source_type_changes():
    document_one = Document(
        page_content="FastAPI",
        metadata={
            "source_type": "txt",
            "source": "sample",
        },
    )

    document_two = Document(
        page_content="FastAPI",
        metadata={
            "source_type": "pdf",
            "source": "sample",
        },
    )

    assert create_document_id(
        document_one
    ) != create_document_id(
        document_two
    )


def test_create_chunk_id_is_deterministic():
    first = create_chunk_id(
        "document-123",
        0,
    )

    second = create_chunk_id(
        "document-123",
        0,
    )

    assert first == second
    assert isinstance(first, str)
    assert len(first) == 24


def test_create_chunk_id_changes_with_index():
    first = create_chunk_id(
        "document-123",
        0,
    )

    second = create_chunk_id(
        "document-123",
        1,
    )

    assert first != second


def test_create_chunk_id_changes_with_document():
    first = create_chunk_id(
        "document-1",
        0,
    )

    second = create_chunk_id(
        "document-2",
        0,
    )

    assert first != second


def test_build_pdf_citation():
    document = Document(
        page_content="FastAPI",
        metadata={
            "source_type": "pdf",
            "source": "data/guide.pdf",
            "file_name": "guide.pdf",
            "page": 5,
            "content_type": "table",
        },
    )

    citation = build_citation(document)

    assert citation == {
        "source_type": "pdf",
        "source": "data/guide.pdf",
        "page": 5,
        "file_name": "guide.pdf",
        "content_type": "table",
    }


def test_build_url_citation():
    document = Document(
        page_content="FastAPI",
        metadata={
            "source_type": "url",
            "source": "https://example.com/fastapi",
            "url": "https://example.com/fastapi",
            "title": "FastAPI Guide",
        },
    )

    citation = build_citation(document)

    assert citation == {
        "source_type": "url",
        "source": "https://example.com/fastapi",
        "url": "https://example.com/fastapi",
        "title": "FastAPI Guide",
    }


def test_build_youtube_citation():
    document = Document(
        page_content="FastAPI tutorial",
        metadata={
            "source_type": "youtube",
            "source": "https://youtube.com/watch?v=abc",
            "url": "https://youtube.com/watch?v=abc",
            "title": "FastAPI Tutorial",
            "video_id": "abc",
        },
    )

    citation = build_citation(document)

    assert citation["source_type"] == "youtube"
    assert citation["url"] == (
        "https://youtube.com/watch?v=abc"
    )
    assert citation["title"] == "FastAPI Tutorial"
    assert citation["video_id"] == "abc"


def test_build_file_citation():
    for source_type in (
        "docx",
        "csv",
        "txt",
        "json",
    ):
        document = Document(
            page_content="Sample",
            metadata={
                "source_type": source_type,
                "source": f"data/sample.{source_type}",
                "file_name": f"sample.{source_type}",
            },
        )

        citation = build_citation(document)

        assert citation["source_type"] == source_type
        assert citation["source"] == (
            f"data/sample.{source_type}"
        )
        assert citation["file_name"] == (
            f"sample.{source_type}"
        )


def test_build_csv_citation_includes_row():
    document = Document(
        page_content="Python,FastAPI",
        metadata={
            "source_type": "csv",
            "source": "data.csv",
            "file_name": "data.csv",
            "row": 4,
        },
    )

    citation = build_citation(document)

    assert citation["row"] == 4


def test_enrich_document_normalizes_source_type():
    document = Document(
        page_content="FastAPI",
        metadata={
            "source_type": "PDF",
            "source": " guide.pdf ",
        },
    )

    result = enrich_document(document)

    assert result.metadata["source_type"] == "pdf"
    assert result.metadata["source"] == "guide.pdf"


def test_enrich_document_adds_document_id():
    document = Document(
        page_content="FastAPI",
        metadata={
            "source_type": "txt",
            "source": "sample.txt",
        },
    )

    result = enrich_document(document)

    assert "document_id" in result.metadata
    assert isinstance(
        result.metadata["document_id"],
        str,
    )


def test_enrich_document_adds_citation():
    document = Document(
        page_content="FastAPI",
        metadata={
            "source_type": "txt",
            "source": "sample.txt",
            "file_name": "sample.txt",
        },
    )

    result = enrich_document(document)

    assert "citation" in result.metadata
    assert (
        result.metadata["citation"]["source_type"]
        == "txt"
    )


def test_enrich_document_normalizes_path_metadata():
    document = Document(
        page_content="FastAPI",
        metadata={
            "source_type": "txt",
            "source": Path("data/sample.txt"),
        },
    )

    result = enrich_document(document)

    assert result.metadata["source"] == (
        "data/sample.txt"
    )


def test_enrich_document_unknown_source_type():
    document = Document(
        page_content="Unknown source",
        metadata={
            "source_type": "something_else",
            "source": "sample",
        },
    )

    result = enrich_document(document)

    assert result.metadata["source_type"] == "unknown"


def test_enrich_documents():
    documents = [
        Document(
            page_content="FastAPI",
            metadata={
                "source_type": "txt",
                "source": "one.txt",
            },
        ),
        Document(
            page_content="ChromaDB",
            metadata={
                "source_type": "txt",
                "source": "two.txt",
            },
        ),
    ]

    results = enrich_documents(documents)

    assert len(results) == 2

    for document in results:
        assert "document_id" in document.metadata
        assert "citation" in document.metadata


def test_enrich_documents_empty():
    assert enrich_documents([]) == []