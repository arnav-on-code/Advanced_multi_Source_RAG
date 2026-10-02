from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.loaders.pdf_loader import load_pdf


# =========================================================
# FILE VALIDATION
# =========================================================


def test_pdf_file_not_found():
    """Loader should reject a PDF path that does not exist."""

    with pytest.raises(
        FileNotFoundError,
        match="PDF file not found",
    ):
        load_pdf("non_existent_file.pdf")


def test_pdf_invalid_extension(tmp_path: Path):
    """Loader should reject files that are not PDFs."""

    file_path = tmp_path / "sample.txt"
    file_path.write_text("test")

    with pytest.raises(
        ValueError,
        match="Expected a PDF file",
    ):
        load_pdf(file_path)


# =========================================================
# SUCCESSFUL LOADING
# =========================================================


def test_pdf_loader_returns_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Loader should return documents with normalized metadata."""

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"fake pdf")

    mock_documents = [
        Document(
            page_content="Sample PDF content",
            metadata={
                "page": 0,
            },
        )
    ]

    class MockPDFLoader:
        def __init__(self, path: str):
            assert path == str(pdf_path)
            self.path = path

        def load(self):
            return mock_documents

    monkeypatch.setattr(
        "app.loaders.pdf_loader.PyPDFLoader",
        MockPDFLoader,
    )

    documents = load_pdf(pdf_path)

    assert len(documents) == 1

    document = documents[0]

    assert document.page_content == (
        "Sample PDF content"
    )

    assert document.metadata["source_type"] == "pdf"
    assert document.metadata["source"] == str(pdf_path)
    assert document.metadata["file_name"] == pdf_path.name
    assert document.metadata["page"] == 1
    assert document.metadata["has_text"] is True


# =========================================================
# EMPTY PAGE
# =========================================================


def test_pdf_loader_detects_empty_page(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Loader should correctly identify pages without text."""

    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"fake pdf")

    mock_documents = [
        Document(
            page_content="   ",
            metadata={
                "page": 0,
            },
        )
    ]

    class MockPDFLoader:
        def __init__(self, path: str):
            self.path = path

        def load(self):
            return mock_documents

    monkeypatch.setattr(
        "app.loaders.pdf_loader.PyPDFLoader",
        MockPDFLoader,
    )

    documents = load_pdf(pdf_path)

    assert len(documents) == 1
    assert documents[0].metadata["has_text"] is False


# =========================================================
# MULTIPLE PAGES
# =========================================================


def test_pdf_loader_handles_multiple_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Each PDF page should receive the correct 1-based page number."""

    pdf_path = tmp_path / "multi_page.pdf"
    pdf_path.write_bytes(b"fake pdf")

    mock_documents = [
        Document(
            page_content="Page one",
            metadata={"page": 0},
        ),
        Document(
            page_content="Page two",
            metadata={"page": 1},
        ),
        Document(
            page_content="Page three",
            metadata={"page": 2},
        ),
    ]

    class MockPDFLoader:
        def __init__(self, path: str):
            self.path = path

        def load(self):
            return mock_documents

    monkeypatch.setattr(
        "app.loaders.pdf_loader.PyPDFLoader",
        MockPDFLoader,
    )

    documents = load_pdf(pdf_path)

    assert len(documents) == 3

    assert [
        document.metadata["page"]
        for document in documents
    ] == [1, 2, 3]

    assert all(
        document.metadata["source_type"] == "pdf"
        for document in documents
    )

    assert all(
        document.metadata["file_name"]
        == pdf_path.name
        for document in documents
    )


# =========================================================
# METADATA PRESERVATION
# =========================================================


def test_pdf_loader_preserves_existing_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Existing loader metadata should not be discarded."""

    pdf_path = tmp_path / "metadata.pdf"
    pdf_path.write_bytes(b"fake pdf")

    mock_documents = [
        Document(
            page_content="PDF content",
            metadata={
                "page": 0,
                "producer": "test-producer",
                "custom_field": "custom-value",
            },
        )
    ]

    class MockPDFLoader:
        def __init__(self, path: str):
            self.path = path

        def load(self):
            return mock_documents

    monkeypatch.setattr(
        "app.loaders.pdf_loader.PyPDFLoader",
        MockPDFLoader,
    )

    documents = load_pdf(pdf_path)

    metadata = documents[0].metadata

    assert metadata["producer"] == "test-producer"
    assert metadata["custom_field"] == "custom-value"

    assert metadata["source_type"] == "pdf"
    assert metadata["page"] == 1


# =========================================================
# LOADER FAILURE
# =========================================================


def test_pdf_loader_raises_loader_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Unexpected PDF loading failures should be surfaced."""

    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"fake pdf")

    class MockPDFLoader:
        def __init__(self, path: str):
            self.path = path

        def load(self):
            raise RuntimeError(
                "Failed to parse PDF"
            )

    monkeypatch.setattr(
        "app.loaders.pdf_loader.PyPDFLoader",
        MockPDFLoader,
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to parse PDF",
    ):
        load_pdf(pdf_path)


# =========================================================
# EMPTY DOCUMENT RESULT
# =========================================================


def test_pdf_loader_returns_empty_list_when_loader_returns_none(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    The loader should safely return an empty list if
    PyPDFLoader returns no documents.
    """

    pdf_path = tmp_path / "empty_result.pdf"
    pdf_path.write_bytes(b"fake pdf")

    class MockPDFLoader:
        def __init__(self, path: str):
            self.path = path

        def load(self):
            return []

    monkeypatch.setattr(
        "app.loaders.pdf_loader.PyPDFLoader",
        MockPDFLoader,
    )

    documents = load_pdf(pdf_path)

    assert documents == []