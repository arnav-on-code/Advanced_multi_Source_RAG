from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.loaders.docx_loader import load_docx


# =========================================================
# FILE VALIDATION
# =========================================================


def test_docx_file_not_found():
    """Loader should reject a DOCX path that does not exist."""

    with pytest.raises(
        FileNotFoundError,
        match="DOCX file not found",
    ):
        load_docx("missing.docx")


def test_docx_invalid_extension(
    tmp_path: Path,
):
    """Loader should reject non-DOCX files."""

    file_path = tmp_path / "sample.txt"
    file_path.write_text("test")

    with pytest.raises(
        ValueError,
        match="Expected a DOCX file",
    ):
        load_docx(file_path)


# =========================================================
# SUCCESSFUL LOADING
# =========================================================


def test_docx_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """DOCX loader should return documents with normalized metadata."""

    docx_path = tmp_path / "sample.docx"
    docx_path.write_bytes(b"fake docx")

    mock_documents = [
        Document(
            page_content="Sample DOCX content",
            metadata={
                "author": "Test Author",
            },
        )
    ]

    class MockDocxLoader:
        def __init__(self, path: str):
            assert path == str(docx_path)
            self.path = path

        def load(self):
            return mock_documents

    monkeypatch.setattr(
        "app.loaders.docx_loader.Docx2txtLoader",
        MockDocxLoader,
    )

    documents = load_docx(docx_path)

    assert len(documents) == 1

    document = documents[0]
    metadata = document.metadata

    assert document.page_content == (
        "Sample DOCX content"
    )

    assert metadata["source_type"] == "docx"
    assert metadata["source"] == str(docx_path)
    assert metadata["file_name"] == docx_path.name
    assert metadata["has_text"] is True

    # Existing metadata must be preserved.
    assert metadata["author"] == "Test Author"


# =========================================================
# EMPTY CONTENT
# =========================================================


def test_docx_loader_empty_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Loader should identify DOCX documents with no text."""

    docx_path = tmp_path / "empty.docx"
    docx_path.write_bytes(b"fake docx")

    mock_documents = [
        Document(
            page_content="   ",
            metadata={},
        )
    ]

    class MockDocxLoader:
        def __init__(self, path: str):
            self.path = path

        def load(self):
            return mock_documents

    monkeypatch.setattr(
        "app.loaders.docx_loader.Docx2txtLoader",
        MockDocxLoader,
    )

    documents = load_docx(docx_path)

    assert len(documents) == 1
    assert documents[0].metadata["has_text"] is False


# =========================================================
# MULTIPLE DOCUMENTS
# =========================================================


def test_docx_loader_multiple_documents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """All documents returned by the loader should be preserved."""

    docx_path = tmp_path / "multi.docx"
    docx_path.write_bytes(b"fake docx")

    mock_documents = [
        Document(
            page_content="First section",
            metadata={},
        ),
        Document(
            page_content="Second section",
            metadata={},
        ),
    ]

    class MockDocxLoader:
        def __init__(self, path: str):
            self.path = path

        def load(self):
            return mock_documents

    monkeypatch.setattr(
        "app.loaders.docx_loader.Docx2txtLoader",
        MockDocxLoader,
    )

    documents = load_docx(docx_path)

    assert len(documents) == 2

    assert documents[0].page_content == "First section"
    assert documents[1].page_content == "Second section"

    assert all(
        document.metadata["source_type"] == "docx"
        for document in documents
    )

    assert all(
        document.metadata["source"] == str(docx_path)
        for document in documents
    )


# =========================================================
# LOADER FAILURE
# =========================================================


def test_docx_loader_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Underlying DOCX loader failures should be propagated."""

    docx_path = tmp_path / "broken.docx"
    docx_path.write_bytes(b"fake docx")

    class MockDocxLoader:
        def __init__(self, path: str):
            self.path = path

        def load(self):
            raise RuntimeError(
                "DOCX parsing failed"
            )

    monkeypatch.setattr(
        "app.loaders.docx_loader.Docx2txtLoader",
        MockDocxLoader,
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to load DOCX",
    ):
        load_docx(docx_path)


# =========================================================
# EMPTY RESULT
# =========================================================


def test_docx_loader_empty_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """An empty loader result should remain an empty list."""

    docx_path = tmp_path / "empty_result.docx"
    docx_path.write_bytes(b"fake docx")

    class MockDocxLoader:
        def __init__(self, path: str):
            self.path = path

        def load(self):
            return []

    monkeypatch.setattr(
        "app.loaders.docx_loader.Docx2txtLoader",
        MockDocxLoader,
    )

    documents = load_docx(docx_path)

    assert documents == []