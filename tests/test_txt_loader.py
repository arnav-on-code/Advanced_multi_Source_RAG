from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.loaders.txt_loader import load_txt


# =========================================================
# FILE VALIDATION
# =========================================================


def test_txt_file_not_found():
    """Loader should reject a TXT path that does not exist."""

    with pytest.raises(
        FileNotFoundError,
        match="TXT file not found",
    ):
        load_txt("data/sample/missing.txt")


def test_txt_invalid_extension(
    tmp_path: Path,
):
    """Loader should reject non-TXT files."""

    file_path = tmp_path / "sample.csv"

    file_path.write_text(
        "name,age\nArnav,21",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Expected a TXT file",
    ):
        load_txt(file_path)


# =========================================================
# SUCCESSFUL LOADING
# =========================================================


def test_txt_loader_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """TXT loader should return documents with normalized metadata."""

    file_path = tmp_path / "sample.txt"

    file_path.write_text(
        "This is a sample text document.",
        encoding="utf-8",
    )

    class MockTextLoader:
        def __init__(
            self,
            path,
            encoding="utf-8",
            autodetect_encoding=True,
        ):
            assert path == str(file_path)
            assert encoding == "utf-8"
            assert autodetect_encoding is True

        def load(self):
            return [
                Document(
                    page_content=(
                        "This is a sample text document."
                    ),
                    metadata={
                        "custom_field": "test",
                    },
                )
            ]

    monkeypatch.setattr(
        "app.loaders.txt_loader.TextLoader",
        MockTextLoader,
    )

    documents = load_txt(file_path)

    assert len(documents) == 1

    document = documents[0]
    metadata = document.metadata

    assert document.page_content == (
        "This is a sample text document."
    )

    assert metadata["source_type"] == "txt"
    assert metadata["source"] == str(file_path)
    assert metadata["file_name"] == "sample.txt"
    assert metadata["has_text"] is True

    # Existing metadata must be preserved.
    assert metadata["custom_field"] == "test"


# =========================================================
# EMPTY CONTENT
# =========================================================


def test_txt_loader_empty_document(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Loader should correctly identify empty text."""

    file_path = tmp_path / "empty.txt"

    file_path.write_text(
        "",
        encoding="utf-8",
    )

    class MockTextLoader:
        def __init__(
            self,
            path,
            encoding="utf-8",
            autodetect_encoding=True,
        ):
            pass

        def load(self):
            return [
                Document(
                    page_content="",
                    metadata={},
                )
            ]

    monkeypatch.setattr(
        "app.loaders.txt_loader.TextLoader",
        MockTextLoader,
    )

    documents = load_txt(file_path)

    assert len(documents) == 1

    document = documents[0]

    assert document.page_content == ""
    assert document.metadata["source_type"] == "txt"
    assert document.metadata["source"] == str(file_path)
    assert document.metadata["file_name"] == "empty.txt"
    assert document.metadata["has_text"] is False


# =========================================================
# WHITESPACE CONTENT
# =========================================================


def test_txt_loader_whitespace_document(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Whitespace-only text should be treated as empty."""

    file_path = tmp_path / "whitespace.txt"
    file_path.write_text(
        "   \n\t  ",
        encoding="utf-8",
    )

    class MockTextLoader:
        def __init__(
            self,
            path,
            encoding="utf-8",
            autodetect_encoding=True,
        ):
            pass

        def load(self):
            return [
                Document(
                    page_content="   \n\t  ",
                    metadata={},
                )
            ]

    monkeypatch.setattr(
        "app.loaders.txt_loader.TextLoader",
        MockTextLoader,
    )

    documents = load_txt(file_path)

    assert documents[0].metadata["has_text"] is False


# =========================================================
# MULTIPLE DOCUMENTS
# =========================================================


def test_txt_loader_multiple_documents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """All documents returned by TextLoader should be preserved."""

    file_path = tmp_path / "multiple.txt"

    file_path.write_text(
        "First section\nSecond section",
        encoding="utf-8",
    )

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

    class MockTextLoader:
        def __init__(
            self,
            path,
            encoding="utf-8",
            autodetect_encoding=True,
        ):
            pass

        def load(self):
            return mock_documents

    monkeypatch.setattr(
        "app.loaders.txt_loader.TextLoader",
        MockTextLoader,
    )

    documents = load_txt(file_path)

    assert len(documents) == 2

    assert documents[0].page_content == "First section"
    assert documents[1].page_content == "Second section"

    assert all(
        document.metadata["source_type"] == "txt"
        for document in documents
    )

    assert all(
        document.metadata["source"] == str(file_path)
        for document in documents
    )


# =========================================================
# LOADER FAILURE
# =========================================================


def test_txt_loader_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """TextLoader failures should be converted to RuntimeError."""

    file_path = tmp_path / "broken.txt"

    file_path.write_text(
        "test content",
        encoding="utf-8",
    )

    class MockTextLoader:
        def __init__(
            self,
            path,
            encoding="utf-8",
            autodetect_encoding=True,
        ):
            pass

        def load(self):
            raise RuntimeError(
                "Text loading failed"
            )

    monkeypatch.setattr(
        "app.loaders.txt_loader.TextLoader",
        MockTextLoader,
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to load TXT",
    ):
        load_txt(file_path)


# =========================================================
# EMPTY RESULT
# =========================================================


def test_txt_loader_empty_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """An empty TextLoader result should remain an empty list."""

    file_path = tmp_path / "empty_result.txt"

    file_path.write_text(
        "",
        encoding="utf-8",
    )

    class MockTextLoader:
        def __init__(
            self,
            path,
            encoding="utf-8",
            autodetect_encoding=True,
        ):
            pass

        def load(self):
            return []

    monkeypatch.setattr(
        "app.loaders.txt_loader.TextLoader",
        MockTextLoader,
    )

    documents = load_txt(file_path)

    assert documents == []