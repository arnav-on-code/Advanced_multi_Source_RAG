from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.loaders.csv_loader import load_csv


# =========================================================
# FILE VALIDATION
# =========================================================


def test_csv_file_not_found():
    """Loader should reject a CSV path that does not exist."""

    with pytest.raises(
        FileNotFoundError,
        match="CSV file not found",
    ):
        load_csv("data/sample/missing.csv")


def test_csv_invalid_extension(
    tmp_path: Path,
):
    """Loader should reject non-CSV files."""

    file_path = tmp_path / "sample.txt"
    file_path.write_text(
        "name,age\nArnav,21",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Expected a CSV file",
    ):
        load_csv(file_path)


# =========================================================
# SUCCESSFUL LOADING
# =========================================================


def test_csv_loader_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """CSV loader should return row documents with metadata."""

    file_path = tmp_path / "sample.csv"

    file_path.write_text(
        "name,age\nArnav,21\nRahul,22",
        encoding="utf-8",
    )

    mock_documents = [
        Document(
            page_content="name: Arnav\nage: 21",
            metadata={},
        ),
        Document(
            page_content="name: Rahul\nage: 22",
            metadata={},
        ),
    ]

    class MockCSVLoader:
        def __init__(
            self,
            file_path,
            encoding="utf-8",
        ):
            assert file_path == str(file_path_expected)
            assert encoding == "utf-8"

        def load(self):
            return mock_documents

    file_path_expected = file_path

    monkeypatch.setattr(
        "app.loaders.csv_loader.CSVLoader",
        MockCSVLoader,
    )

    documents = load_csv(file_path)

    assert len(documents) == 2

    assert documents[0].page_content == (
        "name: Arnav\nage: 21"
    )

    assert documents[1].page_content == (
        "name: Rahul\nage: 22"
    )

    assert documents[0].metadata["source_type"] == "csv"
    assert documents[0].metadata["source"] == str(file_path)
    assert documents[0].metadata["file_name"] == "sample.csv"
    assert documents[0].metadata["row"] == 1

    assert documents[1].metadata["row"] == 2


# =========================================================
# MULTIPLE ROWS
# =========================================================


def test_csv_loader_assigns_row_numbers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Each returned CSV document should receive a row number."""

    file_path = tmp_path / "data.csv"
    file_path.write_text(
        "name,score\nA,90\nB,80\nC,70",
        encoding="utf-8",
    )

    mock_documents = [
        Document(page_content="A,90", metadata={}),
        Document(page_content="B,80", metadata={}),
        Document(page_content="C,70", metadata={}),
    ]

    class MockCSVLoader:
        def __init__(
            self,
            file_path,
            encoding="utf-8",
        ):
            pass

        def load(self):
            return mock_documents

    monkeypatch.setattr(
        "app.loaders.csv_loader.CSVLoader",
        MockCSVLoader,
    )

    documents = load_csv(file_path)

    assert len(documents) == 3

    assert [
        document.metadata["row"]
        for document in documents
    ] == [1, 2, 3]


# =========================================================
# EMPTY CONTENT
# =========================================================


def test_csv_loader_empty_document(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Empty CSV rows should be handled without crashing."""

    file_path = tmp_path / "empty.csv"
    file_path.write_text(
        "",
        encoding="utf-8",
    )

    mock_documents = [
        Document(
            page_content="",
            metadata={},
        )
    ]

    class MockCSVLoader:
        def __init__(
            self,
            file_path,
            encoding="utf-8",
        ):
            pass

        def load(self):
            return mock_documents

    monkeypatch.setattr(
        "app.loaders.csv_loader.CSVLoader",
        MockCSVLoader,
    )

    documents = load_csv(file_path)

    assert len(documents) == 1
    assert documents[0].page_content == ""

    assert documents[0].metadata["source_type"] == "csv"
    assert documents[0].metadata["source"] == str(file_path)
    assert documents[0].metadata["file_name"] == "empty.csv"
    assert documents[0].metadata["row"] == 1


# =========================================================
# METADATA PRESERVATION
# =========================================================


def test_csv_loader_preserves_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Existing CSVLoader metadata should not be removed."""

    file_path = tmp_path / "metadata.csv"
    file_path.write_text(
        "name,age\nArnav,21",
        encoding="utf-8",
    )

    mock_documents = [
        Document(
            page_content="name: Arnav\nage: 21",
            metadata={
                "source": "original-source",
                "custom_field": "custom-value",
            },
        )
    ]

    class MockCSVLoader:
        def __init__(
            self,
            file_path,
            encoding="utf-8",
        ):
            pass

        def load(self):
            return mock_documents

    monkeypatch.setattr(
        "app.loaders.csv_loader.CSVLoader",
        MockCSVLoader,
    )

    documents = load_csv(file_path)

    metadata = documents[0].metadata

    assert metadata["custom_field"] == "custom-value"

    # Loader should normalize these fields.
    assert metadata["source_type"] == "csv"
    assert metadata["source"] == str(file_path)
    assert metadata["file_name"] == "metadata.csv"
    assert metadata["row"] == 1


# =========================================================
# LOADER FAILURE
# =========================================================


def test_csv_loader_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """CSV loader failures should be converted to RuntimeError."""

    file_path = tmp_path / "broken.csv"
    file_path.write_text(
        "name,age\nArnav,21",
        encoding="utf-8",
    )

    class MockCSVLoader:
        def __init__(
            self,
            file_path,
            encoding="utf-8",
        ):
            pass

        def load(self):
            raise RuntimeError(
                "CSV parsing failed"
            )

    monkeypatch.setattr(
        "app.loaders.csv_loader.CSVLoader",
        MockCSVLoader,
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to load CSV",
    ):
        load_csv(file_path)


# =========================================================
# EMPTY RESULT
# =========================================================


def test_csv_loader_empty_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """An empty result from CSVLoader should remain an empty list."""

    file_path = tmp_path / "empty_result.csv"
    file_path.write_text(
        "name,age\n",
        encoding="utf-8",
    )

    class MockCSVLoader:
        def __init__(
            self,
            file_path,
            encoding="utf-8",
        ):
            pass

        def load(self):
            return []

    monkeypatch.setattr(
        "app.loaders.csv_loader.CSVLoader",
        MockCSVLoader,
    )

    documents = load_csv(file_path)

    assert documents == []