from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from app.loaders.json_loader import load_json


# =========================================================
# FILE VALIDATION
# =========================================================


def test_json_file_not_found():
    """Loader should reject a JSON path that does not exist."""

    with pytest.raises(
        FileNotFoundError,
        match="JSON file not found",
    ):
        load_json("data/sample/missing.json")


def test_json_invalid_extension(
    tmp_path: Path,
):
    """Loader should reject non-JSON files."""

    file_path = tmp_path / "sample.txt"

    file_path.write_text(
        '{"name": "Arnav"}',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Expected a JSON file",
    ):
        load_json(file_path)


# =========================================================
# SUCCESSFUL LOADING
# =========================================================


def test_json_loader_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """JSON loader should return documents with normalized metadata."""

    file_path = tmp_path / "sample.json"

    file_path.write_text(
        json.dumps(
            {
                "name": "Arnav",
                "age": 21,
                "skills": [
                    "Python",
                    "FastAPI",
                ],
            }
        ),
        encoding="utf-8",
    )

    class MockJSONLoader:
        def __init__(
            self,
            loader_path,
            jq_schema=".",
            text_content=False,
        ):
            assert loader_path == str(file_path)
            assert jq_schema == "."
            assert text_content is False

        def load(self):
            return [
                Document(
                    page_content=(
                        '{"name": "Arnav", "age": 21}'
                    ),
                    metadata={},
                )
            ]

    monkeypatch.setattr(
        "app.loaders.json_loader.JSONLoader",
        MockJSONLoader,
    )

    documents = load_json(file_path)

    assert len(documents) == 1

    document = documents[0]
    metadata = document.metadata

    assert document.page_content == (
        '{"name": "Arnav", "age": 21}'
    )

    assert metadata["source_type"] == "json"
    assert metadata["source"] == str(file_path)
    assert metadata["file_name"] == "sample.json"
    assert metadata["has_text"] is True


# =========================================================
# EMPTY DOCUMENT
# =========================================================


def test_json_loader_empty_document(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Empty JSON content should be detected."""

    file_path = tmp_path / "empty.json"

    file_path.write_text(
        "{}",
        encoding="utf-8",
    )

    class MockJSONLoader:
        def __init__(
            self,
            loader_path,
            jq_schema=".",
            text_content=False,
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
        "app.loaders.json_loader.JSONLoader",
        MockJSONLoader,
    )

    documents = load_json(file_path)

    assert len(documents) == 1

    document = documents[0]

    assert document.page_content == ""
    assert document.metadata["source_type"] == "json"
    assert document.metadata["source"] == str(file_path)
    assert document.metadata["file_name"] == "empty.json"
    assert document.metadata["has_text"] is False


# =========================================================
# MULTIPLE DOCUMENTS
# =========================================================


def test_json_loader_multiple_documents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Multiple JSON documents should all be preserved."""

    file_path = tmp_path / "multiple.json"

    file_path.write_text(
        json.dumps(
            [
                {"name": "Arnav"},
                {"name": "Rahul"},
            ]
        ),
        encoding="utf-8",
    )

    mock_documents = [
        Document(
            page_content='{"name": "Arnav"}',
            metadata={},
        ),
        Document(
            page_content='{"name": "Rahul"}',
            metadata={},
        ),
    ]

    class MockJSONLoader:
        def __init__(
            self,
            loader_path,
            jq_schema=".",
            text_content=False,
        ):
            pass

        def load(self):
            return mock_documents

    monkeypatch.setattr(
        "app.loaders.json_loader.JSONLoader",
        MockJSONLoader,
    )

    documents = load_json(file_path)

    assert len(documents) == 2

    assert documents[0].page_content == (
        '{"name": "Arnav"}'
    )

    assert documents[1].page_content == (
        '{"name": "Rahul"}'
    )

    assert all(
        document.metadata["source_type"] == "json"
        for document in documents
    )

    assert all(
        document.metadata["source"] == str(file_path)
        for document in documents
    )


# =========================================================
# METADATA PRESERVATION
# =========================================================


def test_json_loader_preserves_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Existing JSONLoader metadata should be preserved."""

    file_path = tmp_path / "metadata.json"

    file_path.write_text(
        '{"name": "Arnav"}',
        encoding="utf-8",
    )

    mock_documents = [
        Document(
            page_content='{"name": "Arnav"}',
            metadata={
                "custom_field": "custom-value",
            },
        )
    ]

    class MockJSONLoader:
        def __init__(
            self,
            loader_path,
            jq_schema=".",
            text_content=False,
        ):
            pass

        def load(self):
            return mock_documents

    monkeypatch.setattr(
        "app.loaders.json_loader.JSONLoader",
        MockJSONLoader,
    )

    documents = load_json(file_path)

    metadata = documents[0].metadata

    assert metadata["custom_field"] == "custom-value"
    assert metadata["source_type"] == "json"
    assert metadata["source"] == str(file_path)
    assert metadata["file_name"] == "metadata.json"
    assert metadata["has_text"] is True


# =========================================================
# LOADER FAILURE
# =========================================================


def test_json_loader_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """JSONLoader failures should be converted to RuntimeError."""

    file_path = tmp_path / "broken.json"

    file_path.write_text(
        '{"name": "Arnav"}',
        encoding="utf-8",
    )

    class MockJSONLoader:
        def __init__(
            self,
            loader_path,
            jq_schema=".",
            text_content=False,
        ):
            pass

        def load(self):
            raise RuntimeError(
                "Invalid JSON structure"
            )

    monkeypatch.setattr(
        "app.loaders.json_loader.JSONLoader",
        MockJSONLoader,
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to load JSON",
    ):
        load_json(file_path)


# =========================================================
# EMPTY RESULT
# =========================================================


def test_json_loader_empty_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """An empty JSONLoader result should remain an empty list."""

    file_path = tmp_path / "empty_result.json"

    file_path.write_text(
        "{}",
        encoding="utf-8",
    )

    class MockJSONLoader:
        def __init__(
            self,
            loader_path,
            jq_schema=".",
            text_content=False,
        ):
            pass

        def load(self):
            return []

    monkeypatch.setattr(
        "app.loaders.json_loader.JSONLoader",
        MockJSONLoader,
    )

    documents = load_json(file_path)

    assert documents == []