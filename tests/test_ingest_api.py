from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.main import app


client = TestClient(app)


# =========================================================
# VALIDATION
# =========================================================


@pytest.mark.parametrize(
    "source_type",
    [
        "unsupported",
        "audio",
        "image",
        "excel",
    ],
)
def test_ingest_unsupported_source_type(source_type):
    response = client.post(
        "/ingest/",
        data={
            "source_type": source_type,
        },
    )

    assert response.status_code == 400


@pytest.mark.parametrize(
    "source_type",
    [
        "pdf",
        "docx",
        "csv",
        "txt",
        "json",
    ],
)
def test_file_ingest_without_file(source_type):
    response = client.post(
        "/ingest/",
        data={
            "source_type": source_type,
        },
    )

    assert response.status_code == 400


@pytest.mark.parametrize(
    "source_type",
    [
        "url",
        "youtube",
    ],
)
def test_url_ingest_without_source(source_type):
    response = client.post(
        "/ingest/",
        data={
            "source_type": source_type,
        },
    )

    assert response.status_code == 400


@pytest.mark.parametrize(
    "source_type",
    [
        "pdf",
        "docx",
        "csv",
        "txt",
        "json",
    ],
)
def test_file_ingest_rejects_source_parameter(source_type):
    response = client.post(
        "/ingest/",
        data={
            "source_type": source_type,
            "source": "sample.txt",
        },
        files={
            "file": (
                "sample.txt",
                BytesIO(b"sample content"),
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


@pytest.mark.parametrize(
    ("source_type", "filename"),
    [
        ("pdf", "sample.txt"),
        ("docx", "sample.txt"),
        ("csv", "sample.txt"),
        ("txt", "sample.pdf"),
        ("json", "sample.txt"),
    ],
)
def test_ingest_rejects_wrong_file_extension(
    source_type,
    filename,
):
    response = client.post(
        "/ingest/",
        data={
            "source_type": source_type,
        },
        files={
            "file": (
                filename,
                BytesIO(b"sample content"),
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 400


# =========================================================
# FILE INGESTION
# =========================================================


@pytest.mark.parametrize(
    ("source_type", "filename", "content_type", "loader"),
    [
        (
            "txt",
            "sample.txt",
            "text/plain",
            "load_txt",
        ),
        (
            "csv",
            "sample.csv",
            "text/csv",
            "load_csv",
        ),
        (
            "json",
            "sample.json",
            "application/json",
            "load_json",
        ),
        (
            "docx",
            "sample.docx",
            "application/vnd.openxmlformats-officedocument"
            ".wordprocessingml.document",
            "load_docx",
        ),
        (
            "pdf",
            "sample.pdf",
            "application/pdf",
            "load_pdf",
        ),
    ],
)
def test_file_ingestion(
    monkeypatch,
    source_type,
    filename,
    content_type,
    loader,
):
    monkeypatch.setattr(
        f"app.api.ingest.{loader}",
        lambda file_path: [
            Document(
                page_content="FastAPI is a Python framework.",
                metadata={
                    "source_type": source_type,
                    "source": str(file_path),
                    "file_name": filename,
                },
            )
        ],
    )

    response = client.post(
        "/ingest/",
        data={
            "source_type": source_type,
        },
        files={
            "file": (
                filename,
                BytesIO(b"test content"),
                content_type,
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"
    assert data["source_type"] == source_type
    assert data["documents"] == 1
    assert data["chunks"] >= 1
    assert data["message"]


# =========================================================
# URL INGESTION
# =========================================================


@pytest.mark.parametrize(
    ("source_type", "loader"),
    [
        ("url", "load_url"),
        ("youtube", "load_youtube"),
    ],
)
def test_url_ingestion(
    monkeypatch,
    source_type,
    loader,
):
    source = (
        "https://example.com"
        if source_type == "url"
        else "https://youtube.com/watch?v=test"
    )

    monkeypatch.setattr(
        f"app.api.ingest.{loader}",
        lambda value: [
            Document(
                page_content="FastAPI information.",
                metadata={
                    "source_type": source_type,
                    "source": value,
                },
            )
        ],
    )

    response = client.post(
        "/ingest/",
        data={
            "source_type": source_type,
            "source": source,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "success"
    assert data["source_type"] == source_type
    assert data["documents"] == 1
    assert data["chunks"] >= 1


@pytest.mark.parametrize(
    "source_type",
    [
        "url",
        "youtube",
    ],
)
def test_url_ingestion_rejects_file(
    source_type,
):
    source = (
        "https://example.com"
        if source_type == "url"
        else "https://youtube.com/watch?v=test"
    )

    response = client.post(
        "/ingest/",
        data={
            "source_type": source_type,
            "source": source,
        },
        files={
            "file": (
                "sample.txt",
                BytesIO(b"content"),
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


# =========================================================
# EMPTY SOURCE RESULT
# =========================================================


def test_ingest_rejects_empty_loader_result(
    monkeypatch,
):
    monkeypatch.setattr(
        "app.api.ingest.load_txt",
        lambda file_path: [],
    )

    response = client.post(
        "/ingest/",
        data={
            "source_type": "txt",
        },
        files={
            "file": (
                "sample.txt",
                BytesIO(b"content"),
                "text/plain",
            )
        },
    )

    assert response.status_code == 400

    assert (
        "no documents"
        in response.json()["detail"].lower()
    )


# =========================================================
# LOADER FAILURE
# =========================================================


def test_ingest_handles_loader_failure(
    monkeypatch,
):
    def failing_loader(file_path):
        raise ValueError(
            "Unable to process file"
        )

    monkeypatch.setattr(
        "app.api.ingest.load_txt",
        failing_loader,
    )

    response = client.post(
        "/ingest/",
        data={
            "source_type": "txt",
        },
        files={
            "file": (
                "sample.txt",
                BytesIO(b"invalid content"),
                "text/plain",
            )
        },
    )

    assert response.status_code == 400

    assert "Unable to process file" in (
        response.json()["detail"]
    )


# =========================================================
# PROCESSING FAILURE
# =========================================================


def test_ingest_rejects_documents_without_chunks(
    monkeypatch,
):
    monkeypatch.setattr(
        "app.api.ingest.load_txt",
        lambda file_path: [
            Document(
                page_content="",
                metadata={
                    "source_type": "txt",
                },
            )
        ],
    )

    response = client.post(
        "/ingest/",
        data={
            "source_type": "txt",
        },
        files={
            "file": (
                "sample.txt",
                BytesIO(b""),
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


# =========================================================
# UPLOAD SIZE
# =========================================================


def test_ingest_rejects_oversized_file(
    monkeypatch,
):
    monkeypatch.setattr(
        "app.api.ingest.load_txt",
        lambda file_path: [
            Document(
                page_content="test",
                metadata={
                    "source_type": "txt",
                },
            )
        ],
    )

    oversized_content = BytesIO(
        b"x" * (50 * 1024 * 1024 + 1)
    )

    response = client.post(
        "/ingest/",
        data={
            "source_type": "txt",
        },
        files={
            "file": (
                "large.txt",
                oversized_content,
                "text/plain",
            )
        },
    )

    assert response.status_code == 413