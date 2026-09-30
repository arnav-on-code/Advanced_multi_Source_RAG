from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from app.processing.visual_processor import (
    process_visual_documents,
)


# =========================================================
# IMAGE
# =========================================================


def test_process_image_marks_image_ready_for_vision(
    tmp_path,
):
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"fake-image")

    document = Document(
        page_content="",
        metadata={
            "content_type": "image",
            "image_path": str(image_path),
            "source_type": "pdf",
            "page": 2,
        },
    )

    results = process_visual_documents([document])

    assert len(results) == 1

    result = results[0]

    assert result.metadata["visual_status"] == (
        "ready_for_vision"
    )
    assert result.metadata["requires_vision"] is True
    assert result.metadata["requires_ocr"] is True

    assert result.metadata["source_type"] == "pdf"
    assert result.metadata["page"] == 2


def test_process_image_missing_image_path():
    document = Document(
        page_content="",
        metadata={
            "content_type": "image",
            "source_type": "pdf",
        },
    )

    results = process_visual_documents([document])

    assert len(results) == 1

    result = results[0]

    assert "visual_status" not in result.metadata
    assert "requires_vision" not in result.metadata
    assert "requires_ocr" not in result.metadata


def test_process_image_missing_file(tmp_path):
    image_path = tmp_path / "missing.png"

    document = Document(
        page_content="",
        metadata={
            "content_type": "image",
            "image_path": str(image_path),
        },
    )

    results = process_visual_documents([document])

    assert len(results) == 1

    result = results[0]

    assert "visual_status" not in result.metadata
    assert "requires_vision" not in result.metadata


# =========================================================
# DRAWING
# =========================================================


def test_process_drawing():
    document = Document(
        page_content="",
        metadata={
            "content_type": "drawing",
            "source_type": "pdf",
            "page": 3,
        },
    )

    results = process_visual_documents([document])

    assert len(results) == 1

    result = results[0]

    assert result.metadata["visual_status"] == (
        "ready_for_analysis"
    )
    assert result.metadata["requires_vision"] is True
    assert result.metadata["possible_diagram"] is True
    assert result.metadata["page"] == 3


# =========================================================
# TABLE
# =========================================================


def test_process_table():
    document = Document(
        page_content=(
            "Name | Age\n"
            "Arnav | 22\n"
            "Rahul | 23"
        ),
        metadata={
            "content_type": "table",
            "source_type": "pdf",
            "page": 4,
        },
    )

    results = process_visual_documents([document])

    assert len(results) == 1

    result = results[0]

    assert result.page_content.startswith(
        "[PDF TABLE]\n"
    )

    assert "Name | Age" in result.page_content
    assert "Arnav | 22" in result.page_content

    assert result.metadata["visual_status"] == (
        "ready_for_embedding"
    )
    assert result.metadata["requires_vision"] is False
    assert result.metadata["requires_ocr"] is False
    assert result.metadata["page"] == 4


def test_process_empty_table():
    document = Document(
        page_content="   ",
        metadata={
            "content_type": "table",
            "source_type": "pdf",
        },
    )

    results = process_visual_documents([document])

    assert len(results) == 1

    result = results[0]

    assert result.page_content == "   "
    assert result.metadata["visual_status"] == "empty"


# =========================================================
# UNKNOWN CONTENT TYPE
# =========================================================


def test_process_unknown_visual_type():
    document = Document(
        page_content="Some content",
        metadata={
            "content_type": "unknown",
        },
    )

    results = process_visual_documents([document])

    assert results == []


def test_process_missing_content_type():
    document = Document(
        page_content="Some content",
        metadata={},
    )

    results = process_visual_documents([document])

    assert results == []


# =========================================================
# MULTIPLE DOCUMENTS
# =========================================================


def test_process_multiple_visual_documents(
    tmp_path,
):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake-image")

    documents = [
        Document(
            page_content="",
            metadata={
                "content_type": "image",
                "image_path": str(image_path),
            },
        ),
        Document(
            page_content="Flowchart",
            metadata={
                "content_type": "drawing",
            },
        ),
        Document(
            page_content="Name | Age",
            metadata={
                "content_type": "table",
            },
        ),
    ]

    results = process_visual_documents(documents)

    assert len(results) == 3

    assert results[0].metadata["visual_status"] == (
        "ready_for_vision"
    )

    assert results[1].metadata["visual_status"] == (
        "ready_for_analysis"
    )

    assert results[2].metadata["visual_status"] == (
        "ready_for_embedding"
    )


# =========================================================
# EMPTY INPUT
# =========================================================


def test_process_visual_documents_empty_input():
    results = process_visual_documents([])

    assert results == []


# =========================================================
# ORDER PRESERVATION
# =========================================================


def test_process_visual_documents_preserves_order(
    tmp_path,
):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"fake-image")

    documents = [
        Document(
            page_content="First",
            metadata={
                "content_type": "table",
            },
        ),
        Document(
            page_content="",
            metadata={
                "content_type": "image",
                "image_path": str(image_path),
            },
        ),
        Document(
            page_content="Third",
            metadata={
                "content_type": "drawing",
            },
        ),
    ]

    results = process_visual_documents(documents)

    assert len(results) == 3

    assert results[0].metadata["content_type"] == "table"
    assert results[1].metadata["content_type"] == "image"
    assert results[2].metadata["content_type"] == "drawing"