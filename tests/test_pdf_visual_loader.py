from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from app.loaders.pdf_visual_loader import load_pdf_visual


def test_load_pdf_visual_file_not_found(tmp_path):
    pdf_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        load_pdf_visual(pdf_path)


def test_load_pdf_visual_invalid_extension(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("not a pdf")

    with pytest.raises(ValueError):
        load_pdf_visual(file_path)


def test_load_pdf_visual_creates_output_directory(
    tmp_path,
    monkeypatch,
):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF")

    output_dir = tmp_path / "processed"

    mock_pdf = MagicMock()
    mock_page = MagicMock()

    mock_page.get_images.return_value = []
    mock_page.get_drawings.return_value = []

    mock_pdf.__iter__.return_value = iter([mock_page])
    mock_pdf.__enter__.return_value = mock_pdf
    mock_pdf.__exit__.return_value = None

    mock_pdfplumber = MagicMock()
    mock_pdfplumber_pdf = MagicMock()
    mock_pdfplumber_pdf.pages = []

    mock_pdfplumber.open.return_value.__enter__.return_value = (
        mock_pdfplumber_pdf
    )
    mock_pdfplumber.open.return_value.__exit__.return_value = None

    monkeypatch.setattr(
        "app.loaders.pdf_visual_loader.fitz.open",
        lambda *args, **kwargs: mock_pdf,
    )

    monkeypatch.setattr(
        "app.loaders.pdf_visual_loader.pdfplumber",
        mock_pdfplumber,
    )

    load_pdf_visual(
        pdf_path,
        output_dir=output_dir,
    )

    assert output_dir.exists()
    assert (output_dir / "sample").exists()


def test_load_pdf_visual_extracts_image(
    tmp_path,
    monkeypatch,
):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF")

    output_dir = tmp_path / "processed"

    mock_pdf = MagicMock()
    mock_page = MagicMock()

    mock_page.get_images.return_value = [
        (10, 0, 100, 100, 8, "DeviceRGB", "", "png", "")
    ]

    mock_page.get_drawings.return_value = []

    mock_pdf.__iter__.return_value = iter([mock_page])
    mock_pdf.__enter__.return_value = mock_pdf
    mock_pdf.__exit__.return_value = None

    mock_pdf.extract_image.return_value = {
        "image": b"fake-image-data",
        "ext": "png",
    }

    mock_pdfplumber = MagicMock()
    mock_pdfplumber_pdf = MagicMock()
    mock_pdfplumber_pdf.pages = []

    mock_pdfplumber.open.return_value.__enter__.return_value = (
        mock_pdfplumber_pdf
    )
    mock_pdfplumber.open.return_value.__exit__.return_value = None

    monkeypatch.setattr(
        "app.loaders.pdf_visual_loader.fitz.open",
        lambda *args, **kwargs: mock_pdf,
    )

    monkeypatch.setattr(
        "app.loaders.pdf_visual_loader.pdfplumber",
        mock_pdfplumber,
    )

    documents = load_pdf_visual(
        pdf_path,
        output_dir=output_dir,
    )

    image_documents = [
        document
        for document in documents
        if document.metadata.get("content_type") == "image"
    ]

    assert image_documents
    assert image_documents[0].metadata["source_type"] == "pdf"
    assert image_documents[0].metadata["source"] == str(pdf_path)
    assert image_documents[0].metadata["page"] == 1
    assert image_documents[0].metadata["processed"] is False
    assert image_documents[0].metadata["path"]


def test_load_pdf_visual_extracts_table(
    tmp_path,
    monkeypatch,
):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF")

    output_dir = tmp_path / "processed"

    mock_pdf = MagicMock()
    mock_page = MagicMock()

    mock_page.get_images.return_value = []
    mock_page.get_drawings.return_value = []

    mock_pdf.__iter__.return_value = iter([mock_page])
    mock_pdf.__enter__.return_value = mock_pdf
    mock_pdf.__exit__.return_value = None

    mock_table = [
        ["Name", "Age"],
        ["Arnav", "22"],
        ["Rahul", "23"],
    ]

    mock_plumber_page = MagicMock()
    mock_plumber_page.extract_tables.return_value = [
        mock_table
    ]

    mock_pdfplumber_pdf = MagicMock()
    mock_pdfplumber_pdf.pages = [mock_plumber_page]

    mock_pdfplumber = MagicMock()

    mock_pdfplumber.open.return_value.__enter__.return_value = (
        mock_pdfplumber_pdf
    )
    mock_pdfplumber.open.return_value.__exit__.return_value = None

    monkeypatch.setattr(
        "app.loaders.pdf_visual_loader.fitz.open",
        lambda *args, **kwargs: mock_pdf,
    )

    monkeypatch.setattr(
        "app.loaders.pdf_visual_loader.pdfplumber",
        mock_pdfplumber,
    )

    documents = load_pdf_visual(
        pdf_path,
        output_dir=output_dir,
    )

    table_documents = [
        document
        for document in documents
        if document.metadata.get("content_type") == "table"
    ]

    assert table_documents

    table = table_documents[0]

    assert table.metadata["source_type"] == "pdf"
    assert table.metadata["page"] == 1
    assert table.metadata["processed"] is True
    assert "Name" in table.page_content
    assert "Arnav" in table.page_content


def test_load_pdf_visual_extracts_drawings(
    tmp_path,
    monkeypatch,
):
    pdf_path = tmp_path / "diagram.pdf"
    pdf_path.write_bytes(b"%PDF")

    output_dir = tmp_path / "processed"

    mock_pdf = MagicMock()
    mock_page = MagicMock()

    mock_page.get_images.return_value = []

    mock_page.get_drawings.return_value = [
        {"type": "rect"},
        {"type": "line"},
        {"type": "curve"},
    ]

    mock_pdf.__iter__.return_value = iter([mock_page])
    mock_pdf.__enter__.return_value = mock_pdf
    mock_pdf.__exit__.return_value = None

    mock_pdfplumber = MagicMock()
    mock_pdfplumber_pdf = MagicMock()
    mock_pdfplumber_pdf.pages = []

    mock_pdfplumber.open.return_value.__enter__.return_value = (
        mock_pdfplumber_pdf
    )
    mock_pdfplumber.open.return_value.__exit__.return_value = None

    monkeypatch.setattr(
        "app.loaders.pdf_visual_loader.fitz.open",
        lambda *args, **kwargs: mock_pdf,
    )

    monkeypatch.setattr(
        "app.loaders.pdf_visual_loader.pdfplumber",
        mock_pdfplumber,
    )

    documents = load_pdf_visual(
        pdf_path,
        output_dir=output_dir,
    )

    drawing_documents = [
        document
        for document in documents
        if document.metadata.get("content_type") == "drawing"
    ]

    assert drawing_documents

    drawing = drawing_documents[0]

    assert drawing.metadata["source_type"] == "pdf"
    assert drawing.metadata["page"] == 1
    assert drawing.metadata["drawing_count"] == 3
    assert drawing.metadata["possible_diagram"] is True
    assert drawing.metadata["processed"] is False


def test_load_pdf_visual_empty_pdf(
    tmp_path,
    monkeypatch,
):
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"%PDF")

    output_dir = tmp_path / "processed"

    mock_pdf = MagicMock()
    mock_pdf.__iter__.return_value = iter([])
    mock_pdf.__enter__.return_value = mock_pdf
    mock_pdf.__exit__.return_value = None

    mock_pdfplumber = MagicMock()
    mock_pdfplumber_pdf = MagicMock()
    mock_pdfplumber_pdf.pages = []

    mock_pdfplumber.open.return_value.__enter__.return_value = (
        mock_pdfplumber_pdf
    )
    mock_pdfplumber.open.return_value.__exit__.return_value = None

    monkeypatch.setattr(
        "app.loaders.pdf_visual_loader.fitz.open",
        lambda *args, **kwargs: mock_pdf,
    )

    monkeypatch.setattr(
        "app.loaders.pdf_visual_loader.pdfplumber",
        mock_pdfplumber,
    )

    documents = load_pdf_visual(
        pdf_path,
        output_dir=output_dir,
    )

    assert documents == []