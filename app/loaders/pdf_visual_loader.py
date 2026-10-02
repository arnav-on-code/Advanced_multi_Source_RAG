from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
from langchain_core.documents import Document


logger = logging.getLogger(__name__)


def load_pdf_visual(
    file_path: str | Path,
    output_dir: str | Path = "data/processed/images",
) -> list[Document]:
    """
    Extract visual and structured content from a PDF.

    Extracts:
        - Embedded images
        - Tables
        - Vector drawing information

    Normal PDF text extraction is handled by pdf_loader.py.

    Args:
        file_path: Path to the PDF file.
        output_dir: Directory used to store extracted images.

    Returns:
        List of LangChain Documents containing visual/structured
        content and metadata.

    Raises:
        FileNotFoundError:
            If the PDF does not exist.

        ValueError:
            If the file is not a PDF.

        RuntimeError:
            If the PDF cannot be opened or processed.
    """

    path = Path(file_path)

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    if not path.is_file():
        raise FileNotFoundError(
            f"PDF file not found: {path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, got: {path.suffix}"
        )

    output_path = Path(output_dir)

    image_dir = output_path / path.stem

    try:
        image_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Unable to create image output directory "
            f"'{image_dir}': {exc}"
        ) from exc

    documents: list[Document] = []

    # Cache extracted images by PDF XRef.
    #
    # The same embedded image can appear on multiple pages.
    # We extract it only once and reuse the generated file.
    extracted_images: dict[int, Path] = {}

    # ---------------------------------------------------------
    # OPEN PDF
    # ---------------------------------------------------------

    try:
        with (
            fitz.open(path) as pdf,
            pdfplumber.open(path) as plumber_pdf,
        ):

            # Ensure the two PDF readers expose the same
            # number of pages.
            if len(pdf) != len(plumber_pdf.pages):
                logger.warning(
                    "Page count mismatch for '%s': "
                    "PyMuPDF=%s, pdfplumber=%s",
                    path.name,
                    len(pdf),
                    len(plumber_pdf.pages),
                )

            # -------------------------------------------------
            # PAGE PROCESSING
            # -------------------------------------------------

            for page_number, page in enumerate(
                pdf,
                start=1,
            ):

                # ---------------------------------------------
                # PDFPLUMBER PAGE
                # ---------------------------------------------

                plumber_page = None

                if page_number <= len(plumber_pdf.pages):
                    plumber_page = (
                        plumber_pdf.pages[page_number - 1]
                    )

                # ---------------------------------------------
                # IMAGES
                # ---------------------------------------------

                documents.extend(
                    _extract_images(
                        pdf=pdf,
                        page=page,
                        page_number=page_number,
                        source_path=path,
                        image_dir=image_dir,
                        extracted_images=extracted_images,
                    )
                )

                # ---------------------------------------------
                # TABLES
                # ---------------------------------------------

                if plumber_page is not None:
                    documents.extend(
                        _extract_tables(
                            plumber_page=plumber_page,
                            page_number=page_number,
                            source_path=path,
                        )
                    )

                # ---------------------------------------------
                # VECTOR DRAWINGS
                # ---------------------------------------------

                documents.extend(
                    _extract_drawings(
                        page=page,
                        page_number=page_number,
                        source_path=path,
                    )
                )

    except (
        fitz.FileDataError,
        pdfplumber.pdfminer.pdfparser.PDFSyntaxError,
        OSError,
    ) as exc:
        raise RuntimeError(
            f"Failed to process PDF '{path}': {exc}"
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected error while processing PDF '%s'.",
            path,
        )

        raise RuntimeError(
            f"Failed to process PDF '{path}': {exc}"
        ) from exc

    logger.info(
        "Extracted %s visual/structured documents from '%s'.",
        len(documents),
        path.name,
    )

    return documents


# =============================================================
# IMAGE EXTRACTION
# =============================================================


def _extract_images(
    pdf: fitz.Document,
    page: fitz.Page,
    page_number: int,
    source_path: Path,
    image_dir: Path,
    extracted_images: dict[int, Path],
) -> list[Document]:
    """Extract embedded images from a PDF page."""

    documents: list[Document] = []

    images = page.get_images(full=True)

    for image_index, image_info in enumerate(
        images,
        start=1,
    ):

        if not image_info:
            continue

        xref = image_info[0]

        if not xref:
            continue

        try:
            # ---------------------------------------------
            # REUSE PREVIOUSLY EXTRACTED IMAGE
            # ---------------------------------------------

            if xref in extracted_images:
                image_path = extracted_images[xref]

                image_ext = (
                    image_path.suffix.lstrip(".")
                )

            # ---------------------------------------------
            # EXTRACT NEW IMAGE
            # ---------------------------------------------

            else:
                image_data = pdf.extract_image(xref)

                image_bytes = image_data["image"]
                image_ext = image_data["ext"]

                if not image_bytes:
                    logger.warning(
                        "Empty image data for XRef %s "
                        "on page %s of '%s'.",
                        xref,
                        page_number,
                        source_path.name,
                    )
                    continue

                image_hash = hashlib.sha256(
                    image_bytes
                ).hexdigest()[:12]

                image_path = (
                    image_dir
                    / (
                        f"page_{page_number}_"
                        f"image_{image_index}_"
                        f"{image_hash}.{image_ext}"
                    )
                )

                image_path.write_bytes(
                    image_bytes
                )

                extracted_images[xref] = image_path

            # ---------------------------------------------
            # DOCUMENT
            # ---------------------------------------------

            documents.append(
                Document(
                    page_content=(
                        "[PDF IMAGE] "
                        f"Image extracted from page "
                        f"{page_number}."
                    ),
                    metadata={
                        "source_type": "pdf",
                        "content_type": "image",
                        "source": str(source_path),
                        "file_name": source_path.name,
                        "page": page_number,
                        "image_index": image_index,
                        "image_xref": xref,
                        "image_path": str(image_path),
                        "image_extension": image_ext,
                        "processed": False,
                    },
                )
            )

        except (
            KeyError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            logger.warning(
                "Failed to extract image %s from page %s "
                "of '%s': %s",
                image_index,
                page_number,
                source_path.name,
                exc,
            )

    return documents


# =============================================================
# TABLE EXTRACTION
# =============================================================


def _extract_tables(
    plumber_page,
    page_number: int,
    source_path: Path,
) -> list[Document]:
    """Extract and normalize tables from a PDF page."""

    documents: list[Document] = []

    try:
        tables = plumber_page.extract_tables()

    except Exception as exc:
        logger.warning(
            "Failed to extract tables from page %s "
            "of '%s': %s",
            page_number,
            source_path.name,
            exc,
        )

        return documents

    for table_index, table in enumerate(
        tables,
        start=1,
    ):

        if not table:
            continue

        rows: list[list[str]] = []

        for row in table:

            if not row:
                continue

            cleaned_row = [
                str(cell).strip() if cell is not None else ""
                for cell in row
            ]

            # Ignore completely empty rows.
            if not any(cleaned_row):
                continue

            rows.append(cleaned_row)

        if not rows:
            continue

        table_text = "\n".join(
            " | ".join(row)
            for row in rows
        )

        documents.append(
            Document(
                page_content=table_text,
                metadata={
                    "source_type": "pdf",
                    "content_type": "table",
                    "source": str(source_path),
                    "file_name": source_path.name,
                    "page": page_number,
                    "table_index": table_index,
                    "rows": len(rows),
                    "columns": max(
                        len(row)
                        for row in rows
                    ),
                    "processed": True,
                },
            )
        )

    return documents


# =============================================================
# DRAWING EXTRACTION
# =============================================================


def _extract_drawings(
    page: fitz.Page,
    page_number: int,
    source_path: Path,
) -> list[Document]:
    """Extract vector drawing information from a PDF page."""

    try:
        drawings = page.get_drawings()

    except Exception as exc:
        logger.warning(
            "Failed to extract drawings from page %s "
            "of '%s': %s",
            page_number,
            source_path.name,
            exc,
        )

        return []

    if not drawings:
        return []

    return [
        Document(
            page_content=(
                "[PDF DRAWING] "
                f"Page {page_number} contains "
                f"{len(drawings)} vector drawing "
                "objects. These may represent a "
                "diagram, chart, flowchart, or "
                "other graphical content."
            ),
            metadata={
                "source_type": "pdf",
                "content_type": "drawing",
                "source": str(source_path),
                "file_name": source_path.name,
                "page": page_number,
                "drawing_count": len(drawings),
                "possible_diagram": True,
                "processed": False,
            },
        )
    ]