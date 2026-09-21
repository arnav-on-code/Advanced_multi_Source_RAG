from pathlib import Path
import hashlib
import logging

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
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the file is not a PDF.
        RuntimeError: If the PDF cannot be opened or processed.
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    image_dir = Path(output_dir) / path.stem
    image_dir.mkdir(parents=True, exist_ok=True)

    documents: list[Document] = []

    # Cache extracted images by PDF XRef.
    # The same image can appear on multiple pages.
    extracted_images: dict[int, Path] = {}

    try:
        with fitz.open(path) as pdf, pdfplumber.open(path) as plumber_pdf:

            for page_number, page in enumerate(pdf, start=1):

                plumber_page = plumber_pdf.pages[page_number - 1]


                images = page.get_images(full=True)

                for image_index, image_info in enumerate(
                    images,
                    start=1,
                ):
                    xref = image_info[0]

                    try:
                        if xref not in extracted_images:
                            image_data = pdf.extract_image(xref)

                            image_bytes = image_data["image"]
                            image_ext = image_data["ext"]

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

                            image_path.write_bytes(image_bytes)

                            extracted_images[xref] = image_path

                        else:
                            image_path = extracted_images[xref]
                            image_ext = image_path.suffix.lstrip(".")

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
                                    "source": str(path),
                                    "file_name": path.name,
                                    "page": page_number,
                                    "image_index": image_index,
                                    "image_path": str(image_path),
                                    "image_extension": image_ext,
                                    "processed": False,
                                },
                            )
                        )

                    except (KeyError, OSError, RuntimeError) as exc:
                        logger.warning(
                            "Failed to extract image %s from page %s "
                            "of '%s': %s",
                            image_index,
                            page_number,
                            path.name,
                            exc,
                        )

                try:
                    tables = plumber_page.extract_tables()
                except Exception as exc:
                    logger.warning(
                        "Failed to extract tables from page %s "
                        "of '%s': %s",
                        page_number,
                        path.name,
                        exc,
                    )
                    tables = []

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
                            str(cell).strip() if cell else ""
                            for cell in row
                        ]

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
                                "source": str(path),
                                "file_name": path.name,
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


                drawings = page.get_drawings()

                if drawings:
                    documents.append(
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
                                "source": str(path),
                                "file_name": path.name,
                                "page": page_number,
                                "drawing_count": len(drawings),
                                "possible_diagram": True,
                                "processed": False,
                            },
                        )
                    )

    except (fitz.FileDataError, OSError) as exc:
        raise RuntimeError(
            f"Failed to process PDF '{path}': {exc}"
        ) from exc

    return documents