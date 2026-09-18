from pathlib import Path
import logging

from langchain_core.documents import Document


logger = logging.getLogger(__name__)


def process_visual_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Process visual PDF documents into RAG-ready Documents.

    Currently:
        - Validates extracted image paths
        - Identifies visual content types
        - Marks visual documents for downstream processing
        - Preserves source/page metadata

    Vision/OCR model integration can be added later.

    Args:
        documents: Documents produced by pdf_visual_loader.py.

    Returns:
        Processed visual Documents.
    """

    processed_documents: list[Document] = []

    for document in documents:

        content_type = document.metadata.get(
            "content_type"
        )

        if content_type == "image":
            processed_document = _process_image(document)

        elif content_type == "drawing":
            processed_document = _process_drawing(document)

        elif content_type == "table":
            processed_document = _process_table(document)

        else:
            logger.warning(
                "Unknown visual content type: %s",
                content_type,
            )
            continue

        processed_documents.append(processed_document)

    return processed_documents


def _process_image(document: Document) -> Document:
    """Prepare an extracted image for OCR/vision processing."""

    image_path = document.metadata.get("image_path")

    if not image_path:
        logger.warning(
            "Image document is missing image_path."
        )
        return document

    path = Path(image_path)

    if not path.is_file():
        logger.warning(
            "Image file does not exist: %s",
            path,
        )
        return document

    document.metadata.update(
        {
            "visual_status": "ready_for_vision",
            "requires_vision": True,
            "requires_ocr": True,
        }
    )

    return document


def _process_drawing(document: Document) -> Document:
    """
    Prepare vector graphics for diagram/flowchart analysis.

    Vector drawings alone cannot reliably identify whether
    the content is a flowchart, chart, diagram, or decoration.
    """

    document.metadata.update(
        {
            "visual_status": "ready_for_analysis",
            "requires_vision": True,
            "possible_diagram": True,
        }
    )

    return document


def _process_table(document: Document) -> Document:
    """Normalize an extracted table for downstream RAG processing."""

    text = document.page_content.strip()

    if not text:
        document.metadata["visual_status"] = "empty"
        return document

    document.page_content = (
        "[PDF TABLE]\n"
        + text
    )

    document.metadata.update(
        {
            "visual_status": "ready_for_embedding",
            "requires_vision": False,
            "requires_ocr": False,
        }
    )

    return document