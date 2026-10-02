from __future__ import annotations

import re

from langchain_core.documents import Document


# Matches spaces and tabs without touching newlines.
_HORIZONTAL_WHITESPACE = re.compile(r"[ \t]+")

# Removes indentation after a newline.
_LINE_INDENTATION = re.compile(r"\n[ \t]+")

# Maximum of one blank line between paragraphs.
_EXCESSIVE_NEWLINES = re.compile(r"\n{3,}")

# Removes spaces immediately before/after line breaks.
_SPACES_AROUND_NEWLINES = re.compile(r"[ \t]*\n[ \t]*")


def clean_text(text: str) -> str:
    """
    Normalize extracted text while preserving paragraph structure.

    Operations:
        - Normalize line endings.
        - Remove null characters.
        - Normalize spaces and tabs.
        - Remove indentation after line breaks.
        - Remove excessive blank lines.
        - Remove spaces around line breaks.
        - Strip leading/trailing whitespace.
    """

    if not text:
        return ""

    # Normalize Windows/Mac line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove null characters.
    text = text.replace("\x00", "")

    # Normalize horizontal whitespace.
    text = _HORIZONTAL_WHITESPACE.sub(" ", text)

    # Remove spaces/tabs immediately after newlines.
    text = _LINE_INDENTATION.sub("\n", text)

    # Remove spaces around line breaks.
    text = _SPACES_AROUND_NEWLINES.sub("\n", text)

    # Collapse 3+ consecutive newlines into 2.
    text = _EXCESSIVE_NEWLINES.sub("\n\n", text)

    return text.strip()


def clean_document(document: Document) -> Document:
    """
    Clean a single LangChain Document.

    Returns a new Document and preserves the original document.
    """

    if not isinstance(document, Document):
        raise TypeError(
            "document must be a LangChain Document."
        )

    cleaned_content = clean_text(
        document.page_content
    )

    metadata = dict(document.metadata)

    if not cleaned_content:
        metadata["has_text"] = False
        metadata["content_length"] = 0
    else:
        metadata["has_text"] = True
        metadata["content_length"] = len(
            cleaned_content
        )

    return Document(
        page_content=cleaned_content,
        metadata=metadata,
    )


def clean_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Clean multiple LangChain Documents.

    Empty documents are removed while preserving order.
    """

    if not documents:
        return []

    cleaned_documents: list[Document] = []

    for document in documents:
        cleaned_document = clean_document(document)

        if not cleaned_document.page_content:
            continue

        cleaned_documents.append(cleaned_document)

    return cleaned_documents