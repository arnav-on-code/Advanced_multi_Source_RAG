from __future__ import annotations

import re

from langchain_core.documents import Document


# Matches spaces and tabs without touching newlines.
_HORIZONTAL_WHITESPACE = re.compile(r"[ \t]+")

# Removes indentation immediately after a newline.
_LINE_INDENTATION = re.compile(r"\n[ \t]+")

# Removes spaces/tabs immediately before or after a newline.
_SPACES_AROUND_NEWLINES = re.compile(r"[ \t]*\n[ \t]*")

# Maximum of one blank line between paragraphs.
_EXCESSIVE_NEWLINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """
    Normalize extracted text while preserving paragraph structure.
    """

    if not text:
        return ""

    # Normalize Windows/Mac line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove null characters.
    text = text.replace("\x00", "")

    # Normalize spaces and tabs.
    text = _HORIZONTAL_WHITESPACE.sub(" ", text)

    # Remove indentation after line breaks.
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
        raise TypeError("document must be a LangChain Document.")

    return Document(
        page_content=clean_text(document.page_content),
        metadata=dict(document.metadata),
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