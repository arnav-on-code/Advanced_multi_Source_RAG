import re

from langchain_core.documents import Document


def clean_text(text: str) -> str:
    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove null characters and other control characters
    text = text.replace("\x00", "")

    # Normalize whitespace while preserving paragraphs
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove spaces around line breaks
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()


def clean_documents(
    documents: list[Document],
) -> list[Document]:
    """
    Clean LangChain documents while preserving metadata.

    Empty documents are removed.
    """

    cleaned_documents: list[Document] = []

    for document in documents:
        cleaned_content = clean_text(document.page_content)

        if not cleaned_content:
            continue

        document.page_content = cleaned_content

        document.metadata["has_text"] = True
        document.metadata["content_length"] = len(
            cleaned_content
        )

        cleaned_documents.append(document)

    return cleaned_documents