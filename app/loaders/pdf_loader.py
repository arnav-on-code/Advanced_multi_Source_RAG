from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(file_path: str | Path) -> list[Document]:
    """Load a PDF into page-level LangChain documents."""

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    documents = PyPDFLoader(str(path)).load()

    for document in documents:
        page = document.metadata.get("page", 0)

        document.metadata.update(
            {
                "source_type": "pdf",
                "source": str(path),
                "file_name": path.name,
                "page": page + 1,
                "has_text": bool(document.page_content.strip()),
            }
        )

    return documents