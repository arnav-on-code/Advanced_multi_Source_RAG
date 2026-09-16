from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document


def load_docx(file_path: str | Path) -> list[Document]:
    """Load a DOCX file as LangChain Documents."""

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"DOCX file not found: {path}")

    if path.suffix.lower() != ".docx":
        raise ValueError(f"Expected a DOCX file, got: {path.suffix}")

    try:
        documents = Docx2txtLoader(str(path)).load()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load DOCX '{path}': {exc}"
        ) from exc

    for document in documents:
        document.metadata.update(
            {
                "source_type": "docx",
                "source": str(path),
                "file_name": path.name,
            }
        )

    return documents