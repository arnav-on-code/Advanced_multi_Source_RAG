from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


def load_txt(file_path: str | Path) -> list[Document]:
    """Load a TXT file as a LangChain Document."""

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"TXT file not found: {path}")

    if path.suffix.lower() != ".txt":
        raise ValueError(f"Expected a TXT file, got: {path.suffix}")

    try:
        documents = TextLoader(
            str(path),
            encoding="utf-8",
            autodetect_encoding=True,
        ).load()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load TXT '{path}': {exc}"
        ) from exc

    for document in documents:
        document.metadata.update(
            {
                "source_type": "txt",
                "source": str(path),
                "file_name": path.name,
            }
        )

    return documents