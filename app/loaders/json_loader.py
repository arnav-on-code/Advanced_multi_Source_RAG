from pathlib import Path

from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document

from app.core.exceptions import LoaderError


def load_json(file_path: str | Path) -> list[Document]:
    """Load a JSON file into LangChain Documents."""

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    if path.suffix.lower() != ".json":
        raise ValueError(
            f"Expected a JSON file, got: {path.suffix or 'unknown'}"
        )

    try:
        loader = JSONLoader(
            str(path),
            jq_schema=".",
            text_content=False,
        )

        documents = loader.load()

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load JSON '{path}': {exc}"
        ) from exc

    for document in documents:
        text = document.page_content.strip()

        document.metadata.update(
            {
                "source_type": "json",
                "source": str(path),
                "file_name": path.name,
                "has_text": bool(text),
            }
        )

    return documents