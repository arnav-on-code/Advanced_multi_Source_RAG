from pathlib import Path

from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document


def load_json(file_path: str | Path) -> list[Document]:
    """Load a JSON file as LangChain Documents."""

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")

    if path.suffix.lower() != ".json":
        raise ValueError(f"Expected a JSON file, got: {path.suffix}")

    try:
        documents = JSONLoader(
            file_path=str(path),
            jq_schema=".",
            text_content=False,
        ).load()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load JSON '{path}': {exc}"
        ) from exc

    for document in documents:
        document.metadata.update(
            {
                "source_type": "json",
                "source": str(path),
                "file_name": path.name,
            }
        )

    return documents