from pathlib import Path

from langchain_community.document_loaders import CSVLoader
from langchain_core.documents import Document


def load_csv(file_path: str | Path) -> list[Document]:
    """Load a CSV file as row-level LangChain Documents."""

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file, got: {path.suffix}")

    try:
        documents = CSVLoader(
            file_path=str(path),
            encoding="utf-8",
        ).load()
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            f"Unable to decode CSV '{path}' using UTF-8."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load CSV '{path}': {exc}"
        ) from exc

    for row_number, document in enumerate(documents, start=1):
        document.metadata.update(
            {
                "source_type": "csv",
                "source": str(path),
                "file_name": path.name,
                "row": row_number,
            }
        )

    return documents