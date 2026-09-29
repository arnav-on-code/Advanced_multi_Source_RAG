import tempfile
from pathlib import Path
from typing import Callable

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from langchain_core.documents import Document

from app.core.config import settings
from app.core.exceptions import (
    LoaderError,
    ProcessingError,
    UnsupportedSourceError,
)
from app.core.logging import get_logger
from app.loaders.csv_loader import load_csv
from app.loaders.docx_loader import load_docx
from app.loaders.json_loader import load_json
from app.loaders.pdf_loader import load_pdf
from app.loaders.txt_loader import load_txt
from app.loaders.url_loader import load_url
from app.loaders.youtube_loader import load_youtube
from app.processing.chunker import chunk_documents
from app.processing.cleaner import clean_documents
from app.processing.metadata import enrich_documents


router = APIRouter(
    prefix="/ingest",
    tags=["Ingestion"],
)

logger = get_logger(__name__)


MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB


FILE_LOADERS: dict[
    str,
    Callable[[str | Path], list[Document]],
] = {
    "pdf": load_pdf,
    "docx": load_docx,
    "csv": load_csv,
    "txt": load_txt,
    "json": load_json,
}


URL_LOADERS: dict[
    str,
    Callable[[str], list[Document]],
] = {
    "url": load_url,
    "youtube": load_youtube,
}


def _validate_source_type(
    source_type: str,
) -> str:
    """Validate and normalize source type."""

    normalized = source_type.strip().lower()

    if normalized not in settings.source_types:
        raise UnsupportedSourceError(
            normalized
        )

    if (
        normalized not in FILE_LOADERS
        and normalized not in URL_LOADERS
    ):
        raise UnsupportedSourceError(
            normalized
        )

    return normalized


async def _save_upload(
    file: UploadFile,
    extension: str,
) -> Path:
    """
    Save an uploaded file to a temporary location.

    The file is written in chunks to avoid loading the
    complete upload into memory.
    """

    filename = file.filename or "uploaded"

    actual_extension = Path(
        filename
    ).suffix.lower()

    if actual_extension != extension:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected a {extension} file, "
                f"got {actual_extension or 'unknown'}."
            ),
        )

    temporary_file = tempfile.NamedTemporaryFile(
        suffix=extension,
        delete=False,
    )

    temporary_path = Path(
        temporary_file.name
    )

    total_size = 0

    try:
        while chunk := await file.read(
            1024 * 1024
        ):
            total_size += len(chunk)

            if total_size > MAX_UPLOAD_SIZE:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        "Uploaded file exceeds the "
                        "50 MB size limit."
                    ),
                )

            temporary_file.write(chunk)

    except Exception:
        temporary_path.unlink(
            missing_ok=True
        )
        raise

    finally:
        temporary_file.close()
        await file.close()

    return temporary_path


def _load_file_source(
    source_type: str,
    file_path: Path,
) -> list[Document]:
    """Load a supported file source."""

    loader = FILE_LOADERS.get(
        source_type
    )

    if loader is None:
        raise UnsupportedSourceError(
            source_type
        )

    return loader(file_path)


def _load_url_source(
    source_type: str,
    source: str,
) -> list[Document]:
    """Load a supported URL-based source."""

    loader = URL_LOADERS.get(
        source_type
    )

    if loader is None:
        raise UnsupportedSourceError(
            source_type
        )

    return loader(source)


@router.post("/")
async def ingest_source(
    source_type: str = Form(...),
    source: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
):
    """
    Ingest and preprocess a supported source.

    Supported sources:
        PDF, URL, YouTube, DOCX, CSV, TXT, JSON
    """

    temporary_path: Path | None = None

    try:
        source_type = _validate_source_type(
            source_type
        )

        is_file_source = (
            source_type in FILE_LOADERS
        )

        is_url_source = (
            source_type in URL_LOADERS
        )

        # -----------------------------------------------------
        # Validate request combination
        # -----------------------------------------------------

        if is_file_source:

            if file is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"'{source_type}' requires "
                        "a file upload."
                    ),
                )

            if source:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Do not provide 'source' "
                        "when uploading a file."
                    ),
                )

        elif is_url_source:

            if not source:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"'{source_type}' requires "
                        "a source URL."
                    ),
                )

            if file is not None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Do not upload a file for "
                        "URL-based sources."
                    ),
                )

        # -----------------------------------------------------
        # Load source
        # -----------------------------------------------------

        if is_file_source:

            extension = Path(
                file.filename or ""
            ).suffix.lower()

            expected_extension = {
                "pdf": ".pdf",
                "docx": ".docx",
                "csv": ".csv",
                "txt": ".txt",
                "json": ".json",
            }[source_type]

            if extension != expected_extension:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Expected a "
                        f"{expected_extension} file."
                    ),
                )

            temporary_path = await _save_upload(
                file=file,
                extension=expected_extension,
            )

            documents = _load_file_source(
                source_type,
                temporary_path,
            )

        else:
            documents = _load_url_source(
                source_type,
                source.strip(),
            )

        # -----------------------------------------------------
        # Processing
        # -----------------------------------------------------

        if not documents:
            raise ProcessingError(
                "The source produced no documents."
            )

        documents = clean_documents(
            documents
        )

        documents = enrich_documents(
            documents
        )

        chunks = chunk_documents(
            documents,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        if not chunks:
            raise ProcessingError(
                "No usable chunks were produced."
            )

        logger.info(
            "Source ingestion completed | "
            "type=%s | documents=%d | chunks=%d",
            source_type,
            len(documents),
            len(chunks),
        )

        return {
            "status": "success",
            "source_type": source_type,
            "documents": len(documents),
            "chunks": len(chunks),
            "message": (
                "Source loaded and processed successfully."
            ),
        }

    except HTTPException:
        raise

    except (
        FileNotFoundError,
        ValueError,
        LoaderError,
        ProcessingError,
        UnsupportedSourceError,
    ) as exc:

        logger.warning(
            "Source ingestion failed | "
            "type=%s | error=%s",
            source_type,
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        logger.exception(
            "Unexpected ingestion failure | "
            "type=%s",
            source_type,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to process the source.",
        ) from exc

    finally:
        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True
            )