from __future__ import annotations

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


# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

MAX_UPLOAD_SIZE = 50 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024


# ---------------------------------------------------------------------------
# LOADER REGISTRY
# ---------------------------------------------------------------------------

FILE_LOADERS: dict[str, str] = {
    "pdf": "load_pdf",
    "docx": "load_docx",
    "csv": "load_csv",
    "txt": "load_txt",
    "json": "load_json",
}

URL_LOADERS: dict[str, str] = {
    "url": "load_url",
    "youtube": "load_youtube",
}

FILE_EXTENSIONS: dict[str, str] = {
    "pdf": ".pdf",
    "docx": ".docx",
    "csv": ".csv",
    "txt": ".txt",
    "json": ".json",
}


# ---------------------------------------------------------------------------
# SOURCE VALIDATION
# ---------------------------------------------------------------------------

def _validate_source_type(
    source_type: str | None,
) -> str:
    """Validate and normalize the requested source type."""

    if source_type is None:
        raise HTTPException(
            status_code=400,
            detail="source_type is required.",
        )

    normalized = source_type.strip().lower()

    if not normalized:
        raise HTTPException(
            status_code=400,
            detail="source_type cannot be empty.",
        )

    if normalized not in settings.source_types:
        raise UnsupportedSourceError(normalized)

    if (
        normalized not in FILE_LOADERS
        and normalized not in URL_LOADERS
    ):
        raise UnsupportedSourceError(normalized)

    return normalized


# ---------------------------------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------------------------------

async def _save_upload(
    file: UploadFile,
    extension: str,
) -> Path:
    """
    Save an uploaded file to a temporary path.

    The temporary file is closed before writing so the implementation
    behaves correctly on Windows and other platforms with stricter
    file-locking semantics.
    """

    filename = file.filename or "uploaded"

    actual_extension = Path(filename).suffix.lower()

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

    temporary_path = Path(temporary_file.name)

    # Close the NamedTemporaryFile immediately.
    # This is important on Windows because the file must not remain
    # open while we later write/delete it.
    temporary_file.close()

    try:
        total_size = 0

        with temporary_path.open("wb") as output_file:
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_UPLOAD_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Uploaded file exceeds the "
                            "50 MB size limit."
                        ),
                    )

                output_file.write(chunk)

            output_file.flush()

        return temporary_path

    except HTTPException:
        temporary_path.unlink(missing_ok=True)
        raise

    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    finally:
        await file.close()


# ---------------------------------------------------------------------------
# LOADER RESOLUTION
# ---------------------------------------------------------------------------

def _resolve_loader(
    loader_name: str,
) -> Callable:
    """
    Resolve the current loader function.

    Using globals() here means tests can monkeypatch the imported
    loader function and the ingestion endpoint will use the patched
    function instead of a stale function captured at import time.
    """

    loader = globals().get(loader_name)

    if not callable(loader):
        raise UnsupportedSourceError(loader_name)

    return loader


def _load_file_source(
    source_type: str,
    file_path: Path,
) -> list[Document]:
    """Load a supported file source."""

    loader_name = FILE_LOADERS.get(source_type)

    if loader_name is None:
        raise UnsupportedSourceError(source_type)

    loader = _resolve_loader(loader_name)

    return loader(file_path)


def _load_url_source(
    source_type: str,
    source: str,
) -> list[Document]:
    """Load a supported URL-based source."""

    loader_name = URL_LOADERS.get(source_type)

    if loader_name is None:
        raise UnsupportedSourceError(source_type)

    loader = _resolve_loader(loader_name)

    return loader(source)


# ---------------------------------------------------------------------------
# INGESTION ENDPOINT
# ---------------------------------------------------------------------------

@router.post("/")
async def ingest_source(
    source_type: str | None = Form(default=None),
    source: str | None = Form(default=None),
    file: UploadFile | None = File(default=None),
):
    """
    Ingest and preprocess a supported source.

    Supported sources:
        PDF
        URL
        YouTube
        DOCX
        CSV
        TXT
        JSON

    Pipeline:

        Source
          ↓
        Loader
          ↓
        Cleaning
          ↓
        Metadata enrichment
          ↓
        Chunking
          ↓
        Processed documents
    """

    temporary_path: Path | None = None

    try:
        # ---------------------------------------------------------------
        # SOURCE TYPE
        # ---------------------------------------------------------------

        normalized_source_type = _validate_source_type(
            source_type
        )

        source_value = (
            source.strip()
            if source is not None
            else None
        )

        is_file_source = (
            normalized_source_type in FILE_LOADERS
        )

        is_url_source = (
            normalized_source_type in URL_LOADERS
        )

        # ---------------------------------------------------------------
        # REQUEST VALIDATION
        # ---------------------------------------------------------------

        if is_file_source:

            if file is None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"'{normalized_source_type}' "
                        "requires a file upload."
                    ),
                )

            if source_value:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Do not provide 'source' "
                        "when uploading a file."
                    ),
                )

        elif is_url_source:

            if not source_value:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"'{normalized_source_type}' "
                        "requires a source URL."
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

        # ---------------------------------------------------------------
        # LOAD SOURCE
        # ---------------------------------------------------------------

        if is_file_source:
            assert file is not None

            filename = file.filename or ""

            extension = Path(filename).suffix.lower()

            expected_extension = FILE_EXTENSIONS[
                normalized_source_type
            ]

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
                normalized_source_type,
                temporary_path,
            )

        else:
            assert source_value is not None

            documents = _load_url_source(
                normalized_source_type,
                source_value,
            )

        # ---------------------------------------------------------------
        # DOCUMENT VALIDATION
        # ---------------------------------------------------------------

        if not documents:
            raise ProcessingError(
                "The source produced no documents."
            )

        # ---------------------------------------------------------------
        # CLEANING
        # ---------------------------------------------------------------

        documents = clean_documents(documents)

        if not documents:
            raise ProcessingError(
                "No usable documents were produced "
                "after cleaning."
            )

        # ---------------------------------------------------------------
        # METADATA
        # ---------------------------------------------------------------

        documents = enrich_documents(documents)

        # ---------------------------------------------------------------
        # CHUNKING
        # ---------------------------------------------------------------

        chunks = chunk_documents(
            documents,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        if not chunks:
            raise ProcessingError(
                "No usable chunks were produced."
            )

        # ---------------------------------------------------------------
        # SUCCESS
        # ---------------------------------------------------------------

        logger.info(
            "Source ingestion completed | "
            "type=%s | documents=%d | chunks=%d",
            normalized_source_type,
            len(documents),
            len(chunks),
        )

        return {
            "status": "success",
            "source_type": normalized_source_type,
            "documents": len(documents),
            "chunks": len(chunks),
            "message": (
                "Source loaded and processed successfully."
            ),
        }

    # -------------------------------------------------------------------
    # EXPECTED HTTP ERRORS
    # -------------------------------------------------------------------

    except HTTPException:
        raise

    # -------------------------------------------------------------------
    # EXPECTED APPLICATION ERRORS
    # -------------------------------------------------------------------

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

    # -------------------------------------------------------------------
    # UNEXPECTED ERRORS
    # -------------------------------------------------------------------

    except Exception as exc:

        logger.exception(
            "Unexpected ingestion failure | "
            "type=%s | error=%s",
            source_type,
            exc,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to process the source.",
        ) from exc

    # -------------------------------------------------------------------
    # TEMPORARY FILE CLEANUP
    # -------------------------------------------------------------------

    finally:

        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True
            )