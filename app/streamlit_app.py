import os
from typing import Any

import requests
import streamlit as st


# =========================================================
# Configuration
# =========================================================

API_URL = os.getenv(
    "RAG_API_URL",
    "http://localhost:8000",
).rstrip("/")

REQUEST_TIMEOUT = (
    10,
    300,
)

SUPPORTED_SOURCES = (
    "PDF",
    "URL",
    "YouTube",
    "DOCX",
    "CSV",
    "TXT",
    "JSON",
)

FILE_SOURCES = {
    "pdf",
    "docx",
    "csv",
    "txt",
    "json",
}


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="P1 — Advanced Multi-Source RAG",
    page_icon="📚",
    layout="wide",
)


# =========================================================
# Session state
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_status" not in st.session_state:
    st.session_state.api_status = None


# =========================================================
# API Client
# =========================================================

class RAGAPIClient:
    """HTTP client for communicating with the FastAPI backend."""

    def __init__(
        self,
        base_url: str,
        timeout: tuple[int, int] = REQUEST_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self.session = requests.Session()

    def health(self) -> bool:
        """Check whether the FastAPI backend is available."""

        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=10,
            )

            return response.ok

        except requests.RequestException:
            return False

    def ingest_file(
        self,
        source_type: str,
        uploaded_file: Any,
    ) -> dict[str, Any]:
        """Upload and ingest a file source."""

        response = self.session.post(
            f"{self.base_url}/ingest",
            data={
                "source_type": source_type,
            },
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                ),
            },
            timeout=self.timeout,
        )

        return self._parse_response(
            response
        )

    def ingest_url(
        self,
        source_type: str,
        source: str,
    ) -> dict[str, Any]:
        """Ingest URL-based sources."""

        response = self.session.post(
            f"{self.base_url}/ingest",
            data={
                "source_type": source_type,
                "source": source,
            },
            timeout=self.timeout,
        )

        return self._parse_response(
            response
        )

    def query(
        self,
        query: str,
        top_k: int,
        use_reranking: bool,
    ) -> dict[str, Any]:
        """Send a RAG query to the backend."""

        response = self.session.post(
            f"{self.base_url}/query",
            json={
                "query": query,
                "top_k": top_k,
                "use_reranking": use_reranking,
            },
            timeout=self.timeout,
        )

        return self._parse_response(
            response
        )

    @staticmethod
    def _parse_response(
        response: requests.Response,
    ) -> dict[str, Any]:
        """Parse successful and failed API responses."""

        try:
            payload = response.json()
        except ValueError:
            payload = {}

        if not response.ok:
            detail = payload.get(
                "detail",
                "The backend returned an unexpected error.",
            )

            raise RuntimeError(
                f"API error ({response.status_code}): "
                f"{detail}"
            )

        return payload


api = RAGAPIClient(API_URL)


# =========================================================
# UI Helpers
# =========================================================

def render_ingestion_result(
    result: dict[str, Any],
) -> None:
    """Render ingestion statistics."""

    st.success(
        result.get(
            "message",
            "Source ingested successfully.",
        )
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Documents",
            result.get("documents", 0),
        )

    with col2:
        st.metric(
            "Chunks",
            result.get("chunks", 0),
        )

    with col3:
        st.metric(
            "Status",
            result.get("status", "unknown"),
        )


def render_sources(
    sources: list[dict[str, Any]],
) -> None:
    """Render source citations consistently."""

    if not sources:
        return

    with st.expander(
        f"Sources ({len(sources)})"
    ):
        for source in sources:
            rank = source.get(
                "rank",
                "-",
            )

            source_type = source.get(
                "source_type",
                "unknown",
            )

            source_name = (
                source.get("file_name")
                or source.get("source")
                or "Unknown source"
            )

            st.markdown(
                f"**#{rank}** "
                f"{source_type} — "
                f"{source_name}"
            )

            page = source.get("page")

            if page is not None:
                st.caption(
                    f"Page: {page}"
                )

            citation = source.get(
                "citation"
            )

            if citation:
                st.json(citation)


def render_chat_history() -> None:
    """Render previous conversation messages."""

    for message in st.session_state.messages:
        with st.chat_message(
            message["role"]
        ):
            st.markdown(
                message["content"]
            )

            sources = message.get(
                "sources",
                [],
            )

            render_sources(sources)


def handle_ingestion_error(
    exc: Exception,
) -> None:
    """Display a user-friendly ingestion error."""

    if isinstance(
        exc,
        requests.RequestException,
    ):
        st.error(
            "Unable to reach the RAG backend. "
            "Please check whether the API is running."
        )
    else:
        st.error(
            str(exc)
        )


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.title("P1 RAG")

    st.caption(
        "Advanced Multi-Source RAG API"
    )

    st.divider()

    st.subheader("Backend")

    if st.session_state.api_status is None:
        st.session_state.api_status = api.health()

    if st.session_state.api_status:
        st.success("API Online")
    else:
        st.error("API Offline")

    st.caption(
        f"API: `{API_URL}`"
    )

    if st.button("Refresh API Status"):
        st.session_state.api_status = api.health()
        st.rerun()

    st.divider()

    st.subheader("Retrieval")

    top_k = st.slider(
        "Retrieved documents",
        min_value=1,
        max_value=20,
        value=5,
    )

    use_reranking = st.checkbox(
        "Enable re-ranking",
        value=True,
    )

    st.divider()

    st.caption("Supported sources")

    st.caption(
        "PDF · URL · YouTube · DOCX · CSV · TXT · JSON"
    )


# =========================================================
# Main UI
# =========================================================

st.title(
    "Advanced Multi-Source RAG"
)

st.write(
    "Upload or connect multiple data sources and "
    "query them using hybrid retrieval with re-ranking."
)


# =========================================================
# Ingestion
# =========================================================

st.header("1. Add a Source")

source_type = st.selectbox(
    "Source type",
    SUPPORTED_SOURCES,
)

source_key = source_type.lower()


if source_key in FILE_SOURCES:

    uploaded_file = st.file_uploader(
        f"Upload {source_type} file",
        type=[source_key],
    )

    if st.button(
        "Ingest File",
        type="primary",
        disabled=uploaded_file is None,
    ):

        with st.spinner(
            f"Processing {source_type}..."
        ):
            try:
                result = api.ingest_file(
                    source_type=source_key,
                    uploaded_file=uploaded_file,
                )

                render_ingestion_result(
                    result
                )

            except Exception as exc:
                handle_ingestion_error(exc)

else:

    label = (
        "YouTube URL"
        if source_key == "youtube"
        else "Web page URL"
    )

    placeholder = (
        "https://www.youtube.com/watch?v=..."
        if source_key == "youtube"
        else "https://example.com"
    )

    source = st.text_input(
        label,
        placeholder=placeholder,
    )

    if st.button(
        "Ingest Source",
        type="primary",
        disabled=not source.strip(),
    ):

        with st.spinner(
            f"Processing {source_type}..."
        ):
            try:
                result = api.ingest_url(
                    source_type=source_key,
                    source=source.strip(),
                )

                render_ingestion_result(
                    result
                )

            except Exception as exc:
                handle_ingestion_error(exc)


# =========================================================
# Query
# =========================================================

st.divider()

st.header("2. Ask a Question")

render_chat_history()

query = st.chat_input(
    "Ask something about your sources..."
)


if query:

    query = query.strip()

    if not query:
        st.warning(
            "Please enter a question."
        )
        st.stop()

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching and generating..."
        ):
            try:
                result = api.query(
                    query=query,
                    top_k=top_k,
                    use_reranking=use_reranking,
                )

                answer = result.get(
                    "answer",
                    "No answer returned.",
                )

                sources = result.get(
                    "sources",
                    [],
                )

                st.markdown(answer)

                render_sources(
                    sources
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except requests.RequestException:
                st.error(
                    "Unable to reach the RAG API."
                )

            except Exception as exc:
                st.error(
                    f"Query failed: {exc}"
                )