from urllib.parse import urlparse

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document


def load_url(url: str) -> list[Document]:
    """Load a web page and return LangChain Documents."""

    if not url or not url.strip():
        raise ValueError("URL cannot be empty.")

    url = url.strip()
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "Invalid URL. Expected a valid http:// or https:// URL."
        )

    try:
        documents = WebBaseLoader(url).load()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load URL '{url}': {exc}"
        ) from exc

    for document in documents:
        document.metadata.update(
            {
                "source_type": "url",
                "source": url,
                "url": url,
                "title": document.metadata.get("title"),
            }
        )

    return documents