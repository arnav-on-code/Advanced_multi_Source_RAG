from urllib.parse import urlparse

from langchain_community.document_loaders import YoutubeLoader
from langchain_core.documents import Document


def load_youtube(url: str) -> list[Document]:
    """Load YouTube transcript content as LangChain Documents."""

    if not url or not url.strip():
        raise ValueError("YouTube URL cannot be empty.")

    url = url.strip()
    parsed = urlparse(url)

    valid_hosts = {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
        "www.youtu.be",
    }

    if parsed.scheme not in {"http", "https"} or parsed.netloc not in valid_hosts:
        raise ValueError("Invalid YouTube URL.")

    try:
        documents = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True,
        ).load()

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load YouTube video '{url}': {exc}"
        ) from exc

    for document in documents:
        document.metadata.update(
            {
                "source_type": "youtube",
                "source": url,
                "url": url,
            }
        )

    return documents