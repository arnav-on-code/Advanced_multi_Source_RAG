from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi


def _extract_video_id(url: str) -> str:
    """Extract the YouTube video ID from a supported URL."""

    parsed = urlparse(url)

    if parsed.netloc.lower() in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
    }:
        video_id = parse_qs(parsed.query).get("v", [None])[0]

    elif parsed.netloc.lower() in {
        "youtu.be",
        "www.youtu.be",
    }:
        video_id = parsed.path.strip("/").split("/")[0]

    else:
        video_id = None

    if not video_id:
        raise ValueError("Could not extract YouTube video ID.")

    return video_id


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

    if (
        parsed.scheme not in {"http", "https"}
        or parsed.netloc.lower() not in valid_hosts
    ):
        raise ValueError("Invalid YouTube URL.")

    video_id = _extract_video_id(url)

    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)

        text = " ".join(
            snippet.text
            for snippet in transcript
            if snippet.text and snippet.text.strip()
        ).strip()

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load YouTube transcript '{url}': {exc}"
        ) from exc

    if not text:
        raise RuntimeError(
            f"YouTube video '{url}' returned an empty transcript."
        )

    return [
        Document(
            page_content=text,
            metadata={
                "source_type": "youtube",
                "source": url,
                "url": url,
                "video_id": video_id,
                "has_text": True,
            },
        )
    ]