from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.loaders.youtube_loader import load_youtube


# =========================================================
# VALIDATION
# =========================================================


def test_youtube_empty_url():
    """Empty YouTube URLs should be rejected."""

    with pytest.raises(
        ValueError,
        match="YouTube URL cannot be empty",
    ):
        load_youtube("")


def test_youtube_whitespace_url():
    """Whitespace-only URLs should be rejected."""

    with pytest.raises(
        ValueError,
        match="YouTube URL cannot be empty",
    ):
        load_youtube("   ")


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/video",
        "https://google.com",
        "https://vimeo.com/123456",
        "ftp://youtube.com/video",
    ],
)
def test_youtube_invalid_url(url: str):
    """Non-YouTube URLs should be rejected."""

    with pytest.raises(
        ValueError,
        match="Invalid YouTube URL",
    ):
        load_youtube(url)


# =========================================================
# SUCCESSFUL LOADING
# =========================================================


def test_youtube_loader(
    monkeypatch: pytest.MonkeyPatch,
):
    """YouTube loader should return transcript documents with metadata."""

    url = "https://www.youtube.com/watch?v=test123"

    mock_documents = [
        Document(
            page_content="YouTube transcript content",
            metadata={
                "title": "Test Video",
            },
        )
    ]

    class MockYoutubeLoader:
        @classmethod
        def from_youtube_url(
            cls,
            loader_url: str,
            add_video_info: bool = False,
        ):
            assert loader_url == url
            assert add_video_info is True

            instance = cls()
            instance.url = loader_url
            instance.add_video_info = add_video_info

            return instance

        def load(self):
            return mock_documents

    monkeypatch.setattr(
        "app.loaders.youtube_loader.YoutubeLoader",
        MockYoutubeLoader,
    )

    documents = load_youtube(url)

    assert len(documents) == 1

    document = documents[0]
    metadata = document.metadata

    assert document.page_content == (
        "YouTube transcript content"
    )

    assert metadata["source_type"] == "youtube"
    assert metadata["source"] == url
    assert metadata["url"] == url
    assert metadata["title"] == "Test Video"
    assert metadata["has_text"] is True


# =========================================================
# URL NORMALIZATION
# =========================================================


def test_youtube_loader_strips_whitespace(
    monkeypatch: pytest.MonkeyPatch,
):
    """Leading and trailing whitespace should be removed."""

    url = "https://youtu.be/test123"

    mock_documents = [
        Document(
            page_content="Transcript content",
            metadata={},
        )
    ]

    class MockYoutubeLoader:
        @classmethod
        def from_youtube_url(
            cls,
            loader_url: str,
            add_video_info: bool = False,
        ):
            assert loader_url == url
            assert add_video_info is True
            return cls()

        def load(self):
            return mock_documents

    monkeypatch.setattr(
        "app.loaders.youtube_loader.YoutubeLoader",
        MockYoutubeLoader,
    )

    documents = load_youtube(
        "  https://youtu.be/test123  "
    )

    assert documents[0].metadata["source"] == url
    assert documents[0].metadata["url"] == url


# =========================================================
# EMPTY CONTENT
# =========================================================


def test_youtube_loader_empty_content(
    monkeypatch: pytest.MonkeyPatch,
):
    """Empty transcript content should be detected."""

    url = "https://youtu.be/test123"

    mock_documents = [
        Document(
            page_content="   ",
            metadata={},
        )
    ]

    class MockYoutubeLoader:
        @classmethod
        def from_youtube_url(
            cls,
            loader_url: str,
            add_video_info: bool = False,
        ):
            assert loader_url == url
            assert add_video_info is True
            return cls()

        def load(self):
            return mock_documents

    monkeypatch.setattr(
        "app.loaders.youtube_loader.YoutubeLoader",
        MockYoutubeLoader,
    )

    documents = load_youtube(url)

    assert len(documents) == 1

    assert documents[0].metadata["has_text"] is False


# =========================================================
# MULTIPLE DOCUMENTS
# =========================================================


def test_youtube_loader_multiple_documents(
    monkeypatch: pytest.MonkeyPatch,
):
    """All transcript documents should be preserved."""

    url = "https://www.youtube.com/watch?v=test123"

    mock_documents = [
        Document(
            page_content="First transcript section",
            metadata={},
        ),
        Document(
            page_content="Second transcript section",
            metadata={},
        ),
    ]

    class MockYoutubeLoader:
        @classmethod
        def from_youtube_url(
            cls,
            loader_url: str,
            add_video_info: bool = False,
        ):
            return cls()

        def load(self):
            return mock_documents

    monkeypatch.setattr(
        "app.loaders.youtube_loader.YoutubeLoader",
        MockYoutubeLoader,
    )

    documents = load_youtube(url)

    assert len(documents) == 2

    assert documents[0].page_content == (
        "First transcript section"
    )

    assert documents[1].page_content == (
        "Second transcript section"
    )

    assert all(
        document.metadata["source_type"] == "youtube"
        for document in documents
    )

    assert all(
        document.metadata["source"] == url
        for document in documents
    )


# =========================================================
# LOADER FAILURE
# =========================================================


def test_youtube_loader_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    """Underlying YouTube loader failures should be propagated."""

    url = "https://www.youtube.com/watch?v=test123"

    class MockYoutubeLoader:
        @classmethod
        def from_youtube_url(
            cls,
            loader_url: str,
            add_video_info: bool = False,
        ):
            return cls()

        def load(self):
            raise RuntimeError(
                "Transcript unavailable"
            )

    monkeypatch.setattr(
        "app.loaders.youtube_loader.YoutubeLoader",
        MockYoutubeLoader,
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to load YouTube video",
    ):
        load_youtube(url)


# =========================================================
# EMPTY RESULT
# =========================================================


def test_youtube_loader_empty_result(
    monkeypatch: pytest.MonkeyPatch,
):
    """An empty result should remain an empty list."""

    url = "https://www.youtube.com/watch?v=test123"

    class MockYoutubeLoader:
        @classmethod
        def from_youtube_url(
            cls,
            loader_url: str,
            add_video_info: bool = False,
        ):
            return cls()

        def load(self):
            return []

    monkeypatch.setattr(
        "app.loaders.youtube_loader.YoutubeLoader",
        MockYoutubeLoader,
    )

    documents = load_youtube(url)

    assert documents == []