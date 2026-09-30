from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.loaders.url_loader import load_url


# =========================================================
# VALIDATION
# =========================================================


def test_url_empty():
    """Empty URLs should be rejected."""

    with pytest.raises(
        ValueError,
        match="URL cannot be empty",
    ):
        load_url("")


def test_url_whitespace():
    """Whitespace-only URLs should be rejected."""

    with pytest.raises(
        ValueError,
        match="URL cannot be empty",
    ):
        load_url("   ")


@pytest.mark.parametrize(
    "url",
    [
        "example.com",
        "ftp://example.com",
        "file:///example.html",
        "://invalid",
        "http://",
        "https://",
    ],
)
def test_url_invalid(
    url: str,
):
    """Invalid or unsupported URLs should be rejected."""

    with pytest.raises(
        ValueError,
        match="Invalid URL",
    ):
        load_url(url)


# =========================================================
# SUCCESSFUL LOADING
# =========================================================


def test_url_loader(
    monkeypatch: pytest.MonkeyPatch,
):
    """URL loader should return documents with normalized metadata."""

    url = "https://example.com"

    mock_documents = [
        Document(
            page_content="Example web content",
            metadata={
                "title": "Example Page",
            },
        )
    ]

    class MockWebBaseLoader:
        def __init__(self, loader_url: str):
            assert loader_url == url
            self.url = loader_url

        def load(self):
            return mock_documents

    monkeypatch.setattr(
        "app.loaders.url_loader.WebBaseLoader",
        MockWebBaseLoader,
    )

    documents = load_url(url)

    assert len(documents) == 1

    document = documents[0]

    assert document.page_content == (
        "Example web content"
    )

    metadata = document.metadata

    assert metadata["source_type"] == "url"
    assert metadata["source"] == url
    assert metadata["url"] == url
    assert metadata["title"] == "Example Page"


# =========================================================
# URL NORMALIZATION
# =========================================================


def test_url_loader_strips_whitespace(
    monkeypatch: pytest.MonkeyPatch,
):
    """Leading/trailing whitespace should be removed."""

    url = "https://example.com"

    mock_documents = [
        Document(
            page_content="Example content",
            metadata={},
        )
    ]

    class MockWebBaseLoader:
        def __init__(self, loader_url: str):
            assert loader_url == url
            self.url = loader_url

        def load(self):
            return mock_documents

    monkeypatch.setattr(
        "app.loaders.url_loader.WebBaseLoader",
        MockWebBaseLoader,
    )

    documents = load_url(
        "  https://example.com  "
    )

    assert documents[0].metadata["source"] == url
    assert documents[0].metadata["url"] == url


# =========================================================
# EMPTY CONTENT
# =========================================================


def test_url_loader_empty_content(
    monkeypatch: pytest.MonkeyPatch,
):
    """
    Empty web content should still be returned.

    The loader should not silently discard documents;
    downstream processing decides whether they are usable.
    """

    url = "https://example.com"

    mock_documents = [
        Document(
            page_content="   ",
            metadata={},
        )
    ]

    class MockWebBaseLoader:
        def __init__(self, loader_url: str):
            self.url = loader_url

        def load(self):
            return mock_documents

    monkeypatch.setattr(
        "app.loaders.url_loader.WebBaseLoader",
        MockWebBaseLoader,
    )

    documents = load_url(url)

    assert len(documents) == 1
    assert documents[0].page_content.strip() == ""

    assert documents[0].metadata["source_type"] == "url"
    assert documents[0].metadata["source"] == url
    assert documents[0].metadata["url"] == url


# =========================================================
# MULTIPLE DOCUMENTS
# =========================================================


def test_url_loader_multiple_documents(
    monkeypatch: pytest.MonkeyPatch,
):
    """Multiple documents returned by WebBaseLoader should be preserved."""

    url = "https://example.com"

    mock_documents = [
        Document(
            page_content="First section",
            metadata={"title": "Example"},
        ),
        Document(
            page_content="Second section",
            metadata={"title": "Example"},
        ),
    ]

    class MockWebBaseLoader:
        def __init__(self, loader_url: str):
            self.url = loader_url

        def load(self):
            return mock_documents

    monkeypatch.setattr(
        "app.loaders.url_loader.WebBaseLoader",
        MockWebBaseLoader,
    )

    documents = load_url(url)

    assert len(documents) == 2

    assert documents[0].page_content == "First section"
    assert documents[1].page_content == "Second section"

    assert all(
        document.metadata["source_type"] == "url"
        for document in documents
    )

    assert all(
        document.metadata["source"] == url
        for document in documents
    )


# =========================================================
# LOADER FAILURE
# =========================================================


def test_url_loader_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    """Underlying web-loader failures should be surfaced."""

    url = "https://example.com"

    class MockWebBaseLoader:
        def __init__(self, loader_url: str):
            self.url = loader_url

        def load(self):
            raise RuntimeError(
                "Connection failed"
            )

    monkeypatch.setattr(
        "app.loaders.url_loader.WebBaseLoader",
        MockWebBaseLoader,
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to load URL",
    ):
        load_url(url)


# =========================================================
# EMPTY LOADER RESULT
# =========================================================


def test_url_loader_returns_empty_list(
    monkeypatch: pytest.MonkeyPatch,
):
    """An empty result from WebBaseLoader should remain an empty list."""

    url = "https://example.com"

    class MockWebBaseLoader:
        def __init__(self, loader_url: str):
            self.url = loader_url

        def load(self):
            return []

    monkeypatch.setattr(
        "app.loaders.url_loader.WebBaseLoader",
        MockWebBaseLoader,
    )

    documents = load_url(url)

    assert documents == []