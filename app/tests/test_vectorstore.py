from __future__ import annotations

import pytest

from langchain_core.documents import Document

from app.vectorstore.chroma import (
    ChromaVectorStore,
    DEFAULT_COLLECTION_NAME,
)


class MockEmbeddings:
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


class MockCollection:
    def __init__(self):
        self.deleted_ids = None
        self.deleted_source = None

    def count(self):
        return 2

    def delete(self, ids=None, where=None):
        self.deleted_ids = ids
        self.deleted_source = where


class MockChroma:
    def __init__(self, *args, **kwargs):
        self._collection = MockCollection()
        self._embedding_function = kwargs.get(
            "embedding_function"
        )
        self.deleted_ids = None
        self.added_documents = []
        self.updated_documents = []

    def add_documents(
        self,
        documents,
        ids=None,
    ):
        self.added_documents.extend(
            zip(documents, ids or [])
        )
        return ids

    def update_documents(
        self,
        ids,
        documents,
    ):
        self.updated_documents.extend(
            zip(ids, documents)
        )

    def similarity_search(
        self,
        query,
        k=5,
        filter=None,
    ):
        return [
            Document(
                page_content="FastAPI is a Python framework.",
                metadata={"source_type": "txt"},
            )
        ]

    def similarity_search_with_score(
        self,
        query,
        k=5,
        filter=None,
    ):
        return [
            (
                Document(
                    page_content="FastAPI is a Python framework.",
                    metadata={"source_type": "txt"},
                ),
                0.1,
            )
        ]

    def as_retriever(
        self,
        search_type="similarity",
        search_kwargs=None,
    ):
        return {
            "search_type": search_type,
            "search_kwargs": search_kwargs,
        }

    def delete(self, ids=None):
        self.deleted_ids = ids

    def delete_collection(self):
        return None


@pytest.fixture
def mock_chroma(monkeypatch):
    monkeypatch.setattr(
        "app.vectorstore.chroma.Chroma",
        MockChroma,
    )


@pytest.fixture
def store(mock_chroma):
    return ChromaVectorStore(
        embedding_function=MockEmbeddings(),
    )


def test_chroma_vectorstore_initialization(store):
    assert store.collection_name == DEFAULT_COLLECTION_NAME
    assert store.persist_directory.exists()


def test_add_documents(store):
    documents = [
        Document(
            page_content="FastAPI is a Python framework.",
            metadata={"source_type": "txt"},
        )
    ]

    ids = store.add_documents(documents)

    assert len(ids) == 1
    assert isinstance(ids[0], str)


def test_add_documents_with_custom_ids(store):
    documents = [
        Document(page_content="Document 1"),
        Document(page_content="Document 2"),
    ]

    ids = store.add_documents(
        documents,
        ids=["id-1", "id-2"],
    )

    assert ids == ["id-1", "id-2"]


def test_add_documents_rejects_mismatched_ids(store):
    documents = [
        Document(page_content="Document 1"),
        Document(page_content="Document 2"),
    ]

    with pytest.raises(
        ValueError,
        match="Number of IDs must match",
    ):
        store.add_documents(
            documents,
            ids=["id-1"],
        )


def test_add_documents_rejects_duplicate_ids(store):
    documents = [
        Document(page_content="Document 1"),
        Document(page_content="Document 2"),
    ]

    with pytest.raises(
        ValueError,
        match="Document IDs must be unique",
    ):
        store.add_documents(
            documents,
            ids=["same-id", "same-id"],
        )


def test_add_empty_documents(store):
    result = store.add_documents([])

    assert result == []


def test_similarity_search(store):
    results = store.similarity_search(
        query="What is FastAPI?",
        k=1,
    )

    assert len(results) == 1
    assert isinstance(results[0], Document)
    assert "FastAPI" in results[0].page_content


def test_similarity_search_with_score(store):
    results = store.similarity_search_with_score(
        query="What is FastAPI?",
        k=1,
    )

    assert len(results) == 1

    document, score = results[0]

    assert isinstance(document, Document)
    assert score == 0.1


def test_similarity_search_supports_metadata_filter(store):
    results = store.similarity_search(
        query="FastAPI",
        k=1,
        metadata_filter={"source_type": "txt"},
    )

    assert len(results) == 1


@pytest.mark.parametrize(
    "query",
    ["", " ", "\n", "\t"],
)
def test_similarity_search_rejects_empty_query(
    store,
    query,
):
    with pytest.raises(
        ValueError,
        match="Search query cannot be empty",
    ):
        store.similarity_search(
            query=query,
            k=5,
        )


@pytest.mark.parametrize(
    "k",
    [0, -1, 101],
)
def test_similarity_search_rejects_invalid_k(
    store,
    k,
):
    with pytest.raises(
        ValueError,
        match="k must be between 1 and 100",
    ):
        store.similarity_search(
            query="FastAPI",
            k=k,
        )


def test_get_retriever(store):
    retriever = store.get_retriever(k=5)

    assert retriever["search_type"] == "similarity"
    assert retriever["search_kwargs"]["k"] == 5


def test_get_retriever_with_filter(store):
    retriever = store.get_retriever(
        k=5,
        metadata_filter={
            "source_type": "pdf",
        },
    )

    assert retriever["search_kwargs"]["filter"] == {
        "source_type": "pdf",
    }


def test_delete(store):
    store.delete(
        ids=["chunk-1", "chunk-2"],
    )

    assert store.vectorstore.deleted_ids == [
        "chunk-1",
        "chunk-2",
    ]


def test_delete_empty_ids(store):
    store.delete(ids=[])

    assert store.vectorstore.deleted_ids is None


def test_delete_by_source(store):
    store.delete_by_source(
        "documents/example.pdf",
    )

    assert store.vectorstore._collection.deleted_source == {
        "source": "documents/example.pdf",
    }


def test_delete_by_source_rejects_empty_source(store):
    with pytest.raises(
        ValueError,
        match="source cannot be empty",
    ):
        store.delete_by_source("")


def test_count(store):
    assert store.count() == 2


def test_reset(store):
    store.reset()

    assert store.collection_name == DEFAULT_COLLECTION_NAME


def test_invalid_collection_name(mock_chroma):
    with pytest.raises(
        ValueError,
        match="collection_name cannot be empty",
    ):
        ChromaVectorStore(
            embedding_function=MockEmbeddings(),
            collection_name="   ",
        )


def test_invalid_batch_size(store):
    documents = [
        Document(page_content="Test"),
    ]

    with pytest.raises(
        ValueError,
        match="batch_size must be greater than zero",
    ):
        store.add_documents(
            documents,
            batch_size=0,
        )