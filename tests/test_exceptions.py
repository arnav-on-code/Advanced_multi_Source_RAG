from __future__ import annotations

import pytest

from app.core.exceptions import (
    ConfigurationError,
    EmbeddingError,
    LLMError,
    LoaderError,
    ProcessingError,
    RAGException,
    RerankingError,
    RetrievalError,
    UnsupportedSourceError,
    VectorStoreError,
)


@pytest.mark.parametrize(
    ("exception_class", "expected_code"),
    [
        (ConfigurationError, "CONFIGURATION_ERROR"),
        (ProcessingError, "PROCESSING_ERROR"),
        (EmbeddingError, "EMBEDDING_ERROR"),
        (VectorStoreError, "VECTOR_STORE_ERROR"),
        (RetrievalError, "RETRIEVAL_ERROR"),
        (RerankingError, "RERANKING_ERROR"),
        (LLMError, "LLM_ERROR"),
    ],
)
def test_rag_exception_subclasses(
    exception_class,
    expected_code,
):
    error = exception_class("Test error")

    assert isinstance(error, RAGException)
    assert str(error) == "Test error"
    assert error.message == "Test error"
    assert error.code == expected_code


def test_loader_error():
    error = LoaderError(
        "Failed to load document",
        source_type="pdf",
    )

    assert isinstance(error, RAGException)
    assert str(error) == "Failed to load document"
    assert error.message == "Failed to load document"
    assert error.source_type == "pdf"
    assert error.code == "LOADER_ERROR"


def test_loader_error_without_source_type():
    error = LoaderError(
        "Failed to load document",
    )

    assert isinstance(error, RAGException)
    assert str(error) == "Failed to load document"
    assert error.message == "Failed to load document"
    assert error.source_type is None
    assert error.code == "LOADER_ERROR"


@pytest.mark.parametrize(
    "source_type",
    [
        "audio",
        "image",
        "unknown",
        "excel",
    ],
)
def test_unsupported_source_error(source_type):
    error = UnsupportedSourceError(source_type)

    assert isinstance(error, RAGException)
    assert error.source_type == source_type
    assert error.code == "UNSUPPORTED_SOURCE"
    assert str(error) == (
        f"Unsupported source type: {source_type}"
    )


def test_exception_can_be_raised():
    with pytest.raises(
        ConfigurationError,
        match="Invalid configuration",
    ):
        raise ConfigurationError(
            "Invalid configuration"
        )


def test_exception_message_is_preserved():
    message = "Something went wrong."

    error = ProcessingError(message)

    assert error.message == message
    assert str(error) == message


def test_exception_inheritance():
    assert issubclass(
        ConfigurationError,
        RAGException,
    )

    assert issubclass(
        LoaderError,
        RAGException,
    )

    assert issubclass(
        UnsupportedSourceError,
        RAGException,
    )


def test_all_exception_codes_are_unique():
    exception_classes = [
        ConfigurationError,
        EmbeddingError,
        LoaderError,
        ProcessingError,
        VectorStoreError,
        RetrievalError,
        RerankingError,
        LLMError,
        UnsupportedSourceError,
    ]

    codes = [
        exception_class.code
        for exception_class in exception_classes
    ]

    assert len(codes) == len(set(codes))