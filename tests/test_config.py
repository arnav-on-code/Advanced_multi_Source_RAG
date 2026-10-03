from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_default_settings():
    settings = Settings()

    assert settings.app_name == (
        "P1 Advanced Multi-Source RAG API"
    )
    assert settings.app_version == "1.0.0"
    assert settings.environment == "development"
    assert settings.debug is False

    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8000

    assert settings.embedding_model == (
        "BAAI/bge-small-en-v1.5"
    )
    assert settings.embedding_device == "cpu"

    assert settings.chroma_persist_directory == (
        "data/chroma"
    )
    assert settings.chroma_collection_name == (
        "p1_rag_documents"
    )

    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 120

    assert settings.retrieval_top_k == 5
    assert settings.retrieval_candidate_k == 15
    assert settings.vector_weight == 0.6
    assert settings.bm25_weight == 0.4

    assert settings.reranker_model == (
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    assert settings.reranker_device == "cpu"
    assert settings.reranker_top_k == 5

    assert settings.llm_provider == "ollama"
    assert settings.llm_model == "llama3.2"
    assert settings.llm_temperature == 0.0

    assert settings.log_level == "INFO"
    assert settings.allowed_origins == "*"


def test_supported_source_types():
    settings = Settings()

    assert settings.source_types == (
        "pdf",
        "url",
        "youtube",
        "docx",
        "csv",
        "txt",
        "json",
    )


def test_custom_supported_source_types():
    settings = Settings(
        supported_sources=(
            "pdf",
            "txt",
            "json",
        )
    )

    assert settings.source_types == (
        "pdf",
        "txt",
        "json",
    )


def test_source_types_are_normalized():
    settings = Settings(
        supported_sources=(
            " PDF ",
            " TXT ",
            "JSON",
        )
    )

    assert settings.source_types == (
        "pdf",
        "txt",
        "json",
    )


def test_custom_retrieval_settings():
    settings = Settings(
        retrieval_top_k=10,
        retrieval_candidate_k=30,
        vector_weight=0.7,
        bm25_weight=0.3,
    )

    assert settings.retrieval_top_k == 10
    assert settings.retrieval_candidate_k == 30
    assert settings.vector_weight == 0.7
    assert settings.bm25_weight == 0.3


def test_custom_chunk_settings():
    settings = Settings(
        chunk_size=1000,
        chunk_overlap=150,
    )

    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 150


def test_cors_origins_wildcard():
    settings = Settings(
        allowed_origins="*",
    )

    assert settings.cors_origins == ["*"]


def test_cors_origins_multiple_values():
    settings = Settings(
        allowed_origins=(
            "http://localhost:3000, "
            "http://localhost:8501"
        ),
    )

    assert settings.cors_origins == [
        "http://localhost:3000",
        "http://localhost:8501",
    ]


def test_log_level_is_normalized():
    settings = Settings(
        log_level=" debug ",
    )

    assert settings.log_level == "DEBUG"


@pytest.mark.parametrize(
    "log_level",
    [
        "INVALID",
        "TRACE",
        "",
    ],
)
def test_invalid_log_level(log_level):
    with pytest.raises(ValidationError):
        Settings(
            log_level=log_level,
        )


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(
        ValidationError,
        match="chunk_overlap must be smaller",
    ):
        Settings(
            chunk_size=500,
            chunk_overlap=500,
        )


def test_retrieval_candidate_k_must_cover_top_k():
    with pytest.raises(
        ValidationError,
        match=(
            "retrieval_candidate_k must be greater "
            "than or equal to retrieval_top_k"
        ),
    ):
        Settings(
            retrieval_top_k=20,
            retrieval_candidate_k=10,
        )


def test_reranker_top_k_cannot_exceed_candidate_k():
    with pytest.raises(
        ValidationError,
        match="reranker_top_k cannot exceed",
    ):
        Settings(
            retrieval_candidate_k=10,
            reranker_top_k=11,
        )


def test_search_weights_cannot_both_be_zero():
    with pytest.raises(
        ValidationError,
        match="cannot both be zero",
    ):
        Settings(
            vector_weight=0,
            bm25_weight=0,
        )


@pytest.mark.parametrize(
    "chunk_size",
    [0, 99, 10001],
)
def test_invalid_chunk_size(chunk_size):
    with pytest.raises(ValidationError):
        Settings(
            chunk_size=chunk_size,
        )


@pytest.mark.parametrize(
    "chunk_overlap",
    [-1, 800],
)
def test_invalid_chunk_overlap(chunk_overlap):
    with pytest.raises(ValidationError):
        Settings(
            chunk_overlap=chunk_overlap,
        )


@pytest.mark.parametrize(
    "top_k",
    [0, 101],
)
def test_invalid_retrieval_top_k(top_k):
    with pytest.raises(ValidationError):
        Settings(
            retrieval_top_k=top_k,
        )


@pytest.mark.parametrize(
    "candidate_k",
    [0, 201],
)
def test_invalid_candidate_k(candidate_k):
    with pytest.raises(ValidationError):
        Settings(
            retrieval_candidate_k=candidate_k,
        )


@pytest.mark.parametrize(
    "weight",
    [-0.1, 1.1],
)
def test_invalid_vector_weight(weight):
    with pytest.raises(ValidationError):
        Settings(
            vector_weight=weight,
        )


@pytest.mark.parametrize(
    "weight",
    [-0.1, 1.1],
)
def test_invalid_bm25_weight(weight):
    with pytest.raises(ValidationError):
        Settings(
            bm25_weight=weight,
        )


@pytest.mark.parametrize(
    "port",
    [0, 65536],
)
def test_invalid_api_port(port):
    with pytest.raises(ValidationError):
        Settings(
            api_port=port,
        )


def test_custom_environment():
    settings = Settings(
        environment="production",
    )

    assert settings.environment == "production"


def test_custom_llm_settings():
    settings = Settings(
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        llm_temperature=0.2,
    )

    assert settings.llm_provider == "openai"
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.llm_temperature == 0.2


@pytest.mark.parametrize(
    "temperature",
    [-0.1, 2.1],
)
def test_invalid_llm_temperature(temperature):
    with pytest.raises(ValidationError):
        Settings(
            llm_temperature=temperature,
        )


def test_settings_accept_extra_environment_values():
    settings = Settings(
        unrelated_variable="ignored",
    )

    assert settings.app_name