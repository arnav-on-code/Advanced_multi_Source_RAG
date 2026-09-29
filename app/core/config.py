from functools import lru_cache
from typing import Literal

from pydantic import (
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


Environment = Literal[
    "development",
    "testing",
    "staging",
    "production",
]

LLMProvider = Literal[
    "ollama",
    "openai",
    "groq",
    "google",
]


class Settings(BaseSettings):
    """
    Centralized application configuration.

    Configuration is loaded from environment variables
    and the .env file.

    Environment variables take precedence over .env values.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        validate_default=True,
    )

    # =========================================================
    # Application
    # =========================================================

    app_name: str = (
        "P1 Advanced Multi-Source RAG API"
    )

    app_version: str = "1.0.0"

    environment: Environment = "development"

    debug: bool = False

    # =========================================================
    # API
    # =========================================================

    api_host: str = "0.0.0.0"

    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
    )

    # =========================================================
    # Supported Sources
    # =========================================================

    supported_sources: tuple[str, ...] = (
        "pdf",
        "url",
        "youtube",
        "docx",
        "csv",
        "txt",
        "json",
    )

    # =========================================================
    # Embeddings
    # =========================================================

    embedding_model: str = (
        "BAAI/bge-small-en-v1.5"
    )

    embedding_device: str = "cpu"

    # =========================================================
    # ChromaDB
    # =========================================================

    chroma_persist_directory: str = (
        "data/chroma"
    )

    chroma_collection_name: str = (
        "p1_rag_documents"
    )

    # =========================================================
    # Chunking
    # =========================================================

    chunk_size: int = Field(
        default=800,
        ge=100,
        le=10000,
    )

    chunk_overlap: int = Field(
        default=120,
        ge=0,
    )

    # =========================================================
    # Retrieval
    # =========================================================

    retrieval_top_k: int = Field(
        default=5,
        ge=1,
        le=100,
    )

    retrieval_candidate_k: int = Field(
        default=15,
        ge=1,
        le=200,
    )

    vector_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
    )

    bm25_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
    )

    # =========================================================
    # Re-ranking
    # =========================================================

    reranker_model: str = (
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    reranker_device: str = "cpu"

    reranker_top_k: int = Field(
        default=5,
        ge=1,
        le=100,
    )

    # =========================================================
    # LLM
    # =========================================================

    llm_provider: LLMProvider = "ollama"

    llm_model: str = "llama3.2"

    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
    )

    # =========================================================
    # Provider Secrets
    # =========================================================

    openai_api_key: SecretStr | None = Field(
        default=None,
        repr=False,
    )

    groq_api_key: SecretStr | None = Field(
        default=None,
        repr=False,
    )

    google_api_key: SecretStr | None = Field(
        default=None,
        repr=False,
    )

    # =========================================================
    # Logging
    # =========================================================

    log_level: str = "INFO"

    # =========================================================
    # Security / CORS
    # =========================================================

    allowed_origins: str = "*"

    # =========================================================
    # Validators
    # =========================================================

    @field_validator("log_level")
    @classmethod
    def validate_log_level(
        cls,
        value: str,
    ) -> str:
        """Normalize and validate logging level."""

        value = value.strip().upper()

        valid_levels = {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }

        if value not in valid_levels:
            raise ValueError(
                f"Invalid log level: {value}"
            )

        return value

    @field_validator(
        "embedding_device",
        "reranker_device",
    )
    @classmethod
    def normalize_device(
        cls,
        value: str,
    ) -> str:
        """Normalize compute device configuration."""

        return value.strip().lower()

    @field_validator("allowed_origins")
    @classmethod
    def normalize_origins(
        cls,
        value: str,
    ) -> str:
        """Normalize comma-separated CORS origins."""

        return ",".join(
            origin.strip()
            for origin in value.split(",")
            if origin.strip()
        )

    @model_validator(mode="after")
    def validate_configuration(self) -> "Settings":
        """Validate relationships between configuration values."""

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        if (
            self.vector_weight == 0
            and self.bm25_weight == 0
        ):
            raise ValueError(
                "vector_weight and bm25_weight "
                "cannot both be zero."
            )

        if self.retrieval_candidate_k < self.retrieval_top_k:
            raise ValueError(
                "retrieval_candidate_k must be greater than "
                "or equal to retrieval_top_k."
            )

        if self.reranker_top_k > self.retrieval_candidate_k:
            raise ValueError(
                "reranker_top_k cannot exceed "
                "retrieval_candidate_k."
            )

        return self

    # =========================================================
    # Derived Configuration
    # =========================================================

    @property
    def source_types(self) -> tuple[str, ...]:
        """Return normalized supported source types."""

        return tuple(
            source.strip().lower()
            for source in self.supported_sources
            if source.strip()
        )

    @property
    def cors_origins(self) -> list[str]:
        """Return CORS origins as a list."""

        if self.allowed_origins == "*":
            return ["*"]

        return [
            origin.strip()
            for origin in self.allowed_origins.split(",")
            if origin.strip()
        ]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the application settings singleton.

    The cached instance prevents repeated .env parsing
    and configuration construction.
    """

    return Settings()


settings = get_settings()