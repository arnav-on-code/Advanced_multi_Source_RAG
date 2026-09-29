class RAGException(Exception):
    """Base exception for the P1 RAG application."""

    code = "RAG_ERROR"

    def __init__(
        self,
        message: str,
    ) -> None:
        super().__init__(message)
        self.message = message


class ConfigurationError(RAGException):
    """Raised when application configuration is invalid."""

    code = "CONFIGURATION_ERROR"


class LoaderError(RAGException):
    """Raised when a source loader fails."""

    code = "LOADER_ERROR"

    def __init__(
        self,
        message: str,
        *,
        source_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.source_type = source_type


class ProcessingError(RAGException):
    """Raised when document processing fails."""

    code = "PROCESSING_ERROR"


class EmbeddingError(RAGException):
    """Raised when document or query embedding fails."""

    code = "EMBEDDING_ERROR"


class VectorStoreError(RAGException):
    """Raised when vector-store operations fail."""

    code = "VECTOR_STORE_ERROR"


class RetrievalError(RAGException):
    """Raised when document retrieval fails."""

    code = "RETRIEVAL_ERROR"


class RerankingError(RAGException):
    """Raised when document re-ranking fails."""

    code = "RERANKING_ERROR"


class LLMError(RAGException):
    """Raised when LLM generation fails."""

    code = "LLM_ERROR"


class UnsupportedSourceError(RAGException):
    """Raised when an unsupported source type is requested."""

    code = "UNSUPPORTED_SOURCE"

    def __init__(
        self,
        source_type: str,
    ) -> None:
        self.source_type = source_type

        super().__init__(
            f"Unsupported source type: {source_type}"
        )