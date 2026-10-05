from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import settings
from app.core.logging import get_logger
from app.embeddings.embedder import get_embedding_model
from app.llm.generator import LLMGenerator
from app.models.requests import QueryRequest
from app.models.responses import QueryResponse, QuerySource
from app.pipeline.rag_pipeline import RAGPipeline
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.hybrid_search import HybridSearcher
from app.retrieval.reranker import Reranker
from app.retrieval.vector_search import VectorSearcher

from langchain_core.documents import Document

router = APIRouter(
    prefix="/query",
    tags=["RAG Query"],
)

logger = get_logger(__name__)


def _create_llm():
    """Create the configured LangChain chat model."""

    provider = settings.llm_provider

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            base_url=settings.ollama_base_url,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if settings.openai_api_key is None:
            raise RuntimeError(
                "OPENAI_API_KEY is required when "
                "llm_provider='openai'."
            )

        return ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key.get_secret_value(),
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        if settings.groq_api_key is None:
            raise RuntimeError(
                "GROQ_API_KEY is required when "
                "llm_provider='groq'."
            )

        return ChatGroq(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.groq_api_key.get_secret_value(),
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if settings.google_api_key is None:
            raise RuntimeError(
                "GOOGLE_API_KEY is required when "
                "llm_provider='google'."
            )

        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            google_api_key=settings.google_api_key.get_secret_value(),
        )

    raise RuntimeError(
        f"Unsupported LLM provider: {provider}"
    )


def _load_documents_for_bm25(
    vector_searcher: VectorSearcher,
) -> list[Document]:
    """
    Load all persisted documents from ChromaDB
    for BM25 indexing.
    """

    return vector_searcher.vectorstore.get_documents()


@lru_cache(maxsize=1)
def get_rag_pipeline() -> RAGPipeline:
    """
    Build and cache the application's real RAG pipeline.

    Expensive components are initialized only once per process.
    """

    try:
        # ---------------------------------------------------------
        # Vector search
        # ---------------------------------------------------------

        vector_searcher = VectorSearcher(
            persist_directory=settings.chroma_persist_directory,
            collection_name=settings.chroma_collection_name,
            embedding_model=settings.embedding_model,
        )

        # ---------------------------------------------------------
        # BM25 search
        # ---------------------------------------------------------

        documents = _load_documents_for_bm25(
            vector_searcher
        )

        if not documents:
            raise RuntimeError(
                "No documents are available in ChromaDB. "
                "Ingest a source before querying."
            )

        bm25_retriever = BM25Retriever(
            documents=documents
        )

        # ---------------------------------------------------------
        # Hybrid retrieval
        # ---------------------------------------------------------

        hybrid_searcher = HybridSearcher(
            vector_searcher=vector_searcher,
            bm25_retriever=bm25_retriever,
            vector_weight=settings.vector_weight,
            bm25_weight=settings.bm25_weight,
        )

        # ---------------------------------------------------------
        # Re-ranking
        # ---------------------------------------------------------

        reranker = Reranker(
            model_name=settings.reranker_model,
            device=settings.reranker_device,
        )

        # ---------------------------------------------------------
        # LLM
        # ---------------------------------------------------------

        llm_generator = LLMGenerator(
            llm=_create_llm(),
        )

        # ---------------------------------------------------------
        # Complete RAG pipeline
        # ---------------------------------------------------------

        return RAGPipeline(
            hybrid_searcher=hybrid_searcher,
            reranker=reranker,
            llm_generator=llm_generator,
            retrieval_top_k=settings.retrieval_top_k,
            candidate_k=settings.retrieval_candidate_k,
            reranker_top_k=settings.reranker_top_k,
        )

    except Exception as exc:
        logger.exception(
            "Failed to initialize RAG pipeline: %s",
            exc,
        )

        raise RuntimeError(
            "RAG pipeline dependency has not been configured."
        ) from exc


@router.post(
    "",
    response_model=QueryResponse,
)
async def query_rag(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> QueryResponse:
    """Query the multi-source RAG system."""

    try:
        metadata_filter = (
            request.metadata_filter.model_dump(
                exclude_none=True
            )
            if request.metadata_filter
            else None
        )

        result = pipeline.invoke(
            query=request.query,
            top_k=request.top_k,
            use_reranking=request.use_reranking,
            metadata_filter=metadata_filter,
        )

        sources = [
            QuerySource(**source)
            for source in result.sources
        ]

        return QueryResponse(
            answer=result.answer,
            sources=sources,
            retrieved_documents=result.retrieved_documents,
        )

    except ValueError as exc:
        logger.warning(
            "Invalid RAG query: %s",
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        logger.error(
            "RAG pipeline unavailable: %s",
            exc,
        )

        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not available.",
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected RAG query error: %s",
            exc,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to process the query.",
        ) from exc