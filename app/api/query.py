from fastapi import APIRouter, Depends, HTTPException

from app.core.logging import get_logger
from app.models.requests import QueryRequest
from app.models.responses import QueryResponse, QuerySource
from app.pipeline.rag_pipeline import RAGPipeline


router = APIRouter(
    prefix="/query",
    tags=["RAG Query"],
)

logger = get_logger(__name__)


def get_rag_pipeline() -> RAGPipeline:
    """
    Return the application's configured RAG pipeline.

    This function is overridden during application startup
    or testing through FastAPI dependency injection.
    """

    raise RuntimeError(
        "RAG pipeline dependency has not been configured."
    )


@router.post(
    "",
    response_model=QueryResponse,
)
async def query_rag(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(
        get_rag_pipeline
    ),
) -> QueryResponse:
    """
    Query the multi-source RAG system.

    Query
      ↓
    Hybrid Search
      ↓
    Re-ranking
      ↓
    LLM
      ↓
    Answer + Citations
    """

    try:
        result = pipeline.invoke(
            query=request.query,
            top_k=request.top_k,
            use_reranking=request.use_reranking,
            metadata_filter=(
                request.metadata_filter.model_dump(
                    exclude_none=True
                )
                if request.metadata_filter
                else None
            ),
        )

        sources = [
            QuerySource(
                **source
            )
            for source in result.sources
        ]

        return QueryResponse(
            answer=result.answer,
            sources=sources,
            retrieved_documents=(
                result.retrieved_documents
            ),
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
            "Unexpected RAG query error."
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to process the query.",
        ) from exc