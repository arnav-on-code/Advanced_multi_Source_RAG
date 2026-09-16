from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


app = FastAPI(
    title= " Advanced MultiSource RAG API",
    description = "PDF, URL, YOUTUBE, DOCX, CSV, TXT, and JSON data sources are supported. The API allows for advanced retrieval-augmented generation (RAG) capabilities, enabling users to query and retrieve information from multiple sources seamlessly.",
    version = "1.0.0",
)



class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    use_reranking: bool = True

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Query cannot be empty")

        return value


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]

class SourceType(str, Enum):
    PDF = "pdf"
    URL = "url"
    YOUTUBE = "youtube"
    DOCX = "docx"
    CSV = "csv"
    TXT = "txt"
    JSON = "json"

class IngestRequest(BaseModel):
    source_type: SourceType
    source: str = Field(min_length=1)







@app.get("/")
async def root():
    return {
        "project": "Advanced Multi-Source RAG API",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }


@app.post("/ingest")
async def ingest(request: IngestRequest):
    """
    Ingest PDF, URL, or YouTube content.

    Actual loader/vector-store logic will be connected here.
    """
    supported_sources = {source.value for source in SourceType}

    if request.source_type.value not in supported_sources:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported source type. Use one of: {supported_sources}",
        )

    return {
        "message": "Source accepted",
        "source_type": request.source_type,
        "source": request.source,
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Main RAG endpoint.

    Pipeline:
    Query
      -> Hybrid Search
      -> Re-ranking
      -> LLM
      -> Answer + Citations
    """

    # RAG pipeline will be connected here.
    #
    # Example:
    # result = rag_pipeline(
    #     query=request.query,
    #     top_k=request.top_k,
    #     use_reranking=request.use_reranking,
    # )

    return QueryResponse(
        answer="RAG pipeline not connected yet.",
        sources=[],
    )


@app.get("/docs-info")
async def docs_info():
    return {
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "POST /ingest": "Ingest PDF, URL, YouTube, DOCX, CSV, TXT, or JSON source",
            "POST /query": "Query the RAG system",
            "GET /docs": "Swagger API documentation",
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )