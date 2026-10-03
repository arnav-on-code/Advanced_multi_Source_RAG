from fastapi import FastAPI

from app.api import health, ingest, query
from app.core.config import settings


app = FastAPI(
    title=settings.app_name,
    description=(
        "PDF, URL, YOUTUBE, DOCX, CSV, TXT, and JSON data sources are "
        "supported. The API provides advanced retrieval-augmented "
        "generation (RAG) capabilities for querying information from "
        "multiple sources."
    ),
    version=settings.app_version,
)


# ============================================================
# ROUTERS
# ============================================================

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)


# ============================================================
# ROOT
# ============================================================

@app.get("/", tags=["Root"])
async def root() -> dict[str, str]:
    return {
        "project": "Advanced Multi-Source RAG API",
        "status": "running",
        "version": settings.app_version,
    }


# ============================================================
# DOCUMENTATION INFO
# ============================================================

@app.get("/docs-info", tags=["Root"])
async def docs_info() -> dict:
    return {
        "project": settings.app_name,
        "version": settings.app_version,
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /health/ready": "Readiness check",
            "POST /ingest/": (
                "Ingest PDF, URL, YouTube, DOCX, CSV, TXT, or JSON source"
            ),
            "POST /query": "Query the RAG system",
            "GET /docs": "Swagger API documentation",
        },
    }


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )