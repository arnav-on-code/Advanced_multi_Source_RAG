# =========================================================
# Advanced Multi-Source RAG API
# =========================================================

FROM python:3.12-slim

# Prevent Python from creating .pyc files
# and ensure logs appear immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# ---------------------------------------------------------
# System dependencies
# ---------------------------------------------------------

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# Python dependencies
# ---------------------------------------------------------

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------
# Application
# ---------------------------------------------------------

COPY app ./app
COPY data ./data

# Create required runtime directories
RUN mkdir -p \
    /app/data/raw \
    /app/data/processed \
    /app/data/sample \
    /app/chroma_data

# ---------------------------------------------------------
# Non-root user
# ---------------------------------------------------------

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

# ---------------------------------------------------------
# API port
# ---------------------------------------------------------

EXPOSE 8000

# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

HEALTHCHECK --interval=30s \
    --timeout=10s \
    --start-period=20s \
    --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# ---------------------------------------------------------
# Start FastAPI
# ---------------------------------------------------------

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]