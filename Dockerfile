# ==============================================================================
# Enterprise Employee RAG & Multi-Agent System Dockerfile
# Base: Python 3.11 Slim (Debian)
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATABASE_PATH=/app/data/employee_management.db

WORKDIR /app

# Install minimal OS dependencies for health checking
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and assets
COPY config.yaml /app/
COPY sample_policy.md /app/
COPY database.py /app/
COPY vector_db.py /app/
COPY litellm_client.py /app/
COPY main.py /app/
COPY mcp_server.py /app/
COPY agents/ /app/agents/
COPY static/ /app/static/
COPY data/ /app/data/

# Create persistent storage directories for SQLite and ChromaDB
RUN mkdir -p /app/data /app/vector_store

# Expose FastAPI application port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start FastAPI application with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
