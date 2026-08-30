# Build context = REPOSITORY ROOT (so the single-file frontend can be copied in).
# On Render: Root Directory = (empty) and Dockerfile Path = backend/Dockerfile.
# Locally: docker-compose uses context: . with dockerfile: backend/Dockerfile.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FRONTEND_DIR=/app/frontend

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Only these paths are copied into the image: no .env, no tests, no .git,
# no secrets (explicit COPY, never `COPY . .`).
COPY backend/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY backend/main.py .
COPY frontend/ ./frontend/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Render provides a dynamic $PORT; bind to it, with a local fallback to 8000.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8000}/health/live" || exit 1

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]