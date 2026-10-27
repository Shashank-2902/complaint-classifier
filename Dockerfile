FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/deploy/model

WORKDIR /app

RUN useradd --create-home --uid 10001 app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api ./api
COPY deploy/model ./deploy/model

USER app
EXPOSE 8000

# The health check fails (exit 1) when /health does not answer 200, for example when the model is missing.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
