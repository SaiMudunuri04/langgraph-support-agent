# ---- builder ----
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---- runtime ----
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY support_agent/ ./support_agent/
COPY data/ ./data/
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser
ENV MOCK_MODE=true \
    DATA_DIR=/app/data \
    PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "support_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
