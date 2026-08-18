FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install .

COPY data/demo_pdfs ./data/demo_pdfs
COPY eval/corpus/source ./eval/corpus/source
COPY eval/results/latest ./eval/results/latest

RUN mkdir -p /app/data/vector_store && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["swe-google-rag", "demo", "--host", "0.0.0.0", "--port", "8000"]
